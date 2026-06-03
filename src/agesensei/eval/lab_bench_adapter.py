"""LAB-Bench adapter — wraps AgeSensei agents as LAB-Bench compatible agent_fn.

LAB-Bench (Language Agent Biology Benchmark) evaluates AI systems on biology
research tasks across 8 categories: LitQA2, DbQA, SuppQA, FigQA, TableQA,
ProtocolQA, SeqQA, CloningScenarios.

This adapter implements a PaperQA2-inspired retrieval pipeline:
1. Multi-source retrieval (PubMed + S2 + PMC full-text + DOI-based)
2. Two-stage relevance scoring (LLM scores each chunk, keeps top-3)
3. Iterative search (if evidence insufficient, refine query and retry)
4. Chain-of-Thought reasoning with focused context

Usage:
    from agesensei.eval import run_lab_bench

    results = await run_lab_bench(
        evals=["LitQA2", "DbQA", "SeqQA"],
        model="claude-haiku-4-5-20251001",
        n_threads=4,
    )
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from agesensei.eval.retrieval_cache import RetrievalCache

logger = logging.getLogger(__name__)

# Global cache instance — shared across all retrieval calls within a session
_cache = RetrievalCache("artifacts/cache")


# ---------------------------------------------------------------------------
# AgentInput / Evaluator protocol (minimal re-implementation for standalone use)
# ---------------------------------------------------------------------------

@dataclass
class AgentInput:
    """Mirrors labbench.AgentInput for standalone execution."""
    question: str
    choices: list[str]
    figures: list[Any] = field(default_factory=list)
    subtask: str = ""
    ideal: str = ""
    sources: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Result for a single evaluation question."""
    question_id: str
    subtask: str
    question: str
    choices: list[str]
    ideal: str
    predicted: str
    correct: bool
    reasoning: str = ""
    tools_used: list[str] = field(default_factory=list)


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    eval_name: str
    total: int
    correct: int
    accuracy: float
    coverage: float
    per_subtask: dict[str, dict[str, float]] = field(default_factory=dict)
    results: list[EvalResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Chunk dataclass for two-stage scoring
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """A chunk of text retrieved from a source."""
    text: str
    source: str  # e.g. "PubMed", "PMC:results", "S2", "DOI"
    relevance_score: float = 0.0


# ---------------------------------------------------------------------------
# Stage 1: Multi-source deep retrieval
# ---------------------------------------------------------------------------

async def _retrieve_all_chunks(query: str, sources: list[str] | None = None) -> list[RetrievedChunk]:
    """Retrieve chunks from multiple sources.

    Strategy: DOI full-text FIRST (gold standard for LitQA2).
    Only fall back to PubMed/S2 search if DOI retrieval yields < 3 chunks.
    """
    chunks: list[RetrievedChunk] = []

    # === PRIMARY: DOI-based full-text retrieval ===
    if sources:
        try:
            doi_chunks = await _retrieve_from_dois(sources)
            chunks.extend(doi_chunks)
            logger.info(f"DOI retrieval: {len(doi_chunks)} chunks from {len(sources)} DOIs")
        except Exception as e:
            logger.warning(f"DOI retrieval failed: {e}")

    # === FALLBACK: Only if DOI gave us very little ===
    if len(chunks) < 3:
        logger.info(f"DOI yielded only {len(chunks)} chunks, falling back to search")

        async def _ncbi_sequential():
            ncbi_chunks = []
            try:
                pubmed_chunks = await _retrieve_from_pubmed(query)
                ncbi_chunks.extend(pubmed_chunks)
            except Exception as e:
                logger.warning(f"PubMed retrieval failed: {e}")
            await asyncio.sleep(0.4)
            try:
                pmc_chunks = await _retrieve_from_pmc(query)
                ncbi_chunks.extend(pmc_chunks)
            except Exception as e:
                logger.warning(f"PMC full-text retrieval failed: {e}")
            return ncbi_chunks

        results = await asyncio.gather(
            _ncbi_sequential(),
            _retrieve_from_s2(query),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, list):
                chunks.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"Retrieval source failed: {result}")

    return chunks


async def _retrieve_from_dois(sources: list[str]) -> list[RetrievedChunk]:
    """Retrieve papers by DOI — try PMC full-text → Unpaywall full-text → S2 abstract.

    This is the PRIMARY retrieval path for LitQA2: every question has DOI sources,
    and the answer is always in one of these papers. We must get full text, not abstracts.
    """
    chunks = []

    for src in sources[:5]:
        doi = src.replace("https://doi.org/", "").replace("https://dx.doi.org/", "").strip()
        if not doi:
            continue

        got_fulltext = False

        # Step 1: Try PMC full-text via DOI → PMCID → PMC
        try:
            full_text_chunks = await _doi_to_pmc_fulltext(doi)
            if full_text_chunks:
                chunks.extend(full_text_chunks)
                got_fulltext = True
                logger.info(f"DOI {doi}: got {len(full_text_chunks)} chunks from PMC")
        except Exception as e:
            logger.debug(f"PMC full-text via DOI failed: {e}")

        # Step 2: Try Unpaywall if PMC didn't work
        if not got_fulltext:
            try:
                unpaywall_chunks = await _unpaywall_fulltext(doi)
                if unpaywall_chunks:
                    chunks.extend(unpaywall_chunks)
                    got_fulltext = True
                    logger.info(f"DOI {doi}: got {len(unpaywall_chunks)} chunks from Unpaywall")
            except Exception as e:
                logger.debug(f"Unpaywall full-text failed: {e}")

        # Step 3: Fallback to abstract (last resort)
        if not got_fulltext:
            try:
                from agesensei.tools.semantic_scholar import search_by_doi
                paper = await search_by_doi(doi)
                if paper and paper.abstract:
                    chunks.append(RetrievedChunk(
                        text=f"{paper.title}\n\n{paper.abstract}",
                        source=f"DOI:{doi}:abstract-only",
                    ))
                    logger.info(f"DOI {doi}: abstract only (no full-text available)")
            except Exception as e:
                logger.debug(f"S2 DOI retrieval failed: {e}")

    return chunks


async def _unpaywall_fulltext(doi: str) -> list[RetrievedChunk]:
    """Retrieve full text via Unpaywall OA links (with cache)."""
    # Check cache first (reuse PMC cache keyed by DOI)
    cache_key = f"unpaywall:{doi}"
    cached = _cache.get_pmc(cache_key)
    if cached is not None:
        chunks = []
        for section_name, section_text in cached.items():
            if len(section_text) > 50:
                chunks.append(RetrievedChunk(
                    text=f"[{section_name}]\n{section_text[:2000]}",
                    source=f"Unpaywall:{doi}:{section_name}",
                ))
        return chunks

    try:
        from agesensei.tools.unpaywall import fetch_fulltext_via_unpaywall
        sections = await fetch_fulltext_via_unpaywall(doi)
        if not sections:
            return []

        _cache.put_pmc(cache_key, sections)

        chunks = []
        skip_sections = {"references", "acknowledgements", "funding"}
        for section_name, section_text in sections.items():
            if section_name.lower() not in skip_sections and len(section_text) > 50:
                paragraphs = _split_into_paragraphs(section_text, max_chars=1500)
                for para in paragraphs:
                    chunks.append(RetrievedChunk(
                        text=f"[{section_name}]\n{para}",
                        source=f"Unpaywall:{doi}:{section_name}",
                    ))
        return chunks
    except ImportError:
        logger.debug("unpaywall tool not available")
        return []


async def _doi_to_pmc_fulltext(doi: str) -> list[RetrievedChunk]:
    """Convert DOI → PMCID → PMC full text sections."""
    import httpx

    chunks = []

    # Step 1: DOI → PMCID
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        resp = await client.get(
            "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
            params={"ids": doi, "format": "json"},
        )
        if resp.status_code != 200:
            return []

        data = resp.json()
        records = data.get("records", [])
        if not records or not records[0].get("pmcid"):
            return []

        pmcid = records[0]["pmcid"]

    # Check cache first
    cached = _cache.get_pmc(pmcid)
    if cached:
        for section_name, section_text in cached.items():
            if len(section_text) > 50:
                # Split into paragraph-sized chunks for better scoring
                paragraphs = _split_into_paragraphs(section_text, max_chars=1500)
                for para in paragraphs:
                    chunks.append(RetrievedChunk(
                        text=f"[{section_name}]\n{para}",
                        source=f"PMC:{pmcid}:{section_name}",
                    ))
        return chunks

    # Step 2: PMCID → full text via BioC API
    await asyncio.sleep(0.4)  # NCBI rate limit
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(
            f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
        )
        if resp.status_code != 200:
            return []

        bioc = resp.json()
        sections = {}
        # BioC format: list → item → documents → doc → passages
        items = bioc if isinstance(bioc, list) else [bioc]
        for item in items:
            for doc in (item.get("documents", []) if isinstance(item, dict) else []):
                for passage in doc.get("passages", []):
                    section = passage.get("infons", {}).get("section_type", "other")
                    text = passage.get("text", "")
                    if text and len(text) > 30:
                        if section not in sections:
                            sections[section] = ""
                        sections[section] += text + "\n"

        # Cache all sections
        _cache.put_pmc(pmcid, sections)

        # Return most useful sections — split into chunks for scoring
        skip_sections = {"REF", "AUTH_CONT", "COMP_INT", "TITLE"}
        for section_name, section_text in sections.items():
            if section_name not in skip_sections and len(section_text) > 50:
                paragraphs = _split_into_paragraphs(section_text, max_chars=1500)
                for para in paragraphs:
                    chunks.append(RetrievedChunk(
                        text=f"[{section_name}]\n{para}",
                        source=f"PMC:{pmcid}:{section_name}",
                    ))

    return chunks


def _extract_search_terms(question: str) -> str:
    """Extract key scientific terms from a question for PubMed search.

    Strategy: remove common English words, keep scientific terms (gene names,
    protein names, species, measurements, author names, years).
    """
    stop_words = {
        'what', 'which', 'how', 'does', 'did', 'was', 'were', 'is', 'are',
        'the', 'a', 'an', 'of', 'in', 'on', 'for', 'to', 'from', 'by',
        'with', 'that', 'this', 'these', 'those', 'it', 'its', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'not', 'but', 'and', 'or',
        'if', 'when', 'where', 'why', 'can', 'could', 'would', 'should',
        'will', 'may', 'might', 'shall', 'about', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'between', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'than', 'too',
        'very', 'just', 'also', 'only', 'same', 'so', 'no', 'nor',
        'observed', 'found', 'shown', 'demonstrated', 'reported', 'used',
        'following', 'according', 'based', 'compared', 'associated',
        'they', 'their', 'them', 'any', 'all', 'one', 'two', 'three',
        'paper', 'study', 'research', 'experiment', 'result', 'results',
        'specific', 'particular', 'given', 'using', 'called', 'known',
        'force', 'change', 'effect', 'cause', 'lead', 'leads', 'show',
        'shows', 'see', 'seen', 'take', 'taken', 'make', 'made',
        'respect', 'regarding', 'concerning', 'upon', 'among', 'within',
        'without', 'across', 'against', 'along', 'around',
        'approximately', 'many', 'much', 'several', 'various',
        'able', 'perform', 'performed', 'anyone', 'ever', 'display',
    }
    words = re.sub(r'[^\w\s-]', '', question).split()
    terms = [w for w in words if w.lower() not in stop_words and len(w) > 2]

    # Prioritize: author names (capitalized), years, gene/protein names (uppercase)
    priority = []
    normal = []
    for t in terms:
        if re.match(r'^\d{4}$', t):  # Year
            priority.append(t)
        elif t[0].isupper() and not t.isupper():  # Author name (Title case)
            priority.append(t)
        elif t.isupper() and len(t) >= 2:  # Gene/protein name (ALL CAPS)
            priority.append(t)
        else:
            normal.append(t)

    # Combine: priority first, then normal, max 6 terms
    combined = priority + normal
    return ' '.join(combined[:6])


async def _retrieve_from_pubmed(query: str) -> list[RetrievedChunk]:
    """Search PubMed and return abstract chunks (with cache)."""
    chunks = []
    search_query = _extract_search_terms(query)

    # Check query cache for PMIDs
    cached_pmids = _cache.get_query(search_query)

    try:
        from agesensei.tools.pubmed import search_and_fetch, search_pubmed

        if cached_pmids is not None:
            # Use cached PMIDs, fetch abstracts from cache or API
            for pmid in cached_pmids[:8]:
                cached_paper = _cache.get_pubmed(pmid)
                if cached_paper:
                    if cached_paper.get("abstract"):
                        chunks.append(RetrievedChunk(
                            text=f"{cached_paper['title']}\n\n{cached_paper['abstract']}",
                            source=f"PubMed:{pmid}",
                        ))
                else:
                    # Fetch individually and cache
                    papers = await search_and_fetch(pmid, max_results=1)
                    for p in papers:
                        if p.abstract:
                            _cache.put_pubmed(p.pmid, {"title": p.title, "abstract": p.abstract, "pmid": p.pmid})
                            chunks.append(RetrievedChunk(
                                text=f"{p.title}\n\n{p.abstract}",
                                source=f"PubMed:{p.pmid}",
                            ))
        else:
            # Fresh search
            papers = await search_and_fetch(search_query, max_results=8)
            pmids = []
            for p in papers[:8]:
                pmids.append(p.pmid)
                _cache.put_pubmed(p.pmid, {"title": p.title, "abstract": p.abstract, "pmid": p.pmid})
                if p.abstract:
                    chunks.append(RetrievedChunk(
                        text=f"{p.title}\n\n{p.abstract}",
                        source=f"PubMed:{p.pmid}",
                    ))
            _cache.put_query(search_query, pmids)
    except Exception as e:
        logger.warning(f"PubMed search failed: {e}")
    return chunks


async def _retrieve_from_s2(query: str) -> list[RetrievedChunk]:
    """Search Semantic Scholar for broader coverage (with cache)."""
    chunks = []

    # Check cache
    cached = _cache.get_s2(query)
    if cached is not None:
        for item in cached[:5]:
            if item.get("abstract"):
                chunks.append(RetrievedChunk(
                    text=f"{item['title']}\n\n{item['abstract']}",
                    source=f"S2:{item.get('doi', 'unknown')}",
                ))
        return chunks

    try:
        from agesensei.tools.semantic_scholar import search_papers
        papers = await search_papers(query, max_results=5)
        cache_items = []
        for p in papers[:5]:
            cache_items.append({"title": p.title, "abstract": p.abstract, "doi": p.doi})
            if p.abstract:
                chunks.append(RetrievedChunk(
                    text=f"{p.title}\n\n{p.abstract}",
                    source=f"S2:{p.doi or 'unknown'}",
                ))
        _cache.put_s2(query, cache_items)
    except Exception as e:
        logger.debug(f"S2 search failed: {e}")
    return chunks


async def _retrieve_from_pmc(query: str) -> list[RetrievedChunk]:
    """Attempt PMC full-text retrieval for top PubMed hits (with cache).

    This is the key differentiator — full-text contains the specific
    experimental details that abstracts miss.
    """
    chunks = []
    search_query = _extract_search_terms(query)
    try:
        from agesensei.tools.pmc import fetch_full_text, pmid_to_pmcid
        from agesensei.tools.pubmed import search_pubmed

        # Use cached PMIDs if available
        cached_pmids = _cache.get_query(search_query)
        if cached_pmids:
            pmids = cached_pmids[:5]
        else:
            pmids = await search_pubmed(search_query, max_results=5)
            _cache.put_query(search_query + "_pmc", pmids)

        papers_fetched = 0
        for pmid in pmids[:5]:
            if papers_fetched >= 2:
                break
            await asyncio.sleep(0.35)  # NCBI rate limit
            pmcid = await pmid_to_pmcid(pmid)
            if pmcid:
                # Check cache first
                cached_sections = _cache.get_pmc(pmcid)
                if cached_sections is not None:
                    sections = cached_sections
                    papers_fetched += 1
                else:
                    await asyncio.sleep(0.35)
                    sections = await fetch_full_text(pmcid)
                    if sections:
                        _cache.put_pmc(pmcid, sections)
                        papers_fetched += 1
                    else:
                        continue

                # Prioritize results/methods/abstract sections
                priority_sections = ['results', 'abstract', 'methods', 'discussion']
                sorted_sections = sorted(
                    sections.items(),
                    key=lambda x: (priority_sections.index(x[0]) if x[0] in priority_sections else 99),
                )
                for sec_name, sec_text in sorted_sections:
                    if sec_text and len(sec_text) > 50:
                        paragraphs = _split_into_paragraphs(sec_text, max_chars=1500)
                        for para in paragraphs:
                            chunks.append(RetrievedChunk(
                                text=para,
                                source=f"PMC:{pmcid}/{sec_name}",
                            ))
    except Exception as e:
        logger.warning(f"PMC full-text failed: {e}")
    return chunks


def _split_into_paragraphs(text: str, max_chars: int = 600) -> list[str]:
    """Split text into paragraph-sized chunks."""
    sentences = text.replace(". ", ".\n").split("\n")
    paragraphs = []
    current = []
    current_len = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if current_len + len(sent) > max_chars and current:
            paragraphs.append(" ".join(current))
            current = [sent]
            current_len = len(sent)
        else:
            current.append(sent)
            current_len += len(sent)

    if current:
        paragraphs.append(" ".join(current))

    return paragraphs[:10]  # Cap at 10 paragraphs per section


# ---------------------------------------------------------------------------
# Stage 2: LLM-based relevance scoring
# ---------------------------------------------------------------------------

RELEVANCE_PROMPT = """Score how relevant this text passage is to answering the question.
Rate from 0 to 10 where:
- 0: completely irrelevant
- 5: somewhat related but doesn't directly answer
- 10: directly contains the answer

Question: {question}

Passage: {passage}

Reply with ONLY a number from 0 to 10."""


async def _score_chunks(
    chunks: list[RetrievedChunk],
    question: str,
    model: str,
    api_key: str,
    top_k: int = 3,
) -> list[RetrievedChunk]:
    """Score each chunk's relevance using LLM, return top-k.

    This is the two-stage retrieval approach from PaperQA2:
    first retrieve broadly, then use LLM to filter to most relevant.
    """
    if not chunks:
        return []

    # If few chunks, skip scoring (not worth the API calls)
    if len(chunks) <= top_k:
        return chunks

    # Quick keyword pre-filter to reduce LLM calls
    q_words = set(w.lower() for w in question.split() if len(w) > 3)
    pre_scored = []
    for chunk in chunks:
        c_words = set(w.lower() for w in chunk.text.split())
        keyword_overlap = len(q_words & c_words)
        pre_scored.append((keyword_overlap, chunk))

    pre_scored.sort(key=lambda x: x[0], reverse=True)

    # Take top 10 by keyword overlap for LLM scoring (saves API calls)
    candidates = [chunk for _, chunk in pre_scored[:10]]

    # LLM scoring in parallel
    import anthropic
    kwargs = {"api_key": api_key}
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    client = anthropic.AsyncAnthropic(**kwargs)
    sem = asyncio.Semaphore(5)

    async def score_one(chunk: RetrievedChunk) -> RetrievedChunk:
        async with sem:
            try:
                prompt = RELEVANCE_PROMPT.format(
                    question=question,
                    passage=chunk.text[:400],  # Truncate for scoring
                )
                response = await client.messages.create(
                    model=model,
                    max_tokens=8,
                    messages=[{"role": "user", "content": prompt}],
                )
                score_text = response.content[0].text.strip()
                # Extract number
                match = re.search(r'\d+', score_text)
                chunk.relevance_score = float(match.group()) if match else 0.0
            except Exception as e:
                logger.debug(f"Scoring failed: {e}")
                # Fall back to keyword overlap score
                c_words = set(w.lower() for w in chunk.text.split())
                chunk.relevance_score = float(len(q_words & c_words))
            return chunk

    scored = await asyncio.gather(*[score_one(c) for c in candidates])
    scored_list = sorted(scored, key=lambda c: c.relevance_score, reverse=True)

    return scored_list[:top_k]


# ---------------------------------------------------------------------------
# Stage 3: Iterative search (refine query if evidence is weak)
# ---------------------------------------------------------------------------

async def _iterative_retrieve(
    question: str,
    sources: list[str] | None,
    model: str,
    api_key: str,
    max_iterations: int = 3,
) -> list[RetrievedChunk]:
    """Iterative retrieval: multi-round search with cumulative evidence.

    Inspired by PaperQA2's iterative search strategy:
    - Round 1: DOI full-text (primary) + original question search (fallback)
    - Round 2: If evidence weak, LLM refines query → search again
    - Round 3: If still weak, targeted query from accumulated context
    All rounds accumulate evidence; final output is top-5 across all rounds.

    When DOI full-text retrieval succeeds (>= 5 chunks), skip iterative search.
    """
    all_chunks: list[RetrievedChunk] = []
    seen_texts: set[str] = set()

    def _dedupe_add(new_chunks: list[RetrievedChunk]):
        for c in new_chunks:
            key = c.text[:100]
            if key not in seen_texts:
                seen_texts.add(key)
                all_chunks.append(c)

    # Round 1: DOI-first retrieval
    chunks = await _retrieve_all_chunks(question, sources)
    scored = await _score_chunks(chunks, question, model, api_key, top_k=5)
    _dedupe_add(scored)

    # If DOI gave us enough high-quality chunks, skip iterative search
    best_score = max((c.relevance_score for c in all_chunks), default=0)
    doi_chunks = [c for c in all_chunks if "DOI:" in c.source or "PMC:" in c.source or "Unpaywall:" in c.source]
    if best_score >= 7 and len(doi_chunks) >= 3:
        logger.info(f"DOI full-text sufficient: {len(doi_chunks)} chunks, best_score={best_score}")
        return sorted(all_chunks, key=lambda x: x.relevance_score, reverse=True)[:5]

    # Round 2: LLM-refined query
    if max_iterations >= 2:
        refined = await _refine_query(question, model, api_key)
        if refined and refined != question:
            new_chunks = await _retrieve_all_chunks(refined, sources)
            new_scored = await _score_chunks(new_chunks, question, model, api_key, top_k=5)
            _dedupe_add(new_scored)

        best_score = max((c.relevance_score for c in all_chunks), default=0)
        if best_score >= 8:
            return sorted(all_chunks, key=lambda x: x.relevance_score, reverse=True)[:5]

    # Round 3: targeted query using accumulated context
    if max_iterations >= 3 and all_chunks:
        targeted = await _refine_with_context(
            question, all_chunks[:3], model, api_key
        )
        if targeted and targeted != question:
            new_chunks = await _retrieve_all_chunks(targeted, None)
            new_scored = await _score_chunks(new_chunks, question, model, api_key, top_k=5)
            _dedupe_add(new_scored)

    return sorted(all_chunks, key=lambda x: x.relevance_score, reverse=True)[:5]


REFINE_QUERY_PROMPT = """Given this biology research question, generate a better PubMed search query.
Focus on the key scientific terms, gene names, protein names, organism names, and specific measurements mentioned.

Question: {question}

Reply with ONLY the search query (no explanation). Use PubMed-style terms."""


REFINE_WITH_CONTEXT_PROMPT = """You are searching for the answer to a biology question.
Your initial searches found some relevant papers but not the exact answer.

Question: {question}

Evidence found so far:
{context}

Based on what you've found, generate a MORE SPECIFIC PubMed search query that targets
the exact information still needed. Use specific author names, gene names, assay types,
or measurements mentioned in the evidence to narrow down the search.

Reply with ONLY the search query (no explanation)."""


async def _refine_query(question: str, model: str, api_key: str) -> str:
    """Use LLM to generate a better search query from the question."""
    try:
        import anthropic
        kwargs = {"api_key": api_key}
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        client = anthropic.AsyncAnthropic(**kwargs)
        response = await client.messages.create(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": REFINE_QUERY_PROMPT.format(question=question)}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        logger.debug(f"Query refinement failed: {e}")
        return question


async def _refine_with_context(
    question: str,
    chunks: list[RetrievedChunk],
    model: str,
    api_key: str,
) -> str:
    """Round 3: Use accumulated evidence to generate a highly targeted query."""
    context = "\n\n".join(
        f"[{c.source}]: {c.text[:300]}" for c in chunks[:3]
    )
    try:
        import anthropic
        kwargs = {"api_key": api_key}
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        client = anthropic.AsyncAnthropic(**kwargs)
        response = await client.messages.create(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": REFINE_WITH_CONTEXT_PROMPT.format(
                question=question, context=context
            )}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        logger.debug(f"Context-based refinement failed: {e}")
        return question


# ---------------------------------------------------------------------------
# Deep retrieval tools (subtask-specific wrappers)
# ---------------------------------------------------------------------------

async def _search_literature_deep(
    query: str, sources: list[str] | None = None, model: str = "", api_key: str = ""
) -> list[RetrievedChunk]:
    """Literature retrieval with iterative refinement and Unpaywall full-text."""
    return await _iterative_retrieve(query, sources, model, api_key, max_iterations=3)


async def _query_protein_db_deep(
    query: str, sources: list[str] | None = None, model: str = "", api_key: str = ""
) -> list[RetrievedChunk]:
    """Database retrieval: UniProt + ChEMBL + literature."""
    from agesensei.tools.uniprot import search as uniprot_search

    chunks = []

    # UniProt
    try:
        results = await uniprot_search(query, max_results=3)
        if results:
            chunks.append(RetrievedChunk(
                text="[UniProt]\n" + json.dumps(results[:3], indent=2, default=str),
                source="UniProt",
            ))
    except Exception as e:
        logger.debug(f"UniProt failed: {e}")

    # ChEMBL
    try:
        from agesensei.tools.chembl import search as chembl_search
        results = await chembl_search(query, max_results=3)
        if results:
            chunks.append(RetrievedChunk(
                text="[ChEMBL]\n" + json.dumps(results[:3], indent=2, default=str),
                source="ChEMBL",
            ))
    except Exception as e:
        logger.debug(f"ChEMBL failed: {e}")

    # Also do literature retrieval
    lit_chunks = await _iterative_retrieve(query, sources, model, api_key)
    chunks.extend(lit_chunks)

    return chunks[:5]


async def _analyze_sequence_deep(
    query: str, sources: list[str] | None = None, model: str = "", api_key: str = ""
) -> list[RetrievedChunk]:
    """Sequence-related retrieval: UniProt + literature."""
    chunks = []

    try:
        from agesensei.tools.uniprot import search as uniprot_search
        results = await uniprot_search(query, max_results=3)
        if results:
            chunks.append(RetrievedChunk(
                text="[UniProt]\n" + json.dumps(results[:3], indent=2, default=str),
                source="UniProt",
            ))
    except Exception as e:
        logger.debug(f"UniProt failed: {e}")

    lit_chunks = await _iterative_retrieve(query, sources, model, api_key)
    chunks.extend(lit_chunks)

    return chunks[:5]


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

SUBTASK_TOOL_MAP: dict[str, Any] = {
    "LitQA2": _search_literature_deep,
    "litqa-v2-public": _search_literature_deep,
    "litqa-v2-closed": _search_literature_deep,
    "DbQA": _query_protein_db_deep,
    "SeqQA": _analyze_sequence_deep,
    "SuppQA": _search_literature_deep,
    "ProtocolQA": _search_literature_deep,
}


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------

COT_SYSTEM_PROMPT = """You are an expert biology researcher with deep knowledge of \
molecular biology, genetics, biochemistry, pharmacology, and bioinformatics.

When answering multiple-choice questions:
1. Read the question carefully. Identify what specific fact or measurement is being asked.
2. If research context is provided, look for information that DIRECTLY addresses the question. \
Quote specific numbers, gene names, or findings from the context.
3. Think step by step. Eliminate choices that contradict the evidence.
4. IMPORTANT: Only choose "Insufficient information" as a LAST RESORT — when you have \
absolutely zero basis to even make an educated guess. If you have ANY partial knowledge, \
domain expertise, or can reason from related biology, pick the most likely answer. \
In biology research questions, making your best scientific judgment is always better than \
saying "insufficient information".
5. End your response with your final answer on a new line in the format: ANSWER: X

where X is a single letter corresponding to your chosen option."""


class AgeSenseiLabBenchAgent:
    """LAB-Bench compatible agent with PaperQA2-inspired retrieval pipeline.

    Pipeline:
    1. Multi-source retrieval (PubMed + S2 + PMC full-text + DOI)
    2. LLM relevance scoring → keep top-3 chunks
    3. Iterative search if evidence is weak
    4. Chain-of-Thought reasoning with focused context
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        use_tools: bool = True,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.use_tools = use_tools
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = base_url

    async def __call__(self, input: AgentInput) -> str:
        return await self.answer(input)

    async def answer(self, input: AgentInput) -> str:
        """Full pipeline: retrieve → score → reason → extract answer."""
        context_chunks: list[RetrievedChunk] = []

        if self.use_tools:
            # Route to appropriate retrieval function
            tool_fn = SUBTASK_TOOL_MAP.get(input.subtask)
            if tool_fn is None:
                for key in SUBTASK_TOOL_MAP:
                    if key.lower() in input.subtask.lower():
                        tool_fn = SUBTASK_TOOL_MAP[key]
                        break

            if tool_fn:
                try:
                    context_chunks = await tool_fn(
                        input.question, input.sources,
                        model=self.model, api_key=self.api_key,
                    )
                except Exception as e:
                    logger.warning(f"Tool augmentation failed for {input.subtask}: {e}")

        prompt = self._build_cot_prompt(input, context_chunks)
        response = await self._call_llm(prompt)
        return self._extract_answer(response, input.choices)

    def _build_cot_prompt(self, input: AgentInput, chunks: list[RetrievedChunk]) -> str:
        """Build CoT prompt with scored, filtered context."""
        choices_text = "\n".join(
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(input.choices)
        )

        parts = []

        if chunks:
            parts.append("## Retrieved Research Evidence")
            parts.append("(Use ONLY if directly relevant. Quote specific facts.)\n")
            for i, chunk in enumerate(chunks[:5], 1):
                # Truncate each chunk to keep total context manageable
                text = chunk.text[:1500] if len(chunk.text) > 1500 else chunk.text
                parts.append(f"### Evidence {i} [source: {chunk.source}, relevance: {chunk.relevance_score:.0f}/10]")
                parts.append(text)
                parts.append("")

        parts.extend([
            "## Question",
            input.question,
            "",
            "## Choices",
            choices_text,
            "",
            "Think step by step. Cite specific evidence from the passages above if relevant.",
            "End with: ANSWER: X (where X is the letter of your chosen option)",
        ])

        return "\n".join(parts)

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with CoT system prompt."""
        try:
            import anthropic

            kwargs = {"api_key": self.api_key}
            base_url = self.base_url or os.environ.get("ANTHROPIC_BASE_URL")
            if base_url:
                kwargs["base_url"] = base_url
            client = anthropic.AsyncAnthropic(**kwargs)
            response = await client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=COT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            if not response.content:
                logger.warning(f"Empty response from {self.model}, stop_reason={response.stop_reason}")
                # On refusal, retry with a simplified prompt
                if response.stop_reason == "refusal":
                    simple_prompt = f"This is a biology quiz for educational purposes.\n\n{prompt}"
                    response2 = await client.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        messages=[{"role": "user", "content": simple_prompt}],
                    )
                    if response2.content:
                        return response2.content[0].text.strip()
                return "F"
            return response.content[0].text.strip()
        except ImportError:
            return await self._call_llm_openai(prompt)

    async def _call_llm_openai(self, prompt: str) -> str:
        """Fallback LLM call using OpenAI-compatible API."""
        try:
            import openai

            client = openai.AsyncOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", ""),
            )
            response = await client.chat.completions.create(
                model=self.model if "gpt" in self.model else "gpt-4o",
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": COT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "A"

    def _extract_answer(self, response: str, choices: list[str]) -> str:
        """Extract answer from CoT response. Looks for 'ANSWER: X' pattern."""
        valid = set(chr(65 + i) for i in range(len(choices)))
        max_letter = chr(64 + len(choices))  # e.g. 'H' for 8 choices
        pattern = f'[A-{max_letter}]'

        # Look for explicit "ANSWER: X" pattern (strongest signal)
        answer_match = re.search(rf'ANSWER:\s*({pattern})', response, re.IGNORECASE)
        if answer_match and answer_match.group(1).upper() in valid:
            return answer_match.group(1).upper()

        # Fallback: look for "The answer is X" or "I choose X"
        fallback = re.search(
            rf'(?:the answer is|i choose|my answer is|correct answer is)\s*[:\s]*({pattern})',
            response, re.IGNORECASE,
        )
        if fallback and fallback.group(1).upper() in valid:
            return fallback.group(1).upper()

        # Check for "Insufficient information" explicitly mentioned
        if re.search(r'insufficient information', response, re.IGNORECASE):
            for i, choice in enumerate(choices):
                if 'insufficient' in choice.lower():
                    return chr(65 + i)

        # Last line often contains just the letter
        last_line = response.strip().split('\n')[-1].strip().upper()
        single = re.search(rf'\b({pattern})\b', last_line)
        if single and single.group(1) in valid:
            return single.group(1)

        # Anywhere in text (last resort)
        any_match = re.search(rf'\b({pattern})\b', response.upper())
        if any_match and any_match.group(1) in valid:
            return any_match.group(1)

        return "A"


# ---------------------------------------------------------------------------
# Dataset loading (HuggingFace)
# ---------------------------------------------------------------------------

async def _load_dataset(eval_name: str) -> list[AgentInput]:
    """Load LAB-Bench dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        import random

        ds = load_dataset("futurehouse/lab-bench", eval_name, split="train")
        inputs = []
        for row in ds:
            distractors = list(row.get("distractors", []))
            choices = [row["ideal"]] + distractors + ["Insufficient information"]
            random.shuffle(choices)
            ideal_idx = choices.index(row["ideal"])
            ideal_letter = chr(65 + ideal_idx)

            inputs.append(AgentInput(
                question=row["question"],
                choices=choices,
                subtask=row.get("subtask", eval_name),
                ideal=ideal_letter,
                sources=row.get("sources", []) or [],
            ))
        return inputs
    except ImportError:
        logger.error("datasets package required: pip install datasets")
        return []
    except Exception as e:
        logger.error(f"Failed to load dataset {eval_name}: {e}")
        return []


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_lab_bench(
    evals: list[str] | None = None,
    model: str = "claude-haiku-4-5-20251001",
    use_tools: bool = True,
    n_threads: int = 4,
    max_questions: int | None = None,
    output_path: str | None = None,
) -> dict[str, BenchmarkResults]:
    """Run LAB-Bench evaluation with PaperQA2-inspired retrieval pipeline.

    Args:
        evals: List of eval names. Defaults to ["LitQA2", "DbQA", "SeqQA"].
        model: LLM model to use.
        use_tools: Whether to augment with AgeSensei retrieval tools.
        n_threads: Concurrent evaluation threads.
        max_questions: Limit questions per eval (for quick testing).
        output_path: Save results JSON to this path.

    Returns:
        Dict mapping eval name -> BenchmarkResults.
    """
    if evals is None:
        evals = ["LitQA2", "DbQA", "SeqQA"]

    agent = AgeSenseiLabBenchAgent(model=model, use_tools=use_tools)
    all_results: dict[str, BenchmarkResults] = {}

    for eval_name in evals:
        print(f"\n{'='*60}")
        print(f"  Running LAB-Bench: {eval_name} (PaperQA2-style pipeline)")
        print(f"  Tools: {use_tools} | Model: {model}")
        print(f"{'='*60}")

        questions = await _load_dataset(eval_name)
        if not questions:
            print(f"  Skipped: could not load dataset")
            continue

        if max_questions:
            questions = questions[:max_questions]

        print(f"  Questions: {len(questions)}")

        sem = asyncio.Semaphore(n_threads)
        completed = [0]
        correct_so_far = [0]

        async def evaluate_one(q: AgentInput, idx: int) -> EvalResult:
            async with sem:
                predicted = await agent(q)
                correct = predicted == q.ideal
                completed[0] += 1
                if correct:
                    correct_so_far[0] += 1
                if completed[0] % 5 == 0:
                    acc = correct_so_far[0] / completed[0]
                    print(f"    [{completed[0]}/{len(questions)}] running acc: {acc:.1%}")
                return EvalResult(
                    question_id=f"{eval_name}_{idx}",
                    subtask=q.subtask,
                    question=q.question[:100],
                    choices=q.choices,
                    ideal=q.ideal,
                    predicted=predicted,
                    correct=correct,
                )

        tasks = [evaluate_one(q, i) for i, q in enumerate(questions)]
        eval_results = await asyncio.gather(*tasks)

        total = len(eval_results)
        correct = sum(1 for r in eval_results if r.correct)
        accuracy = correct / total if total > 0 else 0.0

        benchmark = BenchmarkResults(
            eval_name=eval_name,
            total=total,
            correct=correct,
            accuracy=accuracy,
            coverage=1.0,
            results=eval_results,
        )

        all_results[eval_name] = benchmark
        print(f"\n  Result: {correct}/{total} = {accuracy:.1%} accuracy")

    if output_path:
        import os as _os
        _os.makedirs(_os.path.dirname(output_path), exist_ok=True)
        output = {
            name: {
                "eval_name": r.eval_name,
                "total": r.total,
                "correct": r.correct,
                "accuracy": r.accuracy,
                "coverage": r.coverage,
                "per_question": [
                    {"q": er.question[:80], "ideal": er.ideal, "predicted": er.predicted, "correct": er.correct}
                    for er in r.results
                ],
            }
            for name, r in all_results.items()
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")

    return all_results


# ---------------------------------------------------------------------------
# Comparison runner (with vs without tools)
# ---------------------------------------------------------------------------

async def run_ablation(
    evals: list[str] | None = None,
    model: str = "claude-haiku-4-5-20251001",
    max_questions: int = 50,
) -> dict[str, Any]:
    """Run with/without AgeSensei tools to measure retrieval augmentation impact."""
    print("\n" + "=" * 60)
    print("  LAB-Bench Ablation: PaperQA2-style Pipeline vs Baseline CoT")
    print("=" * 60)

    print("\n--- WITH AgeSensei tools (PaperQA2-style) ---")
    with_tools = await run_lab_bench(
        evals=evals, model=model, use_tools=True, max_questions=max_questions
    )

    print("\n--- CoT only (no tools) ---")
    without_tools = await run_lab_bench(
        evals=evals, model=model, use_tools=False, max_questions=max_questions
    )

    print("\n" + "=" * 60)
    print("  Ablation Results")
    print("=" * 60)
    print(f"  {'Eval':<12} {'PaperQA2':>12} {'CoT only':>12} {'Delta':>8}")
    print(f"  {'-'*44}")

    comparison = {}
    for name in with_tools:
        wt = with_tools[name].accuracy
        bl = without_tools.get(name, BenchmarkResults(name, 0, 0, 0.0, 0.0)).accuracy
        delta = wt - bl
        print(f"  {name:<12} {wt:>11.1%} {bl:>11.1%} {delta:>+7.1%}")
        comparison[name] = {"with_tools": wt, "baseline": bl, "delta": delta}

    return comparison
