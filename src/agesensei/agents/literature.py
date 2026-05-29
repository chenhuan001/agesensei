"""Literature search agent: discovers relevant papers across multiple sources.

This agent combines several data sources and uses LLM to:
1. Expand queries with domain-specific terms
2. Search across PubMed, Semantic Scholar, and arXiv in parallel
3. Deduplicate results across sources
4. Score and rank papers by relevance to aging/longevity research
5. Optionally iterate (ReAct) if first-pass results are weak
6. Progressively fetch PMC full-text sections for top candidates

LLM scoring and ReAct reflection are optional:
- With LLM (anthropic API key set): high-quality scoring + query refinement
- Without LLM: keyword-based heuristic scoring, ReAct degrades to single-shot
"""

import asyncio
import math
import re
from collections import Counter
from agesensei.schema import Paper, PaperFinding, SectionExcerpt
from agesensei.tools import pubmed, semantic_scholar, arxiv, pmc
from agesensei.config import config

# Try to import anthropic, but don't fail if not available
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# Default aging-related MeSH terms for query expansion
AGING_MESH_TERMS = [
    "Aging[MeSH]",
    "Cellular Senescence[MeSH]",
    "Longevity[MeSH]",
    "Telomere Shortening[MeSH]",
    "DNA Methylation[MeSH]",
    "Epigenesis, Genetic[MeSH]",
]

QUERY_EXPANSION_PROMPT = """You are a biomedical search expert. Given a user's research question about aging/longevity, generate 4 optimized search queries.

Rules:
- Query 1: PubMed query with MeSH terms and boolean operators
- Query 2: Broader PubMed query (more recall)
- Query 3: Semantic Scholar natural language query (optimized for relevance)
- Query 4: arXiv query - prefer short keywords (2-4 terms), no MeSH, target q-bio / cs.LG preprints on aging biomarkers, computational biology, or ML models of aging
- Focus on druggable targets, therapeutic interventions, and molecular mechanisms
- Include recent date filters where appropriate (last 3-5 years)

User question: {query}

Respond in this exact format (no extra text):
PUBMED_PRECISE: <query>
PUBMED_BROAD: <query>
S2_QUERY: <query>
ARXIV_QUERY: <query>"""

RELEVANCE_SCORING_PROMPT = """Score the following paper's relevance to the research question on a scale of 0.0 to 1.0.

Research question: {query}

Paper title: {title}
Abstract: {abstract}

Scoring criteria:
- 0.9-1.0: Directly addresses the question with novel druggable targets or mechanisms
- 0.7-0.8: Highly relevant, discusses related targets/pathways/mechanisms
- 0.5-0.6: Moderately relevant, provides useful background
- 0.3-0.4: Tangentially related
- 0.0-0.2: Not relevant

Respond with ONLY a number between 0.0 and 1.0, nothing else."""


class LiteratureAgent:
    """Search and rank scientific literature on aging/longevity topics.

    Pipeline:
        1. Expand user query with LLM (add synonyms, MeSH terms)
        2. Search PubMed via E-utilities API
        3. Search Semantic Scholar for citation-aware ranking
        4. Deduplicate across sources
        5. LLM-score each paper for relevance
        6. Return ranked, deduplicated paper list

    Example:
        agent = LiteratureAgent()
        papers = await agent.search("novel senolytic drug targets 2024-2026")
    """

    # Aging/longevity related keywords for heuristic scoring
    AGING_KEYWORDS = {
        # High relevance (weight 3)
        "senolytic": 3, "senolytics": 3, "senescence": 3, "senescent": 3,
        "rejuvenation": 3, "reprogramming": 3, "longevity": 3,
        "anti-aging": 3, "antiaging": 3, "healthspan": 3, "lifespan": 3,
        "yamanaka": 3, "epigenetic clock": 3, "biological age": 3,
        # Medium relevance (weight 2)
        "aging": 2, "ageing": 2, "telomere": 2, "nad+": 2, "sirtuin": 2,
        "mtor": 2, "rapamycin": 2, "metformin": 2, "autophagy": 2,
        "inflammaging": 2, "geroscience": 2, "drug target": 2,
        "klotho": 2, "foxo": 2, "ampk": 2,
        # Low relevance (weight 1)
        "oxidative stress": 1, "mitochondri": 1, "apoptosis": 1,
        "inflammation": 1, "stem cell": 1, "regenerat": 1,
        "neurodegenerat": 1, "fibrosis": 1, "metaboli": 1,
    }

    def __init__(self, llm_client=None):
        self.client = None
        self.use_llm = False

        if llm_client:
            self.client = llm_client
            self.use_llm = True
        elif HAS_ANTHROPIC:
            import os
            if os.environ.get("ANTHROPIC_API_KEY"):
                self.client = anthropic.Anthropic()
                self.use_llm = True

        self.model = config.llm.model

    async def search(self, query: str, max_results: int = 100) -> list[Paper]:
        """Search literature and return ranked papers.

        Args:
            query: Natural language research question
            max_results: Maximum total papers to return

        Returns:
            List of Papers sorted by relevance_score (descending)
        """
        # Step 1: Expand query
        queries = await self._expand_query(query)

        # Step 2: Search all sources in parallel (PubMed precise + broad, S2, arXiv)
        pubmed_precise, pubmed_broad, s2_papers, arxiv_papers = await asyncio.gather(
            pubmed.search_and_fetch(queries["pubmed_precise"], max_results=max_results // 2),
            pubmed.search_and_fetch(queries["pubmed_broad"], max_results=max_results // 3),
            semantic_scholar.search_papers(queries["s2_query"], max_results=max_results // 2),
            self._safe_arxiv_search(queries.get("arxiv_query", query), max_results // 4),
        )

        # Tag sources explicitly (legacy sources default to pubmed)
        for p in s2_papers:
            p.source = "s2"

        if config.verbose:
            print(f"  PubMed precise: {len(pubmed_precise)} papers")
            print(f"  PubMed broad:   {len(pubmed_broad)} papers")
            print(f"  Semantic Scholar: {len(s2_papers)} papers")
            print(f"  arXiv: {len(arxiv_papers)} papers")

        # Step 3: Merge and deduplicate
        all_papers = self._deduplicate(
            pubmed_precise + pubmed_broad + s2_papers + arxiv_papers
        )
        if config.verbose:
            print(f"  After dedup: {len(all_papers)} unique papers")

        # Step 4: Score relevance (batch for efficiency)
        scored = await self._score_relevance(all_papers, query)

        # Step 4b: BM25 hybrid re-ranking over title+abstract.
        # Current relevance score is "what a reasoning LLM thinks" — excellent for
        # conceptual match but blind to exact-term hits. BM25 over the candidate
        # pool fills that gap at ~zero extra cost and no external vector store.
        self._apply_bm25_rerank(scored, query)

        # Step 5: Sort by relevance and return top N
        scored.sort(key=lambda p: p.relevance_score, reverse=True)
        return scored[:max_results]

    async def _expand_query(self, query: str) -> dict[str, str]:
        """Use LLM to generate expanded search queries, or fall back to heuristics."""
        if not self.use_llm:
            return self._expand_query_heuristic(query)

        prompt = QUERY_EXPANSION_PROMPT.format(query=query)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()

        # Parse response
        queries = {
            "pubmed_precise": query,  # fallback
            "pubmed_broad": query,
            "s2_query": query,
            "arxiv_query": query,
        }

        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("PUBMED_PRECISE:"):
                queries["pubmed_precise"] = line.split(":", 1)[1].strip()
            elif line.startswith("PUBMED_BROAD:"):
                queries["pubmed_broad"] = line.split(":", 1)[1].strip()
            elif line.startswith("S2_QUERY:"):
                queries["s2_query"] = line.split(":", 1)[1].strip()
            elif line.startswith("ARXIV_QUERY:"):
                queries["arxiv_query"] = line.split(":", 1)[1].strip()

        if config.verbose:
            print(f"  Expanded queries (LLM):")
            for k, v in queries.items():
                print(f"    {k}: {v[:80]}...")

        return queries

    def _expand_query_heuristic(self, query: str) -> dict[str, str]:
        """Expand query without LLM using domain knowledge."""
        # PubMed precise: add aging filter but keep it searchable
        pubmed_precise = f"({query}) AND (aging OR senescence OR longevity)"

        # Broader PubMed query: simplify + recent date filter
        # Extract key terms (words > 3 chars, not common words)
        stop_words = {"novel", "beyond", "targets", "drug", "between", "using", "from", "with", "that", "this"}
        key_terms = [w for w in query.lower().split() if len(w) > 3 and w not in stop_words]
        simple_query = " AND ".join(key_terms[:4]) if key_terms else query
        pubmed_broad = f"({simple_query}) AND (\"2022\"[Date - Publication] : \"2026\"[Date - Publication])"

        # S2 query: natural language works best
        s2_query = query

        # arXiv: short 2-4 keyword query works best with its term search
        arxiv_query = " ".join(key_terms[:3]) if key_terms else query

        if config.verbose:
            print(f"  Expanded queries (heuristic):")
            print(f"    pubmed_precise: {pubmed_precise[:80]}...")
            print(f"    pubmed_broad: {pubmed_broad[:80]}...")
            print(f"    s2_query: {s2_query[:80]}...")
            print(f"    arxiv_query: {arxiv_query[:80]}...")

        return {
            "pubmed_precise": pubmed_precise,
            "pubmed_broad": pubmed_broad,
            "s2_query": s2_query,
            "arxiv_query": arxiv_query,
        }

    def _deduplicate(self, papers: list[Paper]) -> list[Paper]:
        """Deduplicate papers by PMID, DOI, arXiv ID, and normalized title."""
        seen_pmids = set()
        seen_dois = set()
        seen_arxiv = set()
        seen_titles = set()
        unique = []

        for paper in papers:
            if not paper.title:
                continue

            title_key = re.sub(r"[^a-z0-9]+", "", paper.title.lower())[:120]

            if paper.pmid and paper.pmid in seen_pmids:
                continue
            if paper.doi and paper.doi in seen_dois:
                continue
            if paper.arxiv_id and paper.arxiv_id in seen_arxiv:
                continue
            if title_key and title_key in seen_titles:
                continue

            if paper.pmid:
                seen_pmids.add(paper.pmid)
            if paper.doi:
                seen_dois.add(paper.doi)
            if paper.arxiv_id:
                seen_arxiv.add(paper.arxiv_id)
            if title_key:
                seen_titles.add(title_key)
            unique.append(paper)

        return unique

    async def _safe_arxiv_search(self, query: str, max_results: int) -> list[Paper]:
        """Wrap arxiv.search_arxiv to swallow transient errors (unlike PubMed we
        don't want an arXiv outage to kill the whole search)."""
        try:
            return await arxiv.search_arxiv(query, max_results=max_results)
        except Exception as e:
            if config.verbose:
                print(f"  arXiv search failed, skipping: {e}")
            return []

    async def _score_relevance(self, papers: list[Paper], query: str) -> list[Paper]:
        """Score paper relevance. Uses LLM if available, otherwise keyword heuristic."""
        if self.use_llm:
            return await self._score_with_llm(papers, query)
        else:
            return self._score_with_keywords(papers, query)

    def _score_with_keywords(self, papers: list[Paper], query: str) -> list[Paper]:
        """Score papers using keyword matching heuristic."""
        query_words = set(query.lower().split())

        for paper in papers:
            text = f"{paper.title} {paper.abstract}".lower()
            score = 0.0
            max_possible = 0.0

            # Keyword matching against aging vocabulary
            for keyword, weight in self.AGING_KEYWORDS.items():
                max_possible += weight
                if keyword in text:
                    score += weight

            # Query term matching
            query_matches = sum(1 for w in query_words if w in text and len(w) > 3)
            query_score = query_matches / max(len(query_words), 1)

            # Citation boost (log scale)
            import math
            citation_boost = min(0.15, math.log1p(paper.citation_count) / 30)

            # Recency boost
            recency_boost = 0.0
            if paper.year and paper.year >= 2024:
                recency_boost = 0.1
            elif paper.year and paper.year >= 2022:
                recency_boost = 0.05

            # Combine: keyword relevance (40%) + query match (30%) + citation (15%) + recency (15%)
            keyword_norm = min(score / max(max_possible * 0.3, 1), 1.0)
            paper.relevance_score = min(1.0,
                0.4 * keyword_norm +
                0.3 * query_score +
                0.15 * citation_boost / 0.15 +  # normalize to 0-1
                0.15 * (recency_boost / 0.1)
            )

        if config.verbose:
            print(f"  Scored {len(papers)} papers (keyword heuristic)")

        return papers

    async def _score_with_llm(self, papers: list[Paper], query: str) -> list[Paper]:
        """Use LLM to score paper relevance. Processes in batches for efficiency."""
        # For papers without abstracts, assign a low default score
        to_score = []
        no_abstract = []
        for paper in papers:
            if paper.abstract and len(paper.abstract) > 50:
                to_score.append(paper)
            else:
                paper.relevance_score = 0.2
                no_abstract.append(paper)

        if config.verbose:
            print(f"  Scoring {len(to_score)} papers (skipping {len(no_abstract)} without abstracts)")

        # Score in batches of 5 (parallel LLM calls)
        batch_size = 5
        for i in range(0, len(to_score), batch_size):
            batch = to_score[i:i + batch_size]
            scores = await asyncio.gather(*[
                self._score_single(paper, query) for paper in batch
            ])
            for paper, score in zip(batch, scores):
                paper.relevance_score = score

        return to_score + no_abstract

    async def _score_single(self, paper: Paper, query: str) -> float:
        """Score a single paper's relevance using LLM."""
        prompt = RELEVANCE_SCORING_PROMPT.format(
            query=query,
            title=paper.title,
            abstract=paper.abstract[:1000],  # truncate long abstracts
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()
            score = float(text)
            return max(0.0, min(1.0, score))  # clamp to [0, 1]
        except (ValueError, IndexError):
            return 0.5  # default score on parsing failure

    # ------------------------------------------------------------------
    # Progressive / hierarchical reading (DeepXiv-inspired)
    # ------------------------------------------------------------------

    @staticmethod
    def get_brief(paper: Paper) -> str:
        """Cheapest read: title + abstract (~500 tokens)."""
        return f"{paper.title}\n\n{paper.abstract}"

    async def _ensure_full_text(self, paper: Paper) -> bool:
        """Lazily fetch PMC full text into paper.sections. Returns True on success."""
        if paper.full_text_fetched:
            return bool(paper.sections)

        pmc_id = paper.pmc_id
        if not pmc_id and paper.pmid:
            try:
                pmc_id = await pmc.pmid_to_pmcid(paper.pmid)
            except Exception as e:
                if config.verbose:
                    print(f"  PMC lookup failed for PMID {paper.pmid}: {e}")
                pmc_id = None
            paper.pmc_id = pmc_id

        if not pmc_id:
            paper.full_text_fetched = True  # no point retrying
            return False

        try:
            sections = await pmc.fetch_full_text(pmc_id)
        except Exception as e:
            if config.verbose:
                print(f"  PMC fetch failed for PMC{pmc_id}: {e}")
            sections = {}

        paper.sections = sections
        paper.full_text_fetched = True
        return bool(sections)

    async def get_structure(self, paper: Paper) -> list[str]:
        """Return the list of section names available for this paper (empty if
        no full-text could be fetched)."""
        await self._ensure_full_text(paper)
        return list(paper.sections.keys())

    async def get_section(self, paper: Paper, section_name: str) -> str:
        """Return a specific section (methods / results / discussion / ...).

        Falls back to abstract if the requested section is not present."""
        await self._ensure_full_text(paper)
        if not paper.sections:
            return paper.abstract
        key = section_name.lower().strip()
        if key in paper.sections:
            return paper.sections[key]
        # Fuzzy fallback: first section whose canonical key contains the query
        for k, v in paper.sections.items():
            if key in k:
                return v
        return paper.abstract

    async def get_full_text(self, paper: Paper) -> str:
        """Concatenate every section; use sparingly (token heavy)."""
        await self._ensure_full_text(paper)
        if not paper.sections:
            return paper.abstract
        return "\n\n".join(f"## {k}\n{v}" for k, v in paper.sections.items())

    # ------------------------------------------------------------------
    # ReAct multi-iteration search
    # ------------------------------------------------------------------

    REFLECT_PROMPT = """You are a literature-search strategist helping with aging/longevity research.

Original user question:
{query}

After searching, we got {n_total} candidate papers, of which {n_strong} scored >= 0.7 relevance.
Titles of the top {n_top} papers (by score):
{titles}

Your job: decide if the result is STRONG enough (>= {min_quality} high-relevance papers AND topical coverage is broad) or if we should iterate with a refined query.

Pick ONE strategy and output a refined query if iterating:
- add_synonyms: expand acronyms / add synonyms (e.g. senolytic <-> senotherapeutic, NAD+ precursor <-> NR/NMN)
- narrow_species: add a species constraint (mouse / human / C. elegans) if current results mix too many models
- broaden_time: relax date filter to catch seminal older papers
- switch_focus: pivot toward a related sub-topic the top papers collectively suggest
- stop: results are strong enough, no refinement needed

Respond in this EXACT format:
DECISION: <stop | iterate>
STRATEGY: <add_synonyms | narrow_species | broaden_time | switch_focus | stop>
REASON: <one sentence>
REFINED_QUERY: <the new query, or the original if stopping>"""

    async def search_react(
        self,
        query: str,
        max_iterations: int = 3,
        min_quality: int = 10,
        max_results_per_iter: int = 60,
    ) -> list[Paper]:
        """Iterative literature search with LLM reflection between rounds.

        Args:
            query: Natural-language research question.
            max_iterations: Hard cap on search rounds (>= 1).
            min_quality: Number of relevance >= 0.7 papers that counts as "good enough".
            max_results_per_iter: Papers retrieved per round.

        Returns:
            Merged + deduplicated papers sorted by relevance_score desc.
        """
        if max_iterations < 1:
            max_iterations = 1

        current_query = query
        accumulated: list[Paper] = []
        history: list[str] = []

        for iteration in range(1, max_iterations + 1):
            if config.verbose:
                print(f"\n[ReAct iteration {iteration}/{max_iterations}] query: {current_query[:80]}")

            batch = await self.search(current_query, max_results=max_results_per_iter)
            accumulated = self._deduplicate(accumulated + batch)
            accumulated.sort(key=lambda p: p.relevance_score, reverse=True)

            strong = [p for p in accumulated if p.relevance_score >= 0.7]
            if config.verbose:
                print(f"  accumulated {len(accumulated)} unique, {len(strong)} strong (>=0.7)")

            history.append(current_query)

            if not self.use_llm:
                # Without LLM, do one extra heuristic fallback then stop.
                if iteration == 1 and len(strong) < min_quality:
                    current_query = self._heuristic_refine(query, history)
                    continue
                break

            decision = await self._reflect(query, accumulated, strong, min_quality)
            if config.verbose:
                print(f"  decision: {decision.get('decision')} / {decision.get('strategy')} -- {decision.get('reason')}")

            if decision.get("decision") == "stop" or iteration >= max_iterations:
                break

            refined = decision.get("refined_query", "").strip()
            if not refined or refined in history:
                break
            current_query = refined

        return accumulated

    async def _reflect(
        self,
        original_query: str,
        all_papers: list[Paper],
        strong_papers: list[Paper],
        min_quality: int,
    ) -> dict:
        """Ask the LLM to decide if we should iterate, and if so produce a new query."""
        top_titles = "\n".join(f"- [{p.relevance_score:.2f}] {p.title}" for p in all_papers[:10])
        prompt = self.REFLECT_PROMPT.format(
            query=original_query,
            n_total=len(all_papers),
            n_strong=len(strong_papers),
            n_top=min(10, len(all_papers)),
            titles=top_titles or "(none)",
            min_quality=min_quality,
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
        except Exception as e:
            if config.verbose:
                print(f"  Reflection LLM call failed: {e}")
            return {"decision": "stop", "strategy": "stop", "reason": "llm error", "refined_query": ""}

        out = {"decision": "stop", "strategy": "stop", "reason": "", "refined_query": ""}
        for line in text.splitlines():
            line = line.strip()
            if line.upper().startswith("DECISION:"):
                out["decision"] = line.split(":", 1)[1].strip().lower()
            elif line.upper().startswith("STRATEGY:"):
                out["strategy"] = line.split(":", 1)[1].strip().lower()
            elif line.upper().startswith("REASON:"):
                out["reason"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("REFINED_QUERY:"):
                out["refined_query"] = line.split(":", 1)[1].strip()
        return out

    def _heuristic_refine(self, query: str, history: list[str]) -> str:
        """Fallback refinement when no LLM is available: append a broadening term."""
        expansions = ["senotherapeutic", "healthspan", "longevity", "lifespan extension"]
        for exp in expansions:
            candidate = f"{query} OR {exp}"
            if candidate not in history:
                return candidate
        return query

    # ------------------------------------------------------------------
    # BM25 hybrid re-ranking (DeepXiv "BM25 + vector semantic" hybrid)
    # ------------------------------------------------------------------

    _STOPWORDS = {
        "the", "and", "for", "with", "into", "from", "this", "that", "these",
        "those", "onto", "upon", "are", "was", "were", "been", "being", "have",
        "has", "had", "its", "their", "them", "they", "which", "what", "when",
        "how", "why", "can", "could", "would", "should", "using", "used", "use",
        "novel", "new", "recent", "beyond", "between", "via", "about", "study",
        "studies", "review", "reviews", "paper", "papers",
    }

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        if not text:
            return []
        return [
            w for w in re.findall(r"[a-z0-9][a-z0-9\-+]{1,}", text.lower())
            if w not in cls._STOPWORDS
        ]

    def _apply_bm25_rerank(self, papers: list[Paper], query: str, blend: float = 0.25) -> None:
        """Blend a BM25 score over title+abstract into paper.relevance_score.

        Mutates papers in place. Formula:
            new = (1 - blend) * old + blend * bm25_norm

        We normalise BM25 to [0, 1] by dividing by the max observed score within
        the candidate pool, so the blend stays calibrated regardless of corpus
        size.
        """
        if not papers:
            return
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return

        corpus_tokens = [
            self._tokenize(f"{p.title} {p.abstract}") for p in papers
        ]
        n = len(corpus_tokens)
        avgdl = sum(len(d) for d in corpus_tokens) / max(n, 1)
        avgdl = max(avgdl, 1.0)

        df: Counter[str] = Counter()
        for doc in corpus_tokens:
            for t in set(doc):
                df[t] += 1

        k1, b = 1.5, 0.75
        raw_scores = []
        for doc in corpus_tokens:
            tf = Counter(doc)
            dl = len(doc) or 1
            s = 0.0
            for q in query_tokens:
                if q not in tf:
                    continue
                idf = math.log(1 + (n - df[q] + 0.5) / (df[q] + 0.5))
                num = tf[q] * (k1 + 1)
                den = tf[q] + k1 * (1 - b + b * dl / avgdl)
                s += idf * num / den
            raw_scores.append(s)

        max_s = max(raw_scores) if raw_scores else 0.0
        if max_s <= 0:
            return
        for paper, raw in zip(papers, raw_scores):
            bm25_norm = raw / max_s
            paper.relevance_score = min(
                1.0, (1.0 - blend) * paper.relevance_score + blend * bm25_norm
            )

        if config.verbose:
            print(f"  BM25 re-rank applied (blend={blend:.2f}, max_raw={max_s:.2f})")

    # ------------------------------------------------------------------
    # deep_read: skim -> pick sections -> synthesize structured finding
    # ------------------------------------------------------------------

    SECTION_SELECT_PROMPT = """You are deciding which sections of a paper to read to answer a question.

Question: {question}

Paper title: {title}
Available sections: {sections}

Pick the {max_sections} most useful sections. Prefer methods/results for experimental
evidence, introduction/discussion for framing, conclusion for takeaways. Use ONLY
section names from the available list.

Respond on ONE line as a comma-separated list, no prose:
<section1>, <section2>, <section3>"""

    DEEP_READ_SYNTHESIS_PROMPT = """Answer the research question using ONLY the provided excerpts. If the excerpts don't address the question, say so honestly and give a low relevance score.

Research question: {question}

Paper title: {title}
Year: {year}

=== Excerpts ===
{excerpts}

Respond in this EXACT format (do not add prose before/after):
RELEVANCE: <0.0 to 1.0>
KEY_FINDINGS:
- <finding 1>
- <finding 2>
- <finding 3>
METHODS: <one sentence on experimental design / data / model>
LIMITATIONS: <one sentence on caveats or what isn't shown>
QUOTE: <one direct quote from the excerpts, <=200 chars, that best supports the key finding>"""

    async def deep_read(
        self,
        question: str,
        papers: list[Paper] | None = None,
        top_k: int = 3,
        max_sections_per_paper: int = 3,
        section_char_limit: int = 2500,
        fallback_to_abstract: bool = True,
    ) -> list[PaperFinding]:
        """Progressively read the top-K papers and return structured findings.

        Workflow (DeepXiv scenario 一 / 二):
            1. If ``papers`` is None, run ``search_react(question)`` first.
            2. For each of the top-K papers, fetch its section structure (PMC).
            3. LLM picks 2–3 most relevant sections for the question.
            4. Fetch ONLY those sections (token-cheap vs full text).
            5. LLM synthesizes a PaperFinding: relevance, key findings, methods,
               limitations, best quote.
            6. If no full text is available, fall back to the abstract so we
               still return a usable finding (marked ``read_mode="brief"``).

        Without an LLM the method returns abstract-only placeholder findings so
        callers don't have to special-case the no-key path.
        """
        if papers is None:
            papers = await self.search_react(
                question, max_iterations=2, max_results_per_iter=60
            )
        candidates = papers[:top_k]
        if not candidates:
            return []

        sem = asyncio.Semaphore(3)

        async def process(p: Paper) -> PaperFinding | None:
            async with sem:
                try:
                    return await self._deep_read_one(
                        p,
                        question,
                        max_sections_per_paper,
                        section_char_limit,
                        fallback_to_abstract,
                    )
                except Exception as e:
                    if config.verbose:
                        print(f"  deep_read failed for '{p.title[:60]}': {e}")
                    return None

        results = await asyncio.gather(*(process(p) for p in candidates))
        findings = [r for r in results if r is not None]
        findings.sort(key=lambda f: f.relevance_score, reverse=True)
        return findings

    async def _deep_read_one(
        self,
        paper: Paper,
        question: str,
        max_sections: int,
        char_limit: int,
        fallback: bool,
    ) -> PaperFinding | None:
        """Run the per-paper deep-read pipeline."""
        structure = await self.get_structure(paper)

        excerpts: list[tuple[str, str]] = []  # (section_name, text)
        read_mode = "deep"

        if structure:
            picked = await self._pick_sections(paper, question, structure, max_sections)
            for name in picked:
                body = await self.get_section(paper, name)
                if not body or body == paper.abstract:
                    continue
                excerpts.append((name, body[:char_limit]))

        if not excerpts:
            if not fallback:
                return None
            read_mode = "brief"
            if paper.abstract:
                excerpts.append(("abstract", paper.abstract[:char_limit]))
            else:
                return None

        return await self._synthesize_finding(paper, question, excerpts, read_mode)

    async def _pick_sections(
        self,
        paper: Paper,
        question: str,
        available: list[str],
        max_sections: int,
    ) -> list[str]:
        """Ask the LLM which sections to read. Falls back to a static priority
        order if LLM is unavailable."""
        priority = ["results", "discussion", "methods", "introduction", "conclusion"]
        if not self.use_llm:
            return [s for s in priority if s in available][:max_sections]

        prompt = self.SECTION_SELECT_PROMPT.format(
            question=question,
            title=paper.title,
            sections=", ".join(available),
            max_sections=max_sections,
        )
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=80,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
        except Exception:
            return [s for s in priority if s in available][:max_sections]

        picks: list[str] = []
        for raw in text.replace("\n", ",").split(","):
            name = raw.strip().lower().strip(".")
            if not name:
                continue
            if name in available and name not in picks:
                picks.append(name)
            else:
                # fuzzy: prefix match
                for a in available:
                    if a not in picks and (name in a or a in name):
                        picks.append(a)
                        break
            if len(picks) >= max_sections:
                break
        if not picks:
            picks = [s for s in priority if s in available][:max_sections]
        return picks

    async def _synthesize_finding(
        self,
        paper: Paper,
        question: str,
        excerpts: list[tuple[str, str]],
        read_mode: str,
    ) -> PaperFinding:
        excerpts_blob = "\n\n".join(f"[{name}]\n{body}" for name, body in excerpts)
        sections_read = [
            SectionExcerpt(section=name, length_chars=len(body))
            for name, body in excerpts
        ]
        base = PaperFinding(
            pmid=paper.pmid or "",
            doi=paper.doi or "",
            arxiv_id=paper.arxiv_id or "",
            title=paper.title,
            year=paper.year,
            source=paper.source,
            question=question,
            relevance_score=paper.relevance_score,
            sections_read=sections_read,
            read_mode=read_mode,
        )

        if not self.use_llm:
            # Heuristic fallback: pull the abstract's first 2 sentences as findings.
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paper.abstract) if s.strip()]
            base.key_findings = sents[:3]
            base.methods_summary = ""
            base.limitations = ""
            base.best_quote = sents[0] if sents else ""
            return base

        prompt = self.DEEP_READ_SYNTHESIS_PROMPT.format(
            question=question,
            title=paper.title,
            year=paper.year or "",
            excerpts=excerpts_blob[:12000],  # cap prompt to stay cheap
        )
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=600,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
        except Exception as e:
            if config.verbose:
                print(f"  deep_read synthesis LLM error: {e}")
            return base

        base.key_findings = []
        current_section = None
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                current_section = None
                continue
            upper = line.upper()
            if upper.startswith("RELEVANCE:"):
                val = line.split(":", 1)[1].strip()
                try:
                    base.relevance_score = max(0.0, min(1.0, float(val)))
                except ValueError:
                    pass
                current_section = None
            elif upper.startswith("KEY_FINDINGS:"):
                current_section = "findings"
            elif upper.startswith("METHODS:"):
                base.methods_summary = line.split(":", 1)[1].strip()
                current_section = None
            elif upper.startswith("LIMITATIONS:"):
                base.limitations = line.split(":", 1)[1].strip()
                current_section = None
            elif upper.startswith("QUOTE:"):
                base.best_quote = line.split(":", 1)[1].strip().strip('"')
                current_section = None
            elif current_section == "findings" and line.startswith(("-", "*", "•")):
                base.key_findings.append(line.lstrip("-*• ").strip())

        return base
