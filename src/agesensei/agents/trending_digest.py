"""TrendingDigestAgent: weekly 'what's new in aging/longevity' digest.

Inspired by DeepXiv's ``deepxiv-trending-digest`` sub-skill. Workflow:

    1. Pull the last N days of PubMed papers matching an aging topic via a
       date-bounded query.
    2. Pull recent arXiv submissions (q-bio / cs.LG) for the same topic.
    3. Score + BM25-rerank with LiteratureAgent's existing pipeline.
    4. Brief every candidate (title + abstract is already there, no fetch).
    5. For the top-K, invoke ``LiteratureAgent.deep_read`` so the digest
       contains structured findings, not just titles.
    6. Emit a markdown digest file suitable for a weekly newsletter.

The agent degrades gracefully without an LLM: top_k papers still get abstract
fallback findings so the digest remains useful.
"""

from __future__ import annotations

import asyncio
import math
import re
from datetime import datetime, timedelta
from pathlib import Path

from agesensei.config import config
from agesensei.schema import DigestEntry, Paper, PaperFinding, TrendingDigest
from agesensei.tools import arxiv as arxiv_tool
from agesensei.tools import pubmed
from agesensei.agents.literature import LiteratureAgent

DEFAULT_TOPIC = "aging longevity senescence"


class TrendingDigestAgent:
    """Generate a weekly digest of trending aging-research papers."""

    def __init__(self, literature: LiteratureAgent | None = None):
        self.literature = literature or LiteratureAgent()

    async def generate(
        self,
        topic: str = DEFAULT_TOPIC,
        days: int = 7,
        top_k: int = 3,
        max_candidates: int = 60,
        max_brief_entries: int = 10,
    ) -> TrendingDigest:
        """Build a TrendingDigest for the given topic over the last ``days``.

        Args:
            topic: Aging / longevity topic (free text). Default covers the field broadly.
            days: Window length in days (PubMed publication-date filter).
            top_k: How many papers to deep-read.
            max_candidates: Hard cap on papers pulled from sources before scoring.
            max_brief_entries: How many brief entries to include beyond the deep reads.
        """
        end = datetime.now()
        start = end - timedelta(days=days)

        pubmed_query = self._build_pubmed_query(topic, start, end)
        if config.verbose:
            print(f"  TrendingDigest: PubMed query = {pubmed_query}")

        pm_papers, ax_papers = await asyncio.gather(
            self._safe_pubmed(pubmed_query, max_candidates),
            self._recent_arxiv(topic, start, max_candidates // 2),
        )

        all_papers = self.literature._deduplicate(pm_papers + ax_papers)
        if config.verbose:
            print(
                f"  TrendingDigest: {len(pm_papers)} PubMed + {len(ax_papers)} arXiv "
                f"-> {len(all_papers)} unique"
            )

        if not all_papers:
            return TrendingDigest(
                topic=topic,
                period_start=start.strftime("%Y-%m-%d"),
                period_end=end.strftime("%Y-%m-%d"),
                generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                total_scanned=0,
                top_papers=[],
            )

        # Score + BM25 re-rank via the existing pipeline
        scored = await self.literature._score_relevance(all_papers, topic)
        self.literature._apply_bm25_rerank(scored, topic)

        # Rank: relevance dominates, citation count as tie-breaker, recency also helps
        def _rank_key(p: Paper) -> float:
            cite_boost = min(
                0.2,
                0.0 if not p.citation_count else math.log1p(p.citation_count) / 20,
            )
            return p.relevance_score + cite_boost

        scored.sort(key=_rank_key, reverse=True)
        scored = scored[: max(top_k + max_brief_entries, max_candidates // 2)]

        entries = [DigestEntry(paper=p) for p in scored]

        # Deep-read the top-K
        if top_k > 0:
            findings = await self.literature.deep_read(
                topic,
                papers=[e.paper for e in entries[:top_k]],
                top_k=top_k,
            )
            by_key: dict[str, PaperFinding] = {}
            for f in findings:
                for key in (f.pmid, f.arxiv_id, f.doi, f.title):
                    if key:
                        by_key[key] = f
            for e in entries[:top_k]:
                for key in (e.paper.pmid, e.paper.arxiv_id, e.paper.doi, e.paper.title):
                    if key and key in by_key:
                        e.deep_finding = by_key[key]
                        break

        digest = TrendingDigest(
            topic=topic,
            period_start=start.strftime("%Y-%m-%d"),
            period_end=end.strftime("%Y-%m-%d"),
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_scanned=len(all_papers),
            top_papers=entries[: top_k + max_brief_entries],
        )
        return digest

    # ------------------------------------------------------------------
    # Source helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_pubmed_query(topic: str, start: datetime, end: datetime) -> str:
        # PubMed date-range syntax uses YYYY/MM/DD
        date_filter = (
            f'("{start:%Y/%m/%d}"[Date - Publication] : '
            f'"{end:%Y/%m/%d}"[Date - Publication])'
        )
        aging_guard = "(aging[MeSH] OR longevity[MeSH] OR senescence[MeSH])"
        return f"({topic}) AND {aging_guard} AND {date_filter}"

    async def _safe_pubmed(self, query: str, max_results: int) -> list[Paper]:
        try:
            return await pubmed.search_and_fetch(query, max_results=max_results)
        except Exception as e:
            if config.verbose:
                print(f"  TrendingDigest: PubMed fetch failed: {e}")
            return []

    async def _recent_arxiv(self, topic: str, start: datetime, max_results: int) -> list[Paper]:
        try:
            papers = await arxiv_tool.search_arxiv(
                topic, max_results=max_results, sort_by="submittedDate"
            )
        except Exception as e:
            if config.verbose:
                print(f"  TrendingDigest: arXiv fetch failed: {e}")
            return []
        # arXiv API lacks a clean date filter per-category; filter client-side by year
        # (month-level filtering would need parsing published from raw feed — not worth it)
        return [p for p in papers if p.year and p.year >= start.year]

    # ------------------------------------------------------------------
    # Markdown rendering
    # ------------------------------------------------------------------

    def to_markdown(self, digest: TrendingDigest) -> str:
        lines: list[str] = [
            f"# Trending digest — {digest.topic}",
            "",
            f"- **Window**: {digest.period_start} → {digest.period_end}",
            f"- **Generated**: {digest.generated_at}",
            f"- **Papers scanned**: {digest.total_scanned}",
            "",
            "---",
            "",
        ]

        deep = [e for e in digest.top_papers if e.deep_finding is not None]
        brief_only = [e for e in digest.top_papers if e.deep_finding is None]

        if deep:
            lines.append("## Must-read (deep-read)")
            lines.append("")
            for i, e in enumerate(deep, 1):
                lines.extend(self._render_deep_entry(i, e))

        if brief_only:
            lines.append("## Also worth a glance")
            lines.append("")
            for e in brief_only:
                lines.append(self._render_brief_entry(e))
            lines.append("")

        return "\n".join(lines) + "\n"

    def save_markdown(self, digest: TrendingDigest, out_dir: Path | str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        topic_slug = re.sub(r"[^a-z0-9]+", "_", digest.topic.lower()).strip("_")[:30] or "digest"
        path = out / f"trending_{digest.period_start}_to_{digest.period_end}_{topic_slug}.md"
        path.write_text(self.to_markdown(digest), encoding="utf-8")
        return path

    @staticmethod
    def _render_deep_entry(idx: int, e: DigestEntry) -> list[str]:
        p, f = e.paper, e.deep_finding
        assert f is not None
        first_author = p.authors[0] if p.authors else ""
        venue = p.journal or p.source or ""
        ids: list[str] = []
        if p.pmid:
            ids.append(f"PMID [{p.pmid}](https://pubmed.ncbi.nlm.nih.gov/{p.pmid}/)")
        if p.arxiv_id:
            ids.append(f"arXiv [{p.arxiv_id}](https://arxiv.org/abs/{p.arxiv_id})")
        if p.doi:
            ids.append(f"DOI [{p.doi}](https://doi.org/{p.doi})")

        out = [
            f"### {idx}. {p.title}",
            "",
            f"_{first_author} et al. · {p.year or 'n.d.'} · {venue}_",
            "",
            f"{' · '.join(ids)}" if ids else "",
            "",
            f"**Relevance**: {f.relevance_score:.2f}  ·  **Read mode**: {f.read_mode}  ·  "
            f"**Sections**: {', '.join(s.section for s in f.sections_read) or '—'}",
            "",
            "**Key findings**",
        ]
        if f.key_findings:
            out.extend(f"- {bullet}" for bullet in f.key_findings)
        else:
            out.append("- (no structured findings extracted)")
        if f.methods_summary:
            out.extend(["", f"**Methods**: {f.methods_summary}"])
        if f.limitations:
            out.extend(["", f"**Limitations**: {f.limitations}"])
        if f.best_quote:
            out.extend(["", f"> {f.best_quote}"])
        out.append("")
        return out

    @staticmethod
    def _render_brief_entry(e: DigestEntry) -> str:
        p = e.paper
        first_author = p.authors[0] if p.authors else ""
        venue = p.journal or p.source or ""
        link = ""
        if p.pmid:
            link = f"https://pubmed.ncbi.nlm.nih.gov/{p.pmid}/"
        elif p.arxiv_id:
            link = f"https://arxiv.org/abs/{p.arxiv_id}"
        elif p.url:
            link = p.url
        title_md = f"[{p.title}]({link})" if link else p.title
        return (
            f"- **{title_md}** — {first_author} · {p.year or ''} · {venue} · "
            f"score {p.relevance_score:.2f}"
        )
