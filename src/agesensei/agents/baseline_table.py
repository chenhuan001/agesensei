"""BaselineTableAgent: synthesize an aging-intervention comparison table.

Given a topic (e.g., "rapamycin lifespan extension", "senolytic mouse studies"),
this agent:
    1. Uses LiteratureAgent.search_react to find relevant experimental papers
    2. For each top paper with PMC full-text, fetches Methods + Results sections
    3. Extracts structured baseline metrics via LLM into BaselineRow objects
    4. Emits a markdown comparison table

Inspired by DeepXiv's "baseline comparison table" sub-skill, adapted for
aging / longevity interventions rather than ML benchmarks.
"""

import asyncio
import json
import re
from pathlib import Path

from agesensei.config import config
from agesensei.schema import BaselineRow, Paper
from agesensei.agents.literature import LiteratureAgent

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


EXTRACT_PROMPT = """You extract structured aging-intervention data from a research paper.

Topic of interest: {topic}

Paper title: {title}
PMID: {pmid}
Year: {year}

=== Methods section ===
{methods}

=== Results section ===
{results}

If the paper does NOT report an aging / lifespan / healthspan intervention experiment
relevant to the topic, respond with exactly: SKIP

Otherwise respond with a single JSON object (no code fences, no prose) matching this shape:
{{
  "intervention": "<name of drug / compound / intervention>",
  "target_gene": "<primary molecular target symbol, or empty string>",
  "organism": "<species and strain, e.g. 'C57BL/6J mouse' or 'C. elegans N2'>",
  "dose": "<dose / schedule / route>",
  "duration": "<treatment duration>",
  "lifespan_delta_pct": <number or null, median lifespan change % vs control>,
  "max_lifespan_delta_pct": <number or null, maximum lifespan change %>,
  "healthspan_markers": ["<marker1: direction/magnitude>", ...],
  "adverse_effects": "<toxicity / side effects, or empty string>",
  "clinical_stage": "<preclinical | phase1 | phase2 | phase3 | approved>",
  "notes": "<one short caveat the table reader should know>"
}}

Rules:
- If a value is unknown, use "" for strings / [] for lists / null for numbers.
- Numbers must be plain numbers (e.g. 14.2), not strings with % sign.
- Keep each string field under 120 characters."""


class BaselineTableAgent:
    """Builds aging-intervention comparison tables from the literature."""

    def __init__(self, literature: LiteratureAgent | None = None, llm_client=None):
        self.literature = literature or LiteratureAgent()
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

    async def build(
        self,
        topic: str,
        max_papers: int = 10,
        max_candidates: int = 40,
    ) -> list[BaselineRow]:
        """Search literature, fetch Methods/Results of top papers, extract rows.

        Args:
            topic: The intervention / target / mechanism to compare across papers.
            max_papers: Hard cap on successfully extracted rows.
            max_candidates: How many top-ranked papers to attempt extraction on.

        Returns:
            List of BaselineRow. Empty if LLM is unavailable or no papers extracted.
        """
        if not self.use_llm:
            if config.verbose:
                print("  BaselineTable requires an LLM (no ANTHROPIC_API_KEY), skipping.")
            return []

        if config.verbose:
            print(f"  BaselineTable: searching for '{topic}'...")
        papers = await self.literature.search_react(
            topic, max_iterations=2, min_quality=max_papers
        )
        papers = [p for p in papers if p.relevance_score >= 0.5][:max_candidates]
        if config.verbose:
            print(f"  BaselineTable: {len(papers)} candidate papers")

        # Fetch Methods + Results for all candidates in parallel
        rows: list[BaselineRow] = []
        sem = asyncio.Semaphore(5)

        async def process(paper: Paper) -> BaselineRow | None:
            async with sem:
                methods = await self.literature.get_section(paper, "methods")
                results = await self.literature.get_section(paper, "results")
                if not methods and not results:
                    return None  # only abstract available, skip
                return await self._extract_row(topic, paper, methods, results)

        extracted = await asyncio.gather(*(process(p) for p in papers))
        rows = [r for r in extracted if r is not None]

        if config.verbose:
            print(f"  BaselineTable: extracted {len(rows)} rows")

        return rows[:max_papers]

    async def _extract_row(
        self, topic: str, paper: Paper, methods: str, results: str
    ) -> BaselineRow | None:
        prompt = EXTRACT_PROMPT.format(
            topic=topic,
            title=paper.title,
            pmid=paper.pmid or "",
            year=paper.year or "",
            methods=(methods or "")[:4000],
            results=(results or "")[:4000],
        )
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=800,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
        except Exception as e:
            if config.verbose:
                print(f"    extract LLM error: {e}")
            return None

        if text.upper().startswith("SKIP"):
            return None

        # Extract JSON object even if LLM added stray prose
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

        return BaselineRow(
            intervention=str(data.get("intervention", "")).strip() or "unknown",
            target_gene=str(data.get("target_gene", "") or "").strip(),
            organism=str(data.get("organism", "") or "").strip(),
            dose=str(data.get("dose", "") or "").strip(),
            duration=str(data.get("duration", "") or "").strip(),
            lifespan_delta_pct=_safe_float(data.get("lifespan_delta_pct")),
            max_lifespan_delta_pct=_safe_float(data.get("max_lifespan_delta_pct")),
            healthspan_markers=[str(x) for x in (data.get("healthspan_markers") or [])][:8],
            adverse_effects=str(data.get("adverse_effects", "") or "").strip(),
            clinical_stage=str(data.get("clinical_stage", "") or "preclinical").strip(),
            pmid=paper.pmid or "",
            pmc_id=paper.pmc_id or "",
            year=paper.year,
            notes=str(data.get("notes", "") or "").strip(),
        )

    def to_markdown(self, topic: str, rows: list[BaselineRow]) -> str:
        """Render rows as a markdown comparison table."""
        header = (
            "| Intervention | Target | Organism | Dose | Duration | "
            "Median Δ% | Max Δ% | Healthspan | Stage | Year | PMID |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|"
        )
        lines = [f"# Baseline table: {topic}", "", header]
        for r in rows:
            md = _fmt_pct(r.lifespan_delta_pct)
            mx = _fmt_pct(r.max_lifespan_delta_pct)
            hs = "; ".join(r.healthspan_markers) if r.healthspan_markers else ""
            lines.append(
                f"| {_esc(r.intervention)} | {_esc(r.target_gene)} | {_esc(r.organism)} | "
                f"{_esc(r.dose)} | {_esc(r.duration)} | {md} | {mx} | {_esc(hs)} | "
                f"{_esc(r.clinical_stage)} | {r.year or ''} | {r.pmid} |"
            )
        return "\n".join(lines) + "\n"

    def save_markdown(self, topic: str, rows: list[BaselineRow], out_dir: Path | str) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")[:60]
        path = out_dir / f"baseline_{safe}.md"
        path.write_text(self.to_markdown(topic, rows), encoding="utf-8")
        return path


def _safe_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt_pct(v: float | None) -> str:
    return "" if v is None else f"{v:+.1f}%"


def _esc(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ")
