"""Orchestrator: coordinates all agents in the discovery pipeline.

One command to run the entire AgeSensei system:
    orchestrator = Orchestrator()
    report = await orchestrator.discover("novel senolytic targets")
"""

import json
from datetime import datetime
from pathlib import Path
from agesensei.schema import DiscoveryReport
from agesensei.agents.literature import LiteratureAgent
from agesensei.agents.target_extractor import TargetExtractor
from agesensei.agents.protein_analyzer import ProteinAnalyzer
from agesensei.agents.structure_predictor import StructurePredictor
from agesensei.agents.druggability import DruggabilityAgent
from agesensei.agents.pathway import PathwayAgent
from agesensei.agents.baseline_table import BaselineTableAgent


class Orchestrator:
    """Top-level coordinator for the AgeSensei discovery pipeline.

    Workflow:
        1. LiteratureAgent -> search papers
        2. TargetExtractor -> extract gene targets from papers
        3. ProteinAnalyzer -> ESM-2 analysis of top targets
        4. DruggabilityAgent -> assess druggability via ChEMBL + OpenTargets
        5. PathwayAgent -> KEGG aging pathway enrichment
        6. Generate markdown report

    Example:
        orch = Orchestrator()
        report = await orch.discover("novel senolytic drug targets")
        orch.save_report(report, "output/")
    """

    def __init__(self, skip_esm: bool = False, predict_structures: bool = False):
        """
        Args:
            skip_esm: If True, skip ESM-2 protein analysis (faster)
            predict_structures: If True, run Protenix structure prediction (requires GPU)
        """
        self.literature = LiteratureAgent()
        self.extractor = TargetExtractor()
        self.protein = ProteinAnalyzer(skip_esm=skip_esm)
        self.structure_predictor = StructurePredictor()
        self.druggability = DruggabilityAgent()
        self.pathway = PathwayAgent()
        self.baseline = BaselineTableAgent(literature=self.literature)
        self._predict_structures = predict_structures

    async def discover(
        self,
        query: str,
        max_papers: int = 20,
        top_targets: int = 5,
        scan_positions: int = 10,
        deep_search: bool = False,
        baseline_table: bool = False,
        deep_read: bool = False,
        deep_read_top_k: int = 5,
    ) -> DiscoveryReport:
        """Run the full discovery pipeline.

        Args:
            query: Research question (e.g., "novel senolytic drug targets")
            max_papers: Maximum papers to retrieve
            top_targets: Number of top targets to deeply analyze
            scan_positions: ESM-2 mutation scan positions per protein (0 = skip)
            deep_search: If True, use ReAct multi-iteration literature search
                         (more calls, higher recall) instead of single-shot.
            baseline_table: If True, also generate a baseline intervention
                            comparison table for each top target.
            deep_read: If True, after the literature search, run DeepXiv-style
                       progressive section-level reading on the top-K papers and
                       attach structured findings to the report. This gives
                       downstream agents (target extraction, reporter) much richer
                       evidence than abstracts alone.
            deep_read_top_k: How many top papers to deep-read when ``deep_read``
                             is enabled.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mode_bits = []
        mode_bits.append("deep-search (ReAct)" if deep_search else "standard search")
        if deep_read:
            mode_bits.append(f"deep-read top-{deep_read_top_k}")
        if baseline_table:
            mode_bits.append("baseline tables")
        print(f"\n{'='*60}")
        print(f"AgeSensei Discovery Pipeline")
        print(f"Query: {query}")
        print(f"Time: {timestamp}")
        print(f"Mode: {' + '.join(mode_bits)}")
        print(f"{'='*60}")

        # Step 1: Literature search
        print(f"\n[1/5] Literature search (max {max_papers} papers)...")
        if deep_search:
            papers = await self.literature.search_react(
                query, max_iterations=3, min_quality=max(10, max_papers // 2),
                max_results_per_iter=max_papers,
            )
            papers = papers[:max_papers]
        else:
            papers = await self.literature.search(query, max_results=max_papers)
        print(f"      Found {len(papers)} papers")

        # Step 1b: Optional DeepXiv-style progressive reading
        findings = []
        if deep_read and papers:
            k = min(deep_read_top_k, len(papers))
            print(f"\n[1b/5] Deep-reading top {k} papers (section-level)...")
            try:
                findings = await self.literature.deep_read(
                    query, papers=papers, top_k=k
                )
                print(f"      Produced {len(findings)} structured findings")
            except Exception as e:
                print(f"      deep_read failed, continuing without findings ({e})")

        # Step 2: Target extraction
        print(f"\n[2/5] Target extraction...")
        targets = self.extractor.extract_from_papers(papers)
        top = targets[:top_targets]
        print(f"      Extracted {len(targets)} targets, analyzing top {len(top)}")

        # Step 3: Protein analysis
        print(f"\n[3/6] Protein analysis...")
        protein_analyses = await self.protein.analyze_targets(top, top_n=top_targets, scan_positions=scan_positions)

        # Step 3b: Structure prediction (optional, requires Protenix + GPU)
        structure_predictions = {}
        if self._predict_structures:
            print(f"\n[3b/6] Structure prediction (Protenix)...")
            if self.structure_predictor.available:
                structure_predictions = await self.structure_predictor.predict_targets(top, top_n=top_targets)
            else:
                print("      Protenix not installed, skipping. Install: pip install protenix")

        # Step 4: Druggability assessment
        print(f"\n[4/6] Druggability assessment...")
        druggability = await self.druggability.assess_targets(top, top_n=top_targets)

        # Step 5: Pathway analysis
        print(f"\n[5/6] Pathway analysis...")
        pathways = await self.pathway.analyze(top, top_n=top_targets)

        # Compute overall scores
        for target in top:
            drug_score = druggability.get(target.gene_symbol)
            target.druggability_score = drug_score.score if drug_score else 0.0
            target.overall_score = (
                0.4 * target.score +           # literature/HAGR evidence
                0.3 * target.druggability_score +  # druggability
                0.2 * (0.5 if target.in_genage else 0.0) +  # aging DB validation
                0.1 * (1.0 if target.aging_link else 0.0)    # aging link
            )

        top.sort(key=lambda t: t.overall_score, reverse=True)

        # Optional: baseline intervention comparison tables for each top target
        baseline_tables: dict[str, list] = {}
        if baseline_table:
            print(f"\n[6/6] Baseline intervention tables...")
            for t in top:
                topic = f"{t.gene_symbol} aging intervention lifespan"
                try:
                    rows = await self.baseline.build(topic, max_papers=8)
                    if rows:
                        baseline_tables[t.gene_symbol] = rows
                        print(f"      {t.gene_symbol}: {len(rows)} baseline rows")
                except Exception as e:
                    print(f"      {t.gene_symbol}: baseline build failed ({e})")

        report = DiscoveryReport(
            query=query,
            timestamp=timestamp,
            total_papers_analyzed=len(papers),
            targets=top,
            protein_analyses=protein_analyses,
            structure_predictions=structure_predictions,
            druggability=druggability,
            pathways=pathways,
            baseline_tables=baseline_tables,
            findings=findings,
        )

        # Print summary
        print(f"\n{'='*60}")
        print(f"Discovery Complete")
        print(f"{'='*60}")
        print(f"Papers analyzed: {len(papers)}")
        print(f"Targets found:   {len(targets)}")
        print(f"Top targets:")
        for i, t in enumerate(top, 1):
            drug_info = druggability.get(t.gene_symbol)
            n_drugs = len(drug_info.known_drugs) if drug_info else 0
            print(f"  {i}. {t.gene_symbol:10s} overall={t.overall_score:.2f} "
                  f"lit={t.score:.2f} drug={t.druggability_score:.2f} "
                  f"drugs={n_drugs} {'[GenAge]' if t.in_genage else ''}")
        print(f"Aging pathways:  {len(pathways)}")

        return report

    def save_report(self, report: DiscoveryReport, output_dir: str):
        """Save report as markdown and JSON."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save markdown
        md = self._generate_markdown(report)
        md_path = out / "report.md"
        md_path.write_text(md, encoding="utf-8")
        print(f"\nReport saved: {md_path}")

        # Save JSON (for programmatic access)
        json_path = out / "report.json"
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    def _generate_markdown(self, report: DiscoveryReport) -> str:
        """Generate markdown report."""
        lines = [
            f"# AgeSensei Discovery Report",
            f"",
            f"**Query**: {report.query}",
            f"**Time**: {report.timestamp}",
            f"**Papers analyzed**: {report.total_papers_analyzed}",
            f"**Targets discovered**: {len(report.targets)}",
            f"",
            f"---",
            f"",
            f"## Top Targets",
            f"",
            f"| Rank | Gene | Overall | Literature | Druggability | GenAge | Drugs |",
            f"|------|------|---------|-----------|-------------|--------|-------|",
        ]

        for i, t in enumerate(report.targets, 1):
            drug_info = report.druggability.get(t.gene_symbol)
            n_drugs = len(drug_info.known_drugs) if drug_info else 0
            genage = "Yes" if t.in_genage else "-"
            lines.append(
                f"| {i} | **{t.gene_symbol}** | {t.overall_score:.2f} | "
                f"{t.score:.2f} | {t.druggability_score:.2f} | {genage} | {n_drugs} |"
            )

        # Target details
        lines.extend(["", "## Target Details", ""])
        for t in report.targets:
            lines.append(f"### {t.gene_symbol} ({t.gene_name or 'N/A'})")
            if t.aging_link:
                lines.append(f"- **Aging link**: {t.aging_link}")
            if t.role:
                lines.append(f"- **Role**: {t.role}")

            # Protein analysis
            pa = report.protein_analyses.get(t.gene_symbol)
            if pa and pa.sequence_length > 0:
                lines.append(f"- **UniProt**: {pa.uniprot_id} | {pa.sequence_length} aa")

            # Druggability
            da = report.druggability.get(t.gene_symbol)
            if da:
                lines.append(f"- **Druggability**: {da.reasoning}")
            lines.append("")

        # Pathway analysis
        if report.pathways:
            lines.extend(["## Aging Pathway Enrichment", ""])
            lines.append("| Pathway | Genes | Count |")
            lines.append("|---------|-------|-------|")
            for pw in report.pathways:
                genes = ", ".join(pw.genes_in_pathway)
                lines.append(f"| {pw.pathway_name} | {genes} | {len(pw.genes_in_pathway)} |")

        # Deep-read findings (DeepXiv-style structured paper reads)
        if report.findings:
            lines.extend(["", "## Deep-read Findings", ""])
            for i, f in enumerate(report.findings, 1):
                lines.append(f"### {i}. {f.title}")
                ids = []
                if f.pmid:
                    ids.append(f"PMID {f.pmid}")
                if f.arxiv_id:
                    ids.append(f"arXiv {f.arxiv_id}")
                if f.doi:
                    ids.append(f"DOI {f.doi}")
                meta = " · ".join(ids) if ids else ""
                lines.append(f"- **Year**: {f.year or 'n.d.'}  ·  **Source**: {f.source}  ·  {meta}")
                lines.append(
                    f"- **Relevance**: {f.relevance_score:.2f}  ·  **Mode**: {f.read_mode}  ·  "
                    f"**Sections**: {', '.join(s.section for s in f.sections_read) or '—'}"
                )
                if f.key_findings:
                    lines.append("- **Key findings**:")
                    for bullet in f.key_findings:
                        lines.append(f"  - {bullet}")
                if f.methods_summary:
                    lines.append(f"- **Methods**: {f.methods_summary}")
                if f.limitations:
                    lines.append(f"- **Limitations**: {f.limitations}")
                if f.best_quote:
                    lines.append(f"- > {f.best_quote}")
                lines.append("")

        # Baseline intervention tables
        if report.baseline_tables:
            lines.extend(["", "## Baseline Intervention Tables", ""])
            for gene, rows in report.baseline_tables.items():
                lines.append(f"### {gene}")
                lines.append("")
                lines.append(
                    "| Intervention | Organism | Dose | Median Δ% | Max Δ% | Stage | PMID |"
                )
                lines.append("|---|---|---|---|---|---|---|")
                for r in rows:
                    md = "" if r.lifespan_delta_pct is None else f"{r.lifespan_delta_pct:+.1f}%"
                    mx = "" if r.max_lifespan_delta_pct is None else f"{r.max_lifespan_delta_pct:+.1f}%"
                    lines.append(
                        f"| {r.intervention} | {r.organism} | {r.dose} | "
                        f"{md} | {mx} | {r.clinical_stage} | {r.pmid} |"
                    )
                lines.append("")

        lines.extend(["", "---", f"*Generated by AgeSensei*"])
        return "\n".join(lines)
