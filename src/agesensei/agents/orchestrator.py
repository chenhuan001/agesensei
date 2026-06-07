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
from agesensei.agents.structure_predictor import StructurePredictAgent
from agesensei.agents.cadd import CADDAgent
from agesensei.agents.druggability import DruggabilityAgent
from agesensei.agents.pathway import PathwayAgent
from agesensei.agents.baseline_table import BaselineTableAgent


class Orchestrator:
    """Top-level coordinator for the AgeSensei discovery pipeline.

    Workflow (8 steps):
        1. LiteratureAgent        -> search papers
        2. TargetExtractor        -> extract gene targets from papers
        3. ProteinAnalyzer        -> ESM-2 analysis of top targets
        4. StructurePredictAgent  -> Protenix 3D structure prediction
        5. CADDAgent              -> virtual screening (docking + QSAR)
        6. DruggabilityAgent      -> assess druggability via ChEMBL + OpenTargets
        7. PathwayAgent           -> KEGG aging pathway enrichment
        8. Generate report

    Example:
        orch = Orchestrator()
        report = await orch.discover("novel senolytic drug targets")
        orch.save_report(report, "output/")
    """

    def __init__(
        self,
        skip_esm: bool = False,
        predict_structures: bool = True,
        run_cadd: bool = True,
    ):
        """
        Args:
            skip_esm: If True, skip ESM-2 protein analysis (faster).
            predict_structures: If True, run Protenix structure prediction.
            run_cadd: If True, run CADD virtual screening after structure
                      prediction. Requires predict_structures=True for
                      structure-based docking; ligand-only filtering still
                      works without structures.
        """
        self.literature = LiteratureAgent()
        self.extractor = TargetExtractor()
        self.protein = ProteinAnalyzer(skip_esm=skip_esm)
        self.structure_predictor = StructurePredictAgent()
        self.cadd = CADDAgent()
        self.druggability = DruggabilityAgent()
        self.pathway = PathwayAgent()
        self.baseline = BaselineTableAgent(literature=self.literature)
        self._predict_structures = predict_structures
        self._run_cadd = run_cadd

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
        n_steps = 8
        mode_bits = []
        mode_bits.append("deep-search (ReAct)" if deep_search else "standard search")
        if deep_read:
            mode_bits.append(f"deep-read top-{deep_read_top_k}")
        if self._predict_structures:
            mode_bits.append("structure prediction")
        if self._run_cadd:
            mode_bits.append("CADD screening")
        if baseline_table:
            mode_bits.append("baseline tables")
        print(f"\n{'='*60}")
        print(f"AgeSensei Discovery Pipeline")
        print(f"Query: {query}")
        print(f"Time: {timestamp}")
        print(f"Mode: {' + '.join(mode_bits)}")
        print(f"{'='*60}")

        # Step 1: Literature search
        print(f"\n[1/{n_steps}] Literature search (max {max_papers} papers)...")
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
            print(f"\n[1b/{n_steps}] Deep-reading top {k} papers (section-level)...")
            try:
                findings = await self.literature.deep_read(
                    query, papers=papers, top_k=k
                )
                print(f"      Produced {len(findings)} structured findings")
            except Exception as e:
                print(f"      deep_read failed, continuing without findings ({e})")

        # Step 2: Target extraction
        print(f"\n[2/{n_steps}] Target extraction...")
        targets = self.extractor.extract_from_papers(papers)
        top = targets[:top_targets]
        print(f"      Extracted {len(targets)} targets, analyzing top {len(top)}")

        # Step 3: Protein analysis (ESM-2)
        print(f"\n[3/{n_steps}] Protein analysis (ESM-2)...")
        protein_analyses = await self.protein.analyze_targets(top, top_n=top_targets, scan_positions=scan_positions)

        # Step 4: Structure prediction (Protenix)
        structure_predictions = {}
        if self._predict_structures:
            print(f"\n[4/{n_steps}] Structure prediction (Protenix)...")
            if self.structure_predictor.available:
                structure_predictions = await self.structure_predictor.predict_targets(top, top_n=top_targets)
            else:
                print("      Protenix not installed, skipping. Install: pip install protenix")
        else:
            print(f"\n[4/{n_steps}] Structure prediction — skipped (disabled)")

        # Step 5: CADD virtual screening
        cadd_results = {}
        if self._run_cadd:
            print(f"\n[5/{n_steps}] CADD virtual screening...")
            if self.cadd.available:
                cadd_results = await self.cadd.screen_targets(
                    top,
                    structures=structure_predictions if structure_predictions else None,
                    top_n=top_targets,
                )
            else:
                print("      CADD deps not installed, skipping. Install: pip install agesensei[cadd]")
        else:
            print(f"\n[5/{n_steps}] CADD virtual screening — skipped (disabled)")

        # Step 6: Druggability assessment
        print(f"\n[6/{n_steps}] Druggability assessment...")
        druggability = await self.druggability.assess_targets(top, top_n=top_targets)

        # Step 7: Pathway analysis
        print(f"\n[7/{n_steps}] Pathway analysis...")
        pathways = await self.pathway.analyze(top, top_n=top_targets)

        # Compute overall scores (now includes CADD signal)
        for target in top:
            drug_score = druggability.get(target.gene_symbol)
            target.druggability_score = drug_score.score if drug_score else 0.0

            # CADD bonus: targets with potent docking hits score higher
            cadd_bonus = 0.0
            cadd_res = cadd_results.get(target.gene_symbol)
            if cadd_res and cadd_res.top_hits:
                best_pchembl = max(
                    (h.pchembl_value or 0.0 for h in cadd_res.top_hits), default=0.0
                )
                cadd_bonus = min(best_pchembl / 10.0, 1.0)  # normalize to 0-1

            target.overall_score = (
                0.35 * target.score +              # literature/HAGR evidence
                0.25 * target.druggability_score +  # druggability
                0.15 * cadd_bonus +                 # CADD screening signal
                0.15 * (0.5 if target.in_genage else 0.0) +  # aging DB validation
                0.10 * (1.0 if target.aging_link else 0.0)    # aging link
            )

        top.sort(key=lambda t: t.overall_score, reverse=True)

        # Optional: baseline intervention comparison tables for each top target
        baseline_tables: dict[str, list] = {}
        if baseline_table:
            print(f"\n[8/{n_steps}] Baseline intervention tables...")
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
            cadd_results=cadd_results,
            pathways=pathways,
            baseline_tables=baseline_tables,
            findings=findings,
        )

        # Step 8: Print summary
        print(f"\n{'='*60}")
        print(f"Discovery Complete")
        print(f"{'='*60}")
        print(f"Papers analyzed: {len(papers)}")
        print(f"Targets found:   {len(targets)}")
        print(f"Top targets:")
        for i, t in enumerate(top, 1):
            drug_info = druggability.get(t.gene_symbol)
            n_drugs = len(drug_info.known_drugs) if drug_info else 0
            cadd_res = cadd_results.get(t.gene_symbol)
            n_hits = len(cadd_res.top_hits) if cadd_res else 0
            struct = structure_predictions.get(t.gene_symbol)
            plddt = f"pLDDT={struct.plddt_mean:.0f}" if struct and not struct.error else "no-struct"
            print(f"  {i}. {t.gene_symbol:10s} overall={t.overall_score:.2f} "
                  f"lit={t.score:.2f} drug={t.druggability_score:.2f} "
                  f"drugs={n_drugs} hits={n_hits} {plddt} "
                  f"{'[GenAge]' if t.in_genage else ''}")
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
            f"| Rank | Gene | Overall | Literature | Druggability | GenAge | Drugs | CADD Hits |",
            f"|------|------|---------|-----------|-------------|--------|-------|-----------|",
        ]

        for i, t in enumerate(report.targets, 1):
            drug_info = report.druggability.get(t.gene_symbol)
            n_drugs = len(drug_info.known_drugs) if drug_info else 0
            genage = "Yes" if t.in_genage else "-"
            cadd_res = report.cadd_results.get(t.gene_symbol)
            n_hits = len(cadd_res.top_hits) if cadd_res else 0
            lines.append(
                f"| {i} | **{t.gene_symbol}** | {t.overall_score:.2f} | "
                f"{t.score:.2f} | {t.druggability_score:.2f} | {genage} | {n_drugs} | {n_hits} |"
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

            # Structure prediction
            sp = report.structure_predictions.get(t.gene_symbol)
            if sp and not sp.error:
                lines.append(
                    f"- **Structure**: pLDDT={sp.plddt_mean:.1f} pTM={sp.ptm:.3f} "
                    f"({sp.num_residues} residues, {sp.prediction_time_sec:.1f}s)"
                )

            # Druggability
            da = report.druggability.get(t.gene_symbol)
            if da:
                lines.append(f"- **Druggability**: {da.reasoning}")

            # CADD results
            cr = report.cadd_results.get(t.gene_symbol)
            if cr and cr.top_hits:
                lines.append(
                    f"- **CADD**: {cr.compounds_fetched} compounds screened, "
                    f"{cr.compounds_passed_filter} passed Lipinski, "
                    f"{cr.compounds_docked} docked"
                )
                lines.append(f"  - Top hits:")
                for j, hit in enumerate(cr.top_hits[:5], 1):
                    aff = f"affinity={hit.affinity_kcal:.1f}" if hit.affinity_kcal else ""
                    pch = f"pChEMBL={hit.pchembl_value:.1f}" if hit.pchembl_value else ""
                    metric = aff or pch or "N/A"
                    lines.append(
                        f"    {j}. {hit.molecule_chembl_id} QED={hit.qed:.2f} {metric}"
                    )

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
