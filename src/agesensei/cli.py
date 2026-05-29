"""AgeSensei CLI entry point."""

import asyncio
import typer
from rich.console import Console

app = typer.Typer(name="agesensei", help="AI Agent for Anti-Aging Drug Target Discovery")
console = Console()


@app.command()
def discover(
    query: str = typer.Argument(..., help="Research question, e.g. 'novel senolytic targets'"),
    max_papers: int = typer.Option(200, help="Maximum papers to retrieve"),
    top_targets: int = typer.Option(20, help="Number of top targets to analyze"),
    output: str = typer.Option("output", help="Output directory"),
    no_protein: bool = typer.Option(False, help="Skip ESM-2 protein analysis"),
    predict_structures: bool = typer.Option(False, "--predict-structures", help="Run Protenix structure prediction (requires GPU)"),
    deep_search: bool = typer.Option(False, "--deep-search", help="Use ReAct multi-iteration search"),
    deep_read: bool = typer.Option(False, "--deep-read", help="DeepXiv-style progressive paper reading"),
    baseline_table: bool = typer.Option(False, "--baseline-table", help="Generate intervention comparison tables"),
):
    """Discover anti-aging drug targets from literature."""
    from agesensei.agents.orchestrator import Orchestrator

    console.print(f"[bold green]AgeSensei[/] — Discovering targets for: {query}")

    orchestrator = Orchestrator(
        skip_esm=no_protein,
        predict_structures=predict_structures,
    )
    report = asyncio.run(
        orchestrator.discover(
            query=query,
            max_papers=max_papers,
            top_targets=top_targets,
            deep_search=deep_search,
            deep_read=deep_read,
            baseline_table=baseline_table,
        )
    )
    orchestrator.save_report(report, output)
    console.print(f"[bold green]Done![/] Report saved to {output}/")


@app.command()
def analyze_protein(
    gene: str = typer.Argument(..., help="Gene symbol, e.g. TP53"),
    mutation_scan: bool = typer.Option(False, help="Run full mutation scanning"),
):
    """Analyze a single protein with ESM-2."""
    from agesensei.agents.protein_analyzer import ProteinAnalyzer

    console.print(f"[bold green]AgeSensei[/] — Analyzing protein: {gene}")
    agent = ProteinAnalyzer()
    analysis = asyncio.run(agent.analyze_single(gene))
    console.print(analysis)


@app.command()
def predict_structure(
    gene: str = typer.Argument(..., help="Gene symbol, e.g. BCL2L1"),
    sequence: str = typer.Option(None, help="Protein sequence (fetches from UniProt if omitted)"),
    model: str = typer.Option("protenix_base_default_v1.0.0", help="Protenix model checkpoint"),
    output: str = typer.Option("artifacts/structures", help="Output directory"),
):
    """Predict protein structure using Protenix (AlphaFold3-class, 464M params)."""
    from agesensei.agents.structure_predictor import StructurePredictor

    console.print(f"[bold green]AgeSensei[/] — Predicting structure for: {gene}")
    predictor = StructurePredictor(model=model)

    if not predictor.available:
        console.print("[red]Protenix not installed.[/] Install with: pip install protenix")
        raise typer.Exit(1)

    result = asyncio.run(predictor.predict(gene, sequence))
    if result.error:
        console.print(f"[red]Error:[/] {result.error}")
    else:
        console.print(f"[green]Success![/] CIF: {result.cif_path}")
        console.print(f"  pLDDT={result.plddt_mean:.1f}  pTM={result.ptm:.3f}  ipTM={result.iptm:.3f}")
        console.print(f"  Time: {result.prediction_time_sec:.1f}s  Residues: {result.num_residues}")


if __name__ == "__main__":
    app()
