"""AgeSensei CLI entry point."""

import asyncio
from pathlib import Path

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


@app.command()
def finetune(
    data: str = typer.Argument(..., help="Training data CSV path"),
    val_data: str = typer.Option(None, help="Validation data CSV (auto-split if omitted)"),
    task: str = typer.Option("mlm", help="Task: mlm or mutation_cls"),
    model: str = typer.Option("facebook/esm2_t33_650M_UR50D", help="Base model"),
    output: str = typer.Option("artifacts/finetune", help="Output directory"),
    epochs: int = typer.Option(5, help="Training epochs"),
    batch_size: int = typer.Option(4, help="Batch size per GPU"),
    lr: float = typer.Option(2e-5, help="Learning rate"),
    bf16: bool = typer.Option(False, help="Use bfloat16 mixed precision"),
):
    """Fine-tune ESM-2 on aging-related protein data.

    Supports domain-adaptive MLM and mutation effect classification.
    For multi-GPU, use torchrun or deepspeed launcher directly.
    """
    from agesensei.infra.distributed.finetune.trainer import ESMFineTuner, setup_distributed

    console.print(f"[bold green]AgeSensei[/] — Fine-tuning ESM-2 ({task})")
    console.print(f"  Data: {data}")
    console.print(f"  Model: {model}")
    console.print(f"  Output: {output}")

    local_rank = setup_distributed()
    trainer = ESMFineTuner(
        model_name=model,
        task=task,
        output_dir=output,
        learning_rate=lr,
        batch_size=batch_size,
        epochs=epochs,
        bf16=bf16,
        local_rank=local_rank,
    )
    trainer.train(data, val_data)
    console.print(f"[bold green]Done![/] Fine-tuned model saved to {output}/best/")


@app.command()
def eval_lab_bench(
    evals: list[str] = typer.Option(["LitQA2", "DbQA", "SeqQA"], help="Eval categories"),
    model: str = typer.Option("claude-sonnet-4-20250514", help="LLM model"),
    max_questions: int = typer.Option(None, help="Limit questions per eval"),
    n_threads: int = typer.Option(4, help="Concurrent threads"),
    no_tools: bool = typer.Option(False, help="Disable AgeSensei tool augmentation"),
    ablation: bool = typer.Option(False, help="Run with/without tools comparison"),
    output: str = typer.Option("artifacts/eval/lab_bench_results.json", help="Output path"),
):
    """Run LAB-Bench evaluation with AgeSensei tool augmentation.

    Evaluates on biology research tasks (LitQA2, DbQA, SeqQA, etc.)
    and measures the impact of AgeSensei retrieval tools vs baseline LLM.
    """
    from agesensei.eval.lab_bench_adapter import run_ablation, run_lab_bench

    console.print(f"[bold green]AgeSensei[/] — LAB-Bench Evaluation")
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    if ablation:
        asyncio.run(run_ablation(evals=evals, model=model, max_questions=max_questions or 50))
    else:
        asyncio.run(run_lab_bench(
            evals=evals, model=model, use_tools=not no_tools,
            n_threads=n_threads, max_questions=max_questions, output_path=output,
        ))


@app.command()
def prepare_finetune_data(
    output: str = typer.Option("data/finetune", help="Output directory"),
    task: str = typer.Option("both", help="Task: mlm, mutation_cls, or both"),
):
    """Prepare training data from GenAge aging genes.

    Downloads protein sequences from UniProt and generates synthetic
    mutation labels for fine-tuning ESM-2.
    """
    from agesensei.infra.distributed.finetune.prepare_data import prepare_data

    console.print(f"[bold green]AgeSensei[/] — Preparing fine-tuning data")
    asyncio.run(prepare_data(output, task))
    console.print(f"[bold green]Done![/] Data saved to {output}/")


if __name__ == "__main__":
    app()
