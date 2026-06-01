#!/usr/bin/env python3
"""Run LAB-Bench evaluation with AgeSensei tool augmentation.

Examples:
    # Quick test (10 questions per eval)
    python scripts/run_lab_bench.py --max-questions 10

    # Full LitQA2 evaluation
    python scripts/run_lab_bench.py --evals LitQA2 --model claude-sonnet-4-20250514

    # Ablation study (with vs without tools)
    python scripts/run_lab_bench.py --ablation --max-questions 50

    # All supported evals
    python scripts/run_lab_bench.py --evals LitQA2 DbQA SeqQA SuppQA ProtocolQA
"""
import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Run LAB-Bench evaluation with AgeSensei agents"
    )
    parser.add_argument(
        "--evals",
        nargs="+",
        default=["LitQA2", "DbQA", "SeqQA"],
        help="Eval categories to run (default: LitQA2 DbQA SeqQA)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="LLM model for answering (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit questions per eval (for quick testing)",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=4,
        help="Concurrent evaluation threads (default: 4)",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Disable AgeSensei tool augmentation (baseline mode)",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run with/without tools comparison",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/eval/lab_bench_results.json",
        help="Output JSON path (default: artifacts/eval/lab_bench_results.json)",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    from agesensei.eval.lab_bench_adapter import run_ablation, run_lab_bench

    if args.ablation:
        results = asyncio.run(run_ablation(
            evals=args.evals,
            model=args.model,
            max_questions=args.max_questions or 50,
        ))
    else:
        results = asyncio.run(run_lab_bench(
            evals=args.evals,
            model=args.model,
            use_tools=not args.no_tools,
            n_threads=args.n_threads,
            max_questions=args.max_questions,
            output_path=args.output,
        ))

    # Print summary
    print("\n" + "=" * 60)
    print("  LAB-Bench Evaluation Complete")
    print("=" * 60)

    if isinstance(results, dict) and all(hasattr(v, "accuracy") for v in results.values()):
        for name, r in results.items():
            print(f"  {name}: {r.accuracy:.1%} ({r.correct}/{r.total})")


if __name__ == "__main__":
    main()
