"""Opus 4.6 evaluation with Unpaywall + 3-round iterative search."""
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

# Unbuffered output
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Set ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY env vars before running


from agesensei.eval.lab_bench_adapter import (
    AgeSenseiLabBenchAgent, AgentInput, _load_dataset,
)

OUT_DIR = Path("artifacts/eval_opus_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)


async def main():
    random.seed(42)

    # Load dataset
    print("Loading LitQA2 dataset...")
    questions = await _load_dataset("LitQA2")
    if not questions:
        print("Failed to load dataset")
        return

    questions = questions[:50]
    print(f"Loaded {len(questions)} questions")

    for mode_name, use_tools in [("baseline", False), ("with_tools", True)]:
        print(f"\n{'='*60}")
        print(f"  Mode: {mode_name} | Model: claude-opus-4-6")
        print(f"{'='*60}")

        agent = AgeSenseiLabBenchAgent(
            model="claude-opus-4-6",
            use_tools=use_tools,
            api_key=os.environ["ANTHROPIC_API_KEY"],
            base_url=os.environ["ANTHROPIC_BASE_URL"],
        )

        correct = 0
        total = 0
        results = []

        for i, q in enumerate(questions):
            try:
                predicted = await agent.answer(q)
                is_correct = predicted == q.ideal
                if is_correct:
                    correct += 1
                total += 1

                results.append({
                    "q": q.question[:100],
                    "ideal": q.ideal,
                    "predicted": predicted,
                    "correct": is_correct,
                })

                status = "CORRECT" if is_correct else "WRONG"
                print(f"  Q{i+1}: {status} (pred={predicted}, ideal={q.ideal}) acc={correct}/{total}={correct/total:.0%}")

                # Rate limit
                await asyncio.sleep(1)

            except Exception as e:
                print(f"  Q{i+1}: ERROR - {e}")
                results.append({
                    "q": q.question[:100],
                    "ideal": q.ideal,
                    "predicted": "ERROR",
                    "correct": False,
                    "error": str(e),
                })
                total += 1
                await asyncio.sleep(5)

        accuracy = correct / total if total else 0
        print(f"\n  {mode_name}: {correct}/{total} = {accuracy:.0%}")

        # Save results
        output = {
            "mode": mode_name,
            "model": "claude-opus-4-6",
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "results": results,
        }

        out_path = OUT_DIR / f"litqa2_{mode_name}.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"  Saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
