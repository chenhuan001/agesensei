"""Eval: Agentic adapter with MemPalace — 100 questions."""
import asyncio
import json
import os
import sys
import random
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
os.environ["ANTHROPIC_BASE_URL"] = "https://www.micuapi.ai"
os.environ["ANTHROPIC_API_KEY"] = "sk-nVkdz46yKk6tbWWU9KbEOnlCw4AS6BfdPpfvqfBdp14GK7NB"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

from datasets import load_dataset
from agesensei.eval.agentic_adapter import AgenticLabBenchAgent, AgentInput

OUTDIR = Path("artifacts/eval_mempalace")
OUTDIR.mkdir(parents=True, exist_ok=True)
PROGRESS_FILE = OUTDIR / "progress.json"


async def main():
    # Load dataset
    ds = load_dataset("futurehouse/lab-bench", "LitQA2", split="train")
    random.seed(42)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:100]

    # Resume from progress
    progress = {}
    if PROGRESS_FILE.exists():
        progress = json.loads(PROGRESS_FILE.read_text())

    agent = AgenticLabBenchAgent(
        model="claude-opus-4-6",
        max_turns=8,
    )

    results = []
    correct = 0
    total = 0

    for i, idx in enumerate(indices):
        q_id = f"Q{i+1}"
        if q_id in progress:
            r = progress[q_id]
            results.append(r)
            if r["correct"]:
                correct += 1
            total += 1
            continue

        row = ds[idx]
        distractors = list(row.get("distractors", []))
        choices = [row["ideal"]] + distractors + ["Insufficient information"]
        random.shuffle(choices)
        ideal_idx = choices.index(row["ideal"])
        ideal_letter = chr(65 + ideal_idx)

        inp = AgentInput(
            question=row["question"],
            choices=choices,
            sources=row.get("sources", []) or [],
            subtask="LitQA2",
            ideal=ideal_letter,
        )

        try:
            answer = await agent.answer(inp)
        except Exception as e:
            logger.error(f"{q_id} error: {e}")
            answer = "A"

        is_correct = answer == ideal_letter
        if is_correct:
            correct += 1
        total += 1

        result = {
            "question_id": q_id,
            "question": row["question"][:100],
            "answer": answer,
            "expected": ideal_letter,
            "correct": is_correct,
        }
        results.append(result)
        progress[q_id] = result
        PROGRESS_FILE.write_text(json.dumps(progress, indent=2))

        logger.info(f"{q_id}/{len(indices)}: {'✓' if is_correct else '✗'} "
                    f"(answered {answer}, expected {ideal_letter}) "
                    f"running: {correct}/{total} = {correct/total*100:.0f}%")

    # Save final results
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "model": "claude-opus-4-6",
        "adapter": "agentic_mempalace_v10",
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUTDIR / "results.json").write_text(json.dumps(results, indent=2))
    logger.info(f"\nFINAL: {correct}/{total} = {summary['accuracy']*100:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
