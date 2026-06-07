"""Run Agentic LAB-Bench evaluation."""
import asyncio
import json
import random
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Set ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY env vars before running


from datasets import load_dataset
from agesensei.eval.agentic_adapter import AgenticLabBenchAgent, AgentInput

RESUME_FILE = Path("artifacts/eval_agentic/progress.json")


def build_choices(q: dict) -> tuple[list[str], str]:
    """Build shuffled choice list and return (choices, correct_letter)."""
    ideal = q["ideal"]
    distractors = q["distractors"]
    insufficient = "Insufficient information to answer this question"

    # Combine and shuffle (with fixed seed per question for reproducibility)
    options = [ideal] + list(distractors) + [insufficient]
    rng = random.Random(hash(q["question"]))
    # Shuffle all except "Insufficient" which stays last
    non_insuf = options[:-1]
    rng.shuffle(non_insuf)
    choices = non_insuf + [insufficient]

    # Find correct answer letter
    correct_idx = choices.index(ideal)
    correct_letter = chr(65 + correct_idx)
    return choices, correct_letter


async def main():
    n_questions = int(sys.argv[1]) if len(sys.argv) > 1 else 199
    ds = load_dataset("futurehouse/lab-bench", "LitQA2", split="train")
    questions = list(ds)[:n_questions]
    total = len(questions)
    print(f"Running agentic eval on {total} questions", flush=True)

    out_dir = Path("artifacts/eval_agentic")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume support
    results = []
    start_idx = 0
    if RESUME_FILE.exists():
        progress = json.loads(RESUME_FILE.read_text())
        results = progress.get("results", [])
        start_idx = len(results)
        print(f"Resuming from Q{start_idx+1}", flush=True)

    agent = AgenticLabBenchAgent(model="claude-opus-4-6", max_turns=8)

    for i in range(start_idx, total):
        q = questions[i]
        choices, correct_letter = build_choices(q)

        sources = q.get("sources", [])
        dois = []
        for s in sources:
            if isinstance(s, str):
                if "doi.org/" in s:
                    dois.append(s.split("doi.org/")[-1])
                elif s.startswith("10."):
                    dois.append(s)

        inp = AgentInput(
            question=q["question"],
            choices=choices,
            sources=dois,
            subtask=q.get("subtask", "LitQA2"),
        )
        try:
            ans = await agent.answer(inp)
        except Exception as e:
            print(f"  Q{i+1} ERROR: {e}", flush=True)
            ans = ""

        is_correct = ans == correct_letter
        tag = "CORRECT" if is_correct else "WRONG"
        correct_so_far = sum(1 for r in results if r["correct"]) + (1 if is_correct else 0)
        total_so_far = len(results) + 1
        acc = correct_so_far / total_so_far * 100
        print(f"  Q{i+1}/{total} [{tag}] ans={ans} exp={correct_letter} | running={acc:.0f}% ({correct_so_far}/{total_so_far})", flush=True)
        results.append({"q": i+1, "answer": ans, "expected": correct_letter, "correct": is_correct})

        # Save progress every 5 questions
        if (i + 1) % 5 == 0:
            RESUME_FILE.write_text(json.dumps({"results": results}, ensure_ascii=False))

    # Final save
    correct = sum(1 for r in results if r["correct"])
    acc = correct / total * 100
    summary = {"correct": correct, "total": total, "accuracy": acc, "version": "agentic_v9"}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    RESUME_FILE.write_text(json.dumps({"results": results}, ensure_ascii=False))
    print(f"\nFINAL: {correct}/{total} = {acc:.1f}%", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
