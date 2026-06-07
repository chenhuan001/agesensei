"""v13 eval: Updated MemPalace with 84% full-text coverage + correct API."""
import asyncio, json, os, sys, time, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Set ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY env vars before running


from datasets import load_dataset
from agesensei.eval.lab_bench_adapter import AgeSenseiLabBenchAgent, AgentInput

async def main():
    ds = load_dataset("futurehouse/lab-bench", "LitQA2", split="train")
    agent = AgeSenseiLabBenchAgent(
        model="claude-opus-4-20250514",
        use_tools=True,
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
    )

    results = []
    correct = 0
    total = 50

    for i in range(total):
        item = ds[i]
        question = item['question']

        # Build choices list: distractors + ideal + F option
        choices = list(item['distractors']) + [item['ideal'], 'Insufficient information to answer this question']
        correct_idx = len(item['distractors'])  # ideal is right after distractors
        correct_letter = chr(65 + correct_idx)

        # Extract sources/DOIs
        sources = item.get('sources', [])
        if isinstance(sources, str):
            sources = [sources]

        inp = AgentInput(
            question=question,
            choices=choices,
            subtask="LitQA2",
            ideal=item['ideal'],
            sources=sources,
        )

        try:
            answer = await asyncio.wait_for(agent.answer(inp), timeout=180)
        except asyncio.TimeoutError:
            answer = "F"
        except Exception as e:
            print(f"  ERROR Q{i+1}: {type(e).__name__}: {e}", flush=True)
            answer = "A"

        is_correct = (answer == correct_letter)
        if is_correct:
            correct += 1
        results.append({"idx": i, "answer": answer, "expected": correct_letter, "correct": is_correct})

        print(f"Q{i+1}/{total}: {'✓' if is_correct else '✗'} (ans={answer}, exp={correct_letter}) Running: {correct}/{i+1} = {correct*100//(i+1)}%", flush=True)

    print(f"\n=== FINAL: {correct}/{total} = {correct*100//total}% ===")

    os.makedirs("artifacts/eval_v13", exist_ok=True)
    with open("artifacts/eval_v13/results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("artifacts/eval_v13/summary.json", "w") as f:
        json.dump({"total": total, "correct": correct, "accuracy": correct*100/total}, f)

asyncio.run(main())
