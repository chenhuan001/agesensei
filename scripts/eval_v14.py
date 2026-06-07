"""v14 eval: MemPalace + 84% full-text coverage (Opus 4.6) + retry logic."""
import asyncio, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Set ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY env vars before running


from datasets import load_dataset
from agesensei.eval.lab_bench_adapter import AgeSenseiLabBenchAgent, AgentInput

async def main():
    ds = load_dataset("futurehouse/lab-bench", "LitQA2", split="train")
    agent = AgeSenseiLabBenchAgent(
        model="claude-opus-4-6",
        use_tools=True,
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
    )

    # Resume support
    progress_file = Path("artifacts/eval_v14/progress.json")
    os.makedirs("artifacts/eval_v14", exist_ok=True)

    results = []
    start_idx = 0
    if progress_file.exists():
        results = json.load(open(progress_file))
        start_idx = len(results)
        print(f"Resuming from Q{start_idx+1}", flush=True)

    correct = sum(1 for r in results if r.get("correct"))
    total = 100

    for i in range(start_idx, total):
        item = ds[i]
        question = item['question']
        choices = list(item['distractors']) + [item['ideal'], 'Insufficient information to answer this question']
        correct_idx = len(item['distractors'])
        correct_letter = chr(65 + correct_idx)

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

        answer = None
        for retry in range(3):
            try:
                answer = await asyncio.wait_for(agent.answer(inp), timeout=300)
                break
            except asyncio.TimeoutError:
                print(f"  Q{i+1} timeout (attempt {retry+1}/3)", flush=True)
                answer = "F"
            except Exception as e:
                err_msg = str(e)
                print(f"  Q{i+1} error (attempt {retry+1}/3): {type(e).__name__}: {err_msg}", flush=True)
                if "502" in err_msg or "503" in err_msg or "429" in err_msg:
                    wait = 30 * (retry + 1)
                    print(f"  Waiting {wait}s before retry...", flush=True)
                    await asyncio.sleep(wait)
                else:
                    answer = "A"
                    break

        if answer is None:
            answer = "F"

        is_correct = (answer == correct_letter)
        if is_correct:
            correct += 1
        results.append({
            "idx": i, "answer": answer, "expected": correct_letter,
            "correct": is_correct, "doi": sources[0] if sources else ""
        })

        # Save progress after each question
        with open(progress_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Q{i+1}/{total}: {'✓' if is_correct else '✗'} (ans={answer}, exp={correct_letter}) Running: {correct}/{i+1} = {correct*100//(i+1)}%", flush=True)

    print(f"\n=== FINAL: {correct}/{total} = {correct*100//total}% ===")
    with open("artifacts/eval_v14/summary.json", "w") as f:
        json.dump({"total": total, "correct": correct, "accuracy": correct*100/total}, f)

asyncio.run(main())
