"""LAB-Bench LitQA2 evaluation using Opus 4.6 via proxy API (httpx)."""
import asyncio
import json
import random
import re
import time
from pathlib import Path

import httpx

API_URL = "https://co.yes.vg/team/v1/messages"
API_KEY = "team-3f6a1f55fce8bb17787ff6926687b59453d644740100dc80"
MODEL = "claude-opus-4-6"
MAX_QUESTIONS = 50
OUT_DIR = Path("artifacts/eval_opus")


async def call_opus(client: httpx.AsyncClient, messages: list, max_tokens: int = 512) -> str:
    for attempt in range(3):
        try:
            resp = await client.post(
                API_URL,
                headers={
                    "x-api-key": API_KEY,
                    "content-type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                json={"model": MODEL, "max_tokens": max_tokens, "messages": messages},
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("content"):
                    return data["content"][0]["text"]
                return ""
            elif resp.status_code == 429:
                wait = 2 ** attempt * 5
                print(f"  Rate limited, waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"  API error {resp.status_code}: {resp.text[:200]}")
                await asyncio.sleep(2)
        except Exception as e:
            print(f"  Request error: {e}")
            await asyncio.sleep(2)
    return ""


def load_questions(max_q: int = 50) -> list:
    from datasets import load_dataset
    ds = load_dataset("futurehouse/lab-bench", "LitQA2", split="train")
    litqa2 = list(ds)
    random.seed(42)
    random.shuffle(litqa2)
    return litqa2[:max_q]


def build_options(row: dict) -> tuple:
    ideal = row["ideal"]
    distractors = row.get("distractors", []) or []
    if isinstance(distractors, str):
        distractors = [d.strip() for d in distractors.split(",")]

    options = list(distractors) + [ideal, "Insufficient information to answer this question."]
    random.seed(hash(row["question"]))
    random.shuffle(options)

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    correct_letter = ""
    option_lines = []
    for i, opt in enumerate(options):
        letter = letters[i]
        option_lines.append(f"{letter}. {opt}")
        if opt == ideal:
            correct_letter = letter

    return option_lines, correct_letter


async def eval_with_tools(client: httpx.AsyncClient, row: dict, options: list) -> str:
    """Evaluate with AgeSensei retrieval context."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from agesensei.eval.lab_bench_adapter import LABBenchAdapter
    adapter = LABBenchAdapter(model="claude-haiku-4-5-20251001")

    question = row["question"]
    chunks = await adapter._retrieve_all_chunks(question)
    scored = await adapter._score_chunks(chunks, question)
    context = adapter._format_context(scored)

    prompt = f"""You are an expert biomedical scientist. Answer the following multiple-choice question.

Context from scientific literature:
{context}

Question: {question}

Options:
{chr(10).join(options)}

Think step by step. First reason through the evidence, then state your final answer as a single letter.
Your response MUST end with exactly: ANSWER: X (where X is the letter)"""

    response = await call_opus(client, [{"role": "user", "content": prompt}])
    match = re.search(r'ANSWER:\s*([A-Z])', response)
    return match.group(1) if match else ""


async def eval_baseline(client: httpx.AsyncClient, row: dict, options: list) -> str:
    """Evaluate without tools (pure LLM)."""
    prompt = f"""You are an expert biomedical scientist. Answer the following multiple-choice question.

Question: {row["question"]}

Options:
{chr(10).join(options)}

Think step by step. First reason through the evidence, then state your final answer as a single letter.
Your response MUST end with exactly: ANSWER: X (where X is the letter)"""

    response = await call_opus(client, [{"role": "user", "content": prompt}])
    match = re.search(r'ANSWER:\s*([A-Z])', response)
    return match.group(1) if match else ""


async def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading LitQA2 questions...")
    questions = load_questions(MAX_QUESTIONS)
    print(f"Loaded {len(questions)} questions")

    prepared = []
    for row in questions:
        options, correct = build_options(row)
        prepared.append((row, options, correct))

    async with httpx.AsyncClient() as client:
        # --- Baseline ---
        print("\n=== Baseline (Opus 4.6, no tools) ===")
        baseline_correct = 0
        baseline_results = []

        for i, (row, options, correct) in enumerate(prepared):
            print(f"  [{i+1}/{len(prepared)}] {row['question'][:60]}...", end=" ", flush=True)
            answer = await eval_baseline(client, row, options)
            is_correct = answer == correct
            if is_correct:
                baseline_correct += 1
            print(f"{'Y' if is_correct else 'N'} (got={answer}, want={correct})")
            baseline_results.append({
                "question": row["question"][:100], "answer": answer,
                "correct": correct, "is_correct": is_correct,
            })
            await asyncio.sleep(1)

        baseline_acc = baseline_correct / len(prepared) * 100
        print(f"\nBaseline: {baseline_correct}/{len(prepared)} = {baseline_acc:.1f}%")

        with open(OUT_DIR / "baseline_results.json", "w") as f:
            json.dump({"accuracy": baseline_acc, "correct": baseline_correct,
                       "total": len(prepared), "model": MODEL, "results": baseline_results}, f, indent=2)

        # --- With Tools ---
        print("\n=== With AgeSensei Tools (Opus + retrieval) ===")
        tools_correct = 0
        tools_results = []

        for i, (row, options, correct) in enumerate(prepared):
            print(f"  [{i+1}/{len(prepared)}] {row['question'][:60]}...", end=" ", flush=True)
            try:
                answer = await eval_with_tools(client, row, options)
            except Exception as e:
                print(f"ERROR: {e}")
                answer = ""
            is_correct = answer == correct
            if is_correct:
                tools_correct += 1
            print(f"{'Y' if is_correct else 'N'} (got={answer}, want={correct})")
            tools_results.append({
                "question": row["question"][:100], "answer": answer,
                "correct": correct, "is_correct": is_correct,
            })
            await asyncio.sleep(1)

        tools_acc = tools_correct / len(prepared) * 100
        print(f"\nWith Tools: {tools_correct}/{len(prepared)} = {tools_acc:.1f}%")

        with open(OUT_DIR / "tools_results.json", "w") as f:
            json.dump({"accuracy": tools_acc, "correct": tools_correct,
                       "total": len(prepared), "model": MODEL, "results": tools_results}, f, indent=2)

    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY (Opus 4.6, {len(prepared)} questions)")
    print(f"  Baseline:   {baseline_acc:.1f}%")
    print(f"  With Tools: {tools_acc:.1f}%")
    print(f"  Delta:      {tools_acc - baseline_acc:+.1f}%")
    print(f"{'='*50}")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({
            "model": MODEL, "total_questions": len(prepared),
            "baseline_accuracy": baseline_acc, "tools_accuracy": tools_acc,
            "delta": tools_acc - baseline_acc,
        }, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
