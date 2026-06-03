"""LAB-Bench LitQA2 evaluation using Opus 4.6 via micuapi."""
import asyncio
import json
import random
import os
import sys
from pathlib import Path

os.environ['ANTHROPIC_API_KEY'] = 'sk-nVkdz46yKk6tbWWU9KbEOnlCw4AS6BfdPpfvqfBdp14GK7NB'
os.environ['ANTHROPIC_BASE_URL'] = 'https://www.micuapi.ai'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agesensei.eval.lab_bench_adapter import AgeSenseiLabBenchAgent, AgentInput


def build_input(q: dict) -> AgentInput:
    """Convert HuggingFace dataset row to AgentInput."""
    # Build choices list
    choices = []
    ideal = q.get("ideal", "")
    distractors = q.get("distractors", [])
    if isinstance(distractors, str):
        distractors = [d.strip() for d in distractors.split(";") if d.strip()]

    all_options = [ideal] + list(distractors) + ["Insufficient information to answer this question"]
    # Shuffle deterministically
    rng = random.Random(hash(q.get("question", "")))
    rng.shuffle(all_options)

    return AgentInput(
        question=q["question"],
        choices=all_options,
        subtask="LitQA2",
        ideal=ideal,
        sources=q.get("sources", []) if isinstance(q.get("sources"), list) else [],
    )


def check_correct(answer_letter: str, agent_input: AgentInput) -> bool:
    """Check if the answer letter maps to the ideal answer."""
    if not answer_letter or len(answer_letter) != 1:
        return False
    idx = ord(answer_letter.upper()) - ord('A')
    if 0 <= idx < len(agent_input.choices):
        return agent_input.choices[idx] == agent_input.ideal
    return False


async def run_eval(use_tools: bool, questions: list, out_path: Path):
    mode = "with_tools" if use_tools else "baseline"
    print(f"\n=== {'WITH TOOLS' if use_tools else 'BASELINE'} (Opus 4.6) ===")

    agent = AgeSenseiLabBenchAgent(
        model='claude-opus-4-6',
        use_tools=use_tools,
        base_url='https://www.micuapi.ai',
    )

    results = []
    correct = 0

    for i, q in enumerate(questions):
        agent_input = build_input(q)
        try:
            answer = await agent.answer(agent_input)
            is_correct = check_correct(answer, agent_input)
            if is_correct:
                correct += 1
            results.append({
                'question_id': i,
                'question': q['question'][:100],
                'answer': answer,
                'correct': is_correct,
                'ideal': agent_input.ideal[:60],
            })
            print(f'  Q{i+1}/50: {answer!r} {"OK" if is_correct else "WRONG"} ({correct}/{i+1} = {correct/(i+1)*100:.0f}%)', flush=True)
        except Exception as e:
            print(f'  Q{i+1}/50: ERROR {e}', flush=True)
            results.append({'question_id': i, 'answer': '', 'correct': False, 'error': str(e)})

    acc = correct / len(results) if results else 0
    print(f'\n{mode}: {correct}/{len(results)} = {acc*100:.1f}%')

    with open(out_path, 'w') as f:
        json.dump({'accuracy': acc, 'correct': correct, 'total': len(results),
                   'model': 'claude-opus-4-6', 'use_tools': use_tools, 'results': results}, f, indent=2)

    return acc, correct


async def main():
    from datasets import load_dataset
    ds = load_dataset('futurehouse/lab-bench', 'LitQA2', split='train')
    questions = list(ds)
    random.seed(42)
    random.shuffle(questions)
    questions = questions[:50]

    out_dir = Path('artifacts/eval_opus_v2')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline first
    base_acc, base_correct = await run_eval(False, questions, out_dir / 'baseline.json')

    # Then with tools
    tools_acc, tools_correct = await run_eval(True, questions, out_dir / 'with_tools.json')

    # Summary
    summary = {
        'model': 'claude-opus-4-6',
        'baseline': {'accuracy': base_acc, 'correct': base_correct, 'total': 50},
        'with_tools': {'accuracy': tools_acc, 'correct': tools_correct, 'total': 50},
        'delta': tools_acc - base_acc,
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n=== SUMMARY ===')
    print(f'Baseline:   {base_acc*100:.1f}%')
    print(f'With tools: {tools_acc*100:.1f}%')
    print(f'Delta:      {(tools_acc-base_acc)*100:+.1f}%')


if __name__ == '__main__':
    asyncio.run(main())
