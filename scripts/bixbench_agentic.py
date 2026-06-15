"""BixBench Agentic Evaluator: GPT-5.5 + Code Execution + MemPalace.

Uses GPT-5.5 as the reasoning engine, executes code locally via subprocess,
and optionally consults MemPalace for domain knowledge.
"""
import asyncio
import json
import os
import re
import subprocess
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import httpx
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Config
API_KEY = os.environ.get("OPENAI_API_KEY", "sk-u5tv9FaqS9v8gZxlCNryNgvvCH6u3U6AgXdhp0IHtisWjqQ2")
API_BASE = os.environ.get("OPENAI_API_BASE", "https://www.micuapi.ai/v1")
MODEL = os.environ.get("BIXBENCH_MODEL", "gpt-5.5")
MAX_STEPS = 15  # max ReAct iterations per question
MAX_EXAMPLES = int(os.environ.get("BIXBENCH_MAX_EXAMPLES", "-1"))
OUTPUT_DIR = Path(os.environ.get("BIXBENCH_OUTPUT_DIR", "artifacts/bixbench_agentic"))
DATA_DIR = Path("artifacts/bixbench_capsules")
MEMPALACE_DIR = Path(os.environ.get("MEMPALACE_DIR", "artifacts/mempalace_papers"))

SYSTEM_PROMPT = """You are an expert bioinformatics data analyst. You solve questions by writing and executing code on real data files.

## How to use tools

To execute code, write a fenced code block with the language tag. Examples:

```python
import pandas as pd
df = pd.read_csv("data.csv")
print(df.head())
print(df.shape)
```

```r
library(DESeq2)
data <- read.csv("counts.csv")
print(head(data))
```

```bash
ls -la *.csv *.tsv *.h5ad
head -5 metadata.csv
```

To read a small file: read_file("filename.txt")
To search literature: search_literature("RNA-seq DESeq2 analysis")

## STRICT RULES

1. Your FIRST response MUST contain a ```python or ```bash code block to explore the data files. NO EXCEPTIONS.
2. You MUST execute at least 2 code blocks before giving FINAL_ANSWER.
3. NEVER guess or use domain knowledge alone — always verify computationally.
4. Write complete, self-contained code that can run independently.
5. Install packages if needed: `pip install <pkg>` in a bash block.
6. After getting computational results, state: FINAL_ANSWER: <letter>
7. If code fails, debug and retry with a different approach.
8. For R packages like DESeq2, use: ```bash\nRscript -e 'if (!require(DESeq2)) BiocManager::install("DESeq2")'```
"""

MCQ_TEMPLATE = """Question: {question}

Options:
{options}

Data files are available in the working directory: {work_dir}

Analyze the data and determine the correct answer. Show your work step by step.
After your analysis, state your final answer as: FINAL_ANSWER: <letter>
"""


async def call_llm(messages: list[dict], client: httpx.AsyncClient) -> str:
    """Call GPT-5.5 via API."""
    for attempt in range(5):
        try:
            resp = await client.post(
                f"{API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": messages,
                    "max_tokens": 4096,
                    "temperature": 0.2,
                },
                timeout=120.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            elif resp.status_code in (429, 502, 503):
                wait = 2 ** attempt * 5
                print(f"  API {resp.status_code}, retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"  API error {resp.status_code}: {resp.text[:200]}")
                return ""
        except Exception as e:
            print(f"  API exception: {e}, retrying...")
            await asyncio.sleep(5)
    return ""


def execute_code(language: str, code: str, work_dir: str, timeout: int = 120) -> str:
    """Execute Python or R code in a subprocess."""
    if language.lower() in ("python", "python3"):
        cmd = ["python3", "-c", code]
    elif language.lower() == "r":
        cmd = ["Rscript", "-e", code]
    elif language.lower() in ("bash", "shell"):
        cmd = ["bash", "-c", code]
    else:
        return f"Unsupported language: {language}"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=work_dir,
            env={**os.environ, "PYTHONPATH": ""},
        )
        output = ""
        if result.stdout:
            output += result.stdout[:3000]
        if result.stderr:
            # Filter out common warnings
            stderr_lines = [
                l for l in result.stderr.split("\n")
                if not any(w in l.lower() for w in ["warning", "deprecat", "futurewarning"])
            ]
            stderr_filtered = "\n".join(stderr_lines).strip()
            if stderr_filtered:
                output += f"\nSTDERR:\n{stderr_filtered[:1500]}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: Code execution timed out (120s limit)"
    except Exception as e:
        return f"ERROR: {e}"


def list_files(work_dir: str) -> str:
    """List files in work directory."""
    result = []
    for root, dirs, files in os.walk(work_dir):
        rel_root = os.path.relpath(root, work_dir)
        for f in files:
            path = os.path.join(rel_root, f) if rel_root != "." else f
            size = os.path.getsize(os.path.join(root, f))
            size_str = f"{size / 1024:.1f}KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f}MB"
            result.append(f"  {path} ({size_str})")
    return "\n".join(result[:50]) or "(empty directory)"


def read_file(path: str, work_dir: str, max_chars: int = 5000) -> str:
    """Read file contents."""
    full_path = os.path.join(work_dir, path)
    if not os.path.exists(full_path):
        return f"ERROR: File not found: {path}"
    try:
        with open(full_path, "r", errors="replace") as f:
            content = f.read(max_chars)
        if len(content) == max_chars:
            content += "\n... (truncated)"
        return content
    except Exception as e:
        return f"ERROR reading file: {e}"


def search_mempalace(query: str) -> str:
    """Search MemPalace for relevant literature."""
    if not MEMPALACE_DIR.exists():
        return "MemPalace not available."

    # Simple keyword search across cached papers
    results = []
    query_terms = query.lower().split()

    for paper_file in MEMPALACE_DIR.glob("*.txt"):
        try:
            content = paper_file.read_text(errors="replace")[:10000]
            score = sum(1 for term in query_terms if term in content.lower())
            if score > 0:
                results.append((score, paper_file.stem, content[:500]))
        except Exception:
            continue

    results.sort(key=lambda x: -x[0])
    if not results:
        return "No relevant papers found in MemPalace."

    output = f"Found {len(results)} relevant papers:\n"
    for score, name, snippet in results[:3]:
        output += f"\n--- {name} (relevance: {score}) ---\n{snippet}\n"
    return output


def parse_tool_calls(response: str) -> list[dict]:
    """Parse tool calls from LLM response."""
    calls = []

    # Pattern: ```python\n...\n``` or ```r\n...\n```
    code_blocks = re.findall(r"```(python|r|bash|shell)\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
    for lang, code in code_blocks:
        calls.append({"tool": "execute_code", "language": lang, "code": code.strip()})

    # Pattern: list_files() or LIST_FILES
    if re.search(r"list_files\(\)|LIST_FILES", response):
        calls.append({"tool": "list_files"})

    # Pattern: read_file("path") or READ_FILE("path")
    for match in re.finditer(r'read_file\(["\']([^"\']+)["\']\)', response):
        calls.append({"tool": "read_file", "path": match.group(1)})

    # Pattern: search_literature("query")
    for match in re.finditer(r'search_literature\(["\']([^"\']+)["\']\)', response):
        calls.append({"tool": "search_literature", "query": match.group(1)})

    return calls


def extract_final_answer(response: str) -> str | None:
    """Extract FINAL_ANSWER from response."""
    match = re.search(r"FINAL_ANSWER:\s*([A-Z])", response)
    if match:
        return match.group(1)

    # Try to find answer in other formats
    match = re.search(r"(?:answer|option)\s+is\s+([A-Z])\b", response, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


async def prepare_capsule(question: dict) -> str:
    """Find pre-extracted capsule data, return work directory path."""
    zip_filename = question["data_folder"]
    capsule_dir = DATA_DIR / zip_filename.replace(".zip", "")

    # Already extracted
    if capsule_dir.exists() and any(capsule_dir.iterdir()):
        # Find the actual data directory (may be nested)
        # Look for Data/ subfolder first
        data_subfolder = next(
            (p for p in capsule_dir.rglob("*") if p.is_dir() and p.name == "Data"), None
        )
        if data_subfolder:
            return str(data_subfolder)
        # Look for any folder with data files
        for root, dirs, files in os.walk(capsule_dir):
            if any(f.endswith(('.csv', '.tsv', '.h5ad', '.rds', '.txt', '.gz', '.xlsx')) for f in files):
                return root
        return str(capsule_dir)

    return ""  # No data available


def format_mcq_options(question: dict) -> tuple[str, dict]:
    """Format MCQ options and return (options_text, letter_to_answer_map)."""
    import random
    options = [question["ideal"]] + question["distractors"]
    # Shuffle with seed for reproducibility
    rng = random.Random(hash(question["question"]))
    rng.shuffle(options)

    letter_map = {}
    options_text = ""
    for i, opt in enumerate(options):
        letter = chr(65 + i)  # A, B, C, D
        letter_map[letter] = opt
        options_text += f"{letter}. {opt}\n"

    # Find correct letter
    correct_letter = None
    for letter, answer in letter_map.items():
        if answer == question["ideal"]:
            correct_letter = letter
            break

    return options_text, letter_map, correct_letter


async def evaluate_question(
    question: dict, client: httpx.AsyncClient, question_idx: int, total: int
) -> dict:
    """Run agentic evaluation on a single question."""
    print(f"\nQ{question_idx}/{total}: {question['question_id']}")
    print(f"  Question: {question['question'][:100]}...")

    # Prepare data
    work_dir = await prepare_capsule(question)

    # Format MCQ
    options_text, letter_map, correct_letter = format_mcq_options(question)

    # Build initial prompt
    user_prompt = MCQ_TEMPLATE.format(
        question=question["question"],
        options=options_text,
        work_dir=work_dir,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Add file listing as initial context
    files_info = list_files(work_dir)
    messages.append({
        "role": "user",
        "content": f"Here are the available files:\n{files_info}",
    })

    final_answer = None
    steps = []

    for step in range(MAX_STEPS):
        # Call LLM
        response = await call_llm(messages, client)
        if not response:
            print(f"  Step {step+1}: Empty response, skipping")
            break

        messages.append({"role": "assistant", "content": response})

        # Parse tool calls first
        tool_calls = parse_tool_calls(response)
        code_steps_so_far = sum(1 for s in steps if s.get("type") == "code")

        # Check for final answer (but only accept after at least 1 code execution)
        final_answer_candidate = extract_final_answer(response)
        if final_answer_candidate and code_steps_so_far >= 1:
            final_answer = final_answer_candidate
            print(f"  Step {step+1}: Got answer {final_answer} (after {code_steps_so_far} code steps)")
            steps.append({"step": step + 1, "type": "answer", "answer": final_answer})
            break
        elif final_answer_candidate and code_steps_so_far < 1:
            # Force the model to actually run code
            messages.append({
                "role": "user",
                "content": "STOP. You have not executed any code yet. You MUST write a ```python code block to explore and analyze the data files before answering. Start by reading the data files.",
            })
            continue

        if not tool_calls:
            if code_steps_so_far == 0:
                # No code executed yet — force it
                messages.append({
                    "role": "user",
                    "content": "You must write a ```python code block now. Start by listing and reading the data files. For example:\n\n```python\nimport os\nfor f in os.listdir('.'):\n    print(f, os.path.getsize(f))\n```",
                })
            else:
                # Has executed code, just needs to answer
                messages.append({
                    "role": "user",
                    "content": "Based on your code results above, provide your final answer as: FINAL_ANSWER: <letter>",
                })
            continue

        tool_results = []
        for tc in tool_calls:
            if tc["tool"] == "execute_code":
                print(f"  Step {step+1}: Execute {tc['language']} code ({len(tc['code'])} chars)")
                result = execute_code(tc["language"], tc["code"], work_dir)
                tool_results.append(f"Code output ({tc['language']}):\n{result}")
                steps.append({"step": step + 1, "type": "code", "language": tc["language"]})
            elif tc["tool"] == "list_files":
                result = list_files(work_dir)
                tool_results.append(f"Files:\n{result}")
                steps.append({"step": step + 1, "type": "list_files"})
            elif tc["tool"] == "read_file":
                result = read_file(tc["path"], work_dir)
                tool_results.append(f"File content ({tc['path']}):\n{result}")
                steps.append({"step": step + 1, "type": "read_file", "path": tc["path"]})
            elif tc["tool"] == "search_literature":
                result = search_mempalace(tc["query"])
                tool_results.append(f"Literature search:\n{result}")
                steps.append({"step": step + 1, "type": "search_literature"})

        # Add tool results as user message
        messages.append({
            "role": "user",
            "content": "\n\n".join(tool_results),
        })

    # If no answer after all steps, try to force one
    if not final_answer:
        messages.append({
            "role": "user",
            "content": "Based on all your analysis, what is your FINAL_ANSWER? Reply with just: FINAL_ANSWER: <letter>",
        })
        response = await call_llm(messages, client)
        if response:
            final_answer = extract_final_answer(response)

    # Default to A if still no answer
    if not final_answer:
        final_answer = "A"

    is_correct = final_answer == correct_letter
    print(f"  Result: {final_answer} (correct: {correct_letter}) {'✓' if is_correct else '✗'}")

    return {
        "question_id": question["question_id"],
        "question": question["question"][:200],
        "predicted": final_answer,
        "target": correct_letter,
        "correct": is_correct,
        "num_steps": len(steps),
        "categories": question.get("categories", ""),
        "eval_mode": question.get("eval_mode", ""),
    }


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-examples", type=int, default=-1)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--model", type=str, default=MODEL)
    args = parser.parse_args()

    model = args.model
    output_dir = Path(args.output_dir)
    max_examples = args.max_examples if args.max_examples > 0 else None

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset from cache
    cache_path = Path("artifacts/bixbench_dataset.json")
    if cache_path.exists():
        print("Loading BixBench dataset from cache...")
        with open(cache_path) as f:
            questions = json.load(f)
    else:
        print("Loading BixBench dataset from HuggingFace...")
        ds = load_dataset("futurehouse/BixBench", split="train")
        questions = ds.to_list()
        # Save cache
        with open(cache_path, "w") as f:
            json.dump(questions, f)

    # Filter to questions with available capsule data
    available = []
    for q in questions:
        capsule_dir = DATA_DIR / q["data_folder"].replace(".zip", "")
        if capsule_dir.exists() and any(capsule_dir.iterdir()):
            available.append(q)
    print(f"Total: {len(questions)}, With data: {len(available)}, Skipped: {len(questions) - len(available)}")
    questions = available

    if max_examples:
        questions = questions[:max_examples]

    print(f"Running agentic evaluation on {len(questions)} questions with {model}")

    # Run evaluations
    results = []
    progress_file = output_dir / "progress.json"

    async with httpx.AsyncClient() as client:
        for i, question in enumerate(questions):
            try:
                result = await evaluate_question(question, client, i + 1, len(questions))
                results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "question_id": question["question_id"],
                    "predicted": "A",
                    "target": "?",
                    "correct": False,
                    "error": str(e),
                })

            # Save progress
            correct = sum(1 for r in results if r["correct"])
            total = len(results)
            print(f"  Progress: {correct}/{total} = {correct/total*100:.0f}%")

            with open(progress_file, "w") as f:
                json.dump(results, f, indent=2)

    # Final summary
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    print(f"\n{'='*60}")
    print(f"BixBench Agentic ({model}): {correct}/{total} = {correct/total*100:.1f}%")
    print(f"Results saved to {output_dir}")

    # Save final results
    with open(output_dir / "final_results.json", "w") as f:
        json.dump({
            "model": model,
            "accuracy": correct / total,
            "correct": correct,
            "total": total,
            "results": results,
        }, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
