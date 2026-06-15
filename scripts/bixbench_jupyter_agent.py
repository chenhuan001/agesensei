"""BixBench Agentic Evaluator with real Jupyter kernel code execution.

Key design:
1. Real Jupyter kernel for code execution (not subprocess)
2. Multi-step ReAct loop: plan → code → execute → observe → iterate
3. GPT-5.5 as the reasoning model via OpenAI-compatible API
4. Data files mounted into working directory
5. Up to 15 steps per question
"""
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import httpx
import jupyter_client

# --- Config ---
API_KEY = os.environ.get("OPENAI_API_KEY", "sk-u5tv9FaqS9v8gZxlCNryNgvvCH6u3U6AgXdhp0IHtisWjqQ2")
API_BASE = os.environ.get("OPENAI_API_BASE", "https://www.micuapi.ai/v1")
MODEL = os.environ.get("MODEL", "gpt-5.5")
MAX_STEPS = 15
CODE_TIMEOUT = 120  # seconds per code execution
MAX_OUTPUT_CHARS = 8000  # truncate long outputs

CAPSULE_DIR = Path(__file__).parent.parent / "artifacts" / "bixbench_capsules"
DATASET_PATH = Path(__file__).parent.parent / "artifacts" / "bixbench_dataset.json"
RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "bixbench_jupyter"


# --- Jupyter Kernel Manager ---
class JupyterExecutor:
    """Manages a local Jupyter kernel for code execution."""

    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        self.km = jupyter_client.KernelManager(kernel_name="python3")
        self.km.start_kernel(cwd=work_dir)
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=30)
        # Set up the working directory and common imports
        self._setup_env()

    def _setup_env(self):
        """Pre-load common bioinformatics packages."""
        setup_code = f"""
import os
os.chdir("{self.work_dir}")
import warnings
warnings.filterwarnings('ignore')

# Pre-import common packages
import pandas as pd
import numpy as np
print("Environment ready. Working dir:", os.getcwd())
print("Available files:", os.listdir('.'))
"""
        self.execute(setup_code, timeout=30)

    def execute(self, code: str, timeout: int = CODE_TIMEOUT) -> dict:
        """Execute code in the kernel and return results."""
        msg_id = self.kc.execute(code)
        stdout_parts = []
        stderr_parts = []
        result_data = None
        error_info = None

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msg = self.kc.get_iopub_msg(timeout=min(5, deadline - time.time()))
            except Exception:
                break

            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = msg["msg_type"]
            content = msg["content"]

            if msg_type == "stream":
                if content["name"] == "stdout":
                    stdout_parts.append(content["text"])
                elif content["name"] == "stderr":
                    stderr_parts.append(content["text"])
            elif msg_type in ("execute_result", "display_data"):
                if "text/plain" in content.get("data", {}):
                    result_data = content["data"]["text/plain"]
            elif msg_type == "error":
                error_info = {
                    "ename": content.get("ename", ""),
                    "evalue": content.get("evalue", ""),
                    "traceback": "\n".join(content.get("traceback", []))
                }
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)

        output = ""
        if stdout:
            output += stdout
        if result_data:
            output += "\n" + result_data if output else result_data
        if stderr:
            output += "\n[stderr] " + stderr if output else "[stderr] " + stderr

        # Truncate very long outputs
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + f"\n... [truncated, {len(output)} total chars]"

        return {
            "output": output,
            "error": error_info,
            "success": error_info is None
        }

    def shutdown(self):
        """Clean up kernel."""
        try:
            self.kc.stop_channels()
            self.km.shutdown_kernel(now=True)
        except Exception:
            pass


# --- LLM API ---
async def call_llm(messages: list, client: httpx.AsyncClient) -> str:
    """Call GPT-5.5 via Anthropic-compatible API (micuapi wraps as Anthropic format)."""
    # Convert to Anthropic format
    system_msg = ""
    api_messages = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            api_messages.append({"role": m["role"], "content": m["content"]})

    payload = {
        "model": MODEL,
        "max_tokens": 4096,
        "messages": api_messages,
    }
    if system_msg:
        payload["system"] = system_msg

    for attempt in range(3):
        try:
            resp = await client.post(
                f"{API_BASE}/messages",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": API_KEY,
                },
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                # Extract text from Anthropic format
                content = data.get("content", [])
                if isinstance(content, list):
                    return "".join(c.get("text", "") for c in content if c.get("type") == "text")
                return str(content)
            else:
                print(f"  API error {resp.status_code}: {resp.text[:200]}")
                if resp.status_code in (429, 502, 503):
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                return ""
        except Exception as e:
            print(f"  API exception: {e}")
            await asyncio.sleep(5)
    return ""


# --- Agent Logic ---
SYSTEM_PROMPT = """You are an expert computational biologist and data analyst. You solve bioinformatics analysis tasks by writing and executing Python code.

RULES:
1. You MUST write Python code to analyze the data. Do NOT guess answers.
2. Write code in ```python blocks. Each block will be executed in a Jupyter kernel.
3. After seeing execution results, you can write more code to continue analysis.
4. Common packages available: pandas, numpy, scipy, statsmodels, lifelines, sklearn, matplotlib, seaborn, pydeseq2, gseapy, scanpy, openpyxl
5. For R-specific analyses (DESeq2, edgeR, clusterProfiler), use Python equivalents: pydeseq2, gseapy
6. Always print results clearly so you can see exact numerical values.
7. When you have the final numerical result, write: FINAL_RESULT: <your_number_or_value>
   Examples: FINAL_RESULT: 0.0002, FINAL_RESULT: 1.52, FINAL_RESULT: 1-50
8. Work step by step: first explore data structure, then perform analysis, then extract the exact answer.
9. If code errors, read the error carefully, fix and retry. Don't give up.
10. Pay attention to exact column names, data types, and file formats.
11. Print intermediate results at each step to verify your work.
12. Be precise with numerical values - print enough decimal places."""


def build_question_prompt(question: dict, file_list: str) -> str:
    """Build the initial prompt for a BixBench question."""
    q_text = question["question"]

    # Keep all options for matching later
    options = [question["ideal"]] + question["distractors"]

    prompt = f"""## Task
Analyze the provided bioinformatics data to answer this question by writing and executing Python code.

## Question
{q_text}

## Available Data Files
{file_list}

## Instructions
1. First, explore the data files to understand their structure (read headers, shapes, column names, dtypes)
2. Then perform the required analysis step by step, printing results at each stage
3. Be very precise with numerical values
4. When you have the final result, write: FINAL_RESULT: <value>

IMPORTANT: Do NOT guess. You MUST run code to compute the answer from the actual data.

Write your first code block to start exploring the data."""

    return prompt, options


def match_answer_to_option(result_text: str, options: list) -> tuple:
    """Match the agent's computed result to the closest MCQ option."""
    import re

    ideal = options[0]  # First option is always the ideal answer

    # Extract numerical value from result
    result_text = result_text.strip()

    # Try to parse as range (x, y)
    range_match = re.search(r'\(?\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*\)?', result_text)
    if range_match:
        try:
            result_val = (float(range_match.group(1)) + float(range_match.group(2))) / 2
        except:
            result_val = None
    else:
        # Try plain number
        num_match = re.search(r'([\d.eE+-]+)', result_text)
        if num_match:
            try:
                result_val = float(num_match.group(1))
            except:
                result_val = None
        else:
            result_val = None

    # For each option, compute distance
    best_option = None
    best_dist = float('inf')

    for i, opt in enumerate(options):
        opt_str = str(opt).strip()

        # Parse option as range
        opt_range = re.search(r'\(?\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*\)?', opt_str)
        if opt_range:
            opt_low = float(opt_range.group(1))
            opt_high = float(opt_range.group(2))
            opt_mid = (opt_low + opt_high) / 2
            if result_val is not None:
                # Check if result falls in range
                if opt_low <= result_val <= opt_high:
                    return i, opt_str, True
                dist = min(abs(result_val - opt_low), abs(result_val - opt_high))
            else:
                continue
        else:
            # Plain number or string
            try:
                opt_val = float(opt_str.replace('E', 'e'))
                if result_val is not None:
                    dist = abs(result_val - opt_val) / max(abs(opt_val), 1e-10)
                else:
                    # String match
                    if result_text.lower() == opt_str.lower():
                        return i, opt_str, True
                    continue
            except:
                # Non-numeric option (like "1-50", ">100")
                if result_text.strip() == opt_str.strip():
                    return i, opt_str, True
                # Try partial match
                if opt_str.replace('-', ' to ') in result_text or result_text in opt_str:
                    return i, opt_str, True
                continue

        if dist < best_dist:
            best_dist = dist
            best_option = (i, opt_str)

    if best_option is not None:
        is_correct = best_option[0] == 0  # options[0] is ideal
        return best_option[0], best_option[1], is_correct

    return -1, "", False


def extract_code_blocks(text: str) -> list:
    """Extract Python code blocks from LLM response."""
    blocks = []
    parts = text.split("```python")
    for part in parts[1:]:  # skip first part before any code block
        if "```" in part:
            code = part.split("```")[0].strip()
            if code:
                blocks.append(code)
    # Also try ```py
    parts = text.split("```py\n")
    for part in parts[1:]:
        if "```" in part:
            code = part.split("```")[0].strip()
            if code:
                blocks.append(code)
    return blocks


def extract_final_result(text: str) -> str:
    """Extract FINAL_RESULT value from text."""
    import re
    # Look for FINAL_RESULT: <value> pattern
    match = re.search(r"FINAL_RESULT:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Look for "the answer is <value>" pattern
    match = re.search(r"(?:the answer is|final answer is|result is)[:\s]+(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return ""


async def solve_question(question: dict, client: httpx.AsyncClient) -> dict:
    """Solve a single BixBench question using Jupyter kernel execution."""
    capsule_uuid = question["capsule_uuid"]
    capsule_path = CAPSULE_DIR / f"CapsuleFolder-{capsule_uuid}"

    if not capsule_path.exists():
        return {
            "question_id": question["question_id"],
            "correct": False,
            "answer": "",
            "expected": "",
            "reason": "capsule_missing",
            "num_steps": 0,
        }

    # Create temp working directory and copy data files
    work_dir = tempfile.mkdtemp(prefix="bixbench_")
    try:
        # Find data files in capsule
        data_files = []
        for root, dirs, files in os.walk(capsule_path):
            for f in files:
                src = os.path.join(root, f)
                rel = os.path.relpath(src, capsule_path)
                dst = os.path.join(work_dir, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                data_files.append(rel)

        # List files for prompt
        file_list = "\n".join(f"  - {f}" for f in sorted(data_files)[:50])
        if len(data_files) > 50:
            file_list += f"\n  ... and {len(data_files)-50} more files"

        # Build question prompt
        prompt, options = build_question_prompt(question, file_list)
        ideal = question["ideal"]

        # Start Jupyter kernel
        executor = JupyterExecutor(work_dir)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        final_result = ""
        num_steps = 0
        code_executed = False

        for step in range(MAX_STEPS):
            # Get LLM response
            response = await call_llm(messages, client)
            if not response:
                break

            # Check for final result
            result_val = extract_final_result(response)

            # Extract and execute code blocks
            code_blocks = extract_code_blocks(response)

            if code_blocks:
                # Execute each code block
                all_results = []
                for i, code in enumerate(code_blocks):
                    exec_result = executor.execute(code)
                    code_executed = True
                    num_steps = step + 1
                    if exec_result["error"]:
                        err = exec_result["error"]
                        all_results.append(f"ERROR:\n{err['ename']}: {err['evalue']}")
                    elif exec_result["output"]:
                        all_results.append(f"Output:\n{exec_result['output']}")
                    else:
                        all_results.append(f"Executed successfully (no output)")

                execution_feedback = "\n\n".join(all_results)
                messages.append({"role": "assistant", "content": response})

                if result_val:
                    # Has both code and final result - we're done
                    final_result = result_val
                    break
                else:
                    messages.append({"role": "user", "content": f"Execution results:\n{execution_feedback}\n\nContinue your analysis. When done, provide FINAL_RESULT: <value>"})
            elif result_val:
                # Final result without new code
                if code_executed:
                    final_result = result_val
                    num_steps = step + 1
                    break
                else:
                    # Force code execution first
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": "You must run code first before providing a result. Write Python code to compute the answer from the data."})
            else:
                # No code, no result - push for code
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "Please write Python code in a ```python block to analyze the data files. Do not guess."})

        # If no result after all steps, force one final answer
        if not final_result:
            messages.append({"role": "user", "content": "Based on all your analysis, what is the final numerical result? Reply with FINAL_RESULT: <value>"})
            response = await call_llm(messages, client)
            if response:
                final_result = extract_final_result(response)
                if not final_result:
                    # Extract any number from the response
                    import re
                    nums = re.findall(r'[\d.eE+-]+', response)
                    if nums:
                        final_result = nums[-1]

        executor.shutdown()

        # Match result to closest option
        if final_result:
            matched_idx, matched_opt, is_correct = match_answer_to_option(final_result, options)
        else:
            matched_idx, matched_opt, is_correct = -1, "", False

        return {
            "question_id": question["question_id"],
            "question": question["question"][:200],
            "computed_result": final_result,
            "matched_option": matched_opt,
            "ideal": ideal,
            "correct": is_correct,
            "num_steps": num_steps,
            "code_executed": code_executed,
            "reason": "completed",
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "question_id": question["question_id"],
            "correct": False,
            "answer": "",
            "expected": "",
            "reason": f"error: {str(e)[:200]}",
            "num_steps": 0,
        }
    finally:
        # Clean up temp dir
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except:
            pass


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=205)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    # Load dataset
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    # Filter to questions with available capsules
    available = []
    for q in dataset:
        capsule_path = CAPSULE_DIR / f"CapsuleFolder-{q['capsule_uuid']}"
        if capsule_path.exists():
            available.append(q)

    print(f"Total questions: {len(dataset)}, With capsule data: {len(available)}")

    # Select range
    questions = available[args.start:args.start + args.num]
    print(f"Running {len(questions)} questions (start={args.start})")

    # Create results dir
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing progress
    progress_file = RESULTS_DIR / "progress.json"
    completed = {}
    if progress_file.exists():
        existing = json.load(open(progress_file))
        completed = {r["question_id"]: r for r in existing}
        print(f"Resuming: {len(completed)} already completed")

    results = list(completed.values())
    correct_count = sum(1 for r in results if r["correct"])

    async with httpx.AsyncClient() as client:
        for i, q in enumerate(questions):
            if q["question_id"] in completed:
                continue

            print(f"\n{'='*60}")
            print(f"Q{i+1}/{len(questions)}: {q['question_id']}")
            print(f"  {q['question'][:100]}...")

            result = await solve_question(q, client)
            results.append(result)

            if result["correct"]:
                correct_count += 1
            total_done = sum(1 for r in results if r["reason"] != "capsule_missing")
            if total_done > 0:
                print(f"  Computed: {result.get('computed_result','')} → Matched: {result.get('matched_option','')} "
                      f"(ideal: {result.get('ideal','')}) "
                      f"{'✓' if result['correct'] else '✗'} "
                      f"Steps: {result['num_steps']} Code: {result.get('code_executed',False)} "
                      f"Running: {correct_count}/{total_done} = {correct_count/total_done*100:.0f}%")

            # Save progress after each question
            with open(progress_file, "w") as f:
                json.dump(results, f, indent=2)

    # Final summary
    total_done = sum(1 for r in results if r["reason"] != "capsule_missing")
    print(f"\n{'='*60}")
    print(f"FINAL: {correct_count}/{total_done} = {correct_count/total_done*100:.1f}%")
    print(f"Capsule missing: {sum(1 for r in results if r['reason'] == 'capsule_missing')}")

    # Save summary
    summary = {
        "model": MODEL,
        "total": total_done,
        "correct": correct_count,
        "accuracy": correct_count / max(total_done, 1),
        "capsule_missing": sum(1 for r in results if r["reason"] == "capsule_missing"),
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
