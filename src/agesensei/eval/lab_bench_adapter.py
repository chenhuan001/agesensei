"""LAB-Bench adapter — wraps AgeSensei agents as LAB-Bench compatible agent_fn.

LAB-Bench (Language Agent Biology Benchmark) evaluates AI systems on biology
research tasks across 8 categories: LitQA2, DbQA, SuppQA, FigQA, TableQA,
ProtocolQA, SeqQA, CloningScenarios.

This adapter routes each question to the appropriate AgeSensei agent/tool,
augmenting LLM responses with domain-specific retrieval and analysis.

Usage:
    from agesensei.eval import run_lab_bench

    results = await run_lab_bench(
        evals=["LitQA2", "DbQA", "SeqQA"],
        model="claude-sonnet-4-20250514",
        n_threads=4,
    )
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AgentInput / Evaluator protocol (minimal re-implementation for standalone use)
# ---------------------------------------------------------------------------

@dataclass
class AgentInput:
    """Mirrors labbench.AgentInput for standalone execution."""
    question: str
    choices: list[str]
    figures: list[Any] = field(default_factory=list)
    subtask: str = ""
    ideal: str = ""


@dataclass
class EvalResult:
    """Result for a single evaluation question."""
    question_id: str
    subtask: str
    question: str
    choices: list[str]
    ideal: str
    predicted: str
    correct: bool
    reasoning: str = ""
    tools_used: list[str] = field(default_factory=list)


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    eval_name: str
    total: int
    correct: int
    accuracy: float
    coverage: float  # fraction of questions answered (non-abstention)
    per_subtask: dict[str, dict[str, float]] = field(default_factory=dict)
    results: list[EvalResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool-augmented reasoning
# ---------------------------------------------------------------------------

async def _search_literature(query: str) -> str:
    """Use AgeSensei's literature tools to retrieve context."""
    from agesensei.tools.pubmed import search as pubmed_search
    from agesensei.tools.semantic_scholar import search as ss_search

    results = []
    try:
        pubmed_results = await pubmed_search(query, max_results=3)
        results.extend(pubmed_results)
    except Exception as e:
        logger.debug(f"PubMed search failed: {e}")

    try:
        ss_results = await ss_search(query, max_results=3)
        results.extend(ss_results)
    except Exception as e:
        logger.debug(f"Semantic Scholar search failed: {e}")

    if not results:
        return ""

    context_parts = []
    for r in results[:5]:
        title = r.get("title", "") if isinstance(r, dict) else str(r)
        abstract = r.get("abstract", "") if isinstance(r, dict) else ""
        context_parts.append(f"- {title}\n  {abstract[:300]}")

    return "\n".join(context_parts)


async def _query_protein_db(query: str) -> str:
    """Use AgeSensei's protein/DB tools for database questions."""
    from agesensei.tools.uniprot import search as uniprot_search

    try:
        results = await uniprot_search(query, max_results=3)
        if results:
            return json.dumps(results[:3], indent=2, default=str)
    except Exception as e:
        logger.debug(f"UniProt search failed: {e}")

    return ""


async def _analyze_sequence(query: str) -> str:
    """Use ESM-2 for sequence-related questions."""
    # For SeqQA, we can provide sequence embeddings or basic analysis
    # This is a lightweight hook — full ESM-2 inference is optional
    return ""


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

SUBTASK_TOOL_MAP = {
    "LitQA2": _search_literature,
    "DbQA": _query_protein_db,
    "SeqQA": _analyze_sequence,
    "SuppQA": _search_literature,
    "ProtocolQA": _search_literature,
    # FigQA, TableQA, CloningScenarios → LLM-only (no tool augmentation)
}


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------

class AgeSenseiLabBenchAgent:
    """LAB-Bench compatible agent backed by AgeSensei tools + LLM.

    Routes questions to domain-specific tools for retrieval augmentation,
    then uses an LLM to select the best answer.

    Attributes:
        model: LLM model identifier.
        use_tools: Whether to augment with AgeSensei tools.
        api_key: API key for LLM provider.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        use_tools: bool = True,
        api_key: str | None = None,
    ):
        self.model = model
        self.use_tools = use_tools
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    async def __call__(self, input: AgentInput) -> str:
        """Answer a LAB-Bench question. Returns a single letter (A-E).

        This is the LAB-Bench agent_fn interface.
        """
        return await self.answer(input)

    async def answer(self, input: AgentInput) -> str:
        """Core reasoning: retrieve context → build prompt → call LLM → extract answer."""
        tools_used = []
        context = ""

        # Step 1: Tool-augmented retrieval
        if self.use_tools and input.subtask in SUBTASK_TOOL_MAP:
            tool_fn = SUBTASK_TOOL_MAP[input.subtask]
            try:
                context = await tool_fn(input.question)
                if context:
                    tools_used.append(input.subtask)
            except Exception as e:
                logger.warning(f"Tool augmentation failed for {input.subtask}: {e}")

        # Step 2: Build prompt
        prompt = self._build_prompt(input, context)

        # Step 3: Call LLM
        answer = await self._call_llm(prompt)

        # Step 4: Extract single letter
        return self._extract_answer(answer, input.choices)

    def _build_prompt(self, input: AgentInput, context: str) -> str:
        """Construct the LLM prompt with question, choices, and retrieved context."""
        choices_text = "\n".join(
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(input.choices)
        )

        parts = [
            "You are an expert biology researcher. Answer the following multiple-choice question.",
            "Select the single best answer and respond with ONLY the letter (A, B, C, D, or E).",
            "",
        ]

        if context:
            parts.extend([
                "## Retrieved Context",
                context,
                "",
            ])

        parts.extend([
            "## Question",
            input.question,
            "",
            "## Choices",
            choices_text,
            "",
            "Answer (single letter only):",
        ])

        return "\n".join(parts)

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM API."""
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            response = await client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except ImportError:
            # Fallback: try openai-compatible
            return await self._call_llm_openai(prompt)

    async def _call_llm_openai(self, prompt: str) -> str:
        """Fallback LLM call using OpenAI-compatible API."""
        try:
            import openai

            client = openai.AsyncOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", ""),
            )
            response = await client.chat.completions.create(
                model=self.model if "gpt" in self.model else "gpt-4o",
                max_tokens=1,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "A"  # Default fallback

    def _extract_answer(self, response: str, choices: list[str]) -> str:
        """Extract a single letter answer from LLM response."""
        import re

        response = response.strip().upper()
        valid = set(chr(65 + i) for i in range(len(choices)))

        # Direct single letter
        if len(response) == 1 and response in valid:
            return response

        # Look for standalone letter patterns: "B", "(B)", "B.", "B)"
        match = re.search(r'\b([A-E])\b', response)
        if match and match.group(1) in valid:
            return match.group(1)

        # Last resort: last valid letter in response
        for char in reversed(response):
            if char in valid:
                return char

        return "A"  # Fallback


# ---------------------------------------------------------------------------
# Dataset loading (HuggingFace or local)
# ---------------------------------------------------------------------------

async def _load_dataset(eval_name: str) -> list[AgentInput]:
    """Load LAB-Bench dataset from HuggingFace."""
    try:
        from datasets import load_dataset

        ds = load_dataset("futurehouse/lab-bench", eval_name, split="train")
        inputs = []
        for row in ds:
            # Choices = ideal + distractors, shuffled
            import random
            choices = [row["ideal"]] + list(row.get("distractors", []))
            random.shuffle(choices)
            ideal_idx = choices.index(row["ideal"])
            ideal_letter = chr(65 + ideal_idx)

            inputs.append(AgentInput(
                question=row["question"],
                choices=choices,
                subtask=eval_name,
                ideal=ideal_letter,
            ))
        return inputs
    except ImportError:
        logger.error("datasets package required: pip install datasets")
        return []
    except Exception as e:
        logger.error(f"Failed to load dataset {eval_name}: {e}")
        return []


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_lab_bench(
    evals: list[str] | None = None,
    model: str = "claude-sonnet-4-20250514",
    use_tools: bool = True,
    n_threads: int = 4,
    max_questions: int | None = None,
    output_path: str | None = None,
) -> dict[str, BenchmarkResults]:
    """Run LAB-Bench evaluation with AgeSensei tool augmentation.

    Args:
        evals: List of eval names. Defaults to ["LitQA2", "DbQA", "SeqQA"].
        model: LLM model to use.
        use_tools: Whether to augment with AgeSensei retrieval tools.
        n_threads: Concurrent evaluation threads.
        max_questions: Limit questions per eval (for quick testing).
        output_path: Save results JSON to this path.

    Returns:
        Dict mapping eval name -> BenchmarkResults.
    """
    if evals is None:
        evals = ["LitQA2", "DbQA", "SeqQA"]

    agent = AgeSenseiLabBenchAgent(model=model, use_tools=use_tools)
    all_results: dict[str, BenchmarkResults] = {}

    for eval_name in evals:
        print(f"\n{'='*60}")
        print(f"  Running LAB-Bench: {eval_name}")
        print(f"{'='*60}")

        questions = await _load_dataset(eval_name)
        if not questions:
            print(f"  Skipped: could not load dataset")
            continue

        if max_questions:
            questions = questions[:max_questions]

        print(f"  Questions: {len(questions)} | Model: {model} | Tools: {use_tools}")

        # Run with semaphore for concurrency control
        sem = asyncio.Semaphore(n_threads)
        eval_results: list[EvalResult] = []

        async def evaluate_one(q: AgentInput, idx: int) -> EvalResult:
            async with sem:
                predicted = await agent(q)
                correct = predicted == q.ideal
                if (idx + 1) % 10 == 0:
                    print(f"    [{idx+1}/{len(questions)}] acc so far: "
                          f"{sum(1 for r in eval_results if r.correct)}/{len(eval_results)}")
                return EvalResult(
                    question_id=f"{eval_name}_{idx}",
                    subtask=q.subtask,
                    question=q.question[:100],
                    choices=q.choices,
                    ideal=q.ideal,
                    predicted=predicted,
                    correct=correct,
                )

        tasks = [evaluate_one(q, i) for i, q in enumerate(questions)]
        eval_results = await asyncio.gather(*tasks)

        # Compute metrics
        total = len(eval_results)
        correct = sum(1 for r in eval_results if r.correct)
        accuracy = correct / total if total > 0 else 0.0

        benchmark = BenchmarkResults(
            eval_name=eval_name,
            total=total,
            correct=correct,
            accuracy=accuracy,
            coverage=1.0,  # We always answer
            results=eval_results,
        )

        all_results[eval_name] = benchmark
        print(f"\n  Result: {correct}/{total} = {accuracy:.1%} accuracy")

    # Save results
    if output_path:
        output = {
            name: {
                "eval_name": r.eval_name,
                "total": r.total,
                "correct": r.correct,
                "accuracy": r.accuracy,
                "coverage": r.coverage,
            }
            for name, r in all_results.items()
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return all_results


# ---------------------------------------------------------------------------
# Comparison runner (with vs without tools)
# ---------------------------------------------------------------------------

async def run_ablation(
    evals: list[str] | None = None,
    model: str = "claude-sonnet-4-20250514",
    max_questions: int = 50,
) -> dict[str, Any]:
    """Run with/without AgeSensei tools to measure retrieval augmentation impact.

    Returns:
        Comparison dict with accuracy for both conditions.
    """
    print("\n" + "=" * 60)
    print("  LAB-Bench Ablation: AgeSensei Tools vs Baseline")
    print("=" * 60)

    # With tools
    print("\n--- WITH AgeSensei tools ---")
    with_tools = await run_lab_bench(
        evals=evals, model=model, use_tools=True, max_questions=max_questions
    )

    # Without tools
    print("\n--- WITHOUT tools (baseline) ---")
    without_tools = await run_lab_bench(
        evals=evals, model=model, use_tools=False, max_questions=max_questions
    )

    # Compare
    print("\n" + "=" * 60)
    print("  Ablation Results")
    print("=" * 60)
    print(f"  {'Eval':<12} {'With Tools':>12} {'Baseline':>12} {'Delta':>8}")
    print(f"  {'-'*44}")

    comparison = {}
    for name in with_tools:
        wt = with_tools[name].accuracy
        bl = without_tools.get(name, BenchmarkResults(name, 0, 0, 0.0, 0.0)).accuracy
        delta = wt - bl
        print(f"  {name:<12} {wt:>11.1%} {bl:>11.1%} {delta:>+7.1%}")
        comparison[name] = {"with_tools": wt, "baseline": bl, "delta": delta}

    return comparison
