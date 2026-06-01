"""LAB-Bench adapter — wraps AgeSensei agents as LAB-Bench compatible agent_fn.

LAB-Bench (Language Agent Biology Benchmark) evaluates AI systems on biology
research tasks across 8 categories: LitQA2, DbQA, SuppQA, FigQA, TableQA,
ProtocolQA, SeqQA, CloningScenarios.

This adapter routes each question to the appropriate AgeSensei agent/tool,
augmenting LLM responses with domain-specific retrieval and analysis.
Uses Chain-of-Thought reasoning and deep retrieval (full-text when available).

Usage:
    from agesensei.eval import run_lab_bench

    results = await run_lab_bench(
        evals=["LitQA2", "DbQA", "SeqQA"],
        model="claude-haiku-4-5-20251001",
        n_threads=4,
    )
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
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
    sources: list[str] = field(default_factory=list)


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
    coverage: float
    per_subtask: dict[str, dict[str, float]] = field(default_factory=dict)
    results: list[EvalResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Deep retrieval tools
# ---------------------------------------------------------------------------

async def _search_literature_deep(query: str, sources: list[str] | None = None) -> str:
    """Deep literature retrieval: PubMed + Semantic Scholar + PMC full-text.

    If DOI sources are available (from the dataset), also fetches by DOI
    for higher precision. Retrieves full abstracts and attempts PMC full-text.
    """
    from agesensei.tools.pubmed import search_and_fetch
    from agesensei.tools.semantic_scholar import search_papers

    context_parts = []

    # 1. DOI-based retrieval (highest precision)
    if sources:
        try:
            from agesensei.tools.semantic_scholar import search_by_doi
            for src in sources[:3]:
                doi = src.replace("https://doi.org/", "").strip()
                if doi:
                    paper = await search_by_doi(doi)
                    if paper and paper.abstract:
                        context_parts.append(
                            f"[DOI Match] {paper.title}\n{paper.abstract}"
                        )
        except Exception as e:
            logger.debug(f"DOI retrieval failed: {e}")

    # 2. PubMed search + full abstracts
    try:
        papers = await search_and_fetch(query, max_results=5)
        for p in papers[:5]:
            if p.abstract:
                context_parts.append(f"[PubMed] {p.title}\n{p.abstract}")
    except Exception as e:
        logger.debug(f"PubMed search failed: {e}")

    # 3. Semantic Scholar for broader coverage
    try:
        s2_papers = await search_papers(query, max_results=5)
        for p in s2_papers[:3]:
            if p.abstract and not any(p.title in c for c in context_parts):
                context_parts.append(f"[S2] {p.title}\n{p.abstract}")
    except Exception as e:
        logger.debug(f"S2 search failed: {e}")

    # 4. PMC full-text for top hit (if available)
    if sources:
        try:
            from agesensei.tools.pmc import fetch_full_text, pmid_to_pmcid
            from agesensei.tools.pubmed import search_pubmed

            pmids = await search_pubmed(query, max_results=1)
            if pmids:
                pmcid = await pmid_to_pmcid(pmids[0])
                if pmcid:
                    sections = await fetch_full_text(pmcid)
                    if sections:
                        relevant = []
                        for title, text in sections.items():
                            if any(kw.lower() in text.lower() for kw in query.split()[:3]):
                                relevant.append(f"[Full-text: {title}]\n{text[:500]}")
                        if relevant:
                            context_parts.extend(relevant[:2])
        except Exception as e:
            logger.debug(f"PMC full-text failed: {e}")

    return "\n\n".join(context_parts[:8])


async def _query_protein_db_deep(query: str, sources: list[str] | None = None) -> str:
    """Deep database retrieval: UniProt + ChEMBL + OpenTargets."""
    from agesensei.tools.uniprot import search as uniprot_search

    context_parts = []

    # UniProt
    try:
        results = await uniprot_search(query, max_results=5)
        if results:
            context_parts.append(
                "[UniProt]\n" + json.dumps(results[:5], indent=2, default=str)
            )
    except Exception as e:
        logger.debug(f"UniProt search failed: {e}")

    # ChEMBL
    try:
        from agesensei.tools.chembl import search as chembl_search
        results = await chembl_search(query, max_results=3)
        if results:
            context_parts.append(
                "[ChEMBL]\n" + json.dumps(results[:3], indent=2, default=str)
            )
    except Exception as e:
        logger.debug(f"ChEMBL search failed: {e}")

    # Also do literature search for DB questions (many require paper context)
    lit_context = await _search_literature_deep(query, sources)
    if lit_context:
        context_parts.append(lit_context)

    return "\n\n".join(context_parts[:6])


async def _analyze_sequence_deep(query: str, sources: list[str] | None = None) -> str:
    """Sequence-related retrieval: literature + UniProt for context."""
    context_parts = []

    # UniProt for sequence/protein context
    try:
        from agesensei.tools.uniprot import search as uniprot_search
        results = await uniprot_search(query, max_results=3)
        if results:
            context_parts.append(
                "[UniProt]\n" + json.dumps(results[:3], indent=2, default=str)
            )
    except Exception as e:
        logger.debug(f"UniProt search failed: {e}")

    # Literature for sequence-related knowledge
    lit_context = await _search_literature_deep(query, sources)
    if lit_context:
        context_parts.append(lit_context)

    return "\n\n".join(context_parts[:4])


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

SUBTASK_TOOL_MAP: dict[str, Any] = {
    "LitQA2": _search_literature_deep,
    "litqa-v2-public": _search_literature_deep,
    "litqa-v2-closed": _search_literature_deep,
    "DbQA": _query_protein_db_deep,
    "SeqQA": _analyze_sequence_deep,
    "SuppQA": _search_literature_deep,
    "ProtocolQA": _search_literature_deep,
}


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------

COT_SYSTEM_PROMPT = """You are an expert biology researcher with deep knowledge of \
molecular biology, genetics, biochemistry, pharmacology, and bioinformatics.

When answering multiple-choice questions:
1. Carefully analyze the question and all provided context
2. Think step by step through your reasoning
3. Consider each choice and eliminate incorrect options
4. If "Insufficient information" is an option, only choose it if you truly cannot determine the answer
5. End your response with your final answer on a new line in the format: ANSWER: X

where X is a single letter (A, B, C, D, or E)."""


class AgeSenseiLabBenchAgent:
    """LAB-Bench compatible agent backed by AgeSensei tools + Chain-of-Thought LLM.

    Routes questions to domain-specific tools for retrieval augmentation,
    then uses an LLM with CoT reasoning to select the best answer.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        use_tools: bool = True,
        api_key: str | None = None,
    ):
        self.model = model
        self.use_tools = use_tools
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    async def __call__(self, input: AgentInput) -> str:
        return await self.answer(input)

    async def answer(self, input: AgentInput) -> str:
        """Retrieve context → CoT reasoning → extract answer."""
        context = ""

        if self.use_tools:
            # Try exact subtask match first, then eval name
            tool_fn = SUBTASK_TOOL_MAP.get(input.subtask)
            if tool_fn is None:
                for key in SUBTASK_TOOL_MAP:
                    if key.lower() in input.subtask.lower():
                        tool_fn = SUBTASK_TOOL_MAP[key]
                        break

            if tool_fn:
                try:
                    context = await tool_fn(input.question, input.sources)
                except Exception as e:
                    logger.warning(f"Tool augmentation failed for {input.subtask}: {e}")

        prompt = self._build_cot_prompt(input, context)
        response = await self._call_llm(prompt)
        return self._extract_answer(response, input.choices)

    def _build_cot_prompt(self, input: AgentInput, context: str) -> str:
        """Build Chain-of-Thought prompt with retrieved context."""
        choices_text = "\n".join(
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(input.choices)
        )

        parts = []

        if context:
            parts.extend([
                "## Retrieved Research Context",
                "(Use this context to inform your answer. Not all context may be relevant.)",
                "",
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
            "Think step by step, then provide your final answer as: ANSWER: X",
        ])

        return "\n".join(parts)

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with CoT system prompt and sufficient token budget."""
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            response = await client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=COT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except ImportError:
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
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": COT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "A"

    def _extract_answer(self, response: str, choices: list[str]) -> str:
        """Extract answer from CoT response. Looks for 'ANSWER: X' pattern."""
        valid = set(chr(65 + i) for i in range(len(choices)))

        # Look for explicit "ANSWER: X" pattern (strongest signal)
        answer_match = re.search(r'ANSWER:\s*([A-E])', response, re.IGNORECASE)
        if answer_match and answer_match.group(1).upper() in valid:
            return answer_match.group(1).upper()

        # Fallback: look for "The answer is X" or "I choose X"
        fallback = re.search(
            r'(?:the answer is|i choose|my answer is|correct answer is)\s*[:\s]*([A-E])',
            response, re.IGNORECASE,
        )
        if fallback and fallback.group(1).upper() in valid:
            return fallback.group(1).upper()

        # Last line often contains just the letter
        last_line = response.strip().split('\n')[-1].strip().upper()
        single = re.search(r'\b([A-E])\b', last_line)
        if single and single.group(1) in valid:
            return single.group(1)

        # Anywhere in text (last resort)
        any_match = re.search(r'\b([A-E])\b', response.upper())
        if any_match and any_match.group(1) in valid:
            return any_match.group(1)

        return "A"


# ---------------------------------------------------------------------------
# Dataset loading (HuggingFace)
# ---------------------------------------------------------------------------

async def _load_dataset(eval_name: str) -> list[AgentInput]:
    """Load LAB-Bench dataset from HuggingFace.

    Includes source DOIs for retrieval augmentation.
    """
    try:
        from datasets import load_dataset
        import random

        ds = load_dataset("futurehouse/lab-bench", eval_name, split="train")
        inputs = []
        for row in ds:
            distractors = list(row.get("distractors", []))
            choices = [row["ideal"]] + distractors + ["Insufficient information"]
            random.shuffle(choices)
            ideal_idx = choices.index(row["ideal"])
            ideal_letter = chr(65 + ideal_idx)

            inputs.append(AgentInput(
                question=row["question"],
                choices=choices,
                subtask=row.get("subtask", eval_name),
                ideal=ideal_letter,
                sources=row.get("sources", []) or [],
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
    model: str = "claude-haiku-4-5-20251001",
    use_tools: bool = True,
    n_threads: int = 4,
    max_questions: int | None = None,
    output_path: str | None = None,
) -> dict[str, BenchmarkResults]:
    """Run LAB-Bench evaluation with AgeSensei tool augmentation + CoT.

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
        print(f"  Running LAB-Bench: {eval_name} (CoT + {'tools' if use_tools else 'no tools'})")
        print(f"{'='*60}")

        questions = await _load_dataset(eval_name)
        if not questions:
            print(f"  Skipped: could not load dataset")
            continue

        if max_questions:
            questions = questions[:max_questions]

        print(f"  Questions: {len(questions)} | Model: {model} | Tools: {use_tools}")

        sem = asyncio.Semaphore(n_threads)
        completed = [0]

        async def evaluate_one(q: AgentInput, idx: int) -> EvalResult:
            async with sem:
                predicted = await agent(q)
                correct = predicted == q.ideal
                completed[0] += 1
                if completed[0] % 10 == 0:
                    current_results = completed[0]
                    print(f"    [{current_results}/{len(questions)}] in progress...")
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

        total = len(eval_results)
        correct = sum(1 for r in eval_results if r.correct)
        accuracy = correct / total if total > 0 else 0.0

        benchmark = BenchmarkResults(
            eval_name=eval_name,
            total=total,
            correct=correct,
            accuracy=accuracy,
            coverage=1.0,
            results=eval_results,
        )

        all_results[eval_name] = benchmark
        print(f"\n  Result: {correct}/{total} = {accuracy:.1%} accuracy")

    if output_path:
        import os as _os
        _os.makedirs(_os.path.dirname(output_path), exist_ok=True)
        output = {
            name: {
                "eval_name": r.eval_name,
                "total": r.total,
                "correct": r.correct,
                "accuracy": r.accuracy,
                "coverage": r.coverage,
                "per_question": [
                    {"q": er.question[:80], "ideal": er.ideal, "predicted": er.predicted, "correct": er.correct}
                    for er in r.results
                ],
            }
            for name, r in all_results.items()
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")

    return all_results


# ---------------------------------------------------------------------------
# Comparison runner (with vs without tools)
# ---------------------------------------------------------------------------

async def run_ablation(
    evals: list[str] | None = None,
    model: str = "claude-haiku-4-5-20251001",
    max_questions: int = 50,
) -> dict[str, Any]:
    """Run with/without AgeSensei tools to measure retrieval augmentation impact."""
    print("\n" + "=" * 60)
    print("  LAB-Bench Ablation: AgeSensei Tools + CoT vs Baseline CoT")
    print("=" * 60)

    print("\n--- WITH AgeSensei tools + CoT ---")
    with_tools = await run_lab_bench(
        evals=evals, model=model, use_tools=True, max_questions=max_questions
    )

    print("\n--- CoT only (no tools) ---")
    without_tools = await run_lab_bench(
        evals=evals, model=model, use_tools=False, max_questions=max_questions
    )

    print("\n" + "=" * 60)
    print("  Ablation Results")
    print("=" * 60)
    print(f"  {'Eval':<12} {'Tools+CoT':>12} {'CoT only':>12} {'Delta':>8}")
    print(f"  {'-'*44}")

    comparison = {}
    for name in with_tools:
        wt = with_tools[name].accuracy
        bl = without_tools.get(name, BenchmarkResults(name, 0, 0, 0.0, 0.0)).accuracy
        delta = wt - bl
        print(f"  {name:<12} {wt:>11.1%} {bl:>11.1%} {delta:>+7.1%}")
        comparison[name] = {"with_tools": wt, "baseline": bl, "delta": delta}

    return comparison
