"""Tests for LAB-Bench adapter."""
import pytest
from agesensei.eval.lab_bench_adapter import (
    AgeSenseiLabBenchAgent,
    AgentInput,
    BenchmarkResults,
    EvalResult,
)


class TestAgentInput:
    def test_basic_creation(self):
        inp = AgentInput(
            question="What is the function of TP53?",
            choices=["Tumor suppressor", "Kinase", "Protease", "Receptor"],
            subtask="LitQA2",
            ideal="A",
        )
        assert inp.question == "What is the function of TP53?"
        assert len(inp.choices) == 4
        assert inp.subtask == "LitQA2"


class TestAgeSenseiLabBenchAgent:
    def test_prompt_building(self):
        agent = AgeSenseiLabBenchAgent(model="test", use_tools=False)
        inp = AgentInput(
            question="Which protein is a senolytic target?",
            choices=["BCL-xL", "Insulin", "Hemoglobin", "Collagen"],
            subtask="LitQA2",
        )
        prompt = agent._build_prompt(inp, "")
        assert "BCL-xL" in prompt
        assert "A." in prompt
        assert "single letter" in prompt.lower() or "single best" in prompt.lower()

    def test_prompt_with_context(self):
        agent = AgeSenseiLabBenchAgent(model="test", use_tools=True)
        inp = AgentInput(
            question="Test question?",
            choices=["A", "B", "C"],
            subtask="LitQA2",
        )
        context = "BCL-xL is a key anti-apoptotic protein."
        prompt = agent._build_prompt(inp, context)
        assert "Retrieved Context" in prompt
        assert "BCL-xL" in prompt

    def test_extract_answer_single_letter(self):
        agent = AgeSenseiLabBenchAgent(model="test")
        assert agent._extract_answer("B", ["a", "b", "c", "d"]) == "B"
        assert agent._extract_answer("c", ["a", "b", "c", "d"]) == "C"

    def test_extract_answer_with_text(self):
        agent = AgeSenseiLabBenchAgent(model="test")
        assert agent._extract_answer("The answer is B", ["a", "b", "c"]) == "B"

    def test_extract_answer_fallback(self):
        agent = AgeSenseiLabBenchAgent(model="test")
        assert agent._extract_answer("???", ["a", "b", "c"]) == "A"


class TestBenchmarkResults:
    def test_creation(self):
        r = BenchmarkResults(
            eval_name="LitQA2",
            total=100,
            correct=65,
            accuracy=0.65,
            coverage=1.0,
        )
        assert r.accuracy == 0.65
        assert r.eval_name == "LitQA2"


class TestEvalResult:
    def test_correct(self):
        r = EvalResult(
            question_id="LitQA2_0",
            subtask="LitQA2",
            question="test",
            choices=["A", "B", "C"],
            ideal="B",
            predicted="B",
            correct=True,
        )
        assert r.correct is True

    def test_incorrect(self):
        r = EvalResult(
            question_id="LitQA2_1",
            subtask="LitQA2",
            question="test",
            choices=["A", "B", "C"],
            ideal="A",
            predicted="C",
            correct=False,
        )
        assert r.correct is False
