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

    def test_with_sources(self):
        inp = AgentInput(
            question="Test?",
            choices=["A", "B"],
            sources=["https://doi.org/10.1234/test"],
        )
        assert len(inp.sources) == 1


class TestAgeSenseiLabBenchAgent:
    def test_cot_prompt_building(self):
        agent = AgeSenseiLabBenchAgent(model="test", use_tools=False)
        inp = AgentInput(
            question="Which protein is a senolytic target?",
            choices=["BCL-xL", "Insulin", "Hemoglobin", "Collagen"],
            subtask="LitQA2",
        )
        prompt = agent._build_cot_prompt(inp, "")
        assert "BCL-xL" in prompt
        assert "A." in prompt
        assert "ANSWER: X" in prompt

    def test_cot_prompt_with_context(self):
        agent = AgeSenseiLabBenchAgent(model="test", use_tools=True)
        inp = AgentInput(
            question="Test question?",
            choices=["A", "B", "C"],
            subtask="LitQA2",
        )
        context = "BCL-xL is a key anti-apoptotic protein."
        prompt = agent._build_cot_prompt(inp, context)
        assert "Retrieved Research Context" in prompt
        assert "BCL-xL" in prompt

    def test_extract_answer_explicit_format(self):
        agent = AgeSenseiLabBenchAgent(model="test")
        response = """Let me think through this step by step.
        The protein BCL-xL is known for its anti-apoptotic role.
        Based on the evidence, the best answer is BCL-xL.
        ANSWER: A"""
        assert agent._extract_answer(response, ["a", "b", "c", "d"]) == "A"

    def test_extract_answer_with_reasoning(self):
        agent = AgeSenseiLabBenchAgent(model="test")
        response = "After careful analysis, the answer is B because..."
        assert agent._extract_answer(response, ["a", "b", "c"]) == "B"

    def test_extract_answer_single_letter(self):
        agent = AgeSenseiLabBenchAgent(model="test")
        assert agent._extract_answer("B", ["a", "b", "c", "d"]) == "B"
        assert agent._extract_answer("c", ["a", "b", "c", "d"]) == "C"

    def test_extract_answer_fallback(self):
        agent = AgeSenseiLabBenchAgent(model="test")
        assert agent._extract_answer("???", ["a", "b", "c"]) == "A"

    def test_extract_answer_last_line(self):
        agent = AgeSenseiLabBenchAgent(model="test")
        response = "Some reasoning here.\nMore analysis.\nC"
        assert agent._extract_answer(response, ["a", "b", "c"]) == "C"


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
