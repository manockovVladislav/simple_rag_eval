from __future__ import annotations

from types import SimpleNamespace
import unittest

from rag_eval.config import AppConfig, MetricsConfig
from rag_eval.custom_judge_evaluator import CustomJudgeEvaluator, _parse_judge_json, _stable_score


class _FlakyJudge:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("isolated API timeout")
        return SimpleNamespace(
            content='```json\n{"score": 0.71, "reason": "в основном верно", "evidence": ["опора"]}\n```'
        )


class CustomJudgeTests(unittest.TestCase):
    def test_custom_judge_retries_api_error_and_normalizes_score(self) -> None:
        config = AppConfig(metrics=MetricsConfig(ragas_max_retries=1))
        evaluator = CustomJudgeEvaluator(config)
        evaluator._llm = _FlakyJudge()

        result = evaluator._evaluate_metric(
            "answer_relevancy",
            {"question": "Q", "answer": "A", "ground_truth": "", "contexts": []},
        )

        self.assertEqual(evaluator._llm.calls, 2)
        self.assertEqual(result["score"], 0.7)

    def test_parser_extracts_first_balanced_json_object(self) -> None:
        parsed = _parse_judge_json('text {"score": "0,5", "reason": "ok", "evidence": []} tail {bad}')
        self.assertEqual(_stable_score(parsed["score"]), 0.5)
