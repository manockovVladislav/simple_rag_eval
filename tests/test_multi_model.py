from __future__ import annotations

import time
import unittest

import pandas as pd

from rag_eval.config import AppConfig, GenerationConfig, MetricsConfig
from rag_eval.generation import RagAnswerGenerator
from rag_eval.metrics import MetricsCalculator, _provider_judge_column
from rag_eval.schemas import ContextItem, PipelineResult


class _DelayedClient:
    def __init__(self, answer: str | None = None, error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error

    def chat(self, _messages) -> str:
        time.sleep(0.15)
        if self.error:
            raise self.error
        return self.answer or ""


class MultiModelGenerationTests(unittest.TestCase):
    def test_parallel_generation_is_concurrent_and_isolates_model_error(self) -> None:
        config = AppConfig(
            generation=GenerationConfig(
                provider="gigachat",
                providers=["gigachat", "qwen", "glm"],
                parallel=True,
                max_workers=3,
                context_source="retriever",
            )
        )
        generator = RagAnswerGenerator(config)
        generator._clients = {
            "gigachat": _DelayedClient(answer="giga answer"),
            "qwen": _DelayedClient(error=TimeoutError("qwen timeout")),
            "glm": _DelayedClient(answer="glm answer"),
        }
        result = PipelineResult(retriever_contexts=[ContextItem(text="shared chunk")])

        started = time.monotonic()
        outputs = generator.generate_many("question", result)
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.35)
        self.assertEqual(outputs["gigachat"]["answer"], "giga answer")
        self.assertEqual(outputs["glm"]["answer"], "glm answer")
        self.assertIsNone(outputs["qwen"]["answer"])
        self.assertIn("TimeoutError", outputs["qwen"]["error"])

    def test_metrics_are_namespaced_for_every_answer_model(self) -> None:
        config = AppConfig(
            generation=GenerationConfig(
                provider="gigachat",
                providers=["gigachat", "qwen", "glm"],
                parallel=True,
            ),
            metrics=MetricsConfig(answer_enabled=True, ragas_enabled=False),
        )
        row = pd.Series(
            {
                "question_id": 1,
                "question": "Q",
                "ground_truth": "right answer",
                "answer": "right answer",
                "answer_gigachat": "right answer",
                "answer_qwen": "wrong",
                "answer_glm": "right answer",
                "error_qwen": "TimeoutError: timeout",
                "retriever_contexts": [],
                "reranker_contexts": [],
                "error": None,
            }
        )

        metrics = MetricsCalculator(config)._row_metrics(row)

        self.assertEqual(metrics["answer_gigachat_exact_match"], 1.0)
        self.assertEqual(metrics["answer_qwen_exact_match"], 0.0)
        self.assertTrue(metrics["answer_qwen_has_error"])
        self.assertEqual(
            _provider_judge_column("ragas_faithfulness_reason", "glm"),
            "ragas_glm_faithfulness_reason",
        )


if __name__ == "__main__":
    unittest.main()
