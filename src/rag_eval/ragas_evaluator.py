from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from rag_eval.config import AppConfig, ModelConfig
from rag_eval.io import contexts_from_json
from rag_eval.llm_clients import GigaChatApiClient


@dataclass(slots=True)
class RagasEvaluationRow:
    question_id: Any
    metrics: dict[str, float | None]
    error: str | None = None


class RagasEvaluator:
    def __init__(self, config: AppConfig):
        self.config = config

    def evaluate_rows(self, frame: pd.DataFrame) -> list[RagasEvaluationRow]:
        self._ensure_ragas_installed()
        results: list[RagasEvaluationRow] = []
        for _, row in frame.iterrows():
            results.append(self._evaluate_row(row))
        return results

    def _evaluate_row(self, row: pd.Series) -> RagasEvaluationRow:
        question_id = row.get("question_id")
        all_metrics: dict[str, float | None] = {}
        errors: list[str] = []

        for source in ("retriever", "reranker"):
            sample = self._make_sample(row, source)
            if sample is None:
                continue
            metrics = self._context_metrics_for_sample(sample)
            if not metrics:
                continue
            try:
                values = self._evaluate_one(sample, metrics)
                all_metrics.update({f"ragas_{source}_{key}": value for key, value in values.items()})
            except Exception as exc:
                errors.append(f"{source}: {type(exc).__name__}: {exc}")

        answer_sample = self._make_sample(row, self.config.metrics.ragas_context_source, fallback=True)
        if answer_sample is not None:
            metrics = self._answer_metrics_for_sample(answer_sample)
            if metrics:
                try:
                    values = self._evaluate_one(answer_sample, metrics)
                    all_metrics.update({f"ragas_{key}": value for key, value in values.items()})
                except Exception as exc:
                    errors.append(f"answer: {type(exc).__name__}: {exc}")

        error = "; ".join(errors) if errors else None
        if not all_metrics and not error:
            error = "Not enough data for Ragas metrics."
        return RagasEvaluationRow(question_id, all_metrics, error)

    def _evaluate_one(self, sample: dict[str, Any], metrics: list[Any]) -> dict[str, float | None]:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.run_config import RunConfig

        dataset = Dataset.from_list([sample])
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self._make_langchain_llm(),
            run_config=RunConfig(
                timeout=self.config.metrics.ragas_timeout_seconds,
                max_workers=self.config.metrics.ragas_max_workers,
                max_retries=self.config.metrics.ragas_max_retries,
            ),
            batch_size=1,
            raise_exceptions=False,
            show_progress=False,
        )
        result_frame = result.to_pandas()
        if result_frame.empty:
            return {}
        first = result_frame.iloc[0]
        return {
            metric.name: _none_if_nan(first.get(metric.name))
            for metric in metrics
        }

    def _make_sample(self, row: pd.Series, source: str, fallback: bool = False) -> dict[str, Any] | None:
        question = _clean_text(row.get("question"))
        answer = _clean_text(row.get("answer"))
        contexts = self._contexts(row, source, fallback)
        reference = _clean_text(row.get("expected_answer"))
        if not question or not answer or not contexts:
            return None
        sample = {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
        }
        if reference:
            sample["reference"] = reference
        return sample

    def _contexts(self, row: pd.Series, source: str, fallback: bool) -> list[str]:
        primary_column = "reranker_contexts" if source == "reranker" else "retriever_contexts"
        fallback_column = "retriever_contexts" if primary_column == "reranker_contexts" else "reranker_contexts"
        contexts = _context_texts(row.get(primary_column))
        if contexts or not fallback:
            return contexts
        return _context_texts(row.get(fallback_column))

    def _context_metrics_for_sample(self, sample: dict[str, Any]) -> list[Any]:
        from ragas.metrics import context_precision, context_recall

        configured = set(self.config.metrics.ragas_metrics)
        metrics: list[Any] = []
        if "reference" in sample:
            if "context_precision" in configured:
                metrics.append(context_precision)
            if "context_recall" in configured:
                metrics.append(context_recall)
        return metrics

    def _answer_metrics_for_sample(self, sample: dict[str, Any]) -> list[Any]:
        from ragas.metrics import answer_correctness, faithfulness

        configured = set(self.config.metrics.ragas_metrics)
        metrics: list[Any] = []
        if "faithfulness" in configured:
            metrics.append(faithfulness)
        if "reference" in sample:
            if "answer_correctness" in configured:
                metrics.append(answer_correctness)
        return metrics

    def _make_langchain_llm(self):
        provider = self.config.metrics.ragas_judge_provider
        if provider not in self.config.models:
            raise ValueError(f"Ragas judge provider '{provider}' is not configured.")
        model_config = self.config.models[provider]
        if model_config.provider == "qwen_transformers":
            return _make_transformers_llm(model_config)
        return _make_openai_compatible_llm(model_config)

    def _ensure_ragas_installed(self) -> None:
        try:
            import datasets  # noqa: F401
            import ragas  # noqa: F401
        except ImportError as exc:
            raise ImportError("Ragas metrics require packages: ragas and datasets.") from exc


def _make_openai_compatible_llm(model_config: ModelConfig):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_config.model,
        base_url=_openai_base_url(model_config.base_url),
        api_key=_api_key(model_config),
        temperature=model_config.temperature,
        timeout=model_config.timeout_seconds,
        max_retries=0,
    )


def _make_transformers_llm(model_config: ModelConfig):
    from langchain_community.llms import HuggingFacePipeline

    model_kwargs: dict[str, Any] = {}
    if model_config.device_map:
        model_kwargs["device_map"] = model_config.device_map
    if model_config.torch_dtype:
        model_kwargs["torch_dtype"] = model_config.torch_dtype

    pipeline_kwargs = {
        "max_new_tokens": model_config.max_new_tokens,
        "do_sample": model_config.do_sample,
        "temperature": model_config.temperature,
        "return_full_text": model_config.return_full_text,
    }

    kwargs: dict[str, Any] = {
        "model_id": model_config.model,
        "task": model_config.task,
        "model_kwargs": model_kwargs,
        "pipeline_kwargs": pipeline_kwargs,
        "batch_size": 1,
    }
    if model_config.device is not None:
        kwargs["device"] = model_config.device
    return HuggingFacePipeline.from_model_id(**kwargs)


def _api_key(config: ModelConfig) -> str:
    if config.provider == "qwen_local" or config.auth_type == "none":
        return "EMPTY"
    if config.provider == "gigachat_api":
        return GigaChatApiClient(config).get_bearer_token()
    if config.auth_type == "bearer_env" and config.api_key_env:
        return os.getenv(config.api_key_env, "")
    return "EMPTY"


def _openai_base_url(value: str) -> str:
    suffix = "/chat/completions"
    return value[: -len(suffix)] if value.endswith(suffix) else value


def _context_texts(value: Any) -> list[str]:
    contexts = contexts_from_json(value)
    texts = []
    for context in contexts:
        text = _clean_text(context.get("text") or context.get("content") or context.get("page_content"))
        if text:
            texts.append(text)
    return texts


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _none_if_nan(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return float(value)
