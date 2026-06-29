from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from rag_eval.config import AppConfig, ModelConfig
from rag_eval.io import contexts_from_json
from rag_eval.logging_utils import setup_file_logger

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda value, **_: value


@dataclass(slots=True)
class RagasEvaluationRow:
    question_id: Any
    metrics: dict[str, float | None]
    error: str | None = None


class RagasEvaluator:
    def __init__(self, config: AppConfig):
        self.config = config
        self._langchain_llm = None
        self._langchain_embeddings = None

    def evaluate_rows(self, frame: pd.DataFrame) -> list[RagasEvaluationRow]:
        self._ensure_ragas_installed()
        results: list[RagasEvaluationRow] = []
        logger = setup_file_logger(self.config.paths.log_file)
        iterator = tqdm(frame.iterrows(), total=len(frame), desc="Running judge")
        for _, row in iterator:
            result = self._evaluate_row(row)
            logger.info(
                "judge_result question_id=%s metrics=%s error=%s",
                result.question_id,
                result.metrics,
                result.error,
            )
            results.append(result)
        return results

    def _evaluate_row(self, row: pd.Series) -> RagasEvaluationRow:
        logger = setup_file_logger(self.config.paths.log_file)
        question_id = row.get("question_id")
        all_metrics: dict[str, float | None] = {}
        errors: list[str] = []

        answer_sample = self._make_answer_sample(row, self.config.metrics.ragas_context_source, fallback=True)
        if answer_sample is not None:
            logger.info("judge_request question_id=%s sample=%s", question_id, answer_sample)
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
            llm=self._make_langchain_llm() if _metrics_need_llm(metrics) else None,
            embeddings=self._make_embeddings() if _metrics_need_embeddings(metrics) else None,
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

    def _make_answer_sample(self, row: pd.Series, source: str, fallback: bool = False) -> dict[str, Any] | None:
        question = _clean_text(row.get("question"))
        answer = _clean_text(row.get("answer"))
        contexts = self._contexts(row, source, fallback)
        reference = _clean_text(row.get("ground_truth")) or _clean_text(row.get("expected_answer"))
        if not question or not answer:
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

    def _answer_metrics_for_sample(self, sample: dict[str, Any]) -> list[Any]:
        configured = set(self.config.metrics.ragas_metrics)
        metrics: list[Any] = []
        has_reference = "reference" in sample
        has_contexts = bool(sample.get("retrieved_contexts"))
        if "faithfulness" in configured and has_contexts:
            metrics.append(_ragas_metric("faithfulness"))
        if "answer_correctness" in configured and has_reference:
            metrics.append(_ragas_metric("answer_correctness"))
        if "answer_relevancy" in configured:
            metrics.append(_ragas_metric("answer_relevancy"))
        if "answer_similarity" in configured and has_reference:
            metrics.append(_ragas_metric("answer_similarity"))
        if "context_precision" in configured and has_contexts and has_reference:
            metrics.append(_ragas_metric("context_precision"))
        if "context_recall" in configured and has_contexts and has_reference:
            metrics.append(_ragas_metric("context_recall"))
        return metrics

    def _make_langchain_llm(self):
        if self._langchain_llm is not None:
            return self._langchain_llm

        provider = self.config.metrics.ragas_judge_provider
        if provider not in self.config.models:
            raise ValueError(f"Ragas judge provider '{provider}' is not configured.")
        model_config = self.config.models[provider]
        if model_config.provider == "qwen_transformers":
            self._langchain_llm = _make_transformers_llm(model_config)
        elif model_config.provider == "gigachat_api" or provider == "gigachat":
            self._langchain_llm = _make_gigachat_langchain_llm(model_config)
        else:
            self._langchain_llm = _make_openai_compatible_llm(model_config)
        return self._langchain_llm

    def _make_embeddings(self):
        provider = self.config.metrics.ragas_embeddings_provider
        if not provider:
            return None
        if self._langchain_embeddings is not None:
            return self._langchain_embeddings
        if provider not in self.config.models:
            raise ValueError(f"Ragas embeddings provider '{provider}' is not configured.")
        model_config = self.config.models[provider]
        if model_config.provider != "huggingface_embeddings":
            raise ValueError(
                f"Ragas embeddings provider '{provider}' must use provider='huggingface_embeddings'."
            )
        self._langchain_embeddings = _make_huggingface_embeddings(model_config)
        return self._langchain_embeddings

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


def _make_gigachat_langchain_llm(model_config: ModelConfig):
    try:
        from langchain_gigachat.chat_models import GigaChat
    except ImportError:
        from langchain_community.chat_models.gigachat import GigaChat

    return GigaChat(
        base_url=model_config.base_url,
        access_token=model_config.access_token,
        model=model_config.model,
        temperature=model_config.temperature,
        verify_ssl_certs=model_config.verify_ssl,
        rate_limiter=_rate_limiter(model_config),
    )


def _make_transformers_llm(model_config: ModelConfig):
    from langchain_community.llms import HuggingFacePipeline

    model_kwargs: dict[str, Any] = {}
    if model_config.device_map:
        model_kwargs["device_map"] = model_config.device_map
    if model_config.torch_dtype:
        model_kwargs["torch_dtype"] = model_config.torch_dtype
    if model_config.local_files_only:
        model_kwargs["local_files_only"] = True

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


def _make_huggingface_embeddings(model_config: ModelConfig):
    model_kwargs: dict[str, Any] = {}
    if model_config.local_files_only:
        model_kwargs["local_files_only"] = True

    try:
        from ragas.embeddings import HuggingFaceEmbeddings as RagasHuggingFaceEmbeddings

        embeddings = RagasHuggingFaceEmbeddings(
            model=model_config.model,
            device=model_config.embedding_device,
            normalize_embeddings=model_config.normalize_embeddings,
            **model_kwargs,
        )
        return _SyncEmbeddingsAdapter(embeddings)
    except ImportError:
        pass

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    if model_config.embedding_device:
        model_kwargs["device"] = model_config.embedding_device

    embeddings = HuggingFaceEmbeddings(
        model_name=model_config.model,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": model_config.normalize_embeddings},
        show_progress=False,
    )
    return _SyncEmbeddingsAdapter(embeddings)


class _SyncEmbeddingsAdapter:
    def __init__(self, embeddings: Any):
        self.embeddings = embeddings

    def embed_text(self, text: str) -> list[float]:
        if hasattr(self.embeddings, "embed_text"):
            return self.embeddings.embed_text(text)
        return self.embeddings.embed_query(text)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text)

    async def aembed_text(self, text: str) -> list[float]:
        return self.embed_text(text)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if hasattr(self.embeddings, "embed_texts"):
            return self.embeddings.embed_texts(texts)
        return self.embeddings.embed_documents(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


def _rate_limiter(model_config: ModelConfig):
    if model_config.min_seconds_between_requests <= 0:
        return None
    from langchain_core.rate_limiters import InMemoryRateLimiter

    return InMemoryRateLimiter(
        requests_per_second=1 / model_config.min_seconds_between_requests,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )


def _api_key(config: ModelConfig) -> str:
    if config.provider == "qwen_local" or config.auth_type == "none":
        return "EMPTY"
    if config.auth_type == "bearer_env" and config.api_key_env:
        return os.getenv(config.api_key_env, "")
    return "EMPTY"


def _openai_base_url(value: str) -> str:
    suffix = "/chat/completions"
    return value[: -len(suffix)] if value.endswith(suffix) else value


def _ragas_metric(name: str):
    import ragas.metrics as ragas_metrics

    try:
        return getattr(ragas_metrics, name)
    except AttributeError as exc:
        raise ValueError(f"Ragas metric '{name}' is not available in installed ragas.") from exc


def _metrics_need_llm(metrics: list[Any]) -> bool:
    return any(hasattr(metric, "llm") for metric in metrics)


def _metrics_need_embeddings(metrics: list[Any]) -> bool:
    return any(hasattr(metric, "embeddings") for metric in metrics)


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
