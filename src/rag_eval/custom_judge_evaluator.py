from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rag_eval.config import AppConfig, ModelConfig
from rag_eval.custom_judge_prompts import STRICT_JSON_INSTRUCTION, metric_prompt
from rag_eval.huggingface_embeddings import make_huggingface_embeddings
from rag_eval.io import contexts_from_json
from rag_eval.logging_utils import setup_file_logger

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda value, **_: value


@dataclass(slots=True)
class CustomJudgeRow:
    question_id: Any
    metrics: dict[str, float | str | None]
    error: str | None = None


class CustomJudgeEvaluator:
    def __init__(self, config: AppConfig):
        self.config = config
        self._llm = None
        self._embeddings = None

    def evaluate_rows(self, frame: pd.DataFrame) -> list[CustomJudgeRow]:
        results: list[CustomJudgeRow] = []
        iterator = tqdm(frame.iterrows(), total=len(frame), desc="Running custom judge")
        for _, row in iterator:
            results.append(self._evaluate_row(row))
        return results

    def _evaluate_row(self, row: pd.Series) -> CustomJudgeRow:
        logger = setup_file_logger(self.config.paths.log_file)
        question_id = row.get("question_id")
        sample = self._make_sample(row)
        metrics: dict[str, float | str | None] = {}
        errors: list[str] = []

        if sample is None:
            return CustomJudgeRow(question_id, metrics, "Not enough data for custom judge metrics.")

        for metric in self._metric_names(sample):
            try:
                logger.info("custom_judge_metric_started question_id=%s metric=%s", question_id, metric)
                result = self._evaluate_metric(metric, sample)
                prefix = f"ragas_{metric}"
                metrics[prefix] = result["score"]
                metrics[f"{prefix}_reason"] = result["reason"]
                metrics[f"{prefix}_evidence"] = "; ".join(result["evidence"])
                metrics[f"{prefix}_raw"] = result["raw_response"]
                logger.info(
                    "custom_judge_metric_finished question_id=%s metric=%s score=%s reason=%s",
                    question_id,
                    metric,
                    result["score"],
                    result["reason"],
                )
            except Exception as exc:
                errors.append(f"{metric}: {type(exc).__name__}: {exc}")
                logger.exception("custom_judge_metric_failed question_id=%s metric=%s", question_id, metric)

        return CustomJudgeRow(question_id, metrics, "; ".join(errors) if errors else None)

    def _evaluate_metric(self, metric: str, sample: dict[str, Any]) -> dict[str, Any]:
        if metric == "answer_similarity":
            return self._answer_similarity(sample)

        messages = self._messages(metric, sample)
        max_attempts = max(1, self.config.metrics.ragas_max_retries + 1)
        raw_response = ""
        last_error: Exception | None = None

        for _ in range(max_attempts):
            try:
                response = self._llm_model().invoke(messages)
                raw_response = _response_text(getattr(response, "content", ""))
                parsed = _parse_judge_json(raw_response)
                return {
                    "score": _stable_score(parsed["score"]),
                    "reason": str(parsed.get("reason") or "").strip(),
                    "evidence": _string_list(parsed.get("evidence"))[:3],
                    "raw_response": raw_response,
                }
            except Exception as exc:
                last_error = exc
                messages = [
                    *self._messages(metric, sample),
                    AIMessage(content=raw_response or "<вызов завершился ошибкой>"),
                    HumanMessage(content=_repair_prompt(str(exc))),
                ]

        raise ValueError(f"Cannot parse judge JSON after {max_attempts} attempts: {last_error}; raw={raw_response!r}")

    def _answer_similarity(self, sample: dict[str, Any]) -> dict[str, Any]:
        answer = sample.get("answer") or ""
        ground_truth = sample.get("ground_truth") or ""
        if not answer or not ground_truth:
            raise ValueError("answer_similarity requires answer and ground_truth")

        answer_vector = self._embed_text(answer)
        reference_vector = self._embed_text(ground_truth)
        cosine = _cosine_similarity(answer_vector, reference_vector)
        score = _clamp_score((cosine + 1) / 2)
        return {
            "score": score,
            "reason": f"BGE-M3 cosine similarity: {cosine:.4f}",
            "evidence": [],
            "raw_response": f"embedding_cosine={cosine:.6f}",
        }

    def _messages(self, metric: str, sample: dict[str, Any]) -> list[Any]:
        prompt = metric_prompt(metric)
        task = (
            f"Метрика: {prompt.metric}\n"
            f"Инструкция: {prompt.instruction}\n\n"
            f"Вопрос:\n{sample['question']}\n\n"
            f"Ответ модели:\n{sample['answer']}\n\n"
            f"Эталонный ответ:\n{sample.get('ground_truth') or ''}\n\n"
            f"Найденные контексты:\n"
            f"{_format_contexts(sample['contexts'], self.config.metrics.ragas_judge_max_context_chars)}\n\n"
            "Оцени только указанную метрику и верни JSON по схеме."
        )
        return [
            SystemMessage(content=f"{STRICT_JSON_INSTRUCTION}\n\nКритерий текущей метрики:\n{prompt.instruction}"),
            HumanMessage(content=f"Положительный пример:\n{prompt.positive_example}"),
            AIMessage(content='{"score": 1.0, "reason": "пример валидного JSON", "evidence": ["пример"]}'),
            HumanMessage(content=f"Отрицательный пример:\n{prompt.negative_example}"),
            AIMessage(content='{"score": 0.0, "reason": "пример валидного JSON", "evidence": []}'),
            HumanMessage(content=task),
        ]

    def _make_sample(self, row: pd.Series) -> dict[str, Any] | None:
        question = _clean_text(row.get("question"))
        answer = _clean_text(row.get("answer"))
        ground_truth = _clean_text(row.get("ground_truth")) or _clean_text(row.get("expected_answer"))
        contexts = self._contexts(row)
        if not question or not answer:
            return None
        return {
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "contexts": contexts,
        }

    def _contexts(self, row: pd.Series) -> list[str]:
        source = self.config.metrics.ragas_context_source
        primary = "reranker_contexts" if source == "reranker" else "retriever_contexts"
        fallback = "retriever_contexts" if primary == "reranker_contexts" else "reranker_contexts"
        contexts = _context_texts(row.get(primary))
        return contexts or _context_texts(row.get(fallback))

    def _metric_names(self, sample: dict[str, Any]) -> list[str]:
        has_reference = bool(sample.get("ground_truth"))
        has_contexts = bool(sample.get("contexts"))
        result = []
        for metric in self.config.metrics.ragas_metrics:
            if metric in {"answer_correctness", "answer_similarity"} and not has_reference:
                continue
            if metric in {"faithfulness", "context_precision", "context_recall"} and not has_contexts:
                continue
            if metric in {"context_precision", "context_recall"} and not has_reference:
                continue
            result.append(metric)
        return result

    def _llm_model(self):
        if self._llm is not None:
            return self._llm
        provider = self.config.metrics.ragas_judge_provider
        if provider not in self.config.models:
            raise ValueError(f"Custom judge provider '{provider}' is not configured.")
        model_config = self.config.models[provider]
        if model_config.provider != "gigachat_api" and provider != "gigachat":
            raise ValueError("Custom judge is intended for provider='gigachat'.")
        self._llm = _make_gigachat_model(model_config)
        return self._llm

    def _embed_text(self, text: str) -> list[float]:
        embeddings = self._embedding_model()
        if hasattr(embeddings, "embed_text"):
            return embeddings.embed_text(text)
        if hasattr(embeddings, "embed_query"):
            return embeddings.embed_query(text)
        raise TypeError("Configured embeddings object does not support text embedding.")

    def _embedding_model(self):
        if self._embeddings is not None:
            return self._embeddings
        provider = self.config.metrics.ragas_embeddings_provider
        if not provider:
            raise ValueError("answer_similarity requires RAGAS_EMBEDDINGS_PROVIDER.")
        if provider not in self.config.models:
            raise ValueError(f"Custom judge embeddings provider '{provider}' is not configured.")
        model_config = self.config.models[provider]
        if model_config.provider != "huggingface_embeddings":
            raise ValueError(
                f"Custom judge embeddings provider '{provider}' must use provider='huggingface_embeddings'."
            )
        self._embeddings = make_huggingface_embeddings(model_config)
        return self._embeddings


def _make_gigachat_model(model_config: ModelConfig):
    try:
        from langchain_gigachat.chat_models import GigaChat
    except ImportError:
        from langchain_community.chat_models.gigachat import GigaChat

    return GigaChat(
        base_url=model_config.base_url,
        access_token=model_config.access_token,
        model=model_config.model,
        temperature=model_config.temperature,
        timeout=model_config.timeout_seconds,
        verify_ssl_certs=model_config.verify_ssl,
        rate_limiter=_rate_limiter(model_config),
    )


def _rate_limiter(model_config: ModelConfig):
    if model_config.min_seconds_between_requests <= 0:
        return None
    from langchain_core.rate_limiters import InMemoryRateLimiter

    return InMemoryRateLimiter(
        requests_per_second=1 / model_config.min_seconds_between_requests,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )


def _parse_judge_json(value: str) -> dict[str, Any]:
    text = _strip_fence(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = _first_json_object(text)
    if not isinstance(parsed, dict):
        raise ValueError("judge response is not a JSON object")
    missing = {"score", "reason", "evidence"} - set(parsed)
    if missing:
        raise ValueError(f"judge response does not contain required fields: {sorted(missing)}")
    unknown = set(parsed) - {"score", "reason", "evidence"}
    if unknown:
        raise ValueError(f"judge response contains unknown fields: {sorted(unknown)}")
    return parsed


def _first_json_object(value: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", value):
        try:
            parsed, _ = decoder.raw_decode(value[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("judge response does not contain a valid JSON object")


def _strip_fence(value: str) -> str:
    text = value.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def _clamp_score(value: Any) -> float:
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError("judge score must be finite")
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return score


def _stable_score(value: Any) -> float:
    """Normalize model output to the documented decimal scale."""
    score = _clamp_score(value)
    return round(score, 1)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors have different dimensions.")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _repair_prompt(error: str) -> str:
    return (
        "Предыдущий ответ не соответствует строгому JSON-формату. "
        f"Ошибка парсинга: {error}. "
        'Повторно примени тот же критерий и верни только JSON вида '
        '{"score": 0.0, "reason": "кратко", "evidence": []}. '
        'score допустим с шагом 0.1 от 0.0 до 1.0.'
    )


def _response_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return _clean_text(value)


def _format_contexts(contexts: list[str], max_chars: int = 60_000) -> str:
    if not contexts:
        return "<контексты отсутствуют>"
    budget = max(1_000, max_chars)
    per_context = max(200, budget // len(contexts) - 32)
    formatted = []
    for index, context in enumerate(contexts, start=1):
        if len(context) > per_context:
            context = context[:per_context] + "… [сокращено для judge; полный текст в run JSON]"
        formatted.append(f"[{index}] {context}")
    return "\n\n".join(formatted)


def _context_texts(value: Any) -> list[str]:
    contexts = contexts_from_json(value)
    texts: list[str] = []
    for context in contexts:
        text = _clean_text(context.get("text") or context.get("content") or context.get("page_content"))
        if text:
            texts.append(text)
    return texts


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()
