from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from rag_eval.config import AppConfig
from rag_eval.custom_judge_evaluator import CustomJudgeEvaluator
from rag_eval.io import append_json_rows, append_xlsx_rows, contexts_from_json, read_run_table
from rag_eval.logging_utils import setup_file_logger
from rag_eval.ragas_evaluator import RagasEvaluator
from rag_eval.retrieval_metrics import context_chunk_ids, parse_relevant_chunk_ids, retrieval_metrics_for_ids
from rag_eval.text_metrics import contains_expected, exact_match, token_f1

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda value, **_: value


class MetricsCalculator:
    def __init__(self, config: AppConfig):
        self.config = config

    @classmethod
    def from_config(cls, path: str | Path = "config.py") -> "MetricsCalculator":
        return cls(AppConfig.from_file(path))

    def evaluate_latest(self) -> Path:
        run_file = self.latest_run_file()
        return self.evaluate(run_file)

    def evaluate(self, run_file: str | Path) -> Path:
        logger = setup_file_logger(self.config.paths.log_file)
        run_path = Path(run_file)
        frame, data_path = read_run_table(run_path)
        logger.info("metrics_started run_file=%s data_file=%s rows=%s", run_path, data_path, len(frame))
        iterator = tqdm(frame.iterrows(), total=len(frame), desc="Calculating metrics")
        detail_rows = [self._row_metrics(row) for _, row in iterator]
        providers = self._multi_generation_providers(frame)
        model_answer_columns = [f"answer_{provider}" for provider in providers]
        if self.config.metrics.ragas_enabled:
            evaluator = self._judge_evaluator()
            if providers:
                for provider in providers:
                    provider_frame = frame.copy()
                    provider_frame["answer"] = provider_frame[f"answer_{provider}"]
                    self._add_ragas_metrics(
                        provider_frame,
                        detail_rows,
                        evaluator=evaluator,
                        provider=provider,
                        include_legacy=provider == self.config.generation.provider,
                    )
            else:
                self._add_ragas_metrics(frame, detail_rows, evaluator=evaluator)
        details = pd.DataFrame(detail_rows)
        summary = self._summary(details, frame, run_path, data_path)
        created_at = summary["created_at"]

        detail_rows_with_run = []
        for row in detail_rows:
            detail_rows_with_run.append(
                {"created_at": created_at, "run_file": str(run_path), "json_file": str(data_path), **row}
            )

        summary_path = self.config.paths.summary_metrics_file
        summary_json_path = summary_path.with_suffix(".json")
        append_xlsx_rows(
            summary_path,
            "summary",
            [summary],
            leading_columns=[
                "created_at",
                "run_file",
                "json_file",
                "question_count",
                "error_count",
                "retriever_context_count_mean",
                "reranker_context_count_mean",
                *_run_parameter_columns,
                "ragas_judge_score_mean",
            ],
            leading_prefixes=["answer_", "ragas_"],
        )
        append_json_rows(summary_json_path, "summary", [summary])
        details_rows = _select_rows(
            detail_rows_with_run,
            _details_columns,
            keep_columns=[
                "created_at", "run_file", "json_file", "question_id", "question", "ground_truth",
                "answer", "retriever_contexts", "reranker_contexts", "has_error",
            ],
        )
        append_xlsx_rows(
            summary_path,
            "details",
            details_rows,
            leading_columns=[
                "created_at",
                "run_file",
                "json_file",
                "question_id",
                "question",
                "ground_truth",
                "answer",
            ],
            leading_prefixes=["answer_", "ragas_"],
        )
        append_json_rows(summary_json_path, "details", details_rows)
        judge_rows = _select_rows(
            detail_rows_with_run,
            _judge_score_columns,
            keep_columns=[
                "created_at", "run_file", "json_file", "question_id", "question", "ground_truth",
                "answer", *model_answer_columns, "retriever_contexts", "reranker_contexts",
            ],
        )
        append_xlsx_rows(
            summary_path,
            "judge",
            judge_rows,
            leading_columns=[
                "created_at",
                "run_file",
                "json_file",
                "question_id",
                "question",
                "ground_truth",
                "answer",
                *model_answer_columns,
                "retriever_contexts",
                "reranker_contexts",
            ],
            leading_prefixes=["ragas_"],
        )
        append_json_rows(summary_json_path, "judge", judge_rows)
        question_rows = _select_rows(
            detail_rows_with_run,
            _question_columns,
            keep_columns=[
                "created_at", "run_file", "json_file", "question_id", "question", "ground_truth",
                "answer", *model_answer_columns, "retriever_contexts", "reranker_contexts", "ragas_error",
            ],
        )
        append_xlsx_rows(
            summary_path,
            "questions",
            question_rows,
            leading_columns=[
                "created_at",
                "run_file",
                "json_file",
                "question_id",
                "question",
                "ground_truth",
                "answer",
                *model_answer_columns,
                "ragas_error",
                "retriever_contexts",
                "reranker_contexts",
            ],
            leading_prefixes=["ragas_"],
        )
        append_json_rows(summary_json_path, "questions", question_rows)
        debug_rows = _select_rows(
            detail_rows_with_run,
            _judge_debug_columns,
            keep_columns=["created_at", "run_file", "json_file", "question_id", "question", "ragas_error"],
        )
        append_xlsx_rows(
            summary_path,
            "judge_debug",
            debug_rows,
            leading_columns=["created_at", "run_file", "json_file", "question_id", "question", "ragas_error"],
            leading_prefixes=["ragas_"],
        )
        append_json_rows(summary_json_path, "judge_debug", debug_rows)
        retrieval_rows = _select_rows(
            detail_rows_with_run,
            _retrieval_columns,
            keep_columns=[
                "created_at", "run_file", "json_file", "question_id", "question",
                "retriever_contexts", "reranker_contexts",
            ],
        )
        append_xlsx_rows(
            summary_path,
            "retrieval",
            retrieval_rows,
            leading_columns=[
                "created_at",
                "run_file",
                "json_file",
                "question_id",
                "question",
                "retriever_contexts",
                "reranker_contexts",
            ],
        )
        append_json_rows(summary_json_path, "retrieval", retrieval_rows)
        logger.info("metrics_finished output=%s json=%s", summary_path, summary_json_path)
        return summary_path

    def latest_run_file(self) -> Path:
        files = sorted(self.config.paths.run_outputs_dir.glob("rag_run_*.xlsx"))
        if not files:
            raise FileNotFoundError(f"No run files found in {self.config.paths.run_outputs_dir}.")
        return files[-1]

    def _row_metrics(self, row: pd.Series) -> dict[str, Any]:
        ground_truth = _ground_truth(row)
        result: dict[str, Any] = {
            "question_id": row.get("question_id"),
            "question": row.get("question"),
            "ground_truth": ground_truth,
            "answer": row.get("answer"),
            "retriever_contexts": row.get("retriever_contexts"),
            "reranker_contexts": row.get("reranker_contexts"),
            "retriever_context_count": len(contexts_from_json(row.get("retriever_contexts"))),
            "reranker_context_count": len(contexts_from_json(row.get("reranker_contexts"))),
            "has_error": bool(row.get("error")) if not pd.isna(row.get("error")) else False,
        }
        expected_answer = ground_truth
        answer = row.get("answer")
        if self.config.metrics.answer_enabled and _has_text(expected_answer) and _has_text(answer):
            result["answer_exact_match"] = exact_match(expected_answer, answer)
            result["answer_contains_expected"] = contains_expected(expected_answer, answer)
            result["answer_token_f1"] = token_f1(expected_answer, answer)
        for provider in self._multi_generation_providers_from_row(row):
            provider_answer = row.get(f"answer_{provider}")
            result[f"answer_{provider}"] = provider_answer
            result[f"error_{provider}"] = row.get(f"error_{provider}")
            result[f"answer_{provider}_has_error"] = _has_text(row.get(f"error_{provider}"))
            if self.config.metrics.answer_enabled and _has_text(expected_answer) and _has_text(provider_answer):
                result[f"answer_{provider}_exact_match"] = exact_match(expected_answer, provider_answer)
                result[f"answer_{provider}_contains_expected"] = contains_expected(expected_answer, provider_answer)
                result[f"answer_{provider}_token_f1"] = token_f1(expected_answer, provider_answer)
        result.update(self._retrieval_metrics(row))
        return result

    def _retrieval_metrics(self, row: pd.Series) -> dict[str, Any]:
        relevant_ids = parse_relevant_chunk_ids(row.get("relevant_chunk_ids"))
        if not relevant_ids:
            return {}

        metrics: dict[str, Any] = {}
        for source in ("retriever", "reranker"):
            column = f"{source}_contexts"
            ids = context_chunk_ids(row.get(column))
            source_metrics = retrieval_metrics_for_ids(
                ids,
                relevant_ids,
                self.config.metrics.retrieval_k_values,
            )
            metrics[f"{source}_retrieved_chunk_ids"] = ";".join(item for item in ids if item)
            metrics.update({f"{source}_{key}": value for key, value in source_metrics.items()})
        return metrics

    def _judge_evaluator(self):
        backend = self.config.metrics.ragas_backend
        if backend == "custom":
            return CustomJudgeEvaluator(self.config)
        elif backend == "ragas":
            return RagasEvaluator(self.config)
        raise ValueError("RAGAS_BACKEND must be 'custom' or 'ragas'.")

    def _add_ragas_metrics(
        self,
        frame: pd.DataFrame,
        detail_rows: list[dict[str, Any]],
        evaluator: Any,
        provider: str | None = None,
        include_legacy: bool = False,
    ) -> None:
        ragas_results = evaluator.evaluate_rows(frame)
        by_question_id = {result.question_id: result for result in ragas_results}
        for row in detail_rows:
            result = by_question_id.get(row.get("question_id"))
            if result is None:
                continue
            if provider is None:
                row.update(result.metrics)
                row["ragas_error"] = result.error
                continue
            prefixed = {
                _provider_judge_column(column, provider): value
                for column, value in result.metrics.items()
            }
            row.update(prefixed)
            row[f"ragas_{provider}_error"] = result.error
            if include_legacy:
                row.update(result.metrics)
                row["ragas_error"] = result.error

    def _multi_generation_providers(self, frame: pd.DataFrame) -> list[str]:
        return [
            provider
            for provider in dict.fromkeys(
                self.config.generation.providers or [self.config.generation.provider]
            )
            if f"answer_{provider}" in frame.columns
        ]

    def _multi_generation_providers_from_row(self, row: pd.Series) -> list[str]:
        return [
            provider
            for provider in dict.fromkeys(
                self.config.generation.providers or [self.config.generation.provider]
            )
            if f"answer_{provider}" in row.index
        ]

    def _summary(
        self,
        details: pd.DataFrame,
        run_frame: pd.DataFrame,
        run_path: Path,
        data_path: Path,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_file": str(run_path),
            "json_file": str(data_path),
            "question_count": len(details),
            "error_count": int(details["has_error"].sum()) if "has_error" in details else 0,
        }
        if len(run_frame):
            first = run_frame.iloc[0]
            for column in _run_parameter_columns:
                value = first.get(column)
                if _has_value(value):
                    summary[column] = value
        providers = self._multi_generation_providers(run_frame)
        judge_score = _judge_score_mean(details, providers)
        if judge_score is not None:
            summary["ragas_judge_score_mean"] = judge_score
        for provider in providers:
            columns = [
                column
                for column in details.columns
                if column.startswith(f"ragas_{provider}_") and _judge_score_columns(column)
            ]
            provider_score = _mean_of_columns(details, columns)
            if provider_score is not None:
                summary[f"ragas_{provider}_judge_score_mean"] = provider_score

        metric_columns = [
            column
            for column in details.columns
            if column not in {"question_id", "question", "has_error"} and pd.api.types.is_numeric_dtype(details[column])
        ]
        for column in metric_columns:
            valid = details[column].dropna()
            summary[f"{column}_mean"] = valid.mean() if len(valid) else None
        return summary


def _has_text(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and pd.isna(value)) and bool(str(value).strip())


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    return True


_run_parameter_columns = (
    "pipeline_factory",
    "model_name",
    "generation_providers",
    "parallel_generation",
    "temperature",
    "judge_model_name",
    "judge_temperature",
    "k_rrf",
    "fusion_method",
    "alpha",
    "bias",
    "rerank_initial_k",
    "retriever_top_k",
    "rag_context_source",
    "rag_max_contexts",
    "judge_backend",
    "judge_context_source",
    "judge_max_context_chars",
    "retriever_init_kwargs",
    "retriever_search_kwargs",
)


def _ground_truth(row: pd.Series) -> Any:
    for column in ("ground_truth", "expected_answer", "grouth_true", "golden_answer"):
        if column in row and _has_text(row.get(column)):
            return row.get(column)
    return None


def _mean_of_columns(frame: pd.DataFrame, columns: list[str]) -> float | None:
    values = []
    for column in columns:
        if column not in frame:
            continue
        valid = frame[column].dropna()
        if len(valid):
            values.append(valid.mean())
    if not values:
        return None
    return float(sum(values) / len(values))


def _judge_score_mean(frame: pd.DataFrame, providers: list[str] | None = None) -> float | None:
    provider_prefixes = tuple(f"ragas_{provider}_" for provider in (providers or []))
    columns = [
        column
        for column in frame.columns
        if _judge_score_columns(column) and not column.startswith(provider_prefixes)
    ]
    return _mean_of_columns(frame, columns)


def _select_rows(rows: list[dict[str, Any]], column_filter, keep_columns: list[str] | None = None) -> list[dict[str, Any]]:
    selected = []
    leading = set(keep_columns or [])
    for row in rows:
        selected.append({key: value for key, value in row.items() if key in leading or column_filter(key)})
    return selected


def _judge_columns(column: str) -> bool:
    return column.startswith("ragas_")


def _details_columns(column: str) -> bool:
    return (
        column.startswith("answer_")
        or column.startswith("retriever_")
        or column.startswith("reranker_")
        or _judge_score_columns(column)
    )


def _judge_score_columns(column: str) -> bool:
    return column.startswith("ragas_") and not column.endswith("_error") and not _judge_diagnostic_columns(column)


def _question_columns(column: str) -> bool:
    return column.startswith("ragas_") and (
        _judge_score_columns(column)
        or column.endswith("_reason")
        or column.endswith("_evidence")
    )


def _judge_debug_columns(column: str) -> bool:
    return column.startswith("ragas_") and (column.endswith("_raw") or column.endswith("_error"))


def _judge_diagnostic_columns(column: str) -> bool:
    return column.endswith("_reason") or column.endswith("_evidence") or column.endswith("_raw")


def _retrieval_columns(column: str) -> bool:
    return (
        column in {"retriever_contexts", "reranker_contexts"}
        or column.startswith("retriever_")
        or column.startswith("reranker_")
    )


def _provider_judge_column(column: str, provider: str) -> str:
    if column.startswith("ragas_"):
        return f"ragas_{provider}_{column[len('ragas_'):]}"
    return f"{provider}_{column}"
