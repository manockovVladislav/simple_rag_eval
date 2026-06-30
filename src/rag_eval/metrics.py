from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from rag_eval.config import AppConfig
from rag_eval.custom_judge_evaluator import CustomJudgeEvaluator
from rag_eval.io import append_xlsx_rows, read_table
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
        frame = read_table(run_path)
        logger.info("metrics_started run_file=%s rows=%s", run_path, len(frame))
        iterator = tqdm(frame.iterrows(), total=len(frame), desc="Calculating metrics")
        detail_rows = [self._row_metrics(row) for _, row in iterator]
        if self.config.metrics.ragas_enabled:
            self._add_ragas_metrics(frame, detail_rows)
        details = pd.DataFrame(detail_rows)
        summary = self._summary(details, run_path)
        created_at = summary["created_at"]

        detail_rows_with_run = []
        for row in detail_rows:
            detail_rows_with_run.append({"created_at": created_at, "run_file": str(run_path), **row})

        summary_path = self.config.paths.summary_metrics_file
        append_xlsx_rows(
            summary_path,
            "summary",
            [summary],
            leading_columns=[
                "created_at",
                "run_file",
                "question_count",
                "error_count",
                "ragas_judge_score_mean",
            ],
            leading_prefixes=["ragas_"],
        )
        append_xlsx_rows(
            summary_path,
            "details",
            _select_rows(
                detail_rows_with_run,
                _details_columns,
                keep_columns=[
                    "created_at",
                    "run_file",
                    "question_id",
                    "question",
                    "ground_truth",
                    "answer",
                    "retriever_contexts",
                    "reranker_contexts",
                    "has_error",
                ],
            ),
            leading_columns=[
                "created_at",
                "run_file",
                "question_id",
                "question",
                "ground_truth",
                "answer",
            ],
            leading_prefixes=["ragas_"],
        )
        append_xlsx_rows(
            summary_path,
            "judge",
            _select_rows(
                detail_rows_with_run,
                _judge_score_columns,
                keep_columns=[
                    "created_at",
                    "run_file",
                    "question_id",
                    "question",
                    "ground_truth",
                    "answer",
                    "retriever_contexts",
                    "reranker_contexts",
                ],
            ),
            leading_columns=[
                "created_at",
                "run_file",
                "question_id",
                "question",
                "ground_truth",
                "answer",
                "retriever_contexts",
                "reranker_contexts",
            ],
            leading_prefixes=["ragas_"],
        )
        append_xlsx_rows(
            summary_path,
            "questions",
            _select_rows(
                detail_rows_with_run,
                _question_columns,
                keep_columns=[
                    "created_at",
                    "run_file",
                    "question_id",
                    "question",
                    "ground_truth",
                    "answer",
                    "retriever_contexts",
                    "reranker_contexts",
                    "ragas_error",
                ],
            ),
            leading_columns=[
                "created_at",
                "run_file",
                "question_id",
                "question",
                "ground_truth",
                "answer",
                "ragas_error",
                "retriever_contexts",
                "reranker_contexts",
            ],
            leading_prefixes=["ragas_"],
        )
        append_xlsx_rows(
            summary_path,
            "judge_debug",
            _select_rows(
                detail_rows_with_run,
                _judge_debug_columns,
                keep_columns=["created_at", "run_file", "question_id", "question", "ragas_error"],
            ),
            leading_columns=["created_at", "run_file", "question_id", "question", "ragas_error"],
            leading_prefixes=["ragas_"],
        )
        append_xlsx_rows(
            summary_path,
            "retrieval",
            _select_rows(
                detail_rows_with_run,
                _retrieval_columns,
                keep_columns=["created_at", "run_file", "question_id", "question", "retriever_contexts", "reranker_contexts"],
            ),
            leading_columns=[
                "created_at",
                "run_file",
                "question_id",
                "question",
                "retriever_contexts",
                "reranker_contexts",
            ],
        )
        logger.info("metrics_finished output=%s", summary_path)
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
            "has_error": bool(row.get("error")) if not pd.isna(row.get("error")) else False,
        }
        expected_answer = ground_truth
        answer = row.get("answer")
        if self.config.metrics.answer_enabled and _has_text(expected_answer) and _has_text(answer):
            result["answer_exact_match"] = exact_match(expected_answer, answer)
            result["answer_contains_expected"] = contains_expected(expected_answer, answer)
            result["answer_token_f1"] = token_f1(expected_answer, answer)
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

    def _add_ragas_metrics(self, frame: pd.DataFrame, detail_rows: list[dict[str, Any]]) -> None:
        backend = self.config.metrics.ragas_backend
        if backend == "custom":
            ragas_results = CustomJudgeEvaluator(self.config).evaluate_rows(frame)
        elif backend == "ragas":
            ragas_results = RagasEvaluator(self.config).evaluate_rows(frame)
        else:
            raise ValueError("RAGAS_BACKEND must be 'custom' or 'ragas'.")
        by_question_id = {result.question_id: result for result in ragas_results}
        for row in detail_rows:
            result = by_question_id.get(row.get("question_id"))
            if result is None:
                continue
            row.update(result.metrics)
            row["ragas_error"] = result.error

    def _summary(self, details: pd.DataFrame, run_path: Path) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_file": str(run_path),
            "question_count": len(details),
            "error_count": int(details["has_error"].sum()) if "has_error" in details else 0,
        }
        judge_score = _judge_score_mean(details)
        if judge_score is not None:
            summary["ragas_judge_score_mean"] = judge_score

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


def _judge_score_mean(frame: pd.DataFrame) -> float | None:
    columns = [column for column in frame.columns if _judge_score_columns(column)]
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
    return column.startswith("ragas_") and column != "ragas_error" and not _judge_diagnostic_columns(column)


def _question_columns(column: str) -> bool:
    return column.startswith("ragas_") and (
        _judge_score_columns(column)
        or column.endswith("_reason")
        or column.endswith("_evidence")
    )


def _judge_debug_columns(column: str) -> bool:
    return column.startswith("ragas_") and (column.endswith("_raw") or column == "ragas_error")


def _judge_diagnostic_columns(column: str) -> bool:
    return column.endswith("_reason") or column.endswith("_evidence") or column.endswith("_raw")


def _retrieval_columns(column: str) -> bool:
    return (
        column in {"retriever_contexts", "reranker_contexts"}
        or column.startswith("retriever_")
        or column.startswith("reranker_")
    )
