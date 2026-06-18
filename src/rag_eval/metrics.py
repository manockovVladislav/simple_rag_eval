from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from rag_eval.config import AppConfig
from rag_eval.io import append_xlsx_rows, contexts_from_json, parse_expected_ids, read_table
from rag_eval.ragas_evaluator import RagasEvaluator
from rag_eval.text_metrics import contains_expected, exact_match, token_f1


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
        run_path = Path(run_file)
        frame = read_table(run_path)
        detail_rows = [self._row_metrics(row) for _, row in frame.iterrows()]
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
            leading_columns=["created_at", "run_file", "question_count", "error_count"],
            leading_prefixes=["ragas_"],
        )
        append_xlsx_rows(
            summary_path,
            "details",
            detail_rows_with_run,
            leading_columns=["created_at", "run_file", "question_id", "question"],
            leading_prefixes=["ragas_"],
        )
        return summary_path

    def latest_run_file(self) -> Path:
        files = sorted(self.config.paths.run_outputs_dir.glob("rag_run_*.xlsx"))
        if not files:
            raise FileNotFoundError(f"No run files found in {self.config.paths.run_outputs_dir}.")
        return files[-1]

    def _row_metrics(self, row: pd.Series) -> dict[str, Any]:
        result: dict[str, Any] = {
            "question_id": row.get("question_id"),
            "question": row.get("question"),
            "has_error": bool(row.get("error")) if not pd.isna(row.get("error")) else False,
        }
        expected_ids = parse_expected_ids(row.get("expected_context_ids"))
        if expected_ids:
            if "retriever_contexts" in row and not pd.isna(row.get("retriever_contexts")):
                result.update(_context_metrics("retriever", expected_ids, contexts_from_json(row.get("retriever_contexts")), self.config.metrics.context_ks))
            if "reranker_contexts" in row and not pd.isna(row.get("reranker_contexts")):
                result.update(_context_metrics("reranker", expected_ids, contexts_from_json(row.get("reranker_contexts")), self.config.metrics.context_ks))

        expected_answer = row.get("expected_answer")
        answer = row.get("answer")
        if self.config.metrics.answer_enabled and _has_text(expected_answer) and _has_text(answer):
            result["answer_exact_match"] = exact_match(expected_answer, answer)
            result["answer_contains_expected"] = contains_expected(expected_answer, answer)
            result["answer_token_f1"] = token_f1(expected_answer, answer)
        return result

    def _add_ragas_metrics(self, frame: pd.DataFrame, detail_rows: list[dict[str, Any]]) -> None:
        ragas_results = RagasEvaluator(self.config).evaluate_rows(frame)
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
        metric_columns = [
            column
            for column in details.columns
            if column not in {"question_id", "question", "has_error"} and pd.api.types.is_numeric_dtype(details[column])
        ]
        for column in metric_columns:
            valid = details[column].dropna()
            summary[f"{column}_mean"] = valid.mean() if len(valid) else None
            summary[f"{column}_count"] = int(len(valid))
        return summary

def _context_metrics(prefix: str, expected_ids: list[str], contexts: list[dict[str, Any]], ks: list[int]) -> dict[str, float | None]:
    predicted_ids = [_context_id(context) for context in contexts]
    predicted_ids = [item_id for item_id in predicted_ids if item_id]
    if not predicted_ids:
        return {f"{prefix}_{metric}@{k}": None for k in ks for metric in ("precision", "recall", "hit", "mrr", "ndcg")}

    expected_set = set(expected_ids)
    result: dict[str, float | None] = {}
    for k in ks:
        top_k = predicted_ids[:k]
        hits = [1 if item_id in expected_set else 0 for item_id in top_k]
        hit_count = sum(hits)
        result[f"{prefix}_precision@{k}"] = hit_count / len(top_k) if top_k else None
        result[f"{prefix}_recall@{k}"] = hit_count / len(expected_set) if expected_set else None
        result[f"{prefix}_hit@{k}"] = float(hit_count > 0)
        result[f"{prefix}_mrr@{k}"] = _mrr(hits)
        result[f"{prefix}_ndcg@{k}"] = _ndcg(hits, min(len(expected_set), k))
    return result


def _context_id(context: dict[str, Any]) -> str | None:
    value = context.get("id") or context.get("doc_id") or context.get("document_id")
    metadata = context.get("metadata") or {}
    value = value or metadata.get("id") or metadata.get("doc_id") or metadata.get("source")
    return str(value) if value is not None else None


def _mrr(hits: list[int]) -> float:
    for index, hit in enumerate(hits, start=1):
        if hit:
            return 1 / index
    return 0.0


def _ndcg(hits: list[int], ideal_hits: int) -> float:
    if ideal_hits == 0:
        return 0.0
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
    ideal_dcg = sum(1 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def _has_text(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and pd.isna(value)) and bool(str(value).strip())
