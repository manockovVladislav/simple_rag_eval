from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from rag_eval.config import AppConfig
from rag_eval.io import append_xlsx_rows, read_table
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


def _has_text(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and pd.isna(value)) and bool(str(value).strip())
