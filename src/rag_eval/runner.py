from __future__ import annotations

import time
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from rag_eval.config import AppConfig
from rag_eval.io import contexts_to_json, read_table, write_xlsx
from rag_eval.pipeline import load_pipeline
from rag_eval.schemas import PipelineResult


class EvaluationRunner:
    def __init__(self, config: AppConfig, pipeline: Any | None = None):
        self.config = config
        self.pipeline = pipeline

    @classmethod
    def from_config(cls, path: str | Path = "config.py") -> "EvaluationRunner":
        return cls(AppConfig.from_file(path))

    def run(self) -> Path:
        self.config.paths.run_outputs_dir.mkdir(parents=True, exist_ok=True)
        questions = read_table(self.config.paths.golden_questions)
        questions = self._validate_questions(questions)
        if self.config.run.max_questions:
            questions = questions.head(self.config.run.max_questions)

        pipeline = self.pipeline or load_pipeline(self.config)
        rows: list[dict[str, Any]] = []

        for index, row in questions.iterrows():
            question = str(row[self.config.golden_columns.question])
            started_at = datetime.now().isoformat(timespec="seconds")
            error = None
            result = PipelineResult()
            try:
                raw_result = self._ask(pipeline, question)
                result = PipelineResult.from_any(raw_result)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            rows.append(
                {
                    "question_id": row.get("id", index + 1),
                    "question": question,
                    "expected_answer": row.get(self.config.golden_columns.expected_answer),
                    "answer": result.answer,
                    "retriever_contexts": contexts_to_json(result.retriever_contexts),
                    "reranker_contexts": contexts_to_json(result.reranker_contexts),
                    "pipeline_metadata": json.dumps(result.metadata, ensure_ascii=False),
                    "error": error,
                    "started_at": started_at,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            if self.config.run.sleep_seconds:
                time.sleep(self.config.run.sleep_seconds)

        run_path = self._make_run_path()
        write_xlsx(run_path, {"results": pd.DataFrame(rows)})
        return run_path

    def _validate_questions(self, questions: pd.DataFrame) -> pd.DataFrame:
        question_column = self.config.golden_columns.question
        if question_column not in questions.columns:
            raise ValueError(f"Golden file must contain '{question_column}' column.")
        return questions[questions[question_column].notna()].reset_index(drop=True)

    def _ask(self, pipeline: Any, question: str) -> PipelineResult | dict[str, Any]:
        if hasattr(pipeline, "answer_question"):
            return pipeline.answer_question(question)
        if callable(pipeline):
            return pipeline(question)
        raise TypeError("Pipeline must be callable or have answer_question(question).")

    def _make_run_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return self.config.paths.run_outputs_dir / f"rag_run_{timestamp}.xlsx"
