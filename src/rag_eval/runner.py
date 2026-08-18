from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from rag_eval.config import AppConfig
from rag_eval.generation import RagAnswerGenerator
from rag_eval.io import contexts_to_json, read_table, to_json_text, write_model_run_json, write_xlsx
from rag_eval.logging_utils import setup_file_logger
from rag_eval.pipeline import load_pipeline
from rag_eval.schemas import PipelineResult

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda value, **_: value


class EvaluationRunner:
    def __init__(self, config: AppConfig, pipeline: Any | None = None):
        self.config = config
        self.pipeline = pipeline

    @classmethod
    def from_config(cls, path: str | Path = "config.py") -> "EvaluationRunner":
        return cls(AppConfig.from_file(path))

    def run(self) -> Path:
        logger = setup_file_logger(self.config.paths.log_file)
        self.config.paths.run_outputs_dir.mkdir(parents=True, exist_ok=True)
        questions = read_table(self.config.paths.golden_questions)
        questions = self._validate_questions(questions)
        if self.config.run.max_questions:
            questions = questions.head(self.config.run.max_questions)

        pipeline = self.pipeline or load_pipeline(self.config)
        run_parameters = self._run_parameters(pipeline)
        generator = RagAnswerGenerator(self.config) if self.config.generation.enabled else None
        generation_providers = self._generation_providers()
        multi_generation = self.config.generation.parallel and len(generation_providers) > 1
        rows: list[dict[str, Any]] = []

        logger.info("run_started questions=%s pipeline=%s", len(questions), type(pipeline).__name__)
        iterator = tqdm(questions.iterrows(), total=len(questions), desc="Running RAG")
        for index, row in iterator:
            question = str(row[self.config.golden_columns.question])
            question_id = row.get("id", index + 1)
            ground_truth = self._ground_truth(row)
            relevant_chunk_ids = self._relevant_chunk_ids(row)
            started_at = datetime.now().isoformat(timespec="seconds")
            error = None
            model_outputs: dict[str, dict[str, str | None]] = {}
            result = PipelineResult()
            try:
                logger.info("question_started question_id=%s question=%s", question_id, question)
                raw_result = self._ask(pipeline, question)
                result = PipelineResult.from_any(raw_result)
                logger.info(
                    "contexts question_id=%s retriever=%s reranker=%s",
                    question_id,
                    contexts_to_json(result.retriever_contexts),
                    contexts_to_json(result.reranker_contexts),
                )
                if generator is not None and multi_generation:
                    generation_contexts = generator.contexts_for_generation(result)
                    logger.info(
                        "rag_llm_parallel_request question_id=%s providers=%s contexts=%s",
                        question_id,
                        generation_providers,
                        contexts_to_json(generation_contexts),
                    )
                    model_outputs = generator.generate_many(question, result)
                    primary_provider = (
                        self.config.generation.provider
                        if self.config.generation.provider in model_outputs
                        else generation_providers[0]
                    )
                    result.answer = model_outputs[primary_provider]["answer"]
                    logger.info("rag_llm_parallel_response question_id=%s outputs=%s", question_id, model_outputs)
                elif generator is not None and not result.answer:
                    generation_contexts = generator.contexts_for_generation(result)
                    logger.info(
                        "rag_llm_request question_id=%s provider=%s contexts=%s",
                        question_id,
                        self.config.generation.provider,
                        contexts_to_json(generation_contexts),
                    )
                    result.answer = generator.generate(question, result)
                    logger.info("rag_llm_response question_id=%s answer=%s", question_id, result.answer)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                logger.exception("question_failed question_id=%s", question_id)

            output_row = {
                "question_id": question_id,
                "question": question,
                "ground_truth": ground_truth,
                "expected_answer": ground_truth,
                "relevant_chunk_ids": relevant_chunk_ids,
                "answer": result.answer,
                "retriever_contexts": [item.to_record() for item in result.retriever_contexts],
                "reranker_contexts": [item.to_record() for item in result.reranker_contexts],
                "retriever_context_count": len(result.retriever_contexts),
                "reranker_context_count": len(result.reranker_contexts),
                "pipeline_metadata": result.metadata,
                **run_parameters,
                "error": error,
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }
            if multi_generation:
                for provider in generation_providers:
                    provider_output = model_outputs.get(provider, {})
                    output_row[f"answer_{provider}"] = provider_output.get("answer")
                    output_row[f"error_{provider}"] = provider_output.get("error")
            rows.append(output_row)
            if self.config.run.sleep_seconds:
                time.sleep(self.config.run.sleep_seconds)

        run_path = self._make_run_path()
        json_paths: dict[str, Path] = {}
        for provider in generation_providers:
            json_path = run_path.with_name(f"{run_path.stem}_{provider}.json")
            provider_rows = [self._provider_json_row(row, provider) for row in rows]
            provider_parameters = {
                **run_parameters,
                "answer_provider": provider,
                "answer_model": self.config.models[provider].model if provider in self.config.models else None,
            }
            write_model_run_json(
                json_path,
                provider_rows,
                run_path,
                provider,
                parameters=provider_parameters,
            )
            json_paths[provider] = json_path
        excel_rows = []
        for row in rows:
            excel_row = {
                **{f"json_file_{provider}": str(path) for provider, path in json_paths.items()},
                **row,
            }
            for column in ("retriever_contexts", "reranker_contexts", "pipeline_metadata"):
                excel_row[column] = to_json_text(excel_row[column])
            excel_rows.append(excel_row)
        write_xlsx(run_path, {"results": pd.DataFrame(excel_rows)})
        logger.info("run_finished output=%s model_json=%s", run_path, json_paths)
        return run_path

    def _provider_json_row(self, row: dict[str, Any], provider: str) -> dict[str, Any]:
        """Create a self-contained row with exactly one model answer."""
        excluded = {
            key
            for key in row
            if key.startswith("answer_") or key.startswith("error_")
        }
        provider_error = row.get(f"error_{provider}")
        pipeline_error = row.get("error")
        return {
            **{key: value for key, value in row.items() if key not in excluded and key != "answer"},
            "provider": provider,
            "model": self.config.models[provider].model if provider in self.config.models else None,
            "answer": row.get(f"answer_{provider}", row.get("answer")),
            "generation_error": provider_error,
            "pipeline_error": pipeline_error,
            "error": provider_error or pipeline_error,
        }

    def _run_parameters(self, pipeline: Any) -> dict[str, Any]:
        init = self.config.retriever_adapter.init_kwargs
        search = self.config.retriever_adapter.search_kwargs
        generation_model = self.config.models.get(self.config.generation.provider)
        judge_model = self.config.models.get(self.config.metrics.ragas_judge_provider)

        def actual(name: str, default: Any = None) -> Any:
            value = getattr(pipeline, name, None)
            if value is not None:
                return value
            if name in search:
                return search[name]
            return init.get(name, default)

        return {
            "pipeline_factory": self.config.pipeline.factory,
            "model_name": generation_model.model if generation_model else None,
            "generation_providers": self._generation_providers(),
            "parallel_generation": self.config.generation.parallel,
            "temperature": generation_model.temperature if generation_model else None,
            "judge_model_name": judge_model.model if judge_model else None,
            "judge_temperature": 0.0 if self.config.metrics.ragas_backend == "custom" else (
                judge_model.temperature if judge_model else None
            ),
            "k_rrf": actual("k_rrf", 60),
            "fusion_method": actual("fusion_method", "rrf"),
            "alpha": actual("alpha", 0.5),
            "bias": actual("bias", 0.0),
            "rerank_initial_k": actual("rerank_initial_k"),
            "retriever_top_k": self.config.retriever_adapter.top_k,
            "rag_context_source": self.config.generation.context_source,
            "rag_max_contexts": self.config.generation.max_contexts,
            "judge_backend": self.config.metrics.ragas_backend,
            "judge_context_source": self.config.metrics.ragas_context_source,
            "judge_max_context_chars": self.config.metrics.ragas_judge_max_context_chars,
            "retriever_init_kwargs": init,
            "retriever_search_kwargs": search,
        }

    def _generation_providers(self) -> list[str]:
        providers = (
            self.config.generation.providers
            if self.config.generation.parallel
            else [self.config.generation.provider]
        )
        providers = providers or [self.config.generation.provider]
        # Preserve config order while preventing duplicate calls/columns.
        return list(dict.fromkeys(provider for provider in providers if provider))

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

    def _ground_truth(self, row: pd.Series) -> Any:
        for column in (
            self.config.golden_columns.ground_truth,
            self.config.golden_columns.expected_answer,
            "ground_truth",
            "grouth_true",
            "golden_answer",
            "expected_answer",
        ):
            if column in row and _has_value(row.get(column)):
                return row.get(column)
        return None

    def _relevant_chunk_ids(self, row: pd.Series) -> Any:
        for column in (
            self.config.golden_columns.chunk_id,
            "chunk_ids",
            "relevant_chunk_ids",
            "chank_id",
            "chunk_id",
        ):
            if column in row and _has_value(row.get(column)):
                return row.get(column)
        return None

    def _make_run_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return self.config.paths.run_outputs_dir / f"rag_run_{timestamp}.xlsx"


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True
