from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PathsConfig:
    golden_questions: Path = Path("data/golden_questions.xlsx")
    run_outputs_dir: Path = Path("outputs/runs")
    metrics_outputs_dir: Path = Path("outputs/metrics")
    summary_metrics_file: Path = Path("outputs/metrics/metrics_summary.xlsx")
    model_comparison_file: Path = Path("outputs/metrics/model_comparison.xlsx")
    log_file: Path = Path("outputs/logs/rag_eval.log")


@dataclass(slots=True)
class GoldenColumnsConfig:
    question: str = "question"
    expected_answer: str = "expected_answer"
    ground_truth: str = "ground_truth"
    chunk_id: str = "chunk_id"


@dataclass(slots=True)
class PipelineConfig:
    factory: str = ""


@dataclass(slots=True)
class RetrieverAdapterConfig:
    init_args: list[Any] = field(default_factory=list)
    init_kwargs: dict[str, Any] = field(default_factory=dict)
    search_kwargs: dict[str, Any] = field(default_factory=dict)
    top_k: int = 10


@dataclass(slots=True)
class RunConfig:
    max_questions: int | None = None
    sleep_seconds: float = 0


@dataclass(slots=True)
class GenerationConfig:
    enabled: bool = True
    provider: str = "qwen"
    providers: list[str] = field(default_factory=list)
    parallel: bool = False
    max_workers: int = 3
    context_source: str = "reranker"
    max_contexts: int = 10
    system_prompt: str = (
        "Ты отвечаешь на вопрос только по переданному контексту. "
        "Если в контексте нет ответа, так и скажи."
    )


@dataclass(slots=True)
class MetricsConfig:
    answer_enabled: bool = True
    ragas_enabled: bool = True
    ragas_backend: str = "ragas"
    ragas_judge_provider: str = "qwen"
    ragas_embeddings_provider: str = ""
    ragas_context_source: str = "reranker"
    ragas_timeout_seconds: int = 60
    ragas_max_workers: int = 1
    ragas_max_retries: int = 0
    ragas_judge_max_context_chars: int = 60_000
    retrieval_k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    ragas_metrics: list[str] = field(
        default_factory=lambda: [
            "faithfulness",
            "answer_correctness",
            "answer_relevancy",
            "answer_similarity",
        ]
    )


@dataclass(slots=True)
class ModelConfig:
    model: str
    base_url: str = ""
    provider: str = "openai_compatible"
    access_token: str = ""
    api_key_env: str = ""
    auth_type: str = "none"
    timeout_seconds: int = 60
    temperature: float = 0
    verify_ssl: bool = True
    min_seconds_between_requests: float = 0
    local_files_only: bool = False
    max_new_tokens: int = 1024
    embedding_device: str | None = None
    normalize_embeddings: bool = True


@dataclass(slots=True)
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    golden_columns: GoldenColumnsConfig = field(default_factory=GoldenColumnsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    retriever_adapter: RetrieverAdapterConfig = field(default_factory=RetrieverAdapterConfig)
    run: RunConfig = field(default_factory=RunConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    models: dict[str, ModelConfig] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "AppConfig":
        config_path = Path(path)
        if config_path.suffix.lower() == ".py":
            return cls.from_py(config_path)
        raise ValueError(f"Unsupported config format: {config_path}. Use .py.")

    @classmethod
    def from_py(cls, path: str | Path) -> "AppConfig":
        config_path = Path(path)
        module_name = f"_rag_eval_user_config_{config_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, config_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load config file: {config_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if _has_flat_config(module):
            raw = _config_from_flat_variables(module)
        else:
            raise ValueError(
                "Python config must define flat uppercase variables."
            )

        return cls.from_dict(raw, config_path.parent)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], base_dir: str | Path = ".") -> "AppConfig":
        base_path = Path(base_dir)
        paths = _paths(raw.get("paths", {}), base_path)
        columns = _dataclass_from_dict(GoldenColumnsConfig, raw.get("golden_columns", {}))
        pipeline = _dataclass_from_dict(PipelineConfig, raw.get("pipeline", {}))
        retriever_adapter = _retriever_adapter(raw.get("retriever_adapter", {}), base_path)
        run = _dataclass_from_dict(RunConfig, raw.get("run", {}))
        generation = _dataclass_from_dict(GenerationConfig, raw.get("generation", {}))
        metrics = _dataclass_from_dict(MetricsConfig, raw.get("metrics", {}))
        models = {
            name: _dataclass_from_dict(ModelConfig, value)
            for name, value in (raw.get("models") or {}).items()
        }
        return cls(
            paths=paths,
            golden_columns=columns,
            pipeline=pipeline,
            retriever_adapter=retriever_adapter,
            run=run,
            generation=generation,
            metrics=metrics,
            models=models,
        )

def _dataclass_from_dict(class_: type, values: dict[str, Any]):
    allowed = set(class_.__dataclass_fields__.keys())
    return class_(**{key: value for key, value in values.items() if key in allowed})


def _has_flat_config(module: Any) -> bool:
    return any(
        hasattr(module, name)
        for name in (
            "GOLDEN_QUESTIONS_PATH",
            "PIPELINE_FACTORY",
            "RETRIEVER_INIT_KWARGS",
            "QWEN_MODEL",
            "GIGACHAT_BASE_URL",
            "RAGAS_JUDGE_PROVIDER",
        )
    )


def _config_from_flat_variables(module: Any) -> dict[str, Any]:
    return {
        "paths": {
            "golden_questions": getattr(module, "GOLDEN_QUESTIONS_PATH", "data/golden_questions.xlsx"),
            "run_outputs_dir": getattr(module, "RUN_OUTPUTS_DIR", "outputs/runs"),
            "metrics_outputs_dir": getattr(module, "METRICS_OUTPUTS_DIR", "outputs/metrics"),
            "summary_metrics_file": getattr(module, "SUMMARY_METRICS_FILE", "outputs/metrics/metrics_summary.xlsx"),
            "model_comparison_file": getattr(module, "MODEL_COMPARISON_FILE", "outputs/metrics/model_comparison.xlsx"),
            "log_file": getattr(module, "LOG_FILE", "outputs/logs/rag_eval.log"),
        },
        "golden_columns": {
            "question": getattr(module, "QUESTION_COLUMN", "question"),
            "expected_answer": getattr(module, "EXPECTED_ANSWER_COLUMN", "expected_answer"),
            "ground_truth": getattr(module, "GROUND_TRUTH_COLUMN", "ground_truth"),
            "chunk_id": getattr(module, "CHUNK_ID_COLUMN", "chunk_id"),
        },
        "pipeline": {
            "factory": getattr(module, "PIPELINE_FACTORY", ""),
        },
        "retriever_adapter": {
            "init_args": getattr(module, "RETRIEVER_INIT_ARGS", []),
            "init_kwargs": getattr(module, "RETRIEVER_INIT_KWARGS", {}),
            "search_kwargs": getattr(module, "RETRIEVER_SEARCH_KWARGS", {}),
            "top_k": getattr(module, "RETRIEVER_TOP_K", 10),
        },
        "run": {
            "max_questions": getattr(module, "MAX_QUESTIONS", None),
            "sleep_seconds": getattr(module, "SLEEP_SECONDS", 0),
        },
        "generation": {
            "enabled": getattr(module, "RAG_ANSWER_ENABLED", True),
            "provider": getattr(module, "RAG_LLM_PROVIDER", "qwen"),
            "providers": getattr(module, "RAG_LLM_PROVIDERS", []),
            "parallel": getattr(module, "RAG_PARALLEL_GENERATION_ENABLED", False),
            "max_workers": getattr(module, "RAG_GENERATION_MAX_WORKERS", 3),
            "context_source": getattr(module, "RAG_CONTEXT_SOURCE", "reranker"),
            "max_contexts": getattr(module, "RAG_MAX_CONTEXTS", 10),
            "system_prompt": getattr(
                module,
                "RAG_SYSTEM_PROMPT",
                GenerationConfig().system_prompt,
            ),
        },
        "metrics": {
            "answer_enabled": getattr(module, "ANSWER_METRICS_ENABLED", True),
            "ragas_enabled": getattr(module, "RAGAS_ENABLED", True),
            "ragas_backend": getattr(module, "RAGAS_BACKEND", "ragas"),
            "ragas_judge_provider": getattr(module, "RAGAS_JUDGE_PROVIDER", "qwen"),
            "ragas_embeddings_provider": getattr(module, "RAGAS_EMBEDDINGS_PROVIDER", ""),
            "ragas_context_source": getattr(module, "RAGAS_CONTEXT_SOURCE", "reranker"),
            "ragas_timeout_seconds": getattr(module, "RAGAS_TIMEOUT_SECONDS", 60),
            "ragas_max_workers": getattr(module, "RAGAS_MAX_WORKERS", 1),
            "ragas_max_retries": getattr(module, "RAGAS_MAX_RETRIES", 0),
            "ragas_judge_max_context_chars": getattr(module, "RAGAS_JUDGE_MAX_CONTEXT_CHARS", 60_000),
            "retrieval_k_values": getattr(module, "RETRIEVAL_K_VALUES", [1, 3, 5, 10]),
            "ragas_metrics": getattr(
                module,
                "RAGAS_METRICS",
                ["faithfulness", "answer_correctness", "answer_relevancy", "answer_similarity"],
            ),
        },
        "models": {
            "qwen": {
                "provider": "gigachat_api",
                "model": getattr(module, "QWEN_MODEL", "Qwen"),
                "base_url": getattr(module, "QWEN_BASE_URL", ""),
                "access_token": getattr(module, "QWEN_ACCESS_TOKEN", ""),
                "timeout_seconds": getattr(module, "QWEN_TIMEOUT_SECONDS", 60),
                "temperature": getattr(module, "QWEN_TEMPERATURE", 0),
                "verify_ssl": getattr(module, "QWEN_VERIFY_SSL", True),
                "min_seconds_between_requests": getattr(module, "QWEN_MIN_SECONDS_BETWEEN_REQUESTS", 0),
            },
            "bge_m3": {
                "provider": "huggingface_embeddings",
                "model": getattr(module, "BGE_M3_MODEL_PATH", "/home/vladislav/models/bge-m3"),
                "local_files_only": getattr(module, "BGE_M3_LOCAL_FILES_ONLY", True),
                "embedding_device": getattr(module, "BGE_M3_DEVICE", None),
                "normalize_embeddings": getattr(module, "BGE_M3_NORMALIZE_EMBEDDINGS", True),
            },
            "gigachat": {
                "provider": "gigachat_api",
                "base_url": getattr(
                    module,
                    "GIGACHAT_BASE_URL",
                    "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
                ),
                "access_token": getattr(module, "GIGACHAT_ACCESS_TOKEN", ""),
                "model": getattr(module, "GIGACHAT_MODEL", "GigaChat"),
                "temperature": getattr(module, "GIGACHAT_TEMPERATURE", 0),
                "verify_ssl": getattr(module, "GIGACHAT_VERIFY_SSL", True),
                "min_seconds_between_requests": getattr(module, "GIGACHAT_MIN_SECONDS_BETWEEN_REQUESTS", 10),
            },
            "glm": {
                "provider": "gigachat_api",
                "base_url": getattr(module, "GLM_BASE_URL", ""),
                "access_token": getattr(module, "GLM_ACCESS_TOKEN", ""),
                "model": getattr(module, "GLM_MODEL", "glm"),
                "timeout_seconds": getattr(module, "GLM_TIMEOUT_SECONDS", 60),
                "temperature": getattr(module, "GLM_TEMPERATURE", 0),
                "verify_ssl": getattr(module, "GLM_VERIFY_SSL", True),
                "min_seconds_between_requests": getattr(module, "GLM_MIN_SECONDS_BETWEEN_REQUESTS", 0),
            },
        },
    }


def _paths(values: dict[str, Any], base_dir: Path) -> PathsConfig:
    config = PathsConfig()
    raw = {
        "golden_questions": values.get("golden_questions", config.golden_questions),
        "run_outputs_dir": values.get("run_outputs_dir", config.run_outputs_dir),
        "metrics_outputs_dir": values.get("metrics_outputs_dir", config.metrics_outputs_dir),
        "summary_metrics_file": values.get("summary_metrics_file", config.summary_metrics_file),
        "model_comparison_file": values.get("model_comparison_file", config.model_comparison_file),
        "log_file": values.get("log_file", config.log_file),
    }
    return PathsConfig(
        golden_questions=_resolve(base_dir, raw["golden_questions"]),
        run_outputs_dir=_resolve(base_dir, raw["run_outputs_dir"]),
        metrics_outputs_dir=_resolve(base_dir, raw["metrics_outputs_dir"]),
        summary_metrics_file=_resolve(base_dir, raw["summary_metrics_file"]),
        model_comparison_file=_resolve(base_dir, raw["model_comparison_file"]),
        log_file=_resolve(base_dir, raw["log_file"]),
    )


def _retriever_adapter(values: dict[str, Any], base_dir: Path) -> RetrieverAdapterConfig:
    return RetrieverAdapterConfig(
        init_args=list(values.get("init_args") or []),
        init_kwargs=dict(values.get("init_kwargs") or {}),
        search_kwargs=dict(values.get("search_kwargs") or {}),
        top_k=int(values.get("top_k", 10)),
    )


def _resolve(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path
