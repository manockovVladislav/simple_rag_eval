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


@dataclass(slots=True)
class GoldenColumnsConfig:
    question: str = "question"
    expected_answer: str = "expected_answer"
    expected_context_ids: str = "expected_context_ids"


@dataclass(slots=True)
class PipelineConfig:
    factory: str = ""


@dataclass(slots=True)
class RunConfig:
    max_questions: int | None = None
    sleep_seconds: float = 0


@dataclass(slots=True)
class MetricsConfig:
    context_ks: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    answer_enabled: bool = True
    ragas_enabled: bool = True
    ragas_judge_provider: str = "qwen"
    ragas_context_source: str = "reranker"
    ragas_timeout_seconds: int = 60
    ragas_max_workers: int = 1
    ragas_max_retries: int = 0
    ragas_metrics: list[str] = field(
        default_factory=lambda: [
            "faithfulness",
            "context_precision",
            "context_recall",
            "answer_correctness",
        ]
    )


@dataclass(slots=True)
class ModelConfig:
    model: str
    base_url: str = ""
    provider: str = "openai_compatible"
    api_key_env: str = ""
    auth_type: str = "none"
    token_url: str = ""
    credentials_env: str = ""
    scope: str = "GIGACHAT_API_PERS"
    timeout_seconds: int = 60
    temperature: float = 0
    verify_ssl: bool = True
    task: str = "text-generation"
    device: int | None = None
    device_map: str | None = "auto"
    torch_dtype: str = "auto"
    max_new_tokens: int = 1024
    do_sample: bool = False
    return_full_text: bool = False


@dataclass(slots=True)
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    golden_columns: GoldenColumnsConfig = field(default_factory=GoldenColumnsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    run: RunConfig = field(default_factory=RunConfig)
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
        run = _dataclass_from_dict(RunConfig, raw.get("run", {}))
        metrics = _dataclass_from_dict(MetricsConfig, raw.get("metrics", {}))
        models = {
            name: _dataclass_from_dict(ModelConfig, value)
            for name, value in (raw.get("models") or {}).items()
        }
        return cls(paths=paths, golden_columns=columns, pipeline=pipeline, run=run, metrics=metrics, models=models)

def _dataclass_from_dict(class_: type, values: dict[str, Any]):
    allowed = set(class_.__dataclass_fields__.keys())
    return class_(**{key: value for key, value in values.items() if key in allowed})


def _has_flat_config(module: Any) -> bool:
    return any(
        hasattr(module, name)
        for name in (
            "GOLDEN_QUESTIONS_PATH",
            "PIPELINE_FACTORY",
            "QWEN_MODEL_PATH",
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
        },
        "golden_columns": {
            "question": getattr(module, "QUESTION_COLUMN", "question"),
            "expected_answer": getattr(module, "EXPECTED_ANSWER_COLUMN", "expected_answer"),
            "expected_context_ids": getattr(module, "EXPECTED_CONTEXT_IDS_COLUMN", "expected_context_ids"),
        },
        "pipeline": {
            "factory": getattr(module, "PIPELINE_FACTORY", ""),
        },
        "run": {
            "max_questions": getattr(module, "MAX_QUESTIONS", None),
            "sleep_seconds": getattr(module, "SLEEP_SECONDS", 0),
        },
        "metrics": {
            "context_ks": getattr(module, "CONTEXT_KS", [1, 3, 5, 10]),
            "answer_enabled": getattr(module, "ANSWER_METRICS_ENABLED", True),
            "ragas_enabled": getattr(module, "RAGAS_ENABLED", True),
            "ragas_judge_provider": getattr(module, "RAGAS_JUDGE_PROVIDER", "qwen"),
            "ragas_context_source": getattr(module, "RAGAS_CONTEXT_SOURCE", "reranker"),
            "ragas_timeout_seconds": getattr(module, "RAGAS_TIMEOUT_SECONDS", 60),
            "ragas_max_workers": getattr(module, "RAGAS_MAX_WORKERS", 1),
            "ragas_max_retries": getattr(module, "RAGAS_MAX_RETRIES", 0),
            "ragas_metrics": getattr(
                module,
                "RAGAS_METRICS",
                ["faithfulness", "context_precision", "context_recall", "answer_correctness"],
            ),
        },
        "models": {
            "qwen": {
                "provider": getattr(module, "QWEN_PROVIDER", "qwen_transformers"),
                "model": getattr(module, "QWEN_MODEL_PATH", getattr(module, "QWEN_MODEL", "Qwen/Qwen3-8B")),
                "auth_type": "none",
                "timeout_seconds": getattr(module, "QWEN_TIMEOUT_SECONDS", 60),
                "temperature": getattr(module, "QWEN_TEMPERATURE", 0),
                "task": getattr(module, "QWEN_TASK", "text-generation"),
                "device": getattr(module, "QWEN_DEVICE", None),
                "device_map": getattr(module, "QWEN_DEVICE_MAP", "auto"),
                "torch_dtype": getattr(module, "QWEN_TORCH_DTYPE", "auto"),
                "max_new_tokens": getattr(module, "QWEN_MAX_NEW_TOKENS", 1024),
                "do_sample": getattr(module, "QWEN_DO_SAMPLE", False),
                "return_full_text": getattr(module, "QWEN_RETURN_FULL_TEXT", False),
            },
            "gigachat": {
                "provider": "gigachat_api",
                "base_url": getattr(module, "GIGACHAT_BASE_URL", "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"),
                "model": getattr(module, "GIGACHAT_MODEL", "GigaChat"),
                "auth_type": getattr(module, "GIGACHAT_AUTH_TYPE", "bearer_env"),
                "api_key_env": getattr(module, "GIGACHAT_API_KEY_ENV", "GIGACHAT_API_KEY"),
                "token_url": getattr(module, "GIGACHAT_TOKEN_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"),
                "credentials_env": getattr(module, "GIGACHAT_CREDENTIALS_ENV", "GIGACHAT_CREDENTIALS"),
                "scope": getattr(module, "GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
                "timeout_seconds": getattr(module, "GIGACHAT_TIMEOUT_SECONDS", 60),
                "temperature": getattr(module, "GIGACHAT_TEMPERATURE", 0),
                "verify_ssl": getattr(module, "GIGACHAT_VERIFY_SSL", True),
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
    }
    return PathsConfig(
        golden_questions=_resolve(base_dir, raw["golden_questions"]),
        run_outputs_dir=_resolve(base_dir, raw["run_outputs_dir"]),
        metrics_outputs_dir=_resolve(base_dir, raw["metrics_outputs_dir"]),
        summary_metrics_file=_resolve(base_dir, raw["summary_metrics_file"]),
    )


def _resolve(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path
