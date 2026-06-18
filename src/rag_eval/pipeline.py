from __future__ import annotations

import importlib
from typing import Any, Callable

from rag_eval.config import AppConfig


def load_pipeline(config: AppConfig) -> Any:
    if not config.pipeline.factory:
        raise ValueError("PIPELINE_FACTORY is empty. Set it in config.py.")
    factory = import_object(config.pipeline.factory)
    return factory(config) if _accepts_config(factory) else factory()


def import_object(dotted_path: str) -> Callable[..., Any]:
    if ":" in dotted_path:
        module_name, object_name = dotted_path.split(":", 1)
    else:
        module_name, object_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def _accepts_config(factory: Callable[..., Any]) -> bool:
    try:
        return factory.__code__.co_argcount > 0
    except AttributeError:
        return True
