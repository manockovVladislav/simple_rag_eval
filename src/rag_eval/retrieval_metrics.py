from __future__ import annotations

import json
import math
import re
from typing import Any

from rag_eval.io import contexts_from_json

_ID_KEYS = ("chunk_id", "chank_id", "chunk_ids", "id")


def parse_relevant_chunk_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, list | tuple | set):
        return [_clean_id(item) for item in value if _clean_id(item)]

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [_clean_id(item) for item in parsed if _clean_id(item)]

    return [_clean_id(item) for item in re.split(r"[;,\n|]+", text) if _clean_id(item)]


def context_chunk_ids(value: Any) -> list[str]:
    ids: list[str] = []
    for context in contexts_from_json(value):
        chunk_id = _extract_chunk_id(context)
        ids.append(chunk_id or "")
    return ids


def retrieval_metrics_for_ids(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k_values: list[int],
) -> dict[str, float]:
    relevant = {item for item in relevant_ids if item}
    if not relevant:
        return {}

    result: dict[str, float] = {}
    for k in sorted({int(value) for value in k_values if int(value) > 0}):
        top = retrieved_ids[:k]
        hits = [item for item in top if item and item in relevant]
        first_rank = next(
            (rank for rank, item in enumerate(top, start=1) if item and item in relevant),
            None,
        )
        dcg = sum(1 / math.log2(rank + 1) for rank, item in enumerate(top, start=1) if item and item in relevant)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

        result[f"hit_rate_at_{k}"] = 1.0 if hits else 0.0
        result[f"precision_at_{k}"] = len(hits) / k
        result[f"recall_at_{k}"] = len(set(hits)) / len(relevant)
        result[f"mrr_at_{k}"] = 1 / first_rank if first_rank is not None else 0.0
        result[f"ndcg_at_{k}"] = dcg / idcg if idcg else 0.0
    return result


def _extract_chunk_id(context: dict[str, Any]) -> str:
    for key in _ID_KEYS:
        value = context.get(key)
        if _clean_id(value):
            return _clean_id(value)

    metadata = context.get("metadata")
    if isinstance(metadata, dict):
        for key in _ID_KEYS:
            value = metadata.get(key)
            if _clean_id(value):
                return _clean_id(value)
    return ""


def _clean_id(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()
