from __future__ import annotations

import re
from collections import Counter


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(expected: object, actual: object) -> float | None:
    expected_text = normalize_text(expected)
    actual_text = normalize_text(actual)
    if not expected_text or not actual_text:
        return None
    return float(expected_text == actual_text)


def contains_expected(expected: object, actual: object) -> float | None:
    expected_text = normalize_text(expected)
    actual_text = normalize_text(actual)
    if not expected_text or not actual_text:
        return None
    return float(expected_text in actual_text)


def token_f1(expected: object, actual: object) -> float | None:
    expected_tokens = normalize_text(expected).split()
    actual_tokens = normalize_text(actual).split()
    if not expected_tokens or not actual_tokens:
        return None
    expected_counter = Counter(expected_tokens)
    actual_counter = Counter(actual_tokens)
    overlap = sum((expected_counter & actual_counter).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(actual_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)
