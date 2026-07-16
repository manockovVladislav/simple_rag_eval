from __future__ import annotations

import json
import math
import re
import traceback
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from rag_eval.schemas import ContextItem


EXCEL_MAX_CELL_LENGTH = 32_767
EXCEL_TRUNCATION_MARKER = "\n… [полное значение сохранено в JSON]"
EXCEL_ILLEGAL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            for key in ("results", "rows", "details"):
                if isinstance(payload.get(key), list):
                    return pd.DataFrame(payload[key])
        raise ValueError(f"JSON table {path} must contain a list of rows or a 'results' list.")
    raise ValueError(f"Unsupported file format: {path}. Use xlsx, csv, json, or jsonl.")


def read_run_table(path: Path) -> tuple[pd.DataFrame, Path]:
    """Read a run, preferring its lossless JSON companion over the Excel preview."""
    if path.suffix.lower() == ".json":
        return read_table(path), path

    frame = read_table(path)
    candidates: list[Path] = []
    if "json_file" in frame and len(frame):
        value = frame.iloc[0].get("json_file")
        if isinstance(value, str) and value.strip():
            configured = Path(value.strip())
            candidates.append(configured if configured.is_absolute() else path.parent / configured)
    candidates.append(path.with_suffix(".json"))
    for candidate in candidates:
        if candidate.exists():
            return read_table(candidate), candidate
    return frame, path


def write_xlsx(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            _excel_safe_frame(frame).to_excel(writer, sheet_name=sheet_name[:31], index=False)


def write_run_json(
    path: Path,
    rows: list[dict[str, Any]],
    xlsx_path: Path,
    parameters: dict[str, Any] | None = None,
) -> None:
    write_json(
        path,
        {
            "format_version": 1,
            "type": "rag_run",
            "xlsx_file": str(xlsx_path),
            "parameters": parameters or {},
            "results": rows,
        },
    )


def write_json(path: Path, payload: Any) -> None:
    """Write UTF-8 JSON atomically so an interrupted run cannot leave half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def append_json_rows(path: Path, section: str, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"format_version": 1, "type": "metrics_summary"}
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Metrics JSON {path} must contain an object.")
        payload.update(loaded)
    current = payload.get(section, [])
    if not isinstance(current, list):
        raise ValueError(f"Metrics JSON section '{section}' must contain a list.")
    payload[section] = [*current, *rows]
    write_json(path, payload)


def append_xlsx_rows(
    path: Path,
    sheet_name: str,
    rows: list[dict[str, Any]],
    leading_columns: list[str] | None = None,
    leading_prefixes: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_frame = pd.DataFrame(rows)
    sheets: dict[str, pd.DataFrame] = {}
    if path.exists():
        sheets = pd.read_excel(path, sheet_name=None)
        try:
            old_frame = sheets[sheet_name]
        except ValueError:
            old_frame = pd.DataFrame()
        except KeyError:
            old_frame = pd.DataFrame()
        frame = pd.concat([old_frame, new_frame], ignore_index=True)
    else:
        frame = new_frame
    frame = _order_columns(frame, leading_columns or [], leading_prefixes or [])
    sheets[sheet_name] = frame
    write_xlsx(path, sheets)


def _order_columns(frame: pd.DataFrame, leading_columns: list[str], leading_prefixes: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    explicit = [column for column in leading_columns if column in frame.columns]
    prefixed = [
        column
        for column in frame.columns
        if column not in explicit and any(column.startswith(prefix) for prefix in leading_prefixes)
    ]
    rest = [column for column in frame.columns if column not in explicit and column not in prefixed]
    return frame[explicit + prefixed + rest]


def _excel_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    safe = frame.copy()
    for column in safe.columns:
        safe[column] = safe[column].map(_excel_safe_value)
    return safe


def _excel_safe_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False)
    if isinstance(value, str):
        value = EXCEL_ILLEGAL_CHARACTERS.sub("�", value)
    if not isinstance(value, str) or len(value) <= EXCEL_MAX_CELL_LENGTH:
        return value
    available = EXCEL_MAX_CELL_LENGTH - len(EXCEL_TRUNCATION_MARKER)
    return value[:available] + EXCEL_TRUNCATION_MARKER


def _json_safe(value: Any, seen: set[int] | None = None) -> Any:
    """Convert a value to JSON data, replacing only broken values with diagnostics."""
    if seen is None:
        seen = set()
    try:
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, str):
            # json.dumps accepts lone surrogates, but a subsequent UTF-8 write does not.
            value.encode("utf-8")
            return value
        if isinstance(value, float):
            return None if math.isnan(value) or math.isinf(value) else value
        if isinstance(value, Path):
            return _json_safe(str(value), seen)
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, dict):
            identity = id(value)
            if identity in seen:
                raise ValueError("Circular reference detected while serializing JSON")
            seen.add(identity)
            try:
                result: dict[str, Any] = {}
                for key, item in value.items():
                    safe_key = _json_key(key)
                    result[safe_key] = _json_safe(item, seen)
                return result
            finally:
                seen.remove(identity)
        if isinstance(value, (list, tuple)):
            identity = id(value)
            if identity in seen:
                raise ValueError("Circular reference detected while serializing JSON")
            seen.add(identity)
            try:
                return [_json_safe(item, seen) for item in value]
            finally:
                seen.remove(identity)
        if hasattr(value, "item"):
            return _json_safe(value.item(), seen)
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
    except Exception as exc:
        return _serialization_error(exc)


def _json_key(value: Any) -> str:
    try:
        key = str(value)
        key.encode("utf-8")
        return key
    except Exception as exc:
        error = _serialization_error(exc)
        return f"__serialization_error__: {error['serialization_error']}"


def _serialization_error(exc: Exception) -> dict[str, str]:
    def utf8_safe(text: str) -> str:
        return text.encode("utf-8", errors="replace").decode("utf-8")

    return {
        "serialization_error": utf8_safe(f"{type(exc).__name__}: {exc}"),
        "traceback": utf8_safe(traceback.format_exc()),
    }


def to_json_text(value: Any, *, indent: int | None = None) -> str:
    """Serialize arbitrary application data without letting one bad value abort a run."""
    return json.dumps(_json_safe(value), ensure_ascii=False, indent=indent, allow_nan=False)


def contexts_to_json(contexts: list[ContextItem]) -> str:
    return to_json_text([item.to_record() for item in contexts])


def contexts_from_json(value: Any) -> list[dict[str, Any]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]
