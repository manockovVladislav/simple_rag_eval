from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from rag_eval.schemas import ContextItem


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported file format: {path}. Use xlsx, csv, or jsonl.")


def write_xlsx(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)


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


def contexts_to_json(contexts: list[ContextItem]) -> str:
    return json.dumps([item.to_record() for item in contexts], ensure_ascii=False)


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
