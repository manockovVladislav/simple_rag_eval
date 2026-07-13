from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from rag_eval.io import EXCEL_MAX_CELL_LENGTH, read_run_table, write_run_json, write_xlsx


class RunIoTests(unittest.TestCase):
    def test_run_json_preserves_context_larger_than_excel_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            xlsx_path = Path(directory) / "rag_run_test.xlsx"
            json_path = xlsx_path.with_suffix(".json")
            long_context = json.dumps([{"text": "я" * 70_000}], ensure_ascii=False)
            rows = [{"question_id": 1, "retriever_contexts": long_context}]

            write_run_json(json_path, rows, xlsx_path)
            write_xlsx(
                xlsx_path,
                {"results": pd.DataFrame([{"json_file": str(json_path), **rows[0]}])},
            )

            excel_frame = pd.read_excel(xlsx_path)
            self.assertLessEqual(len(excel_frame.loc[0, "retriever_contexts"]), EXCEL_MAX_CELL_LENGTH)
            full_frame, source_path = read_run_table(xlsx_path)
            self.assertEqual(source_path, json_path)
            self.assertEqual(full_frame.loc[0, "retriever_contexts"], long_context)

    def test_run_table_finds_sibling_json_after_files_are_moved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            xlsx_path = Path(directory) / "rag_run_test.xlsx"
            json_path = xlsx_path.with_suffix(".json")
            write_run_json(json_path, [{"question_id": 7}], xlsx_path)
            write_xlsx(
                xlsx_path,
                {"results": pd.DataFrame([{"json_file": "/missing/rag_run_test.json", "question_id": 7}])},
            )

            frame, source_path = read_run_table(xlsx_path)
            self.assertEqual(source_path, json_path)
            self.assertEqual(frame.loc[0, "question_id"], 7)
