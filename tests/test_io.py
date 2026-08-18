from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from rag_eval.io import (
    EXCEL_MAX_CELL_LENGTH,
    read_model_run_tables,
    read_run_table,
    write_model_run_json,
    write_run_json,
    write_xlsx,
)


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

    def test_json_replaces_invalid_unicode_context_with_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            xlsx_path = Path(directory) / "rag_run_test.xlsx"
            json_path = xlsx_path.with_suffix(".json")
            rows = [{"retriever_contexts": [{"text": "valid emoji 😊"}, {"text": "bad \ud800 text"}]}]

            write_run_json(json_path, rows, xlsx_path)

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            contexts = payload["results"][0]["retriever_contexts"]
            self.assertEqual(contexts[0]["text"], "valid emoji 😊")
            self.assertIn("UnicodeEncodeError", contexts[1]["text"]["serialization_error"])
            self.assertIn("Traceback", contexts[1]["text"]["traceback"])

    def test_xlsx_replaces_xml_control_characters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            xlsx_path = Path(directory) / "rag_run_test.xlsx"

            write_xlsx(xlsx_path, {"results": pd.DataFrame([{"answer": "hello\x00world 😊"}])})

            frame = pd.read_excel(xlsx_path)
            self.assertEqual(frame.loc[0, "answer"], "hello�world 😊")

    def test_each_model_answer_is_saved_to_its_own_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xlsx_path = root / "rag_run_test.xlsx"
            giga_path = root / "rag_run_test_gigachat.json"
            qwen_path = root / "rag_run_test_qwen.json"
            write_model_run_json(giga_path, [{"question_id": 1, "answer": "giga"}], xlsx_path, "gigachat")
            write_model_run_json(qwen_path, [{"question_id": 1, "answer": "qwen"}], xlsx_path, "qwen")
            write_xlsx(
                xlsx_path,
                {"results": pd.DataFrame([{
                    "json_file_gigachat": str(giga_path),
                    "json_file_qwen": str(qwen_path),
                }])},
            )

            runs = read_model_run_tables(xlsx_path, ["gigachat", "qwen", "glm"])

            self.assertEqual(set(runs), {"gigachat", "qwen"})
            self.assertEqual(runs["gigachat"][0].loc[0, "answer"], "giga")
            self.assertEqual(runs["qwen"][0].loc[0, "answer"], "qwen")
