from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_PATH):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from rag_eval import EvaluationRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run golden questions through configured RAG pipeline.")
    parser.add_argument("--config", default="config.py", help="Path to Python config file.")
    args = parser.parse_args()

    output = EvaluationRunner.from_config(args.config).run()
    print(output)


if __name__ == "__main__":
    main()
