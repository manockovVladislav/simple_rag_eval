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

from rag_eval import MetricsCalculator


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate metrics for saved RAG run xlsx.")
    parser.add_argument("--config", default="config.py", help="Path to Python config file.")
    parser.add_argument("--run-file", help="Path to run xlsx. If omitted, latest run is used.")
    args = parser.parse_args()

    calculator = MetricsCalculator.from_config(args.config)
    output = calculator.evaluate(args.run_file) if args.run_file else calculator.evaluate_latest()
    print(output)


if __name__ == "__main__":
    main()
