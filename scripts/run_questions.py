from __future__ import annotations

import argparse

from rag_eval import EvaluationRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run golden questions through configured RAG pipeline.")
    parser.add_argument("--config", default="config.py", help="Path to Python config file.")
    args = parser.parse_args()

    output = EvaluationRunner.from_config(args.config).run()
    print(output)


if __name__ == "__main__":
    main()
