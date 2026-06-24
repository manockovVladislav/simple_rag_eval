from __future__ import annotations

import logging
from pathlib import Path


def setup_file_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("rag_eval")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    target = str(log_file)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == target:
            return logger

    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    return logger
