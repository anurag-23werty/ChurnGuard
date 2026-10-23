"""Single logging setup used by every CLI entry point."""

from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def setup_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=(level or os.environ.get("CHURNGUARD_LOG_LEVEL", "INFO")).upper(),
        format=_FORMAT,
        datefmt="%H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
