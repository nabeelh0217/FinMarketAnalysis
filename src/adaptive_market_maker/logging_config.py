"""Structured, single-line log records (key=value) to stderr."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("ts=%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
