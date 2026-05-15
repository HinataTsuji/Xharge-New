"""Logging setup helpers."""
from __future__ import annotations

import logging


def configure_logging() -> None:
    """Configure a concise production-friendly log format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
