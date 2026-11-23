"""
Logging configuration for the pipeline.

Provides:
- `configure_logging()` to set up console + rotating file handlers once.
- `get_agent_logger(name)` for a namespaced logger per agent, e.g. "pipeline.Research".
- `log_timing` decorator/context manager to log start/completion + elapsed time
  in the exact style requested:
      [Research] Started
      [Research] Completed (2.31s)
"""
from __future__ import annotations

import functools
import logging
import os
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from typing import Callable, Iterator

from app.config import settings

_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    os.makedirs(settings.log_dir, exist_ok=True)
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    root = logging.getLogger("pipeline")
    root.setLevel(level)
    root.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        os.path.join(settings.log_dir, "pipeline.log"),
        maxBytes=5_000_000,
        backupCount=3,
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_agent_logger(agent_name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"pipeline.{agent_name}")


@contextmanager
def log_timing(agent_name: str, action: str = "") -> Iterator[Callable[[str], None]]:
    """
    Context manager that logs "[Agent] Started" / "[Agent] Completed (Xs)".

    Usage:
        with log_timing("Research") as note:
            ...
            note("17/20 claims supported")  # optional extra detail line
    """
    logger = get_agent_logger(agent_name)
    label = f"[{agent_name}]" + (f" {action}" if action else "")
    logger.info(f"{label} Started")
    start = time.perf_counter()

    def note(msg: str) -> None:
        logger.info(f"[{agent_name}] {msg}")

    try:
        yield note
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        logger.error(f"[{agent_name}] Failed after {elapsed:.2f}s: {exc}")
        raise
    else:
        elapsed = time.perf_counter() - start
        logger.info(f"[{agent_name}] Completed ({elapsed:.2f}s)")


def timed_agent(agent_name: str):
    """Decorator version of log_timing for simple sync methods."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with log_timing(agent_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator
