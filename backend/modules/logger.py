"""Centralized logging for the SfM Platform.

Logs are written to:
  - Platform/log/app.log      — all application-level events
  - Platform/log/requests.log  — HTTP request/response summary
  - Platform/log/errors.log    — errors and exceptions only
  - stdout (console)           — same as app.log, for dev convenience

Usage:
    from modules.logger import get_logger
    logger = get_logger(__name__)
    logger.info("something happened")
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

LOG_DIR = Path(__file__).parent.parent.parent / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MAX_BYTES = 5 * 1024 * 1024   # 5 MB per file
BACKUP_COUNT = 3               # keep 3 rotated files

# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

DETAIL_FMT = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
BRIEF_FMT = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Root logger setup (called once)
# ---------------------------------------------------------------------------

_initialized = False


def _init_logging() -> None:
    global _initialized
    if _initialized:
        return
    _initialized = True

    root = logging.getLogger("platform")
    root.setLevel(logging.DEBUG)

    # --- App log (all levels) ---
    app_handler = RotatingFileHandler(
        LOG_DIR / "app.log", maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    app_handler.setLevel(logging.DEBUG)
    app_handler.setFormatter(DETAIL_FMT)
    root.addHandler(app_handler)

    # --- Error log (WARNING+) ---
    err_handler = RotatingFileHandler(
        LOG_DIR / "errors.log", maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    err_handler.setLevel(logging.WARNING)
    err_handler.setFormatter(DETAIL_FMT)
    root.addHandler(err_handler)

    # --- Console (INFO+) ---
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(DETAIL_FMT)
    root.addHandler(console)

    # --- Request log (separate logger) ---
    req_logger = logging.getLogger("platform.requests")
    req_logger.setLevel(logging.INFO)
    req_logger.propagate = False  # don't duplicate in app log
    req_handler = RotatingFileHandler(
        LOG_DIR / "requests.log", maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    req_handler.setLevel(logging.INFO)
    req_handler.setFormatter(BRIEF_FMT)
    req_logger.addHandler(req_handler)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Get a logger under the 'platform' hierarchy.

    Typical usage:
        logger = get_logger(__name__)
    """
    _init_logging()
    # Strip leading "modules." or absolute prefix for cleaner names
    short = name.replace("modules.", "").replace("Platform.backend.", "")
    return logging.getLogger(f"platform.{short}")


def request_log(method: str, path: str, status: int, duration_ms: float) -> None:
    """Log an HTTP request summary (to requests.log only)."""
    _init_logging()
    logger = logging.getLogger("platform.requests")
    logger.info("%s %s → %d  (%.1fms)", method, path, status, duration_ms)
