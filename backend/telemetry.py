"""
Telemetry data capture for the Text-to-CAD Critic Loop.

Logs every Machinist attempt as a structured JSONL record for future
fine-tuning and failure-mode analysis. Each record captures the full
context of a code generation attempt: the part being built, the domain
classification, which attempt number it was, the generated code, any
error traceback, and whether the attempt succeeded.
"""

import json
import logging
import pathlib
import time
from logging.handlers import RotatingFileHandler
from typing import Optional

_LOG_DIR = pathlib.Path(__file__).parent / "telemetry_logs"
_LOG_FILE = _LOG_DIR / "critic_loop.jsonl"

_MAX_BYTES = 50 * 1024 * 1024   # 50 MB per file
_BACKUP_COUNT = 5                # keep 5 rotated files
_MAX_CODE_LEN = 10_000           # truncate generated_code per record
_MAX_ERROR_LEN = 2_000           # truncate traceback per record

_logger = logging.getLogger("mirum.telemetry")
_handler_initialized = False


def _ensure_handler() -> None:
    global _handler_initialized
    if _handler_initialized:
        return
    _LOG_DIR.mkdir(exist_ok=True)
    handler = RotatingFileHandler(
        str(_LOG_FILE),
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
    _handler_initialized = True


def log_attempt(
    part_id: str,
    domain: str,
    attempt: int,
    code: str,
    error: Optional[str],
    success: bool,
) -> None:
    """Append a single critic-loop attempt record to the JSONL log."""
    _ensure_handler()

    record = {
        "timestamp": time.time(),
        "part_id": part_id,
        "domain_classification": domain,
        "attempt_number": attempt,
        "generated_code": code[:_MAX_CODE_LEN],
        "traceback_error": error[:_MAX_ERROR_LEN] if error else None,
        "success_status": success,
    }

    _logger.info(json.dumps(record))
