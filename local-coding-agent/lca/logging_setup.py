"""Structured JSON-lines logging to ~/.lca/logs. Logs record what happened (tools, timings, errors),
never prompt or file contents, and every message passes through secret redaction."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from lca.security import redact_secrets


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            entry.update(fields)
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return redact_secrets(json.dumps(entry, default=str))


def setup_logging(log_dir: Path, level: str = "INFO") -> Path | None:
    """Configure the 'lca' logger. Returns the log file path, or None if it cannot be written."""
    logger = logging.getLogger("lca")
    logger.setLevel(level)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"lca-{datetime.now():%Y%m%d}.jsonl"
        handler = logging.FileHandler(path, encoding="utf-8")
    except OSError:
        logger.addHandler(logging.NullHandler())
        return None
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return path
