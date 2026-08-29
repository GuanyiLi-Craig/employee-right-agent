"""Logging setup.

Two formats: human-readable for a terminal in front of a room, and JSON lines
for a container log pipeline (``RIGHTS_LOG_JSON=true``).  A ``request_id`` is bound
to the calling thread so every line emitted while serving a request can be
correlated with its metrics row and its trace.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from contextlib import contextmanager
from typing import Any, Iterator

_local = threading.local()

_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"message", "asctime", "taskName"}


def request_id() -> str:
    """The request id bound to this thread, or ``""``."""
    return getattr(_local, "request_id", "")


@contextmanager
def bind_request(request_id_value: str) -> Iterator[None]:
    """Bind ``request_id`` for the duration of the block."""
    previous = getattr(_local, "request_id", "")
    _local.request_id = request_id_value
    try:
        yield
    finally:
        _local.request_id = previous


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if getattr(record, "request_id", ""):
            payload["request_id"] = record.request_id
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key != "request_id":
                payload[key] = value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rid = getattr(record, "request_id", "")
        prefix = f"[{rid}] " if rid else ""
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} {record.name:<22} {prefix}{record.getMessage()}"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


_configured = False
_configure_lock = threading.Lock()


def configure_logging(level: str | None = None, *, json_format: bool | None = None) -> None:
    """Install handlers on the ``rights_agent`` logger.  Idempotent and thread-safe."""
    global _configured
    with _configure_lock:
        if level is None or json_format is None:
            # Imported lazily: config imports nothing from here, but reading
            # settings during a failed bootstrap should not mask the real error.
            from rights_agent.config import settings

            try:
                resolved = settings()
                level = level or resolved.log_level
                json_format = resolved.log_json if json_format is None else json_format
            except Exception:  # pragma: no cover - fall back to env defaults
                level = level or os.environ.get("RIGHTS_LOG_LEVEL", "INFO").upper()
                json_format = bool(json_format)

        root = logging.getLogger("rights_agent")
        root.setLevel(level)
        root.propagate = False
        for handler in list(root.handlers):
            root.removeHandler(handler)
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter() if json_format else _TextFormatter())
        handler.addFilter(_RequestIdFilter())
        root.addHandler(handler)

        # Chroma and its dependencies are chatty at INFO and would drown the
        # demo's own output.
        for noisy in ("chromadb", "httpx", "urllib3", "opentelemetry", "phoenix"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        # The OTLP exporter retries loudly when the collector is unreachable.
        # Killing Phoenix mid-demo is expected behaviour here and must be a
        # non-event, so its retry chatter is silenced; whether spans are being
        # exported is reported by ``telemetry.telemetry_status()`` and shown on
        # the dashboard instead.
        logging.getLogger("opentelemetry.exporter").setLevel(logging.CRITICAL)
        logging.getLogger("opentelemetry.sdk.trace.export").setLevel(logging.CRITICAL)
        _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured child of the ``rights_agent`` logger."""
    if not _configured:
        configure_logging()
    return logging.getLogger(name if name.startswith("rights_agent") else f"rights_agent.{name}")
