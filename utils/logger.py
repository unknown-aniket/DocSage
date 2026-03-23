"""
utils/logger.py
---------------
Structured JSON logging with query/response audit trail.
All LLM queries are written to logs/query_log.jsonl for eval and replay.
"""

import json
import logging
import structlog
from datetime import datetime, UTC
from pathlib import Path
from config.settings import get_settings


def setup_logging():
    """Configure structlog for the application."""
    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(message)s",
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    """Get a named structured logger."""
    return structlog.get_logger(name)


class QueryAuditLogger:
    """Writes query/response pairs to JSONL for evaluation and replay."""

    def __init__(self):
        settings = get_settings()
        self.log_path = Path(settings.log_dir) / "query_log.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        session_id: str,
        user_id: str,
        query: str,
        answer: str,
        sources: list[dict],
        latency_ms: float,
        model: str,
    ):
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": session_id,
            "user_id": user_id,
            "query": query,
            "answer": answer,
            "sources": sources,
            "latency_ms": latency_ms,
            "model": model,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# Module-level singletons
_audit_logger: QueryAuditLogger | None = None


def get_audit_logger() -> QueryAuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = QueryAuditLogger()
    return _audit_logger
