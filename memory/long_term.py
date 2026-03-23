"""
memory/long_term.py
-------------------
SQLite-backed long-term memory for persistent conversation history.
Schema:
  sessions(id, user_id, created_at, updated_at)
  messages(id, session_id, role, content, timestamp)
"""

from __future__ import annotations
import sqlite3
from datetime import datetime, UTC
from pathlib import Path
from typing import List

from config.settings import get_settings
from utils.logger import get_logger

log = get_logger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    timestamp   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user    ON sessions(user_id);
"""


class LongTermMemory:
    """Persistent conversation history backed by SQLite."""

    def __init__(self):
        settings = get_settings()
        self._db_path = settings.chat_db_path
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(DDL)

    # ── Session management ────────────────────────────────────

    def create_session(self, session_id: str, user_id: str):
        now = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions(id, user_id, created_at, updated_at) VALUES (?,?,?,?)",
                (session_id, user_id, now, now),
            )
        log.info("session_created", session_id=session_id, user_id=user_id)

    def get_user_sessions(self, user_id: str) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE user_id=? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Message storage ───────────────────────────────────────

    def save_message(self, session_id: str, role: str, content: str):
        now = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO messages(session_id, role, content, timestamp) VALUES (?,?,?,?)",
                (session_id, role, content, now),
            )
            conn.execute(
                "UPDATE sessions SET updated_at=? WHERE id=?",
                (now, session_id),
            )

    def get_session_messages(
        self, session_id: str, limit: int = 50
    ) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content, timestamp FROM messages "
                "WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def delete_session(self, session_id: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        log.info("session_deleted", session_id=session_id)


_ltm: LongTermMemory | None = None


def get_long_term_memory() -> LongTermMemory:
    global _ltm
    if _ltm is None:
        _ltm = LongTermMemory()
    return _ltm
