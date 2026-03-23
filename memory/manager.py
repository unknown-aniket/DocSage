"""
memory/manager.py
-----------------
MemoryManager: unified interface combining short-term + long-term memory.
One instance per session, keyed by session_id.
"""

from __future__ import annotations
from typing import Dict

from memory.short_term import ShortTermMemory
from memory.long_term import get_long_term_memory
from utils.logger import get_logger

log = get_logger(__name__)


class MemoryManager:
    """
    Manages conversation memory for a single session.
    - Short-term: in-process sliding window (fast, for prompt injection)
    - Long-term: SQLite persistence (durable, for history recall)
    """

    def __init__(self, session_id: str, user_id: str):
        self.session_id = session_id
        self.user_id = user_id
        self._short = ShortTermMemory()
        self._long = get_long_term_memory()
        # Ensure session exists in DB
        self._long.create_session(session_id, user_id)
        # Replay recent history into short-term buffer
        self._reload_from_db()

    def _reload_from_db(self, limit: int = 20):
        """Warm-start short-term buffer from persisted history."""
        messages = self._long.get_session_messages(self.session_id, limit=limit)
        for msg in messages:
            if msg["role"] == "user":
                self._short.add_user(msg["content"])
            elif msg["role"] == "assistant":
                self._short.add_assistant(msg["content"])

    # ── Write ─────────────────────────────────────────────────

    def record_user_turn(self, content: str):
        self._short.add_user(content)
        self._long.save_message(self.session_id, "user", content)

    def record_assistant_turn(self, content: str):
        self._short.add_assistant(content)
        self._long.save_message(self.session_id, "assistant", content)

    # ── Read ──────────────────────────────────────────────────

    def get_context_for_prompt(self) -> str:
        """Return formatted history string to inject into the system prompt."""
        return self._short.get_history_text()

    def get_full_history(self) -> list[dict]:
        """Return all persisted messages for this session."""
        return self._long.get_session_messages(self.session_id)

    def clear_short_term(self):
        self._short.clear()


class SessionRegistry:
    """
    Global registry of active MemoryManager instances.
    Keyed by session_id. Creates new managers on demand.
    """

    def __init__(self):
        self._sessions: Dict[str, MemoryManager] = {}

    def get_or_create(self, session_id: str, user_id: str) -> MemoryManager:
        if session_id not in self._sessions:
            log.info("new_session", session_id=session_id, user_id=user_id)
            self._sessions[session_id] = MemoryManager(session_id, user_id)
        return self._sessions[session_id]

    def delete(self, session_id: str):
        self._sessions.pop(session_id, None)
        get_long_term_memory().delete_session(session_id)


_registry: SessionRegistry | None = None


def get_session_registry() -> SessionRegistry:
    global _registry
    if _registry is None:
        _registry = SessionRegistry()
    return _registry
