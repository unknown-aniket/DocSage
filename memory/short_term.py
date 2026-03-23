"""
memory/short_term.py
--------------------
In-process short-term (session) memory.
Uses a sliding window of the last N turns to keep the context
within the LLM's context window.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Deque

from config.settings import get_settings


@dataclass
class Turn:
    """A single conversational exchange."""
    role: str          # "user" | "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}


class ShortTermMemory:
    """
    Sliding-window conversation buffer per session.
    Stores the last `window_size` turns (pairs of user+assistant).
    """

    def __init__(self, window_size: int | None = None):
        settings = get_settings()
        self._window = window_size or settings.memory_window_size
        # deque automatically drops oldest entries beyond maxlen
        self._turns: Deque[Turn] = deque(maxlen=self._window * 2)

    def add_user(self, content: str):
        self._turns.append(Turn(role="user", content=content))

    def add_assistant(self, content: str):
        self._turns.append(Turn(role="assistant", content=content))

    def get_history_text(self) -> str:
        """Format history as a readable string for prompt injection."""
        if not self._turns:
            return "No previous conversation."
        lines = []
        for turn in self._turns:
            prefix = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.content}")
        return "\n".join(lines)

    def get_turns(self) -> list[dict]:
        return [t.to_dict() for t in self._turns]

    def clear(self):
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
