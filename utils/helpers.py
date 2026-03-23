"""
utils/helpers.py
----------------
Shared helper utilities used across the application.
"""

import hashlib
import time
import uuid
from pathlib import Path
from typing import Any


def generate_session_id() -> str:
    """Generate a new unique session ID."""
    return str(uuid.uuid4())


def generate_user_id() -> str:
    """Generate a stable anonymous user ID."""
    return str(uuid.uuid4())


def file_checksum(path: str | Path) -> str:
    """MD5 checksum of a file for deduplication."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitize_filename(filename: str) -> str:
    """Remove path traversal characters from uploaded filenames."""
    return Path(filename).name


def truncate_text(text: str, max_chars: int = 300) -> str:
    """Truncate text with ellipsis for display."""
    return text[:max_chars] + "…" if len(text) > max_chars else text


def milliseconds_since(start: float) -> float:
    """Elapsed milliseconds since a time.time() snapshot."""
    return round((time.time() - start) * 1000, 2)


def chunk_metadata(
    source: str,
    page: int | None,
    chunk_index: int,
    total_chunks: int,
) -> dict[str, Any]:
    """Build a standardised metadata dict for a document chunk."""
    return {
        "source": source,
        "page": page,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
    }
