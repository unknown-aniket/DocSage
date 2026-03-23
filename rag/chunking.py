"""
rag/chunking.py
---------------
Text chunking strategy.

Strategy: RecursiveCharacterTextSplitter with semantic-aware separators.
- chunk_size=1000 chars (≈ 250 tokens) — balanced for context and retrieval
- chunk_overlap=200 — prevents losing context at boundaries
- Separators: paragraph > sentence > word — preserves semantic units
"""

from __future__ import annotations
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_settings
from utils.logger import get_logger

log = get_logger(__name__)


class TextChunker:
    """Splits Documents into semantically coherent chunks."""

    def __init__(self):
        settings = get_settings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        """Split a list of Documents into chunks with enriched metadata."""
        chunks = self._splitter.split_documents(documents)

        # Enrich metadata with chunk index per source document
        source_counters: dict[str, int] = {}
        for chunk in chunks:
            src = chunk.metadata.get("source", "unknown")
            idx = source_counters.get(src, 0)
            chunk.metadata["chunk_index"] = idx
            source_counters[src] = idx + 1

        # Back-fill total_chunks per source
        for chunk in chunks:
            src = chunk.metadata.get("source", "unknown")
            chunk.metadata["total_chunks"] = source_counters[src]

        log.info(
            "chunking_complete",
            input_docs=len(documents),
            output_chunks=len(chunks),
        )
        return chunks
