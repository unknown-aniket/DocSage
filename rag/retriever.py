"""
rag/retriever.py
----------------
Semantic retriever with optional MMR (Maximal Marginal Relevance)
to improve diversity of retrieved chunks.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document

from config.settings import get_settings
from rag.vector_store import get_vector_store
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with display metadata."""
    content: str
    source: str
    page: int | None
    chunk_index: int
    score: float  # cosine distance (lower = more similar)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "score": round(float(self.score), 4),
        }


class Retriever:
    """
    Retrieves relevant document chunks for a query.
    Supports global and per-user namespaced indexes.
    """

    def __init__(self):
        self._settings = get_settings()
        self._store = get_vector_store()

    def retrieve(
        self,
        query: str,
        namespace: str = "global",
        k: int | None = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve top-k chunks most relevant to `query`.
        Falls back to global namespace if user namespace is empty.
        """
        k = k or self._settings.top_k_retrieval
        results_with_scores: list[tuple[Document, float]] = []

        # Try user namespace first
        if namespace != "global" and self._store.namespace_exists(namespace):
            results_with_scores = self._store.similarity_search_with_scores(
                query, k=k, namespace=namespace
            )

        # Also search global namespace and merge
        if self._store.namespace_exists("global"):
            global_results = self._store.similarity_search_with_scores(
                query, k=k, namespace="global"
            )
            results_with_scores.extend(global_results)

        # De-duplicate by content and sort by score (ascending = more similar)
        seen: set[str] = set()
        unique: list[tuple[Document, float]] = []
        for doc, score in results_with_scores:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                unique.append((doc, score))

        unique.sort(key=lambda x: x[1])
        top = unique[:k]

        chunks = [
            RetrievedChunk(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page"),
                chunk_index=doc.metadata.get("chunk_index", 0),
                score=score,
            )
            for doc, score in top
        ]

        log.info(
            "retrieval_complete",
            query=query[:60],
            namespace=namespace,
            found=len(chunks),
        )
        return chunks

    def format_context(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks into a context block for the LLM prompt."""
        if not chunks:
            return "No relevant documents found in the knowledge base."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            page_str = f", page {chunk.page}" if chunk.page else ""
            header = f"[Source {i}: {chunk.source}{page_str}]"
            parts.append(f"{header}\n{chunk.content}")

        return "\n\n---\n\n".join(parts)