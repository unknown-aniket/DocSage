"""
rag/vector_store.py
-------------------
FAISS vector store manager with persistent local storage.

- Index is saved to disk after every batch ingest
- Supports per-namespace (per-user) indexes
- Thread-safe for concurrent requests
"""

from __future__ import annotations
import threading
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config.settings import get_settings
from rag.embeddings import get_embeddings
from utils.logger import get_logger

log = get_logger(__name__)


class VectorStoreManager:
    """
    Manages FAISS vector stores with:
    - Lazy loading from disk
    - Thread-safe upsert
    - Namespace support (one index per user_id)
    """

    def __init__(self):
        self._settings = get_settings()
        self._base_path = Path(self._settings.vector_db_path)
        self._stores: dict[str, FAISS] = {}
        self._lock = threading.Lock()

    # ── Public API ───────────────────────────────────────────

    def add_documents(
        self, documents: List[Document], namespace: str = "global"
    ) -> int:
        """Embed and add documents. Returns number of chunks added."""
        with self._lock:
            store = self._get_or_create(namespace)
            store.add_documents(documents)
            self._save(namespace, store)
            log.info("docs_added", namespace=namespace, count=len(documents))
            return len(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        namespace: str = "global",
    ) -> List[Document]:
        """Return top-k most similar document chunks for a query."""
        store = self._get_or_create(namespace)
        if store is None:
            return []
        results = store.similarity_search(query, k=k)
        log.info("retrieval", namespace=namespace, query_len=len(query), k=len(results))
        return results

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 5,
        namespace: str = "global",
    ) -> List[tuple[Document, float]]:
        """Return top-k results with similarity scores (lower = more similar for L2)."""
        store = self._get_or_create(namespace)
        if store is None:
            return []
        return store.similarity_search_with_score(query, k=k)

    def namespace_exists(self, namespace: str) -> bool:
        """Check if a vector index exists for the given namespace."""
        if namespace in self._stores:
            return True
        index_path = self._index_path(namespace)
        return index_path.exists()

    def document_count(self, namespace: str = "global") -> int:
        """Approximate number of vectors in namespace."""
        store = self._get_or_create(namespace)
        if store is None:
            return 0
        return store.index.ntotal

    # ── Internal ─────────────────────────────────────────────

    def _index_path(self, namespace: str) -> Path:
        return self._base_path / namespace

    def _get_or_create(self, namespace: str) -> Optional[FAISS]:
        if namespace in self._stores:
            return self._stores[namespace]

        path = self._index_path(namespace)
        if path.exists():
            log.info("loading_index", namespace=namespace)
            store = FAISS.load_local(
                str(path),
                get_embeddings(),
                allow_dangerous_deserialization=True,
            )
            self._stores[namespace] = store
            return store

        return None

    def _save(self, namespace: str, store: FAISS):
        path = self._index_path(namespace)
        path.mkdir(parents=True, exist_ok=True)
        store.save_local(str(path))
        self._stores[namespace] = store

    def _create_new(self, namespace: str, documents: List[Document]) -> FAISS:
        store = FAISS.from_documents(documents, get_embeddings())
        self._save(namespace, store)
        return store

    # Override add_documents to handle creation of new index
    def add_documents(
        self, documents: List[Document], namespace: str = "global"
    ) -> int:
        with self._lock:
            existing = self._get_or_create(namespace)
            if existing is None:
                store = self._create_new(namespace, documents)
            else:
                existing.add_documents(documents)
                self._save(namespace, existing)
            log.info("docs_added", namespace=namespace, count=len(documents))
            return len(documents)


# Module-level singleton
_vector_store: VectorStoreManager | None = None


def get_vector_store() -> VectorStoreManager:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreManager()
    return _vector_store
