"""
rag/embeddings.py
-----------------
Embedding model factory.
Supports OpenAI embeddings (default) and a local HuggingFace fallback.

Model choice: text-embedding-3-small
  - 1536 dimensions, cheapest OpenAI model
  - 62.3% MTEB benchmark — strong for retrieval tasks
  - ~$0.02 / 1M tokens
"""

from __future__ import annotations
from functools import lru_cache

from config.settings import get_settings
from utils.logger import get_logger

log = get_logger(__name__)


@lru_cache()
def get_embeddings():
    """Return a cached embedding model instance."""
    settings = get_settings()

    if settings.llm_provider == "openai" or settings.openai_api_key:
        from langchain_openai import OpenAIEmbeddings
        log.info("embedding_model", provider="openai", model=settings.openai_embedding_model)
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
        )

    # Fallback: local HuggingFace sentence-transformers (no API key needed)
    log.info("embedding_model", provider="huggingface", model="all-MiniLM-L6-v2")
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except ImportError:
        raise RuntimeError(
            "No embedding provider available. "
            "Set OPENAI_API_KEY or install sentence-transformers."
        )
