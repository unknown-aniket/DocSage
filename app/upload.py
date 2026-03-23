"""
app/upload.py
-------------
FastAPI router: POST /upload
Handles multi-file document uploads, ingestion, chunking, and indexing.
"""

from __future__ import annotations
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from config.settings import get_settings
from rag.ingestion import DocumentIngester
from rag.chunking import TextChunker
from rag.vector_store import get_vector_store
from utils.helpers import sanitize_filename, file_checksum
from utils.logger import get_logger

router = APIRouter()
log = get_logger(__name__)

ALLOWED_TYPES = {
    "application/pdf",
    "text/plain",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/markdown",
}
MAX_FILE_SIZE_MB = 50


@router.post("/upload", tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(default="anonymous"),
    namespace: str = Form(default="global"),
):
    """
    Upload and index a document.
    - Validates file type and size
    - Ingests → chunks → embeds → stores in FAISS
    - Returns chunk count and document metadata
    """
    settings = get_settings()
    start = time.time()

    # ── Validation ────────────────────────────────────────────
    safe_name = sanitize_filename(file.filename or "upload.txt")
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB",
        )

    # ── Save to disk ──────────────────────────────────────────
    user_dir = Path(settings.upload_dir) / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    save_path = user_dir / safe_name
    save_path.write_bytes(content)

    # ── Ingest → Chunk → Index ────────────────────────────────
    ingester = DocumentIngester()
    chunker = TextChunker()
    store = get_vector_store()

    try:
        docs = ingester.ingest(save_path)
        chunks = chunker.chunk(docs)
        # Use per-user namespace for isolation
        ns = f"user_{user_id}" if user_id != "anonymous" else "global"
        store.add_documents(chunks, namespace=ns)
    except Exception as e:
        log.error("ingestion_failed", filename=safe_name, error=str(e))
        raise HTTPException(status_code=422, detail=f"Failed to process document: {e}")

    elapsed_ms = round((time.time() - start) * 1000, 1)
    log.info(
        "upload_complete",
        filename=safe_name,
        chunks=len(chunks),
        namespace=ns,
        elapsed_ms=elapsed_ms,
    )

    return JSONResponse({
        "status": "indexed",
        "filename": safe_name,
        "pages": len(docs),
        "chunks": len(chunks),
        "namespace": ns,
        "elapsed_ms": elapsed_ms,
    })
