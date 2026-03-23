"""
app/chat.py
-----------
FastAPI router: POST /chat (JSON) and GET /chat/stream (SSE)
Implements the full RAG pipeline per request:
  Memory → Retrieval → Prompt → LLM → Memory → Response
"""

from __future__ import annotations
import time
import json
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config.settings import get_settings
from rag.retriever import Retriever
from rag.generator import get_generator
from memory.manager import get_session_registry
from utils.helpers import generate_session_id
from utils.logger import get_logger, get_audit_logger

router = APIRouter()
log = get_logger(__name__)
settings = get_settings()


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(default_factory=generate_session_id)
    user_id: str = Field(default="anonymous")
    stream: bool = Field(default=True)
    top_k: int = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[dict]
    latency_ms: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    """Non-streaming chat endpoint. Returns complete response as JSON."""
    start = time.time()

    # 1. Memory
    registry = get_session_registry()
    memory = registry.get_or_create(req.session_id, req.user_id)
    history = memory.get_context_for_prompt()

    # 2. Retrieval
    ns = f"user_{req.user_id}" if req.user_id != "anonymous" else "global"
    retriever = Retriever()
    chunks = retriever.retrieve(req.query, namespace=ns, k=req.top_k)
    context = retriever.format_context(chunks)

    # 3. Generate
    generator = get_generator()
    answer = await generator.generate(req.query, context, history)

    # 4. Persist turns
    memory.record_user_turn(req.query)
    memory.record_assistant_turn(answer)

    latency = round((time.time() - start) * 1000, 2)
    sources = [c.to_dict() for c in chunks]

    # 5. Audit log
    get_audit_logger().log(
        session_id=req.session_id,
        user_id=req.user_id,
        query=req.query,
        answer=answer,
        sources=sources,
        latency_ms=latency,
        model=settings.openai_model,
    )

    return ChatResponse(
        answer=answer,
        session_id=req.session_id,
        sources=sources,
        latency_ms=latency,
    )


@router.post("/chat/stream", tags=["Chat"])
async def chat_stream(req: ChatRequest):
    """
    Streaming chat endpoint (Server-Sent Events).
    Yields tokens as they arrive from the LLM.
    Final event contains JSON with sources and session metadata.
    """
    registry = get_session_registry()
    memory = registry.get_or_create(req.session_id, req.user_id)
    history = memory.get_context_for_prompt()

    ns = f"user_{req.user_id}" if req.user_id != "anonymous" else "global"
    retriever = Retriever()
    chunks = retriever.retrieve(req.query, namespace=ns, k=req.top_k)
    context = retriever.format_context(chunks)

    # Record user turn before streaming begins
    memory.record_user_turn(req.query)

    async def event_stream() -> AsyncGenerator[str, None]:
        start = time.time()
        full_answer = ""
        generator = get_generator()

        try:
            async for token in generator.stream(req.query, context, history, chunks):
                full_answer += token
                # SSE format: "data: <token>\n\n"
                yield f"data: {json.dumps({'token': token})}\n\n"

            # Persist assistant turn
            memory.record_assistant_turn(full_answer)

            latency = round((time.time() - start) * 1000, 2)
            sources = [c.to_dict() for c in chunks]

            # Final metadata event
            yield f"data: {json.dumps({'done': True, 'session_id': req.session_id, 'sources': sources, 'latency_ms': latency})}\n\n"

            get_audit_logger().log(
                session_id=req.session_id,
                user_id=req.user_id,
                query=req.query,
                answer=full_answer,
                sources=sources,
                latency_ms=latency,
                model=settings.openai_model,
            )

        except Exception as e:
            log.error("stream_error", error=str(e))
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history/{session_id}", tags=["Chat"])
async def get_history(session_id: str, user_id: str = "anonymous"):
    """Return full conversation history for a session."""
    registry = get_session_registry()
    memory = registry.get_or_create(session_id, user_id)
    return {"session_id": session_id, "messages": memory.get_full_history()}


@router.delete("/session/{session_id}", tags=["Chat"])
async def delete_session(session_id: str):
    """Delete a session and all its messages."""
    registry = get_session_registry()
    registry.delete(session_id)
    return {"status": "deleted", "session_id": session_id}
