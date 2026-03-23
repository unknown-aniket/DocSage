"""
main.py
-------
FastAPI application entry point.
Mounts all routers and configures CORS, middleware, and startup tasks.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import get_settings
from app.chat import router as chat_router
from app.upload import router as upload_router
from utils.logger import setup_logging, get_logger

setup_logging()
log = get_logger(__name__)
settings = get_settings()

app = FastAPI(
    title="RAG Chatbot API",
    description="Context-Aware AI Chatbot with Memory using RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(chat_router, prefix="/api/v1")
app.include_router(upload_router, prefix="/api/v1")


@app.on_event("startup")
async def on_startup():
    log.info("server_startup", host=settings.app_host, port=settings.app_port)


@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
