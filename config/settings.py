"""
config/settings.py — updated with Groq support
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    # ── LLM Provider ──────────────────────────────────────────
    llm_provider: str = Field(default="groq")

    # ── Groq (FREE) ───────────────────────────────────────────
    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="llama-3.3-70b-versatile")

    # ── OpenAI ────────────────────────────────────────────────
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4o-mini")
    openai_embedding_model: str = Field(default="text-embedding-3-small")

    # ── Anthropic ─────────────────────────────────────────────
    anthropic_api_key: str = Field(default="")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022")

    # ── Server ────────────────────────────────────────────────
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    allowed_origins: str = Field(default="http://localhost:8501")

    # ── RAG ───────────────────────────────────────────────────
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    top_k_retrieval: int = Field(default=5)
    llm_temperature: float = Field(default=0.3)
    max_tokens: int = Field(default=2048)
    memory_window_size: int = Field(default=10)

    # ── Paths ─────────────────────────────────────────────────
    vector_db_path: str = Field(default="./database/vector_db")
    upload_dir: str = Field(default="./uploads")
    log_dir: str = Field(default="./logs")
    chat_db_path: str = Field(default="./database/chat_history.db")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    def ensure_dirs(self):
        for path in [self.vector_db_path, self.upload_dir, self.log_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
        Path(self.chat_db_path).parent.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s