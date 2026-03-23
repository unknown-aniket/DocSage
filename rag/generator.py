"""
rag/generator.py  (Groq + OpenAI multi-provider)
"""
from __future__ import annotations
from typing import AsyncGenerator, List

from config.settings import get_settings
from rag.retriever import RetrievedChunk
from utils.logger import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = """You are a helpful, accurate AI assistant with access to a \
knowledge base. Your job is to answer questions using ONLY the provided context.

## Instructions
1. Base your answer strictly on the CONTEXT section provided below.
2. If the context does not contain enough information, say so clearly.
3. At the end of every answer, cite the sources you used as:
   **Sources:** [filename, page X] — one per line.
4. Never hallucinate facts not present in the context.
5. Be concise but complete. Use markdown for formatting when helpful.
6. If the user asks a follow-up, use the conversation history to maintain continuity.

## Context
{context}

## Conversation History
{history}
"""


def build_messages(query: str, context: str, history: str) -> list[dict]:
    system_content = SYSTEM_PROMPT.format(context=context, history=history)
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
    ]


class LLMGenerator:
    def __init__(self):
        self._settings = get_settings()
        self._provider, self._client = self._build_client()

    def _build_client(self):
        s = self._settings

        if s.llm_provider == "groq" or s.groq_api_key:
            from groq import AsyncGroq
            log.info("llm_provider", provider="groq", model=s.groq_model)
            return "groq", AsyncGroq(api_key=s.groq_api_key)

        if s.llm_provider == "anthropic" or s.anthropic_api_key:
            import anthropic
            log.info("llm_provider", provider="anthropic")
            return "anthropic", anthropic.AsyncAnthropic(api_key=s.anthropic_api_key)

        import openai
        log.info("llm_provider", provider="openai", model=s.openai_model)
        return "openai", openai.AsyncOpenAI(api_key=s.openai_api_key)

    async def stream(
        self,
        query: str,
        context: str,
        history: str,
        chunks: List[RetrievedChunk],
    ) -> AsyncGenerator[str, None]:
        messages = build_messages(query, context, history)
        s = self._settings

        if self._provider == "groq":
            async for token in self._stream_groq(messages, s):
                yield token
        elif self._provider == "anthropic":
            async for token in self._stream_anthropic(messages, s):
                yield token
        else:
            async for token in self._stream_openai(messages, s):
                yield token

        # Source citations
        if chunks:
            yield "\n\n---\n**📚 Sources:**\n"
            seen: set[str] = set()
            for chunk in chunks:
                key = f"{chunk.source}-{chunk.page}"
                if key not in seen:
                    page_str = f", page {chunk.page}" if chunk.page else ""
                    yield f"- `{chunk.source}{page_str}`\n"
                    seen.add(key)

    async def _stream_groq(self, messages, settings):
        stream = await self._client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def _stream_openai(self, messages, settings):
        stream = await self._client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def _stream_anthropic(self, messages, settings):
        system_msg = messages[0]["content"]
        user_msgs = messages[1:]
        async with self._client.messages.stream(
            model=settings.anthropic_model,
            max_tokens=settings.max_tokens,
            system=system_msg,
            messages=user_msgs,
            temperature=settings.llm_temperature,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def generate(self, query: str, context: str, history: str) -> str:
        full = ""
        async for token in self.stream(query, context, history, []):
            full += token
        return full


_generator: LLMGenerator | None = None


def get_generator() -> LLMGenerator:
    global _generator
    if _generator is None:
        _generator = LLMGenerator()
    return _generator