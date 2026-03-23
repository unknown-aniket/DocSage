# 🧠 Context-Aware AI Chatbot with Memory using RAG

> A production-ready Retrieval-Augmented Generation (RAG) chatbot with persistent conversation memory, multi-user support, streaming responses, and source citations.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                  STREAMLIT FRONTEND (8501)               │
│  Chat UI │ Document Upload │ Source Citations │ History  │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP / SSE
                       ▼
┌─────────────────────────────────────────────────────────┐
│               FASTAPI GATEWAY (8000)                     │
│  POST /chat  │  POST /chat/stream  │  POST /upload       │
└──┬──────────────────────────────────────────────────────┘
   │
   ├──► MEMORY MANAGER
   │      ├── ShortTermMemory (sliding window, in-process)
   │      └── LongTermMemory  (SQLite persistence)
   │
   ├──► RETRIEVER
   │      ├── Query embedding (OpenAI text-embedding-3-small)
   │      ├── FAISS similarity search (top-k=5)
   │      └── Per-user + global namespace merge
   │
   └──► LLM GENERATOR
          ├── Prompt engineering (RAG system prompt)
          ├── Context injection
          └── Streaming (OpenAI / Anthropic)
```

---

## ✨ Features

| Feature | Details |
|---|---|
| **RAG Pipeline** | PDF / TXT / DOCX ingestion → chunking → embedding → FAISS |
| **Memory** | Short-term (sliding window) + Long-term (SQLite) per session |
| **Streaming** | Server-Sent Events, token-by-token rendering |
| **Multi-User** | Per-user vector namespaces + session isolation |
| **Source Citations** | Every answer cites retrieved document chunks |
| **Evaluation Logging** | JSONL audit trail for every query/response |
| **Docker** | Single `docker-compose up` deployment |

---

## 🗂️ Project Structure

```
rag_chatbot/
├── app/
│   ├── chat.py          # FastAPI chat endpoints (streaming + JSON)
│   ├── upload.py        # Document upload & indexing endpoint
│   └── frontend.py      # Streamlit UI
├── rag/
│   ├── ingestion.py     # PDF/TXT/DOCX document loader
│   ├── chunking.py      # RecursiveCharacterTextSplitter
│   ├── embeddings.py    # OpenAI / HuggingFace embedding factory
│   ├── vector_store.py  # FAISS manager (persist + namespace)
│   ├── retriever.py     # Semantic retrieval with score ranking
│   └── generator.py     # LLM streaming generator + prompt engineering
├── memory/
│   ├── short_term.py    # In-process sliding window buffer
│   ├── long_term.py     # SQLite-backed persistence
│   └── manager.py       # Unified MemoryManager + SessionRegistry
├── config/
│   └── settings.py      # Pydantic settings from .env
├── utils/
│   ├── logger.py        # Structlog + JSONL audit logger
│   └── helpers.py       # Shared utilities
├── database/
│   ├── vector_db/       # FAISS indexes (one per namespace)
│   └── chat_history.db  # SQLite conversation history
├── uploads/             # Uploaded documents (per-user)
├── logs/
│   └── query_log.jsonl  # Evaluation audit trail
├── main.py              # FastAPI application entry point
├── requirements.txt
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

---

## 🚀 Quick Start

### 1. Clone & Configure

```bash
git clone https://github.com/yourname/rag-chatbot.git
cd rag-chatbot

cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Backend API

```bash
python main.py
# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 4. Run the Streamlit Frontend

```bash
streamlit run app/frontend.py
# UI available at http://localhost:8501
```

---

## 🐳 Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# API: http://localhost:8000
# UI:  http://localhost:8501
```

---

## 📡 API Reference

### Upload a Document
```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@mydoc.pdf" \
  -F "user_id=alice"
```

### Chat (streaming)
```bash
curl -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings?",
    "session_id": "abc-123",
    "user_id": "alice",
    "top_k": 5
  }'
```

### Chat (JSON)
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the document", "user_id": "alice"}'
```

### Get Session History
```bash
curl http://localhost:8000/api/v1/history/abc-123?user_id=alice
```

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for OpenAI provider |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `5` | Retrieved chunks per query |
| `LLM_TEMPERATURE` | `0.3` | Generation temperature |
| `MEMORY_WINDOW_SIZE` | `10` | Conversation turns in short-term memory |

---

## 💡 Example Queries

After uploading a research paper:
- *"What methodology did the authors use?"*
- *"What were the key results?"* ← follow-up, uses memory
- *"How does this compare to prior work?"*
- *"Can you summarise section 3?"*

---

## 🔬 Evaluation

Query logs are written to `logs/query_log.jsonl`. Each record contains:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "session_id": "abc-123",
  "query": "What is the main conclusion?",
  "answer": "The authors conclude that...",
  "sources": [{"source": "paper.pdf", "page": 12, "score": 0.234}],
  "latency_ms": 1840,
  "model": "gpt-4o-mini"
}
```

---

## 🛠️ Tech Stack

- **LLM:** OpenAI GPT-4o-mini / Claude 3.5 Sonnet
- **Embeddings:** OpenAI text-embedding-3-small
- **Vector Store:** FAISS (local, persistent)
- **RAG Framework:** LangChain
- **API:** FastAPI + Server-Sent Events
- **Frontend:** Streamlit
- **Memory DB:** SQLite (via SQLAlchemy)
- **Containerisation:** Docker + Docker Compose

---

## 🔮 Future Improvements

- [ ] ChromaDB / Pinecone cloud vector store
- [ ] Re-ranking with Cohere Rerank or cross-encoders
- [ ] Hybrid search (BM25 + dense vectors)
- [ ] User authentication (JWT)
- [ ] Document versioning & deduplication
- [ ] Evaluation dashboard (RAGAS metrics)
- [ ] Multi-modal support (images in PDFs)
