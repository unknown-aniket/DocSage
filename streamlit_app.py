"""
streamlit_app.py — DocSage Cloud Edition
Single-file deployment for Streamlit Community Cloud.
No separate FastAPI backend needed — runs everything in one process.
"""

import json
import uuid
import sqlite3
import threading
import time
from pathlib import Path
from collections import deque
from datetime import datetime, UTC

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocSage — Document Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stApp {
    background-color: #080c14 !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #c8d6e8 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton, [data-testid="stToolbar"] { display: none !important; }
.block-container { padding-top: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] {
    background-color: #060910 !important;
    border-right: 1px solid #0f1824 !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: #0a1828 !important;
    color: #5a9abf !important;
    border: 1px solid #0f1824 !important;
    font-size: 0.78rem !important;
    border-radius: 8px !important;
    width: 100% !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #0f2030 !important;
    color: #00d4ff !important;
}
[data-testid="stFileUploader"] {
    background: #0a1020 !important;
    border: 1px dashed #1a2a40 !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"] {
    background: #0a1020 !important;
    border: 1px solid #1a2a40 !important;
    border-radius: 16px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #e0eaf8 !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg,#00d4ff,#0080ff) !important;
    border-radius: 10px !important;
}
hr { border-color: #0f1824 !important; margin: 10px 0 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #1a2a40; border-radius: 4px; }
.stExpander { background: #060910 !important; border: 1px solid #0f1824 !important; border-radius: 10px !important; }
.stProgress > div > div { background: linear-gradient(90deg,#00d4ff,#0080ff) !important; }
</style>
""", unsafe_allow_html=True)


# ── Lazy imports (only load heavy libs when needed) ───────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings():
    """Load embedding model once and cache it."""
    groq_key = st.secrets.get("GROQ_API_KEY", "")
    openai_key = st.secrets.get("OPENAI_API_KEY", "")

    if openai_key:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_key)

    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def get_groq_client():
    from groq import Groq
    key = st.secrets.get("GROQ_API_KEY", "")
    if not key:
        st.error("⚠️ GROQ_API_KEY not set in Streamlit secrets!")
        st.stop()
    return Groq(api_key=key)


# ── Vector store (in-memory per session via st.cache_resource) ────────────────
def get_vector_store(namespace: str):
    """Get or create FAISS store for a namespace."""
    key = f"vs_{namespace}"
    if key not in st.session_state:
        st.session_state[key] = None
    return st.session_state[key]


def set_vector_store(namespace: str, store):
    st.session_state[f"vs_{namespace}"] = store


# ── Document processing ───────────────────────────────────────────────────────
def process_file(file_bytes: bytes, filename: str, namespace: str) -> tuple[int, str]:
    """Ingest → chunk → embed → store. Returns (chunk_count, error_or_empty)."""
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    suffix = Path(filename).suffix.lower()
    try:
        # Load
        if suffix == ".pdf":
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(file_bytes))
            docs = [
                Document(
                    page_content=p.extract_text() or "",
                    metadata={"source": filename, "page": i+1}
                )
                for i, p in enumerate(reader.pages)
                if (p.extract_text() or "").strip()
            ]
        elif suffix in (".txt", ".md"):
            docs = [Document(page_content=file_bytes.decode("utf-8", errors="replace"),
                             metadata={"source": filename, "page": None})]
        elif suffix == ".docx":
            from docx import Document as DocxDoc
            import io
            d = DocxDoc(io.BytesIO(file_bytes))
            text = "\n".join(p.text for p in d.paragraphs if p.text.strip())
            docs = [Document(page_content=text, metadata={"source": filename, "page": None})]
        else:
            return 0, f"Unsupported file type: {suffix}"

        # Chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        if not chunks:
            return 0, "No text could be extracted from this file."

        # Embed & store
        embeddings = load_embeddings()
        existing = get_vector_store(namespace)
        if existing is None:
            store = FAISS.from_documents(chunks, embeddings)
        else:
            existing.add_documents(chunks)
            store = existing
        set_vector_store(namespace, store)
        return len(chunks), ""

    except Exception as e:
        return 0, str(e)


def retrieve(query: str, namespace: str, k: int = 6) -> list[dict]:
    """Semantic search. Returns list of source dicts."""
    store = get_vector_store(namespace)
    if store is None:
        return []
    try:
        results = store.similarity_search_with_score(query, k=k)
        seen, chunks = set(), []
        for doc, score in results:
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                chunks.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page"),
                    "score": round(float(score), 4),
                })
        return chunks
    except:
        return []


# ── LLM generation ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are DocSage, a helpful AI document assistant.
Answer questions using ONLY the provided context below.
If the context doesn't contain enough information, say so clearly.
At the end cite sources as: **Sources:** [filename, page X]
Never hallucinate facts not present in the context.

## Context
{context}

## Conversation History
{history}
"""

def format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant documents found in the knowledge base."
    parts = []
    for i, c in enumerate(chunks, 1):
        page = f", page {c['page']}" if c.get("page") else ""
        parts.append(f"[Source {i}: {c['source']}{page}]\n{c['content']}")
    return "\n\n---\n\n".join(parts)


def generate_response(query: str, chunks: list[dict], history: list[dict]) -> str:
    """Generate answer using Groq."""
    client = get_groq_client()
    context = format_context(chunks)
    hist_text = "\n".join(
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
        for m in history[-10:]
    ) or "No previous conversation."

    system = SYSTEM_PROMPT.format(context=context, history=hist_text)
    model = st.secrets.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
        temperature=0.3,
        max_tokens=2048,
        stream=False,
    )
    answer = resp.choices[0].message.content or ""

    # Append source citations
    if chunks:
        answer += "\n\n---\n**📚 Sources:**\n"
        seen = set()
        for c in chunks:
            k = f"{c['source']}-{c.get('page','')}"
            if k not in seen:
                seen.add(k)
                page_str = f", page {c['page']}" if c.get("page") else ""
                answer += f"- `{c['source']}{page_str}`\n"
    return answer


# ── Session state ──────────────────────────────────────────────────────────────
if "user_id"      not in st.session_state: st.session_state.user_id      = str(uuid.uuid4())
if "session_id"   not in st.session_state: st.session_state.session_id   = str(uuid.uuid4())
if "messages"     not in st.session_state: st.session_state.messages     = []
if "doc_count"    not in st.session_state: st.session_state.doc_count    = 0
if "last_sources" not in st.session_state: st.session_state.last_sources = []

ns = f"user_{st.session_state.user_id}"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 16px 12px;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:2px;">
        <div style="width:30px;height:30px;background:linear-gradient(135deg,#00d4ff,#0080ff);
             border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:14px;">⚡</div>
        <span style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:#e8f4ff;">DocSage</span>
      </div>
      <span style="font-size:0.68rem;color:#2a4a6a;letter-spacing:0.1em;text-transform:uppercase;margin-left:40px;">
        RAG · Memory · Citations
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin:0 12px 14px;padding:8px 12px;background:#0a1020;
         border:1px solid #0f1824;border-radius:8px;display:flex;align-items:center;gap:8px;">
      <div style="width:7px;height:7px;border-radius:50%;background:#00ff88;box-shadow:0 0 8px #00ff88;"></div>
      <span style="font-size:0.75rem;color:#00ff88;font-weight:500;">System Online</span>
      <span style="margin-left:auto;font-size:0.68rem;color:#1a3050;">{st.session_state.doc_count} docs</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<p style="font-size:0.7rem;color:#2a4a6a;text-transform:uppercase;
       letter-spacing:0.1em;margin:0 12px 6px;font-weight:600;">📄 Upload Documents</p>""",
       unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "docs", type=["pdf","txt","docx","md"],
        accept_multiple_files=True, label_visibility="collapsed"
    )

    if uploaded:
        st.markdown(f"""<div style="margin:4px 12px 6px;font-size:0.73rem;color:#3a6a8a;">
          {len(uploaded)} file{"s" if len(uploaded)>1 else ""} ready</div>""",
          unsafe_allow_html=True)

        if st.button("⚡  Index Documents", key="idx"):
            bar = st.progress(0)
            for i, f in enumerate(uploaded):
                with st.spinner(f"Processing {f.name}…"):
                    count, err = process_file(f.read(), f.name, ns)
                    if err:
                        st.error(f"✗ {f.name}: {err}")
                    else:
                        st.success(f"✓ {f.name} · {count} chunks")
                        st.session_state.doc_count += 1
                bar.progress((i+1)/len(uploaded))
            if st.session_state.doc_count:
                st.balloons()

    st.markdown("<hr>", unsafe_allow_html=True)

    if st.session_state.last_sources:
        st.markdown("""<p style="font-size:0.7rem;color:#2a4a6a;text-transform:uppercase;
           letter-spacing:0.1em;margin:0 12px 8px;font-weight:600;">📚 Last Sources</p>""",
           unsafe_allow_html=True)
        seen = set()
        for src in st.session_state.last_sources[:4]:
            k = f"{src['source']}-{src.get('page','')}"
            if k in seen: continue
            seen.add(k)
            page_str = f" · p.{src['page']}" if src.get("page") else ""
            score = src.get("score", 0)
            rel = max(0, min(100, int((1 - float(score)) * 100))) if score else 75
            st.markdown(f"""
            <div style="margin:0 12px 6px;padding:8px 12px;background:#0a1020;
                 border:1px solid #0f1824;border-radius:8px;">
              <div style="font-size:0.73rem;color:#4a8abf;overflow:hidden;
                   text-overflow:ellipsis;white-space:nowrap;">{src['source']}{page_str}</div>
              <div style="margin-top:5px;height:2px;background:#0f1824;border-radius:2px;">
                <div style="width:{rel}%;height:100%;
                     background:linear-gradient(90deg,#00d4ff,#0080ff);border-radius:2px;"></div>
              </div>
              <div style="font-size:0.63rem;color:#1a3050;margin-top:3px;">{rel}% match</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear"):
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.rerun()
    with col2:
        if st.button("＋ New"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.rerun()

    st.markdown(f"""
    <div style="padding:10px 12px 20px;">
      <div style="font-size:0.62rem;color:#1a3050;font-family:monospace;line-height:2;">
        USER {st.session_state.user_id[:14]}…<br>
        SESS {st.session_state.session_id[:14]}…
      </div>
    </div>""", unsafe_allow_html=True)

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:24px 36px 0;border-bottom:1px solid #0f1824;margin-bottom:0;">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
    <h1 style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
         color:#e8f4ff;letter-spacing:-0.02em;margin:0;">Document Intelligence</h1>
    <span style="font-size:0.65rem;color:#3a5a7a;text-transform:uppercase;
          letter-spacing:0.1em;padding:3px 8px;border:1px solid #1a2a40;
          border-radius:4px;font-weight:600;">DocSage</span>
  </div>
  <p style="font-size:0.82rem;color:#2a4a6a;margin:0 0 18px;">
    Upload documents · Ask anything · Get cited answers instantly
  </p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;padding:64px 20px;text-align:center;">
      <div style="width:56px;height:56px;background:linear-gradient(135deg,#00d4ff18,#0080ff18);
           border:1px solid #1a3a5a;border-radius:14px;display:flex;align-items:center;
           justify-content:center;font-size:24px;margin-bottom:16px;">⚡</div>
      <h2 style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
           color:#3a6a8a;margin-bottom:8px;">Ready to answer</h2>
      <p style="font-size:0.83rem;color:#1a3a5a;max-width:380px;line-height:1.7;margin-bottom:24px;">
        Upload a PDF, Word doc, or text file on the left,<br>then ask me anything about its contents.
      </p>
      <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:center;">
        <div style="padding:8px 16px;background:#0a1020;border:1px solid #0f2030;border-radius:20px;font-size:0.75rem;color:#2a5a7a;">"Summarise the key points"</div>
        <div style="padding:8px 16px;background:#0a1020;border:1px solid #0f2030;border-radius:20px;font-size:0.75rem;color:#2a5a7a;">"What does section 3 say?"</div>
        <div style="padding:8px 16px;background:#0a1020;border:1px solid #0f2030;border-radius:20px;font-size:0.75rem;color:#2a5a7a;">"List all requirements"</div>
      </div>
    </div>""", unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div style="display:flex;justify-content:flex-end;padding:6px 36px;">
          <div style="max-width:68%;padding:12px 18px;
               background:linear-gradient(135deg,#0a2040,#081830);
               border:1px solid #1a3a5a;border-radius:18px 18px 4px 18px;
               font-size:0.88rem;color:#c8e0f8;line-height:1.65;">{msg['content']}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display:flex;gap:10px;padding:6px 36px;align-items:flex-start;">
          <div style="width:26px;height:26px;flex-shrink:0;margin-top:6px;
               background:linear-gradient(135deg,#00d4ff,#0080ff);
               border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:12px;">⚡</div>
          <div style="max-width:78%;padding:12px 18px;background:#0a1020;
               border:1px solid #0f1824;border-radius:4px 18px 18px 18px;
               font-size:0.86rem;color:#a8c8e0;line-height:1.75;">{msg['content']}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

query = st.chat_input("Ask anything about your documents…")

if query:
    st.markdown(f"""
    <div style="display:flex;justify-content:flex-end;padding:6px 36px;">
      <div style="max-width:68%;padding:12px 18px;
           background:linear-gradient(135deg,#0a2040,#081830);
           border:1px solid #1a3a5a;border-radius:18px 18px 4px 18px;
           font-size:0.88rem;color:#c8e0f8;line-height:1.65;">{query}</div>
    </div>""", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "user", "content": query})

    resp_box = st.empty()
    resp_box.markdown("""
    <div style="display:flex;gap:10px;padding:6px 36px;align-items:flex-start;">
      <div style="width:26px;height:26px;flex-shrink:0;margin-top:6px;
           background:linear-gradient(135deg,#00d4ff,#0080ff);
           border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:12px;">⚡</div>
      <div style="padding:12px 18px;background:#0a1020;border:1px solid #0f1824;
           border-radius:4px 18px 18px 18px;font-size:0.86rem;color:#2a5a7a;">
        Searching your documents…
      </div>
    </div>""", unsafe_allow_html=True)

    chunks = retrieve(query, ns)
    st.session_state.last_sources = chunks

    with st.spinner(""):
        answer = generate_response(query, chunks, st.session_state.messages)

    resp_box.markdown(f"""
    <div style="display:flex;gap:10px;padding:6px 36px;align-items:flex-start;">
      <div style="width:26px;height:26px;flex-shrink:0;margin-top:6px;
           background:linear-gradient(135deg,#00d4ff,#0080ff);
           border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:12px;">⚡</div>
      <div style="max-width:78%;padding:12px 18px;background:#0a1020;
           border:1px solid #0f1824;border-radius:4px 18px 18px 18px;
           font-size:0.86rem;color:#a8c8e0;line-height:1.75;">{answer}</div>
    </div>""", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    if chunks:
        seen, unique = set(), []
        for s in chunks:
            k = f"{s['source']}-{s.get('page','')}"
            if k not in seen:
                seen.add(k)
                unique.append(s)
        with st.expander(f"📚 {len(unique)} source{'s' if len(unique)>1 else ''} used", expanded=False):
            for i, src in enumerate(unique, 1):
                page_str = f" — page {src['page']}" if src.get("page") else ""
                score = src.get("score", 0)
                rel = max(0, min(100, int((1 - float(score)) * 100))) if score else 75
                preview = str(src.get("content",""))[:220]
                st.markdown(f"""
                <div style="padding:10px 14px;background:#060910;border:1px solid #0f1824;
                     border-radius:8px;margin-bottom:8px;">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <span style="font-size:0.78rem;color:#00d4ff;font-weight:600;">
                      [{i}] {src['source']}{page_str}</span>
                    <span style="font-size:0.68rem;color:#2a5070;background:#0a1020;
                           padding:2px 8px;border-radius:4px;">{rel}% match</span>
                  </div>
                  <p style="font-size:0.76rem;color:#3a6a8a;line-height:1.55;margin:0;">
                    {preview}{'…' if len(str(src.get('content','')))>220 else ''}
                  </p>
                </div>""", unsafe_allow_html=True)

    st.rerun()