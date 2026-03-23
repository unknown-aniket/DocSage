"""
streamlit_app.py - DocSage Cloud Edition
Python 3.12 | Groq LLM | FAISS vector search | No heavy dependencies
"""

import io
import uuid
import hashlib
import numpy as np
import streamlit as st
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocSage",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
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
    background: #0a1828 !important; color: #5a9abf !important;
    border: 1px solid #0f1824 !important; font-size: 0.78rem !important;
    border-radius: 8px !important; width: 100% !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #0f2030 !important; color: #00d4ff !important;
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
}
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg,#00d4ff,#0080ff) !important;
    border-radius: 10px !important;
}
hr { border-color: #0f1824 !important; margin: 10px 0 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #1a2a40; border-radius: 4px; }
.stExpander {
    background: #060910 !important;
    border: 1px solid #0f1824 !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# GROQ CLIENT
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_groq_client():
    from groq import Groq
    key = st.secrets.get("GROQ_API_KEY", "")
    if not key:
        st.error("⚠️ GROQ_API_KEY not found in Streamlit Secrets. Add it via Settings → Secrets.")
        st.stop()
    return Groq(api_key=key)


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def extract_text_from_file(file_bytes: bytes, filename: str) -> list[dict]:
    """Extract text from PDF, DOCX, TXT, MD. Returns list of {text, page}."""
    suffix = Path(filename).suffix.lower()
    pages = []

    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"text": text, "page": i + 1})

    elif suffix in (".txt", ".md"):
        text = file_bytes.decode("utf-8", errors="replace").strip()
        if text:
            pages.append({"text": text, "page": None})

    elif suffix == ".docx":
        from docx import Document as DocxDocument
        doc = DocxDocument(io.BytesIO(file_bytes))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        if text:
            pages.append({"text": text, "page": None})

    return pages


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks at natural boundaries."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break at natural boundary
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", "! ", "? ", ", ", " "]:
                idx = text.rfind(sep, start + chunk_size // 2, end)
                if idx > start:
                    end = idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS — pure numpy, no API calls needed
# ══════════════════════════════════════════════════════════════════════════════
def embed(text: str, dim: int = 256) -> np.ndarray:
    """
    Fast deterministic text embedding using character n-gram hashing.
    No external API or model required. Works offline.
    """
    vec = np.zeros(dim, dtype=np.float32)
    text_lower = text.lower()
    # Use multiple n-gram sizes for better coverage
    for n in [2, 3, 4, 5]:
        for i in range(len(text_lower) - n + 1):
            gram = text_lower[i:i + n]
            h = int(hashlib.sha256(gram.encode()).hexdigest()[:8], 16)
            idx = h % dim
            vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # vectors are pre-normalised


# ══════════════════════════════════════════════════════════════════════════════
# IN-SESSION VECTOR STORE
# ══════════════════════════════════════════════════════════════════════════════
def get_store(ns: str) -> dict:
    key = f"vs_{ns}"
    if key not in st.session_state:
        st.session_state[key] = {
            "chunks": [],       # list of {content, source, page}
            "embeddings": [],   # list of np.ndarray
        }
    return st.session_state[key]


def add_to_store(ns: str, chunks: list[str], source: str, page) -> int:
    store = get_store(ns)
    for chunk in chunks:
        store["chunks"].append({"content": chunk, "source": source, "page": page})
        store["embeddings"].append(embed(chunk))
    return len(chunks)


def semantic_search(query: str, ns: str, k: int = 6) -> list[dict]:
    store = get_store(ns)
    if not store["chunks"]:
        return []
    q_emb = embed(query)
    scores = [cosine_sim(q_emb, e) for e in store["embeddings"]]
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    results = []
    for i in top_idx:
        if scores[i] > 0.05:
            results.append({
                **store["chunks"][i],
                "score": round(float(scores[i]), 4),
            })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT INDEXING
# ══════════════════════════════════════════════════════════════════════════════
def index_file(file_bytes: bytes, filename: str, ns: str) -> tuple[int, str]:
    try:
        pages = extract_text_from_file(file_bytes, filename)
        if not pages:
            return 0, "Could not extract any text from this file."
        total = 0
        for page_info in pages:
            chunks = chunk_text(page_info["text"])
            count = add_to_store(ns, chunks, filename, page_info["page"])
            total += count
        return total, ""
    except Exception as e:
        return 0, str(e)


# ══════════════════════════════════════════════════════════════════════════════
# LLM RESPONSE
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are DocSage, a precise and helpful AI document assistant.

RULES:
1. Answer ONLY using the CONTEXT provided below.
2. If the context does not contain the answer, say: "I couldn't find that in the uploaded documents."
3. Be concise but thorough. Use markdown formatting where helpful.
4. At the end of your answer, always cite your sources like this:
   **Sources:** `filename.pdf, page 3`

## CONTEXT
{context}

## CONVERSATION HISTORY
{history}
"""

def build_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant documents found in the knowledge base."
    parts = []
    for i, c in enumerate(chunks, 1):
        page_str = f", page {c['page']}" if c.get("page") else ""
        parts.append(f"[{i}] Source: {c['source']}{page_str}\n{c['content']}")
    return "\n\n---\n\n".join(parts)


def build_history(messages: list[dict], max_turns: int = 8) -> str:
    recent = messages[-max_turns * 2:] if len(messages) > max_turns * 2 else messages
    lines = []
    for m in recent:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines) if lines else "No previous conversation."


def generate_answer(query: str, chunks: list[dict], messages: list[dict]) -> str:
    client = get_groq_client()
    model = st.secrets.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    context = build_context(chunks)
    history = build_history(messages)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(
                context=context, history=history
            )},
            {"role": "user", "content": query},
        ],
        temperature=0.2,
        max_tokens=2048,
    )
    return response.choices[0].message.content or "No response generated."


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def user_bubble(text: str):
    st.markdown(f"""
    <div style="display:flex;justify-content:flex-end;padding:6px 36px;">
      <div style="max-width:70%;padding:12px 18px;
           background:linear-gradient(135deg,#0a2040,#081830);
           border:1px solid #1a3a5a;border-radius:18px 18px 4px 18px;
           font-size:0.88rem;color:#c8e0f8;line-height:1.65;">
        {text}
      </div>
    </div>""", unsafe_allow_html=True)


def ai_bubble(text: str, cursor: bool = False):
    tail = "▌" if cursor else ""
    st.markdown(f"""
    <div style="display:flex;gap:10px;padding:6px 36px;align-items:flex-start;">
      <div style="width:26px;height:26px;flex-shrink:0;margin-top:6px;
           background:linear-gradient(135deg,#00d4ff,#0080ff);
           border-radius:7px;display:flex;align-items:center;
           justify-content:center;font-size:12px;">⚡</div>
      <div style="max-width:78%;padding:12px 18px;background:#0a1020;
           border:1px solid #0f1824;border-radius:4px 18px 18px 18px;
           font-size:0.86rem;color:#a8c8e0;line-height:1.75;">
        {text}{tail}
      </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "user_id"      not in st.session_state: st.session_state.user_id      = str(uuid.uuid4())
if "session_id"   not in st.session_state: st.session_state.session_id   = str(uuid.uuid4())
if "messages"     not in st.session_state: st.session_state.messages     = []
if "doc_count"    not in st.session_state: st.session_state.doc_count    = 0
if "last_sources" not in st.session_state: st.session_state.last_sources = []

ns = f"u_{st.session_state.user_id[:8]}"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # Logo
    st.markdown("""
    <div style="padding:20px 16px 12px;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
        <div style="width:32px;height:32px;
             background:linear-gradient(135deg,#00d4ff,#0080ff);
             border-radius:9px;display:flex;align-items:center;
             justify-content:center;font-size:16px;">⚡</div>
        <span style="font-family:'Syne',sans-serif;font-size:1.15rem;
              font-weight:800;color:#e8f4ff;letter-spacing:-0.01em;">DocSage</span>
      </div>
      <span style="font-size:0.67rem;color:#2a4a6a;letter-spacing:0.1em;
            text-transform:uppercase;padding-left:44px;">
        RAG · Memory · Citations
      </span>
    </div>
    """, unsafe_allow_html=True)

    # Status
    store = get_store(ns)
    chunk_count = len(store["chunks"])
    st.markdown(f"""
    <div style="margin:0 12px 14px;padding:9px 14px;background:#0a1020;
         border:1px solid #0f1824;border-radius:9px;
         display:flex;align-items:center;gap:8px;">
      <div style="width:8px;height:8px;border-radius:50%;
           background:#00ff88;box-shadow:0 0 8px #00ff88;flex-shrink:0;"></div>
      <span style="font-size:0.75rem;color:#00ff88;font-weight:500;">Online</span>
      <span style="margin-left:auto;font-size:0.68rem;color:#1a3050;">
        {st.session_state.doc_count} doc{"s" if st.session_state.doc_count != 1 else ""}
        · {chunk_count} chunks
      </span>
    </div>
    """, unsafe_allow_html=True)

    # Upload
    st.markdown("""
    <p style="font-size:0.7rem;color:#2a4a6a;text-transform:uppercase;
       letter-spacing:0.1em;margin:0 12px 8px;font-weight:600;">📄 Upload Documents</p>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(f"""
        <div style="margin:4px 12px 8px;font-size:0.73rem;color:#3a6a8a;">
          {len(uploaded_files)} file{"s" if len(uploaded_files) > 1 else ""} selected
        </div>""", unsafe_allow_html=True)

        if st.button("⚡ Index Documents", key="index"):
            progress = st.progress(0)
            indexed = 0
            for i, f in enumerate(uploaded_files):
                with st.spinner(f"Indexing {f.name}…"):
                    count, err = index_file(f.read(), f.name, ns)
                    if err:
                        st.error(f"✗ {f.name}: {err}")
                    else:
                        st.success(f"✓ {f.name} · {count} chunks")
                        indexed += 1
                        st.session_state.doc_count += 1
                progress.progress((i + 1) / len(uploaded_files))
            if indexed:
                st.balloons()

    st.markdown("<hr>", unsafe_allow_html=True)

    # Last sources panel
    if st.session_state.last_sources:
        st.markdown("""
        <p style="font-size:0.7rem;color:#2a4a6a;text-transform:uppercase;
           letter-spacing:0.1em;margin:0 12px 8px;font-weight:600;">📚 Last Sources</p>
        """, unsafe_allow_html=True)

        seen: set = set()
        for src in st.session_state.last_sources[:4]:
            key = f"{src['source']}-{src.get('page', '')}"
            if key in seen:
                continue
            seen.add(key)
            page_str = f" · p.{src['page']}" if src.get("page") else ""
            score = float(src.get("score", 0.5))
            relevance = max(0, min(100, int(score * 100)))
            st.markdown(f"""
            <div style="margin:0 12px 7px;padding:9px 12px;background:#0a1020;
                 border:1px solid #0f1824;border-radius:9px;">
              <div style="font-size:0.73rem;color:#4a8abf;font-weight:500;
                   overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
                {src['source']}{page_str}
              </div>
              <div style="margin-top:5px;height:2px;background:#0f1824;border-radius:2px;">
                <div style="width:{relevance}%;height:100%;
                     background:linear-gradient(90deg,#00d4ff,#0080ff);
                     border-radius:2px;"></div>
              </div>
              <div style="font-size:0.63rem;color:#1a3050;margin-top:3px;">
                {relevance}% relevance
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear Chat"):
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.rerun()
    with col2:
        if st.button("＋ New Session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.rerun()

    st.markdown(f"""
    <div style="padding:10px 12px 24px;">
      <div style="font-size:0.62rem;color:#1a3050;
           font-family:monospace;line-height:2.2;">
        USER&nbsp;&nbsp;{st.session_state.user_id[:16]}…<br>
        SESSION&nbsp;{st.session_state.session_id[:16]}…
      </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="padding:24px 36px 0;border-bottom:1px solid #0f1824;margin-bottom:4px;">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
    <h1 style="font-family:'Syne',sans-serif;font-size:1.55rem;font-weight:800;
         color:#e8f4ff;letter-spacing:-0.02em;margin:0;">Document Intelligence</h1>
    <span style="font-size:0.65rem;color:#3a5a7a;text-transform:uppercase;
          letter-spacing:0.1em;padding:3px 10px;border:1px solid #1a2a40;
          border-radius:4px;font-weight:600;">DocSage</span>
  </div>
  <p style="font-size:0.82rem;color:#2a4a6a;margin:0 0 18px;">
    Upload documents · Ask anything · Get cited answers
  </p>
</div>
""", unsafe_allow_html=True)

# Empty state
if not st.session_state.messages:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;
         padding:60px 20px;text-align:center;">
      <div style="width:58px;height:58px;
           background:linear-gradient(135deg,#00d4ff14,#0080ff14);
           border:1px solid #1a3a5a;border-radius:16px;
           display:flex;align-items:center;justify-content:center;
           font-size:26px;margin-bottom:18px;">⚡</div>
      <h2 style="font-family:'Syne',sans-serif;font-size:1.1rem;
           font-weight:700;color:#3a6a8a;margin-bottom:8px;">Ready to answer</h2>
      <p style="font-size:0.83rem;color:#1a3a5a;max-width:380px;
           line-height:1.7;margin-bottom:26px;">
        Upload a PDF, Word doc, or text file from the sidebar,<br>
        then ask me anything about its contents.
      </p>
      <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:center;">
        <span style="padding:8px 16px;background:#0a1020;border:1px solid #0f2030;
              border-radius:20px;font-size:0.75rem;color:#2a5a7a;">
          "Summarise the key points"
        </span>
        <span style="padding:8px 16px;background:#0a1020;border:1px solid #0f2030;
              border-radius:20px;font-size:0.75rem;color:#2a5a7a;">
          "What does section 3 say?"
        </span>
        <span style="padding:8px 16px;background:#0a1020;border:1px solid #0f2030;
              border-radius:20px;font-size:0.75rem;color:#2a5a7a;">
          "List all requirements"
        </span>
      </div>
    </div>""", unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        user_bubble(msg["content"])
    else:
        ai_bubble(msg["content"])

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────────────────────
query = st.chat_input("Ask anything about your documents…")

if query:
    # Show user message immediately
    user_bubble(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Thinking placeholder
    resp_slot = st.empty()
    resp_slot.markdown("""
    <div style="display:flex;gap:10px;padding:6px 36px;align-items:flex-start;">
      <div style="width:26px;height:26px;flex-shrink:0;margin-top:6px;
           background:linear-gradient(135deg,#00d4ff,#0080ff);border-radius:7px;
           display:flex;align-items:center;justify-content:center;font-size:12px;">⚡</div>
      <div style="padding:12px 18px;background:#0a1020;border:1px solid #0f1824;
           border-radius:4px 18px 18px 18px;font-size:0.86rem;color:#2a5a7a;">
        Searching documents…
      </div>
    </div>""", unsafe_allow_html=True)

    # Retrieve and generate
    chunks = semantic_search(query, ns, k=6)
    st.session_state.last_sources = chunks

    with st.spinner(""):
        answer = generate_answer(query, chunks, st.session_state.messages)

    # Final answer
    resp_slot.empty()
    ai_bubble(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Sources expander
    if chunks:
        seen: set = set()
        unique_sources = []
        for s in chunks:
            k = f"{s['source']}-{s.get('page', '')}"
            if k not in seen:
                seen.add(k)
                unique_sources.append(s)

        with st.expander(
            f"📚 {len(unique_sources)} source{'s' if len(unique_sources) > 1 else ''} retrieved",
            expanded=False,
        ):
            for i, src in enumerate(unique_sources, 1):
                page_str = f" — page {src['page']}" if src.get("page") else ""
                score = float(src.get("score", 0.5))
                relevance = max(0, min(100, int(score * 100)))
                preview = str(src.get("content", ""))[:250]
                if len(str(src.get("content", ""))) > 250:
                    preview += "…"
                st.markdown(f"""
                <div style="padding:10px 14px;background:#060910;
                     border:1px solid #0f1824;border-radius:8px;margin-bottom:8px;">
                  <div style="display:flex;justify-content:space-between;
                       align-items:center;margin-bottom:6px;">
                    <span style="font-size:0.78rem;color:#00d4ff;font-weight:600;">
                      [{i}] {src['source']}{page_str}
                    </span>
                    <span style="font-size:0.68rem;color:#2a5070;
                           background:#0a1020;padding:2px 8px;border-radius:4px;">
                      {relevance}% match
                    </span>
                  </div>
                  <p style="font-size:0.76rem;color:#3a6a8a;line-height:1.55;margin:0;">
                    {preview}
                  </p>
                </div>""", unsafe_allow_html=True)

    st.rerun()