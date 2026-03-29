"""
DocSage — Premium Document Intelligence UI
Aesthetic: Refined editorial — warm ivory, deep charcoal, gold accents
Typography: Cormorant Garamond (display) + DM Sans (body)
"""

import io, uuid, hashlib
import numpy as np
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="DocSage",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --ivory:    #f8f5ef;
  --ivory2:   #f2ede4;
  --ivory3:   #ede6d9;
  --charcoal: #1c1917;
  --stone:    #78716c;
  --muted:    #a8a29e;
  --border:   #e7e0d5;
  --gold:     #b5975a;
  --gold2:    #d4af70;
  --white:    #ffffff;
  --shadow:   0 1px 3px rgba(28,25,23,0.06), 0 4px 16px rgba(28,25,23,0.04);
  --shadow2:  0 2px 8px rgba(28,25,23,0.10), 0 12px 32px rgba(28,25,23,0.06);
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    background: var(--ivory) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--charcoal) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton, [data-testid="stToolbar"] { display: none !important; }
.block-container { padding-top: 0 !important; max-width: 100% !important; }

/* ── Sidebar ─────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--white) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 2px 0 24px rgba(28,25,23,0.05) !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

/* Sidebar buttons */
section[data-testid="stSidebar"] .stButton > button {
    background: var(--ivory2) !important;
    color: var(--stone) !important;
    border: 1px solid var(--border) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    border-radius: 8px !important;
    width: 100% !important;
    padding: 8px 12px !important;
    transition: all 0.2s ease !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--charcoal) !important;
    color: var(--ivory) !important;
    border-color: var(--charcoal) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(28,25,23,0.15) !important;
}

/* Index button special */
div[data-testid="stButton"]:has(button[kind="primary"]) button,
.index-btn > button {
    background: var(--charcoal) !important;
    color: var(--ivory) !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    font-size: 0.76rem !important;
    text-transform: uppercase !important;
    border-radius: 8px !important;
    padding: 10px 16px !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
div[data-testid="stButton"]:has(button[kind="primary"]) button:hover {
    background: var(--gold) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(181,151,90,0.3) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--ivory2) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--gold) !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: var(--white) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 16px !important;
    box-shadow: var(--shadow2) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(181,151,90,0.12), var(--shadow2) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--charcoal) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.93rem !important;
}
[data-testid="stChatInput"] button {
    background: var(--charcoal) !important;
    border-radius: 10px !important;
    transition: background 0.2s !important;
}
[data-testid="stChatInput"] button:hover {
    background: var(--gold) !important;
}

/* Divider */
hr { border-color: var(--border) !important; margin: 14px 0 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--ivory3); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* Expander */
.stExpander {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    box-shadow: var(--shadow) !important;
}

/* Progress */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--charcoal), var(--gold)) !important;
    border-radius: 4px !important;
}

/* Alerts */
.stSuccess {
    background: #f0fdf4 !important;
    border-left: 3px solid #22c55e !important;
    border-radius: 0 8px 8px 0 !important;
    color: #166534 !important;
}
.stError {
    background: #fff7f0 !important;
    border-left: 3px solid var(--gold) !important;
    border-radius: 0 8px 8px 0 !important;
    color: #7c2d12 !important;
}

/* ── Animations ──────────────────────────── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes slideRight {
    from { opacity: 0; transform: translateX(-12px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes shimmer {
    0%   { opacity: 0.4; }
    50%  { opacity: 1; }
    100% { opacity: 0.4; }
}
@keyframes dotPulse {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40%            { transform: scale(1); opacity: 1; }
}
@keyframes goldLine {
    from { width: 0; }
    to   { width: 40px; }
}

.anim-fade-up   { animation: fadeUp 0.4s cubic-bezier(0.16,1,0.3,1) forwards; }
.anim-fade-in   { animation: fadeIn 0.3s ease forwards; }
.anim-slide-r   { animation: slideRight 0.35s cubic-bezier(0.16,1,0.3,1) forwards; }

.dot {
    display: inline-block;
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--muted);
    margin: 0 2px;
    animation: dotPulse 1.4s ease infinite;
}
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
</style>
""", unsafe_allow_html=True)


# ── Groq ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_groq():
    from groq import Groq
    key = st.secrets.get("GROQ_API_KEY", "")
    if not key:
        st.error("⚠ GROQ_API_KEY not set. Add it in Render → Environment Variables.")
        st.stop()
    return Groq(api_key=key)


# ── Vector search (pure numpy) ────────────────────────────────────────────────
def embed(text: str, dim=256) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    t = text.lower()
    for n in [2, 3, 4]:
        for i in range(len(t) - n + 1):
            h = int(hashlib.md5(t[i:i+n].encode()).hexdigest()[:8], 16)
            vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def cosine(a, b): return float(np.dot(a, b))

def chunk_text(text, size=900, overlap=150):
    if len(text) <= size: return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", " "]:
                idx = text.rfind(sep, start + size//2, end)
                if idx > start: end = idx + len(sep); break
        c = text[start:end].strip()
        if c: chunks.append(c)
        start = end - overlap
    return chunks


# ── Store ──────────────────────────────────────────────────────────────────────
def get_store(ns):
    k = f"vs_{ns}"
    if k not in st.session_state:
        st.session_state[k] = {"chunks": [], "embs": []}
    return st.session_state[k]

def add_to_store(ns, chunks, source, page):
    s = get_store(ns)
    for c in chunks:
        s["chunks"].append({"content": c, "source": source, "page": page})
        s["embs"].append(embed(c))
    return len(chunks)

def search_store(query, ns, k=6):
    s = get_store(ns)
    if not s["chunks"]: return []
    q = embed(query)
    scores = [cosine(q, e) for e in s["embs"]]
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [{**s["chunks"][i], "score": round(float(scores[i]), 4)}
            for i in top if scores[i] > 0.05]


# ── File loading ───────────────────────────────────────────────────────────────
def load_file(data, name):
    sfx = Path(name).suffix.lower()
    pages = []
    if sfx == ".pdf":
        from pypdf import PdfReader
        for i, p in enumerate(PdfReader(io.BytesIO(data)).pages):
            t = (p.extract_text() or "").strip()
            if t: pages.append({"text": t, "page": i+1})
    elif sfx in (".txt", ".md"):
        pages.append({"text": data.decode("utf-8", errors="replace"), "page": None})
    elif sfx == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(data))
        t = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        pages.append({"text": t, "page": None})
    return pages

def index_file(data, name, ns):
    try:
        pages = load_file(data, name)
        if not pages: return 0, "No text found."
        total = sum(add_to_store(ns, chunk_text(p["text"]), name, p["page"]) for p in pages)
        return total, ""
    except Exception as e:
        return 0, str(e)


# ── LLM ───────────────────────────────────────────────────────────────────────
SYSTEM = """You are DocSage, a precise and professional document assistant.
Answer ONLY from the provided context. If information isn't in the context, say so clearly.
Be concise, accurate, and professional. Use markdown formatting when appropriate.
End responses with cited sources: **Sources:** `filename, page X`

## Context
{context}

## Conversation History
{history}
"""

def generate(query, chunks, messages):
    client = get_groq()
    ctx = "\n\n---\n\n".join(
        f"[{c['source']}{f', page {c[\"page\"]}' if c.get('page') else ''}]\n{c['content']}"
        for c in chunks
    ) or "No documents have been uploaded yet."
    hist = "\n".join(
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
        for m in messages[-8:]
    ) or "No previous conversation."
    model = st.secrets.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    r = get_groq().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM.format(context=ctx, history=hist)},
            {"role": "user", "content": query},
        ],
        temperature=0.2, max_tokens=2048,
    )
    ans = r.choices[0].message.content or ""
    if chunks:
        ans += "\n\n---\n**Sources:**\n"
        seen = set()
        for c in chunks:
            k = f"{c['source']}-{c.get('page','')}"
            if k not in seen:
                seen.add(k)
                pg = f", page {c['page']}" if c.get("page") else ""
                ans += f"- `{c['source']}{pg}`\n"
    return ans


# ── Session state ──────────────────────────────────────────────────────────────
defaults = [("uid", str(uuid.uuid4())), ("sid", str(uuid.uuid4())),
            ("messages", []), ("doc_count", 0), ("last_src", [])]
for k, v in defaults:
    if k not in st.session_state:
        st.session_state[k] = v

ns = f"u_{st.session_state.uid[:8]}"
store = get_store(ns)
n_chunks = len(store["chunks"])


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Brand header ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:28px 22px 20px;background:var(--white);
         border-bottom:1px solid var(--border);">
      <div class="anim-slide-r" style="display:flex;align-items:center;gap:12px;">
        <div style="width:36px;height:36px;background:var(--charcoal);
             border-radius:10px;display:flex;align-items:center;
             justify-content:center;flex-shrink:0;">
          <span style="font-family:'Cormorant Garamond',serif;font-size:18px;
                color:var(--gold2);font-weight:600;line-height:1;">◈</span>
        </div>
        <div>
          <div style="font-family:'Cormorant Garamond',serif;font-size:1.25rem;
               font-weight:600;color:var(--charcoal);letter-spacing:-0.01em;
               line-height:1.1;">DocSage</div>
          <div style="font-size:0.62rem;color:var(--muted);letter-spacing:0.12em;
               text-transform:uppercase;margin-top:1px;font-weight:500;">
            Document Intelligence
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Status ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="anim-fade-in" style="margin:16px 16px 12px;padding:10px 14px;
         background:var(--ivory2);border:1px solid var(--border);border-radius:10px;
         display:flex;align-items:center;gap:10px;">
      <div style="position:relative;width:8px;height:8px;flex-shrink:0;">
        <div style="width:8px;height:8px;border-radius:50%;background:#22c55e;"></div>
        <div style="position:absolute;inset:0;border-radius:50%;background:#22c55e;
             animation:shimmer 2s ease infinite;opacity:0.4;transform:scale(1.8);"></div>
      </div>
      <span style="font-size:0.74rem;color:var(--stone);font-weight:500;">
        System ready
      </span>
      <span style="margin-left:auto;font-size:0.68rem;color:var(--muted);
            font-feature-settings:'tnum';">
        {st.session_state.doc_count} doc{"s" if st.session_state.doc_count!=1 else ""}
        &nbsp;·&nbsp; {n_chunks} chunks
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload section ────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:0 16px 8px;">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
        <div style="width:18px;height:1px;background:var(--gold);"></div>
        <span style="font-size:0.65rem;color:var(--muted);text-transform:uppercase;
              letter-spacing:0.14em;font-weight:600;">Upload Documents</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "docs", type=["pdf","txt","docx","md"],
        accept_multiple_files=True, label_visibility="collapsed"
    )

    if uploaded:
        st.markdown(f"""
        <div style="margin:4px 16px 10px;padding:6px 12px;background:var(--ivory3);
             border-radius:6px;font-size:0.72rem;color:var(--stone);">
          {len(uploaded)} file{"s" if len(uploaded)>1 else ""} selected
        </div>""", unsafe_allow_html=True)

        if st.button("Index Documents", key="idx", type="primary"):
            bar = st.progress(0)
            ok = 0
            for i, f in enumerate(uploaded):
                with st.spinner(f"Processing {f.name}…"):
                    cnt, err = index_file(f.read(), f.name, ns)
                    if err:
                        st.error(f"Failed: {f.name}")
                    else:
                        st.success(f"✓ {f.name}  ·  {cnt} chunks")
                        ok += 1
                        st.session_state.doc_count += 1
                bar.progress((i+1)/len(uploaded))
            if ok: st.balloons()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Sources panel ─────────────────────────────────────────────────────────
    if st.session_state.last_src:
        st.markdown("""
        <div style="padding:0 16px 10px;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
            <div style="width:18px;height:1px;background:var(--gold);"></div>
            <span style="font-size:0.65rem;color:var(--muted);text-transform:uppercase;
                  letter-spacing:0.14em;font-weight:600;">Last Retrieved</span>
          </div>
        </div>""", unsafe_allow_html=True)

        seen = set()
        for src in st.session_state.last_src[:3]:
            k = f"{src['source']}-{src.get('page','')}"
            if k in seen: continue
            seen.add(k)
            pg = f" · p.{src['page']}" if src.get("page") else ""
            rel = max(0, min(100, int(float(src.get("score",0.5))*100)))
            st.markdown(f"""
            <div style="margin:0 16px 8px;padding:10px 13px;background:var(--white);
                 border:1px solid var(--border);border-radius:9px;
                 box-shadow:var(--shadow);transition:transform 0.15s;"
                 onmouseover="this.style.transform='translateY(-1px)'"
                 onmouseout="this.style.transform='translateY(0)'">
              <div style="font-size:0.72rem;color:var(--charcoal);font-weight:500;
                   overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
                   margin-bottom:6px;">{src['source']}{pg}</div>
              <div style="height:2px;background:var(--ivory3);border-radius:2px;">
                <div style="width:{rel}%;height:100%;border-radius:2px;
                     background:linear-gradient(90deg,var(--charcoal),var(--gold));
                     transition:width 0.8s ease;"></div>
              </div>
              <div style="font-size:0.62rem;color:var(--muted);margin-top:3px;
                   font-feature-settings:'tnum';">{rel}% relevance</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear Chat", key="clr"):
            st.session_state.messages = []
            st.session_state.last_src = []
            st.rerun()
    with c2:
        if st.button("New Session", key="new"):
            st.session_state.sid = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.last_src = []
            st.rerun()

    st.markdown(f"""
    <div style="padding:12px 16px 28px;">
      <div style="font-size:0.58rem;color:var(--border);
           font-family:monospace;line-height:2.2;user-select:none;">
        {st.session_state.uid[:22]}…
      </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="anim-fade-up" style="padding:36px 52px 0;">
  <div style="display:flex;align-items:baseline;gap:16px;margin-bottom:6px;">
    <h1 style="font-family:'Cormorant Garamond',serif;font-size:2.2rem;
         font-weight:600;color:var(--charcoal);letter-spacing:-0.03em;
         line-height:1;margin:0;">
      Document Intelligence
    </h1>
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:2px;">
      <div style="width:40px;height:1px;background:var(--gold);
           animation:goldLine 0.8s ease 0.3s both;"></div>
      <span style="font-size:0.6rem;color:var(--gold);text-transform:uppercase;
            letter-spacing:0.16em;font-weight:600;">Beta</span>
    </div>
  </div>
  <p style="font-size:0.84rem;color:var(--muted);margin:0 0 22px;
       font-weight:400;letter-spacing:0.01em;">
    Upload documents &nbsp;·&nbsp; Ask questions &nbsp;·&nbsp; Get precise, cited answers
  </p>
  <div style="height:1px;background:linear-gradient(90deg,var(--border),transparent);
       margin-bottom:0;"></div>
</div>
""", unsafe_allow_html=True)


# ── Empty state ───────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="anim-fade-up" style="display:flex;flex-direction:column;
         align-items:center;padding:80px 24px 40px;text-align:center;">

      <!-- Icon -->
      <div style="width:60px;height:60px;border:1px solid var(--border);
           border-radius:16px;background:var(--white);
           display:flex;align-items:center;justify-content:center;
           box-shadow:var(--shadow2);margin-bottom:22px;">
        <span style="font-family:'Cormorant Garamond',serif;font-size:24px;
              color:var(--gold);font-weight:500;">◈</span>
      </div>

      <h2 style="font-family:'Cormorant Garamond',serif;font-size:1.35rem;
           font-weight:500;color:var(--charcoal);margin-bottom:8px;
           letter-spacing:-0.01em;">
        Ready to assist
      </h2>
      <p style="font-size:0.83rem;color:var(--muted);max-width:340px;
           line-height:1.75;margin-bottom:30px;">
        Upload a PDF, Word document, or text file from the sidebar,<br>
        then ask anything about its contents.
      </p>

      <!-- Suggestion chips -->
      <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:center;
           max-width:520px;">
        <span style="padding:8px 18px;background:var(--white);
              border:1px solid var(--border);border-radius:24px;
              font-size:0.75rem;color:var(--stone);
              box-shadow:var(--shadow);cursor:default;
              transition:all 0.2s;"
              onmouseover="this.style.borderColor='var(--gold)';this.style.color='var(--charcoal)'"
              onmouseout="this.style.borderColor='var(--border)';this.style.color='var(--stone)'">
          Summarise this document
        </span>
        <span style="padding:8px 18px;background:var(--white);
              border:1px solid var(--border);border-radius:24px;
              font-size:0.75rem;color:var(--stone);
              box-shadow:var(--shadow);cursor:default;
              transition:all 0.2s;"
              onmouseover="this.style.borderColor='var(--gold)';this.style.color='var(--charcoal)'"
              onmouseout="this.style.borderColor='var(--border)';this.style.color='var(--stone)'">
          What are the key findings?
        </span>
        <span style="padding:8px 18px;background:var(--white);
              border:1px solid var(--border);border-radius:24px;
              font-size:0.75rem;color:var(--stone);
              box-shadow:var(--shadow);cursor:default;
              transition:all 0.2s;"
              onmouseover="this.style.borderColor='var(--gold)';this.style.color='var(--charcoal)'"
              onmouseout="this.style.borderColor='var(--border)';this.style.color='var(--stone)'">
          List all requirements
        </span>
        <span style="padding:8px 18px;background:var(--white);
              border:1px solid var(--border);border-radius:24px;
              font-size:0.75rem;color:var(--stone);
              box-shadow:var(--shadow);cursor:default;
              transition:all 0.2s;"
              onmouseover="this.style.borderColor='var(--gold)';this.style.color='var(--charcoal)'"
              onmouseout="this.style.borderColor='var(--border)';this.style.color='var(--stone)'">
          Compare section 2 and 3
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Chat messages ──────────────────────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.messages):
    delay = min(idx * 0.05, 0.3)
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="anim-fade-up" style="display:flex;justify-content:flex-end;
             padding:8px 52px;animation-delay:{delay}s;">
          <div style="max-width:62%;padding:13px 20px;
               background:var(--charcoal);
               border-radius:20px 20px 4px 20px;
               font-size:0.875rem;color:var(--ivory);line-height:1.65;
               box-shadow:0 4px 16px rgba(28,25,23,0.18);
               font-family:'DM Sans',sans-serif;">
            {msg['content']}
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="anim-fade-up" style="display:flex;gap:14px;padding:8px 52px;
             align-items:flex-start;animation-delay:{delay}s;">
          <div style="width:30px;height:30px;flex-shrink:0;margin-top:2px;
               background:var(--white);border:1px solid var(--border);
               border-radius:9px;box-shadow:var(--shadow);
               display:flex;align-items:center;justify-content:center;">
            <span style="font-family:'Cormorant Garamond',serif;font-size:14px;
                  color:var(--gold);font-weight:500;">◈</span>
          </div>
          <div style="max-width:74%;padding:14px 20px;background:var(--white);
               border:1px solid var(--border);
               border-radius:4px 20px 20px 20px;
               font-size:0.865rem;color:var(--charcoal);line-height:1.78;
               box-shadow:var(--shadow);
               font-family:'DM Sans',sans-serif;">
            {msg['content']}
          </div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)


# ── Input ──────────────────────────────────────────────────────────────────────
query = st.chat_input("Ask anything about your documents…")

if query:
    # User bubble
    st.markdown(f"""
    <div class="anim-fade-up" style="display:flex;justify-content:flex-end;
         padding:8px 52px;">
      <div style="max-width:62%;padding:13px 20px;background:var(--charcoal);
           border-radius:20px 20px 4px 20px;font-size:0.875rem;
           color:var(--ivory);line-height:1.65;
           box-shadow:0 4px 16px rgba(28,25,23,0.18);">
        {query}
      </div>
    </div>""", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "user", "content": query})

    # Thinking indicator
    slot = st.empty()
    slot.markdown("""
    <div class="anim-fade-in" style="display:flex;gap:14px;padding:8px 52px;
         align-items:flex-start;">
      <div style="width:30px;height:30px;flex-shrink:0;margin-top:2px;
           background:var(--white);border:1px solid var(--border);
           border-radius:9px;box-shadow:var(--shadow);
           display:flex;align-items:center;justify-content:center;">
        <span style="font-family:'Cormorant Garamond',serif;font-size:14px;
              color:var(--gold);font-weight:500;">◈</span>
      </div>
      <div style="padding:16px 20px;background:var(--white);
           border:1px solid var(--border);border-radius:4px 20px 20px 20px;
           box-shadow:var(--shadow);">
        <span class="dot"></span>
        <span class="dot"></span>
        <span class="dot"></span>
      </div>
    </div>""", unsafe_allow_html=True)

    # Retrieve & generate
    chunks = search_store(query, ns)
    st.session_state.last_src = chunks

    with st.spinner(""):
        resp = generate(query, chunks, st.session_state.messages)

    slot.empty()

    # AI response
    st.markdown(f"""
    <div class="anim-fade-up" style="display:flex;gap:14px;padding:8px 52px;
         align-items:flex-start;">
      <div style="width:30px;height:30px;flex-shrink:0;margin-top:2px;
           background:var(--white);border:1px solid var(--border);
           border-radius:9px;box-shadow:var(--shadow);
           display:flex;align-items:center;justify-content:center;">
        <span style="font-family:'Cormorant Garamond',serif;font-size:14px;
              color:var(--gold);font-weight:500;">◈</span>
      </div>
      <div style="max-width:74%;padding:14px 20px;background:var(--white);
           border:1px solid var(--border);border-radius:4px 20px 20px 20px;
           font-size:0.865rem;color:var(--charcoal);line-height:1.78;
           box-shadow:var(--shadow);">
        {resp}
      </div>
    </div>""", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": resp})

    # Sources expander
    if chunks:
        seen, uniq = set(), []
        for s in chunks:
            k = f"{s['source']}-{s.get('page','')}"
            if k not in seen:
                seen.add(k)
                uniq.append(s)

        with st.expander(f"◈  {len(uniq)} source{'s' if len(uniq)>1 else ''} consulted", expanded=False):
            for i, src in enumerate(uniq, 1):
                pg = f" — page {src['page']}" if src.get("page") else ""
                rel = max(0, min(100, int(float(src.get("score",0.5))*100)))
                prev = str(src.get("content",""))[:260]
                st.markdown(f"""
                <div style="padding:12px 16px;background:var(--ivory2);
                     border:1px solid var(--border);border-radius:9px;
                     margin-bottom:9px;transition:box-shadow 0.2s;">
                  <div style="display:flex;justify-content:space-between;
                       align-items:center;margin-bottom:7px;">
                    <span style="font-size:0.77rem;color:var(--charcoal);
                          font-weight:600;font-family:'DM Sans',sans-serif;">
                      {i}.&nbsp; {src['source']}{pg}
                    </span>
                    <span style="font-size:0.65rem;color:var(--gold);
                           background:var(--white);padding:2px 9px;
                           border:1px solid var(--border);border-radius:20px;
                           font-weight:600;letter-spacing:0.04em;">
                      {rel}%
                    </span>
                  </div>
                  <div style="height:2px;background:var(--ivory3);
                       border-radius:2px;margin-bottom:8px;">
                    <div style="width:{rel}%;height:100%;border-radius:2px;
                         background:linear-gradient(90deg,var(--charcoal),var(--gold));"></div>
                  </div>
                  <p style="font-size:0.75rem;color:var(--stone);
                       line-height:1.65;margin:0;font-style:italic;">
                    "{prev}{'…' if len(str(src.get('content','')))>260 else ''}"
                  </p>
                </div>""", unsafe_allow_html=True)

    st.rerun()