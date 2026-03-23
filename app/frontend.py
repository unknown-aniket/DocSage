"""
app/frontend.py — Premium RAG Chatbot UI (fixed CSS rendering)
"""

import json
import uuid
import requests
import streamlit as st

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="NeuralDoc AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS via st.html workaround ─────────────────────────────────────────
def inject_css():
    css = """
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"], .stApp {
        background-color: #080c14 !important;
        font-family: 'DM Sans', sans-serif !important;
        color: #c8d6e8 !important;
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    .stDeployButton { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        max-width: 100% !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #060910 !important;
        border-right: 1px solid #0f1824 !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: #0a1828 !important;
        color: #5a9abf !important;
        border: 1px solid #0f1824 !important;
        font-size: 0.78rem !important;
        padding: 6px 12px !important;
        border-radius: 8px !important;
        width: 100% !important;
        font-weight: 500 !important;
        letter-spacing: 0.02em !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #0f2030 !important;
        color: #00d4ff !important;
        border-color: #1a3a5a !important;
    }
    .index-btn > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0080ff 100%) !important;
        color: #000 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.04em !important;
    }
    [data-testid="stFileUploader"] {
        background: #0a1020 !important;
        border: 1px dashed #1a2a40 !important;
        border-radius: 10px !important;
        padding: 4px !important;
    }
    [data-testid="stChatInput"] {
        background: #0a1020 !important;
        border: 1px solid #1a2a40 !important;
        border-radius: 16px !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: #e0eaf8 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #00d4ff, #0080ff) !important;
        border-radius: 10px !important;
    }
    hr { border-color: #0f1824 !important; margin: 10px 0 !important; }
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #1a2a40; border-radius: 4px; }
    .stExpander {
        background: #060910 !important;
        border: 1px solid #0f1824 !important;
        border-radius: 10px !important;
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #00d4ff, #0080ff) !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_css()

# ── Session State ──────────────────────────────────────────────────────────────
if "user_id"      not in st.session_state: st.session_state.user_id      = str(uuid.uuid4())
if "session_id"   not in st.session_state: st.session_state.session_id   = str(uuid.uuid4())
if "messages"     not in st.session_state: st.session_state.messages     = []
if "doc_count"    not in st.session_state: st.session_state.doc_count    = 0
if "last_sources" not in st.session_state: st.session_state.last_sources = []

# ── API health check ───────────────────────────────────────────────────────────
def check_api():
    try:
        r = requests.get(f"{API_BASE.replace('/api/v1','')}/health", timeout=2)
        return r.ok
    except:
        return False

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 16px 12px;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:2px;">
        <div style="width:30px;height:30px;background:linear-gradient(135deg,#00d4ff,#0080ff);
             border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:14px;">
          ⚡
        </div>
        <span style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
              color:#e8f4ff;letter-spacing:-0.01em;">NeuralDoc AI</span>
      </div>
      <span style="font-size:0.68rem;color:#2a4a6a;letter-spacing:0.1em;
            text-transform:uppercase;margin-left:40px;">RAG · Memory · Citations</span>
    </div>
    """, unsafe_allow_html=True)

    # Status badge
    api_ok = check_api()
    c, t = ("#00ff88", "System Online") if api_ok else ("#ff4466", "API Offline — run python main.py")
    st.markdown(f"""
    <div style="margin:0 12px 14px;padding:8px 12px;background:#0a1020;
         border:1px solid #0f1824;border-radius:8px;
         display:flex;align-items:center;gap:8px;">
      <div style="width:7px;height:7px;border-radius:50%;background:{c};
           box-shadow:0 0 8px {c};"></div>
      <span style="font-size:0.75rem;color:{c};font-weight:500;">{t}</span>
      <span style="margin-left:auto;font-size:0.68rem;color:#1a3050;">
        {st.session_state.doc_count} docs
      </span>
    </div>
    """, unsafe_allow_html=True)

    # Upload section
    st.markdown("""
    <p style="font-size:0.7rem;color:#2a4a6a;text-transform:uppercase;
       letter-spacing:0.1em;margin:0 12px 6px;font-weight:600;">
      📄 Upload Documents
    </p>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "docs", type=["pdf","txt","docx","md"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded:
        st.markdown(f"""
        <div style="margin:4px 12px 6px;font-size:0.73rem;color:#3a6a8a;">
          {len(uploaded)} file{"s" if len(uploaded)>1 else ""} ready to index
        </div>
        """, unsafe_allow_html=True)

        if st.button("⚡  Index Documents", key="index_btn"):
            bar = st.progress(0)
            ok = 0
            for i, f in enumerate(uploaded):
                with st.spinner(f"Processing {f.name}…"):
                    try:
                        ns = f"user_{st.session_state.user_id}"
                        resp = requests.post(
                            f"{API_BASE}/upload",
                            files={"file": (f.name, f.read(), f.type or "application/octet-stream")},
                            data={"user_id": st.session_state.user_id, "namespace": ns},
                            timeout=120,
                        )
                        if resp.ok:
                            d = resp.json()
                            st.success(f"✓ {d['filename']} · {d['chunks']} chunks")
                            st.session_state.doc_count += 1
                            ok += 1
                        else:
                            st.error(f"✗ {f.name}: {resp.json().get('detail','Failed')}")
                    except Exception as e:
                        st.error(f"✗ {f.name}: {e}")
                bar.progress((i+1)/len(uploaded))
            if ok:
                st.balloons()

    st.markdown("<hr>", unsafe_allow_html=True)

    # Last sources
    if st.session_state.last_sources:
        st.markdown("""
        <p style="font-size:0.7rem;color:#2a4a6a;text-transform:uppercase;
           letter-spacing:0.1em;margin:0 12px 8px;font-weight:600;">
          📚 Last Sources
        </p>
        """, unsafe_allow_html=True)
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
              <div style="font-size:0.73rem;color:#4a8abf;font-weight:500;
                   overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
                {src['source']}{page_str}
              </div>
              <div style="margin-top:5px;height:2px;background:#0f1824;border-radius:2px;">
                <div style="width:{rel}%;height:100%;
                     background:linear-gradient(90deg,#00d4ff,#0080ff);border-radius:2px;">
                </div>
              </div>
              <div style="font-size:0.63rem;color:#1a3050;margin-top:3px;">{rel}% match</div>
            </div>
            """, unsafe_allow_html=True)
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
    <div style="padding:10px 12px 20px;">
      <div style="font-size:0.62rem;color:#1a3050;font-family:monospace;line-height:2;">
        USER {st.session_state.user_id[:14]}…<br>
        SESS {st.session_state.session_id[:14]}…
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Main Area ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:24px 36px 0;border-bottom:1px solid #0f1824;margin-bottom:0;">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
    <h1 style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
         color:#e8f4ff;letter-spacing:-0.02em;margin:0;">
      Document Intelligence
    </h1>
    <span style="font-size:0.65rem;color:#3a5a7a;text-transform:uppercase;
          letter-spacing:0.1em;padding:3px 8px;border:1px solid #1a2a40;
          border-radius:4px;font-weight:600;">Beta</span>
  </div>
  <p style="font-size:0.82rem;color:#2a4a6a;margin:0 0 18px;">
    Upload documents · Ask anything · Get cited answers instantly
  </p>
</div>
""", unsafe_allow_html=True)

# Empty state
if not st.session_state.messages:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;
         padding:64px 20px;text-align:center;">
      <div style="width:56px;height:56px;
           background:linear-gradient(135deg,#00d4ff18,#0080ff18);
           border:1px solid #1a3a5a;border-radius:14px;
           display:flex;align-items:center;justify-content:center;
           font-size:24px;margin-bottom:16px;">⚡</div>
      <h2 style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
           color:#3a6a8a;margin-bottom:8px;">Ready to answer</h2>
      <p style="font-size:0.83rem;color:#1a3a5a;max-width:380px;line-height:1.7;margin-bottom:24px;">
        Upload a PDF, Word doc, or text file on the left sidebar,
        then ask me anything about its contents.
      </p>
      <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:center;">
        <div style="padding:8px 16px;background:#0a1020;border:1px solid #0f2030;
             border-radius:20px;font-size:0.75rem;color:#2a5a7a;">
          "Summarise the key points"
        </div>
        <div style="padding:8px 16px;background:#0a1020;border:1px solid #0f2030;
             border-radius:20px;font-size:0.75rem;color:#2a5a7a;">
          "What does section 3 say?"
        </div>
        <div style="padding:8px 16px;background:#0a1020;border:1px solid #0f2030;
             border-radius:20px;font-size:0.75rem;color:#2a5a7a;">
          "List all requirements"
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div style="display:flex;justify-content:flex-end;padding:6px 36px;">
          <div style="max-width:68%;padding:12px 18px;
               background:linear-gradient(135deg,#0a2040,#081830);
               border:1px solid #1a3a5a;border-radius:18px 18px 4px 18px;
               font-size:0.88rem;color:#c8e0f8;line-height:1.65;">
            {msg['content']}
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display:flex;gap:10px;padding:6px 36px;align-items:flex-start;">
          <div style="width:26px;height:26px;flex-shrink:0;margin-top:6px;
               background:linear-gradient(135deg,#00d4ff,#0080ff);
               border-radius:7px;display:flex;align-items:center;
               justify-content:center;font-size:12px;">⚡</div>
          <div style="max-width:78%;padding:12px 18px;background:#0a1020;
               border:1px solid #0f1824;border-radius:4px 18px 18px 18px;
               font-size:0.86rem;color:#a8c8e0;line-height:1.75;">
            {msg['content']}
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Chat Input ─────────────────────────────────────────────────────────────────
query = st.chat_input("Ask anything about your documents…")

if query:
    # Show user bubble
    st.markdown(f"""
    <div style="display:flex;justify-content:flex-end;padding:6px 36px;">
      <div style="max-width:68%;padding:12px 18px;
           background:linear-gradient(135deg,#0a2040,#081830);
           border:1px solid #1a3a5a;border-radius:18px 18px 4px 18px;
           font-size:0.88rem;color:#c8e0f8;line-height:1.65;">
        {query}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "user", "content": query})

    resp_placeholder = st.empty()
    full_answer = ""

    # Thinking state
    resp_placeholder.markdown("""
    <div style="display:flex;gap:10px;padding:6px 36px;align-items:flex-start;">
      <div style="width:26px;height:26px;flex-shrink:0;margin-top:6px;
           background:linear-gradient(135deg,#00d4ff,#0080ff);
           border-radius:7px;display:flex;align-items:center;
           justify-content:center;font-size:12px;">⚡</div>
      <div style="padding:12px 18px;background:#0a1020;border:1px solid #0f1824;
           border-radius:4px 18px 18px 18px;font-size:0.86rem;
           color:#2a5a7a;line-height:1.75;">
        Searching your documents…
      </div>
    </div>
    """, unsafe_allow_html=True)

    sources = []

    try:
        with requests.post(
            f"{API_BASE}/chat/stream",
            json={
                "query": query,
                "session_id": st.session_state.session_id,
                "user_id": st.session_state.user_id,
                "top_k": 6,
                "stream": True,
            },
            stream=True,
            timeout=120,
        ) as response:
            for line in response.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                try:
                    payload = json.loads(line[6:])
                except:
                    continue

                if "token" in payload:
                    full_answer += payload["token"]
                    resp_placeholder.markdown(f"""
                    <div style="display:flex;gap:10px;padding:6px 36px;align-items:flex-start;">
                      <div style="width:26px;height:26px;flex-shrink:0;margin-top:6px;
                           background:linear-gradient(135deg,#00d4ff,#0080ff);
                           border-radius:7px;display:flex;align-items:center;
                           justify-content:center;font-size:12px;">⚡</div>
                      <div style="max-width:78%;padding:12px 18px;background:#0a1020;
                           border:1px solid #0f1824;border-radius:4px 18px 18px 18px;
                           font-size:0.86rem;color:#a8c8e0;line-height:1.75;">
                        {full_answer}▌
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                elif payload.get("done"):
                    sources = payload.get("sources", [])
                    st.session_state.last_sources = sources

                elif "error" in payload:
                    full_answer = f"⚠️ {payload['error']}"

    except Exception as e:
        full_answer = "⚠️ Cannot connect to API. Make sure `python main.py` is running in Terminal 1."

    # Final render (no cursor)
    resp_placeholder.markdown(f"""
    <div style="display:flex;gap:10px;padding:6px 36px;align-items:flex-start;">
      <div style="width:26px;height:26px;flex-shrink:0;margin-top:6px;
           background:linear-gradient(135deg,#00d4ff,#0080ff);
           border-radius:7px;display:flex;align-items:center;
           justify-content:center;font-size:12px;">⚡</div>
      <div style="max-width:78%;padding:12px 18px;background:#0a1020;
           border:1px solid #0f1824;border-radius:4px 18px 18px 18px;
           font-size:0.86rem;color:#a8c8e0;line-height:1.75;">
        {full_answer}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": full_answer})

    # Sources panel
    if sources:
        seen = set()
        unique = []
        for s in sources:
            k = f"{s['source']}-{s.get('page','')}"
            if k not in seen:
                seen.add(k)
                unique.append(s)

        with st.expander(f"📚 {len(unique)} source{'s' if len(unique)>1 else ''} used", expanded=False):
            for i, src in enumerate(unique, 1):
                page_str = f" — page {src['page']}" if src.get("page") else ""
                score = src.get("score", 0)
                rel = max(0, min(100, int((1 - float(score)) * 100))) if score else 75
                preview = str(src.get("content", ""))[:220]
                if len(str(src.get("content", ""))) > 220:
                    preview += "…"
                st.markdown(f"""
                <div style="padding:10px 14px;background:#060910;border:1px solid #0f1824;
                     border-radius:8px;margin-bottom:8px;">
                  <div style="display:flex;justify-content:space-between;align-items:center;
                       margin-bottom:6px;">
                    <span style="font-size:0.78rem;color:#00d4ff;font-weight:600;">
                      [{i}] {src['source']}{page_str}
                    </span>
                    <span style="font-size:0.68rem;color:#2a5070;background:#0a1020;
                           padding:2px 8px;border-radius:4px;">{rel}% match</span>
                  </div>
                  <p style="font-size:0.76rem;color:#3a6a8a;line-height:1.55;margin:0;">
                    {preview}
                  </p>
                </div>
                """, unsafe_allow_html=True)

    st.rerun()