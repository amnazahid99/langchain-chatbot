# ┌─────────────────────────────────────────────────────────────────┐
# │  Prerequisites:                                                  │
# │    uv add streamlit langchain-ollama langchain-core             │
# │    ollama pull minimax-m2.5:cloud                               │
# │                                                                  │
# │  Run from project root:                                         │
# │    uv run streamlit run src/frontend/frontend.py                │
# └─────────────────────────────────────────────────────────────────┘

import sys
import os

# Make sure app.py (at project root) is importable from this sub-folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
from app import (
    chat,
    chat_history,
    NUM_CTX,
    MAX_INPUT_TOKENS,
    RESPONSE_RESERVE,
    SYSTEM_OVERHEAD,
    history_token_count,
    context_usage_pct,
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuralChat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080b14 !important;
    font-family: 'DM Sans', sans-serif;
    color: #c8d0e0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1020 !important;
    border-right: 1px solid #1e2540;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.8rem 1.4rem; }

.sidebar-brand {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 1.8rem;
}
.sidebar-brand .icon {
    width: 38px; height: 38px; border-radius: 10px;
    background: linear-gradient(135deg, #4f6ef7, #a855f7);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
    box-shadow: 0 0 16px #4f6ef740;
}
.sidebar-brand h1 {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem; font-weight: 800;
    color: #e8eaf6; letter-spacing: -0.3px;
    line-height: 1.1;
}
.sidebar-brand p {
    font-size: 0.7rem; color: #5a6a8a;
    letter-spacing: 0.5px; text-transform: uppercase;
    margin-top: 1px;
}

/* ── Token Meter ── */
.token-label {
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 0.4rem;
}
.token-label span:first-child {
    font-size: 0.72rem; font-weight: 500; text-transform: uppercase;
    letter-spacing: 0.8px; color: #5a6a8a;
}
.token-label span:last-child {
    font-family: 'Syne', sans-serif;
    font-size: 0.88rem; font-weight: 700; color: #e8eaf6;
}

.meter-track {
    width: 100%; height: 6px; border-radius: 99px;
    background: #1e2540; overflow: hidden; margin-bottom: 0.7rem;
}
.meter-fill {
    height: 100%; border-radius: 99px;
    transition: width 0.6s cubic-bezier(.4,0,.2,1), background 0.4s;
}

.token-stats {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 6px; margin-bottom: 1.4rem;
}
.stat-pill {
    background: #131726; border: 1px solid #1e2540;
    border-radius: 8px; padding: 7px 10px;
}
.stat-pill .label {
    font-size: 0.64rem; text-transform: uppercase; letter-spacing: 0.6px;
    color: #3d4f70; margin-bottom: 2px;
}
.stat-pill .value {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem; font-weight: 700; color: #c8d0e0;
}

/* ── Info chips in sidebar ── */
.info-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 7px 0; border-bottom: 1px solid #131726;
    font-size: 0.78rem;
}
.info-row .key { color: #3d4f70; }
.info-row .val { color: #8b9dd4; font-weight: 500; }

.badge-active {
    display: inline-block; padding: 2px 10px; border-radius: 99px;
    background: #0f2a1a; border: 1px solid #1a5c35;
    color: #3ddc84; font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.5px; text-transform: uppercase;
}

/* ── Clear button ── */
[data-testid="stSidebar"] button {
    width: 100%; margin-top: 1.4rem;
    background: #131726 !important; border: 1px solid #1e2540 !important;
    color: #8b9dd4 !important; border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    padding: 0.55rem 1rem !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebar"] button:hover {
    background: #1e2540 !important; color: #e8eaf6 !important;
    border-color: #4f6ef7 !important;
}

/* ── Main area ── */
.main-header {
    text-align: center; padding: 2.5rem 0 1.5rem;
}
.main-header h2 {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem; font-weight: 800; color: #e8eaf6;
    letter-spacing: -0.5px;
}
.main-header p { font-size: 0.85rem; color: #3d4f70; margin-top: 4px; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important; padding: 0 !important;
    margin-bottom: 0.6rem !important;
}

[data-testid="stChatMessageContent"] {
    border-radius: 16px !important;
    padding: 0.8rem 1.1rem !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
    max-width: 78% !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #0d1020 !important;
    border: 1px solid #1e2540 !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #4f6ef7 !important;
    box-shadow: 0 0 0 3px #4f6ef720 !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important; color: #c8d0e0 !important;
    background: transparent !important;
}

/* ── Alert/info boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    border: none !important;
}

/* ── Divider ── */
hr { border-color: #1e2540 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1e2540; border-radius: 99px; }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # [{role, content}]
if "last_q_tokens" not in st.session_state:
    st.session_state.last_q_tokens = 0
if "last_r_tokens" not in st.session_state:
    st.session_state.last_r_tokens = 0


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="icon">🧠</div>
        <div>
            <h1>NeuralChat</h1>
            <p>LangChain · Ollama</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Live token meter
    usage      = context_usage_pct()
    used_tokens = history_token_count()
    color = "#4f6ef7" if usage < 60 else "#f7a94f" if usage < 85 else "#f75f5f"

    st.markdown(f"""
    <div class="token-label">
        <span>Context Window</span>
        <span>{usage:.0f}%</span>
    </div>
    <div class="meter-track">
        <div class="meter-fill" style="width:{usage:.1f}%; background:{color};"></div>
    </div>
    <div class="token-stats">
        <div class="stat-pill">
            <div class="label">Used</div>
            <div class="value">{used_tokens:,}</div>
        </div>
        <div class="stat-pill">
            <div class="label">Budget</div>
            <div class="value">{MAX_INPUT_TOKENS:,}</div>
        </div>
        <div class="stat-pill">
            <div class="label">Last Q</div>
            <div class="value">{st.session_state.last_q_tokens}</div>
        </div>
        <div class="stat-pill">
            <div class="label">Last R</div>
            <div class="value">{st.session_state.last_r_tokens}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(f"""
    <div class="info-row"><span class="key">Model</span><span class="val">minimax-m2.5:cloud</span></div>
    <div class="info-row"><span class="key">Context</span><span class="val">{NUM_CTX:,} tokens</span></div>
    <div class="info-row"><span class="key">Reply reserve</span><span class="val">{RESPONSE_RESERVE:,} tokens</span></div>
    <div class="info-row"><span class="key">Input budget</span><span class="val">{MAX_INPUT_TOKENS:,} tokens</span></div>
    <div class="info-row" style="border:none">
        <span class="key">Token manager</span>
        <span class="badge-active">● Active</span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("↺  Clear conversation"):
        chat_history.clear()
        st.session_state.messages       = []
        st.session_state.last_q_tokens  = 0
        st.session_state.last_r_tokens  = 0
        st.rerun()


# ── Main Chat Area ────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h2>What can I help you with?</h2>
    <p>Context window managed automatically by token count — no turn limits.</p>
</div>
""", unsafe_allow_html=True)

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask me anything about AI…")

if user_input:
    # Show user bubble immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Call backend
    result = chat(user_input)

    # Update sidebar token stats
    st.session_state.last_q_tokens = result["q_tokens"]
    st.session_state.last_r_tokens = result["r_tokens"]

    # Question too long — can never fit
    if result["response"] is None:
        st.warning(f"⚠️  {result['warning']}")

    else:
        # Auto-trim notice
        if result["trimmed"]:
            st.info(
                "🗂  Some older messages were removed from context to stay "
                "within the token budget. Recent conversation is preserved."
            )

        # Assistant reply
        with st.chat_message("assistant"):
            st.markdown(result["response"])
        st.session_state.messages.append(
            {"role": "assistant", "content": result["response"]}
        )

        # Soft context-full warning
        if result["warning"]:
            st.warning(f"⚠️  {result['warning']}")

    st.rerun()