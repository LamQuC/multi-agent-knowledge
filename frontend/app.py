import streamlit as st
import requests
import os
import json
from datetime import datetime
from pathlib import Path

# ===================== CONFIG =====================
st.set_page_config(
    page_title="Multi-Agent Knowledge Chat",
    page_icon="🤖",
    layout="wide"
)

# Directories
DATA_DIR = Path("frontend/data/chats")
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ===================== UTILITIES =====================
def list_sessions():
    """Trả về danh sách file chat đã lưu (sorted by time desc)."""
    sessions = sorted(DATA_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
    return sessions


def load_session(filepath):
    """Đọc file JSON chat session."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"title": "Untitled", "messages": []}


def save_session(filepath, data):
    """Lưu nội dung hội thoại vào JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def create_new_session(title: str = None):
    """Tạo file session mới."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = title or f"Chat {ts}"
    path = DATA_DIR / f"{ts}.json"
    data = {"title": name, "created_at": ts, "messages": []}
    save_session(path, data)
    return path


# ===================== SESSION STATE =====================
if "current_session" not in st.session_state:
    sessions = list_sessions()
    if sessions:
        st.session_state["current_session"] = sessions[0]
    else:
        st.session_state["current_session"] = create_new_session()

if "messages" not in st.session_state:
    current_data = load_session(st.session_state["current_session"])
    st.session_state["messages"] = current_data.get("messages", [])

if "agent" not in st.session_state:
    st.session_state["agent"] = "knowledge"

if "mode" not in st.session_state:
    st.session_state["mode"] = "Auto"  # "Auto" or "Manual"

if "web_search" not in st.session_state:
    st.session_state["web_search"] = True


# ===================== SIDEBAR =====================
with st.sidebar:
    st.title("🧠 Multi-Agent Chat")
    st.markdown("---")

    # Mode selector (Auto / Manual)
    st.session_state["mode"] = st.radio(
        "Mode",
        ["Auto", "Manual"],
        index=0 if st.session_state["mode"] == "Auto" else 1
    )

    # Manual mode: chọn agent cụ thể
    if st.session_state["mode"] == "Manual":
        agents = ["knowledge", "explain", "code"]
        current = st.session_state.get("agent", "knowledge")
        st.session_state["agent"] = st.selectbox(
            "🧩 Chọn Agent",
            agents,
            index=agents.index(current) if current in agents else 0
        )
    else:
        st.session_state["agent"] = "auto"

    st.markdown("---")

    # 🌍 Web Search toggle – chỉ hiện nếu agent là "knowledge"
    if st.session_state["mode"] == "Manual" and st.session_state["agent"] == "knowledge":
        prev_state = st.session_state["web_search"]
        new_state = st.checkbox("🌍 Bật Web Search (ép buộc cho KnowledgeAgent)", value=prev_state)
        if new_state != prev_state:
            st.session_state["web_search"] = new_state
            try:
                api_base = "http://localhost:8000"
                requests.post(f"{api_base}/toggle_websearch", params={"enable": new_state})
            except Exception as e:
                st.warning(f"Không thể cập nhật WebSearch backend: {e}")
    else:
        # Tự động tắt web search nếu không phải KnowledgeAgent
        st.session_state["web_search"] = False

    st.markdown("---")

    # New chat section
    st.subheader("➕ New / Manage Chats")
    new_title = st.text_input("Tiêu đề chat mới (để trống cho timestamp)", value="")
    if st.button("➕ Tạo Chat Mới"):
        new_path = create_new_session(title=new_title if new_title.strip() else None)
        st.session_state["current_session"] = new_path
        st.session_state["messages"] = []
        st.rerun()

    # Lịch sử chat
    st.subheader("💬 Lịch sử Chat")
    sessions = list_sessions()
    for i, s in enumerate(sessions):
        data = load_session(s)
        label = data.get("title", f"Chat {i+1}")
        cols = st.columns([1, 4, 1])
        if cols[0].button("Open", key=f"open_{i}"):
            st.session_state["current_session"] = s
            st.session_state["messages"] = data.get("messages", [])
            st.rerun()
        cols[1].markdown(f"**{label}**\n<small>{data.get('created_at','')}</small>", unsafe_allow_html=True)
        if cols[2].button("Xóa", key=f"del_{i}"):
            try:
                os.remove(s)
            except:
                pass
            if s == st.session_state["current_session"]:
                st.session_state["current_session"] = create_new_session()
                st.session_state["messages"] = []
            st.rerun()

    st.markdown("---")
    if st.button("🗑 Xóa tất cả hội thoại"):
        for s in sessions:
            try:
                os.remove(s)
            except:
                pass
        st.session_state["messages"] = []
        st.session_state["current_session"] = create_new_session()
        st.rerun()

    st.markdown("<hr><small>Built with LangChain + FastAPI + Streamlit</small>", unsafe_allow_html=True)


# ===================== STYLES =====================
st.markdown("""
<style>
.main-chat {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.chat-bubble {
    padding: 0.9rem 1rem;
    border-radius: 12px;
    margin: 0.5rem 0;
    max-width: 80%;
    line-height: 1.5;
    font-size: 1rem;
    word-wrap: break-word;
}
.user-bubble {
    background-color: #0052cc;
    color: white;
    margin-left: auto;
}
.bot-bubble {
    background-color: #2b2b2b;
    color: #f5f5f5;
    margin-right: auto;
}
.chat-container {
    display: flex;
    flex-direction: column;
}
.sidebar-title { font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ===================== MAIN CHAT AREA =====================
st.title("🤖 Multi-Agent Knowledge")

# show selected session
current_meta = load_session(st.session_state["current_session"])
st.markdown(f"**{current_meta.get('title','Untitled')}**  •  <small>{current_meta.get('created_at','')}</small>", unsafe_allow_html=True)
st.caption(f"Mode: {st.session_state['mode']}  —  Web Search: {'On' if st.session_state['web_search'] else 'Off'}")

chat_container = st.container()
with chat_container:
    for msg in st.session_state["messages"]:
        role_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
        st.markdown(
            f"<div class='chat-container'><div class='chat-bubble {role_class}'>{msg['content']}</div></div>",
            unsafe_allow_html=True,
        )

# input
query = st.chat_input("Nhập tin nhắn của bạn...")

if query:
    st.session_state["messages"].append({"role": "user", "content": query})

    payload = {
        "query": query,
        "agent": st.session_state["agent"],
        "web_search": st.session_state["web_search"],
        "mode": st.session_state["mode"].lower()
    }

    with st.spinner("🤖 Thinking..."):
        try:
            # api_base = os.getenv("API_URL", "http://backend:8000")
            api_base = "http://localhost:8000"
            if st.session_state["mode"] == "Manual":
                api_url = f"{api_base}/chat"
            else:
                api_url = f"{api_base}/route"
            res = requests.post(api_url, json=payload, timeout=90)
            res.raise_for_status()
            answer = res.json().get("answer") or str(res.json())
        except Exception as e:
            answer = f"⚠️ Error: {e}"

    st.session_state["messages"].append({"role": "assistant", "content": answer})
    data = load_session(st.session_state["current_session"])
    data["messages"] = st.session_state["messages"]
    save_session(st.session_state["current_session"], data)

    st.rerun()

# footer
st.markdown("<hr><center>🚀 Multi-Agent Knowledge System | Phase 6 — Chat Sessions, Auto/Manual</center>", unsafe_allow_html=True)
