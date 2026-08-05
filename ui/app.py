import os
import streamlit as st
import requests
import time
import uuid
import logfire
from dotenv import load_dotenv

# Load environment variables explicitly from the root directory
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=env_path)

# Initialize Logfire
try:
    token = os.getenv("LOGFIRE_TOKEN")
    if not token:
        print("ERROR: LOGFIRE_TOKEN is empty or None!")
    logfire.configure(token=token)
    LOGFIRE_STATUS = "Connected & Tracing"
except Exception as e:
    print(f"Logfire Init Error in UI: {e}")
    LOGFIRE_STATUS = f"Standby (Error: {e})"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="🤖",
    layout="wide",
)

# --- AVATARS ---
AI_AVATAR = "🤖"
USER_AVATAR = "👤"

# --- SESSION MANAGEMENT ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logfire.info(f"✨ New User Session Created: {st.session_state.session_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- BACKEND URL & AUTOMATIC WAKE-UP ---
def get_backend_url() -> str:
    try:
        url = st.secrets.get("BACKEND_URL")
        if url:
            return url.rstrip("/")
    except Exception:
        pass
    return os.getenv("BACKEND_URL", "").rstrip("/")

base_url = get_backend_url()

def wake_and_ping_backend(url: str, retries: int = 6, retry_delay: int = 5) -> bool:
    """
    Pings backend health endpoint. Automatically triggers Render cold-start wake-up
    and continuously retries until backend is fully live.
    """
    if not url:
        return False
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(f"{url}/", timeout=12)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        if attempt < retries:
            time.sleep(retry_delay)
    return False

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 Agent OS")
    st.markdown("---")

    if not base_url:
        st.error("⚠️ BACKEND_URL not set!")
    else:
        with st.spinner("⚡ Connecting to Backend (Waking Render server)..."):
            backend_ok = wake_and_ping_backend(base_url, retries=4, retry_delay=4)
        if backend_ok:
            st.success("✅ Backend Online")
        else:
            st.warning("⚠️ Backend Starting Up... (cold start in progress)")
        st.caption(f"API: {base_url}")

    st.markdown("---")
    st.success(f"Logfire: {LOGFIRE_STATUS}")
    st.info(f"Memory ID: {st.session_state.session_id[:8]}")

    if st.button("🗑️ Clear History & Memory", width="stretch", type="primary"):
        logfire.warning(f"🗑️ Memory Wipe Triggered for session: {st.session_state.session_id}")
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# --- MAIN CHAT ---
st.title("🤖 Enterprise Agentic Assistant")

# Display history
for message in st.session_state.messages:
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask about your documentation..."):
    # START TRACE: User Interaction
    with logfire.span("💬 User Chat Interaction", user_query=prompt, session_id=st.session_state.session_id):

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        # Assistant Response
        with st.chat_message("assistant", avatar=AI_AVATAR):
            data = {}
            with st.status("🔍 Agent is thinking...", expanded=True) as status:
                try:
                    with logfire.span("📡 Calling RAG Backend"):
                        url = f"{base_url}/query"
                        payload = {"q": prompt, "thread_id": st.session_state.session_id}

                        # Attempt POST query, waking up server if sleeping on cold start
                        try:
                            response = requests.post(url, json=payload, timeout=60)
                        except Exception:
                            status.update(label="⚡ Server sleeping. Waking up Render backend...", state="running")
                            if wake_and_ping_backend(base_url, retries=6, retry_delay=5):
                                response = requests.post(url, json=payload, timeout=60)
                            else:
                                raise Exception("Backend server did not respond in time.")

                        if response.status_code != 200:
                            st.error(f"Backend Error: {response.status_code} - {response.text}")
                            st.stop()

                        data = response.json()

                    steps = data.get("thought_process", [])
                    for step in steps:
                        st.write(f"⚙️ {step}")

                    status.update(label="✅ Answer Synthesized", state="complete", expanded=False)

                except Exception as e:
                    logfire.error(f"❌ UI-Backend Connection Failed: {e}")
                    status.update(label="❌ Connection Failed", state="error")
                    st.error("Backend Offline or waking up. Please try again in 10 seconds.")
                    st.stop()

            # Final Answer Streaming
            answer_placeholder = st.empty()
            full_answer = data.get("answer", "No response.")

            curr_text = ""
            for char in full_answer:
                curr_text += char
                answer_placeholder.markdown(curr_text + "▌")
                time.sleep(0.005)

            answer_placeholder.markdown(full_answer)

            # Sources (Expandable)
            sources = data.get("sources", [])
            if sources:
                with st.expander(f"📄 View Retrieved Context ({len(sources)} sources)"):
                    for i, source in enumerate(sources):
                        preview = source[:100].replace("\n", " ") + "..."
                        with st.expander(f"Chunk {i+1}: {preview}"):
                            st.info(source)

            st.session_state.messages.append({"role": "assistant", "content": full_answer})
            logfire.info("✅ Chat cycle completed successfully.")
