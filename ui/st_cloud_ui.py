import os
import time
import uuid
from json import JSONDecodeError

import logfire
import requests
import streamlit as st
from dotenv import load_dotenv
from requests.exceptions import RequestException

try:
    from ui.errors import BackendConnectionError, BackendTimeoutError
except ImportError:
    from errors import BackendConnectionError, BackendTimeoutError

load_dotenv()

# --- BROWSER HEADERS ---
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}

# Initialize Logfire
try:
    logfire.configure(token=st.secrets.get("LOGFIRE_TOKEN", os.getenv("LOGFIRE_TOKEN")))
    logfire.instrument_requests()
    LOGFIRE_STATUS = "Connected & Tracing"
except (AttributeError, KeyError, Exception):
    LOGFIRE_STATUS = "Standby (No Token)"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="🤖",
    layout="wide",
)

# --- AVATARS ---
AI_AVATAR = "🤖"
USER_AVATAR = "👤"

# --- SESSION STATE INIT ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# "idle" | "starting" | "online" | "failed"
if "backend_status" not in st.session_state:
    st.session_state.backend_status = "idle"


# --- BACKEND URL ---
def get_backend_url() -> str:
    try:
        url = st.secrets.get("BACKEND_URL")
        if url:
            return url.rstrip("/")
    except (AttributeError, KeyError, FileNotFoundError):
        pass
    return os.getenv("BACKEND_URL", "").rstrip("/")

base_url = get_backend_url()


def wake_and_ping_backend(url: str, retries: int = 15, retry_delay: int = 5) -> bool:
    """
    Pings backend health endpoint, treating 502 as 'still booting' and retrying.
    Returns True only on HTTP 200.
    """
    if not url:
        return False
    target_url = f"{url.rstrip('/')}/"
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(target_url, headers=HTTP_HEADERS, timeout=15, allow_redirects=True)
            if r.status_code == 200:
                return True
            logfire.debug(f"Backend ping attempt {attempt}: status {r.status_code} (still booting)")
        except RequestException as e:
            logfire.debug(f"Backend ping attempt {attempt}: connection error ({e})")
        if attempt < retries:
            time.sleep(retry_delay)
    return False


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Agent OS")
    st.markdown("---")

    if not base_url:
        st.error("⚠️ BACKEND_URL not configured!")

    # ── IDLE: Show the "Start Backend" button ──
    elif st.session_state.backend_status == "idle":
        st.info(
            "**Backend is on Render Free Tier.**\n\n"
            "It spins down when idle. Click the button below to wake it up before chatting."
        )
        if st.button("🚀 Start Backend", type="primary", use_container_width=True):
            st.session_state.backend_status = "starting"
            st.rerun()

    # ── STARTING: Show live wake-up progress ──
    elif st.session_state.backend_status == "starting":
        progress_slot = st.empty()
        for attempt in range(1, 16):
            elapsed = attempt * 5
            progress_slot.warning(
                f"⚡ Waking up backend... `{elapsed}s` elapsed\n\n"
                f"_Render free-tier cold start takes 30-90 seconds. Please wait..._"
            )
            try:
                r = requests.get(f"{base_url}/", headers=HTTP_HEADERS, timeout=15, allow_redirects=True)
                if r.status_code == 200:
                    st.session_state.backend_status = "online"
                    progress_slot.empty()
                    st.rerun()
                    break
                logfire.debug(f"Ping {attempt}: status {r.status_code}")
            except RequestException as e:
                logfire.debug(f"Ping {attempt}: error {e}")
            if attempt < 15:
                time.sleep(5)
        else:
            st.session_state.backend_status = "failed"
            progress_slot.empty()
            st.rerun()

    # ── ONLINE ──
    elif st.session_state.backend_status == "online":
        st.success("✅ Backend Online")
        st.caption(f"API: {base_url}")

    # ── FAILED ──
    elif st.session_state.backend_status == "failed":
        st.error(
            "❌ Backend did not respond after 75s.\n\n"
            "Render may be overloaded. Try starting again."
        )
        if st.button("🔄 Retry", type="secondary", use_container_width=True):
            st.session_state.backend_status = "starting"
            st.rerun()

    st.markdown("---")
    st.success(f"Logfire: {LOGFIRE_STATUS}")
    st.info(f"Memory ID: `{st.session_state.session_id[:8]}`")

    if st.button("🗑️ Clear History & Memory", type="primary", use_container_width=True):
        logfire.warning(f"🗑️ Memory Wipe: session={st.session_state.session_id}")
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


# ─────────────────────────────────────────────
# MAIN CHAT
# ─────────────────────────────────────────────
st.title("🤖 Enterprise Agentic Assistant")

# If backend not online, show welcome screen
if st.session_state.backend_status != "online":
    st.markdown("---")
    st.markdown(
        """
        ### 👋 Welcome to the Enterprise Agentic RAG Assistant

        This assistant answers questions on **Kubernetes**, **Intel Hardware**, and **Enterprise Networking**
        using a full **LangGraph RAG pipeline** with semantic reranking and memory.

        **To get started:**
        1. Click **🚀 Start Backend** in the sidebar to wake up the Render server.
        2. Wait ~30-60 seconds for the cold start to complete.
        3. Start chatting!
        """
    )
    st.stop()

# Display chat history
for message in st.session_state.messages:
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask about your documentation..."):
    with logfire.span("💬 User Chat Interaction", user_query=prompt, session_id=st.session_state.session_id):

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=AI_AVATAR):
            data = {}
            with st.status("🔍 Agent is thinking...", expanded=True) as status:
                try:
                    with logfire.span("📡 Calling RAG Backend"):
                        query_url = f"{base_url}/query"
                        payload = {"q": prompt, "thread_id": st.session_state.session_id}

                        try:
                            response = requests.post(query_url, json=payload, headers=HTTP_HEADERS, timeout=60)
                        except RequestException as e:
                            logfire.debug(f"Initial POST failed, wake-and-retry: {e}")
                            status.update(label="⚡ Backend sleeping. Retrying...", state="running")
                            if wake_and_ping_backend(base_url):
                                try:
                                    response = requests.post(query_url, json=payload, headers=HTTP_HEADERS, timeout=60)
                                except RequestException as e2:
                                    logfire.error(f"Retry POST failed: {e2}")
                                    raise BackendConnectionError("Backend POST retry failed.") from e2
                            else:
                                st.session_state.backend_status = "failed"
                                raise BackendTimeoutError("Backend did not respond in time.") from e

                        if response.status_code != 200:
                            st.error(f"Backend Error: {response.status_code} - {response.text[:300]}")
                            st.stop()

                        try:
                            data = response.json()
                        except JSONDecodeError as e:
                            logfire.error(f"Invalid JSON from backend: {e}")
                            st.error("Backend sent invalid JSON.")
                            st.stop()

                    for step in data.get("thought_process", []):
                        st.markdown(f"⚙️ {step}", unsafe_allow_html=False)

                    status.update(label="✅ Answer Synthesized", state="complete", expanded=False)

                except (BackendConnectionError, BackendTimeoutError, RequestException) as e:
                    logfire.error(f"❌ UI-Backend Connection Failed: {e}")
                    status.update(label="❌ Connection Failed", state="error")
                    st.error("Backend went offline. Click **🔄 Retry** in the sidebar to reconnect.")
                    st.stop()

            # Streaming answer
            answer_placeholder = st.empty()
            full_answer = data.get("answer", "No response.")
            curr_text = ""
            for char in full_answer:
                curr_text += char
                answer_placeholder.markdown(curr_text + "▌")
                time.sleep(0.005)
            answer_placeholder.markdown(full_answer)

            # Sources
            sources = data.get("sources", [])
            if sources:
                with st.expander(f"📄 Retrieved Context ({len(sources)} chunks)"):
                    for i, source in enumerate(sources):
                        st.caption(f"Chunk {i + 1}")
                        st.info(source)
            else:
                st.caption("ℹ️ No context retrieved — conversational response.")

            st.session_state.messages.append({"role": "assistant", "content": full_answer})
            logfire.info("✅ Chat cycle completed.")