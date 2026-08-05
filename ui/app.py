import os
import streamlit as st
import requests
from requests.exceptions import RequestException
from json import JSONDecodeError
import time
import uuid
import logfire
from dotenv import load_dotenv

try:
    from ui.errors import BackendConnectionError, BackendTimeoutError
except ImportError:
    from errors import BackendConnectionError, BackendTimeoutError

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
except Exception as e:  # noqa: BLE001
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

# --- BROWSER HEADERS (Bypasses Cloudflare bot filtering on Render) ---
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}

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
    except (AttributeError, KeyError, FileNotFoundError):
        pass
    return os.getenv("BACKEND_URL", "").rstrip("/")

base_url = get_backend_url()

def wake_and_ping_backend(
    url: str,
    retries: int = 15,
    retry_delay: int = 5,
    status_placeholder=None,
) -> bool:
    """
    Pings backend health endpoint, triggering Render container cold-start.
    - Render free-tier takes 50-90 seconds to fully boot.
    - A 502 response means the container received the request and IS booting — keep retrying.
    - Only returns True when the backend responds with HTTP 200.
    """
    if not url:
        return False
    target_url = f"{url.rstrip('/')}/"
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(target_url, headers=HTTP_HEADERS, timeout=15, allow_redirects=True)
            if r.status_code == 200:
                return True
            # 502/503/504 = container is booting but Cloudflare/Render proxy received request
            # Keep pinging — it will flip to 200 once the app finishes startup
            logfire.debug(f"Backend ping attempt {attempt}: status {r.status_code} (still booting)")
        except RequestException as e:
            logfire.debug(f"Backend ping attempt {attempt}: connection error ({e})")
        if status_placeholder:
            elapsed = attempt * retry_delay
            status_placeholder.warning(
                f"⚡ Backend waking up... ({elapsed}s elapsed, attempt {attempt}/{retries})\n\n"
                f"Render free-tier cold start takes up to 60-90 seconds. Please wait."
            )
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
        _status_placeholder = st.empty()
        _status_placeholder.info("⚡ Pinging backend (waking Render container)...")
        backend_ok = wake_and_ping_backend(
            base_url,
            retries=15,
            retry_delay=5,
            status_placeholder=_status_placeholder,
        )
        _status_placeholder.empty()  # Clear the live status widget
        if backend_ok:
            st.success("✅ Backend Online")
        else:
            st.error(
                "❌ Backend did not respond after 75s.\n\n"
                "The Render container may be overloaded. Try refreshing the page."
            )
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
                            response = requests.post(url, json=payload, headers=HTTP_HEADERS, timeout=60)
                        except RequestException as e:
                            logfire.debug(f"Initial POST failed, attempting wake-and-retry: {e}")
                            status.update(label="⚡ Server sleeping. Waking up Render backend...", state="running")
                            if wake_and_ping_backend(base_url, retries=8, retry_delay=4):
                                try:
                                    response = requests.post(url, json=payload, headers=HTTP_HEADERS, timeout=60)
                                except RequestException as e2:
                                    logfire.error(f"Retry POST failed: {e2}")
                                    raise BackendConnectionError("Backend POST retry failed.") from e2
                            else:
                                raise BackendTimeoutError("Backend server did not respond in time.") from e

                        if response.status_code != 200:
                            st.error(f"Backend Error: {response.status_code} - {response.text}")
                            st.stop()

                        try:
                            data = response.json()
                        except JSONDecodeError as e:
                            logfire.error(f"Invalid JSON received from backend: {e}")
                            st.error("Backend sent invalid JSON response.")
                            st.stop()

                    steps = data.get("thought_process", [])
                    for step in steps:
                        st.write(f"⚙️ {step}")

                    status.update(label="✅ Answer Synthesized", state="complete", expanded=False)

                except (BackendConnectionError, BackendTimeoutError, RequestException) as e:
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
