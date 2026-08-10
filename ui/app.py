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

# ─────────────────────────────────────────────
# ENV + LOGFIRE
# ─────────────────────────────────────────────
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=env_path)

try:
    token = os.getenv("LOGFIRE_TOKEN")
    logfire.configure(token=token)
    LOGFIRE_STATUS = "Connected & Tracing"
except Exception as e:  # noqa: BLE001
    LOGFIRE_STATUS = f"Standby (Error: {e})"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Kubernetes RAG Assistant",
    page_icon="☸️",
    layout="wide",
)

AI_AVATAR = "☸️"
USER_AVATAR = "👤"

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp { background-color: #0b1220; }

    /* Pipeline trace */
    .pipe-step {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 6px;
        background: rgba(255,255,255,0.03);
        border-left: 3px solid #326ce5;
    }
    .pipe-step.blocked { border-left-color: #e5484d; background: rgba(229,72,77,0.08); }
    .pipe-step.ok { border-left-color: #30a46c; }
    .pipe-icon { font-size: 1.1rem; }
    .pipe-name { font-weight: 600; color: #e6e6e6; }
    .pipe-detail { color: #9aa4b2; font-size: 0.85rem; }
    .pipe-time { margin-left: auto; color: #6b7684; font-size: 0.78rem; white-space: nowrap; }

    /* Guardrail badge */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .badge-pass { background: rgba(48,164,108,0.15); color: #30a46c; border: 1px solid #30a46c; }
    .badge-block { background: rgba(229,72,77,0.15); color: #e5484d; border: 1px solid #e5484d; }
    .badge-degraded { background: rgba(245,197,66,0.15); color: #f5c542; border: 1px solid #f5c542; }

    /* Chunk cards */
    .chunk-card {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        background: rgba(50,108,229,0.05);
    }
    .chunk-rank {
        display: inline-block;
        background: #326ce5;
        color: white;
        border-radius: 6px;
        padding: 1px 8px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 8px;
    }
    .chunk-score { color: #9aa4b2; font-size: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SESSION STATE INIT ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "backend_status" not in st.session_state:
    st.session_state.backend_status = "idle"  # idle | starting | online | failed


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
    if not url:
        return False
    target_url = f"{url.rstrip('/')}/"
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(
                target_url, headers=HTTP_HEADERS, timeout=15, allow_redirects=True
            )
            if r.status_code == 200:
                return True
            logfire.debug(
                f"Backend ping attempt {attempt}: status {r.status_code} (still booting)"
            )
        except RequestException as e:
            logfire.debug(f"Backend ping attempt {attempt}: connection error ({e})")
        if attempt < retries:
            time.sleep(retry_delay)
    return False


# ─────────────────────────────────────────────
# HELPERS FOR PIPELINE / GUARDRAILS / CHUNK RENDERING
# ─────────────────────────────────────────────
STEP_ICONS = {
    "guardrails": "🛡️",
    "planner": "🧠",
    "retrieval": "🔍",
    "reranking": "⚖️",
    "synthesis": "✍️",
    "cache": "🗃️",
    "default": "⚙️",
}


def icon_for(step_name: str) -> str:
    name = step_name.lower()
    for key, icon in STEP_ICONS.items():
        if key in name:
            return icon
    return STEP_ICONS["default"]


def render_guardrails_badge(guardrails: dict):
    """guardrails = {'status': 'passed'|'blocked'|'degraded', 'category': str, 'duration_ms': int}"""
    status = guardrails.get("status", "passed")
    category = guardrails.get("category")
    duration = guardrails.get("duration_ms")
    if status == "blocked":
        label = f'<span class="badge badge-block">🛡️ BLOCKED{f" · {category}" if category else ""}</span>'
    elif status == "degraded":
        label = '<span class="badge badge-degraded">🛡️ GUARDRAILS DEGRADED — FAILED OPEN</span>'
    else:
        label = '<span class="badge badge-pass">🛡️ GUARDRAILS PASSED</span>'
    if duration:
        label += f'  <span class="pipe-time">{duration} ms</span>'
    st.markdown(label, unsafe_allow_html=True)


def render_pipeline(pipeline: list):
    """
    pipeline = [
      {"name": "Guardrails Check", "detail": "...", "duration_ms": 871, "status": "ok"|"blocked"},
      ...
    ]
    """
    for step in pipeline:
        name = step.get("name", "Step")
        detail = step.get("detail", "")
        duration = step.get("duration_ms")
        status = step.get("status", "ok")
        css_class = "blocked" if status == "blocked" else "ok"
        time_html = f'<span class="pipe-time">{duration} ms</span>' if duration else ""
        st.markdown(
            f"""
            <div class="pipe-step {css_class}">
                <span class="pipe-icon">{icon_for(name)}</span>
                <span class="pipe-name">{name}</span>
                <span class="pipe-detail">{detail}</span>
                {time_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sources(sources: list):
    """
    Accepts either:
      - list[str]  (legacy: raw chunk text)
      - list[dict] with keys like text/content, score, rank, source/url
    """
    for i, source in enumerate(sources):
        if isinstance(source, dict):
            text = source.get("text") or source.get("content") or ""
            score = source.get("score")
            origin = (
                source.get("source")
                or source.get("url")
                or source.get("metadata", {}).get("source")
            )
        else:
            text = str(source)
            score = None
            origin = None

        preview = text[:110].replace("\n", " ").strip() + (
            "..." if len(text) > 110 else ""
        )
        score_html = (
            f'<span class="chunk-score">score {score:.3f}</span>'
            if isinstance(score, (int, float))
            else ""
        )

        with st.expander(f"Chunk {i + 1} — {preview}"):
            st.markdown(
                f'<span class="chunk-rank">#{i + 1}</span> {score_html}',
                unsafe_allow_html=True,
            )
            if origin:
                st.caption(f"Source: {origin}")
            st.info(text)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("☸️ Agent OS")
    st.caption("Kubernetes Docs RAG · LangGraph + Qdrant + Jina Rerank")
    st.markdown("---")

    if not base_url:
        st.error("⚠️ BACKEND_URL not configured!")
    elif st.session_state.backend_status == "idle":
        st.info(
            "**Backend is hosted on Render Free Tier.**\n\n"
            "Click the button below. A new tab will open to wake the backend. "
            "Return here while the app waits for it to come online."
        )
        st.link_button("🚀 Start Backend", base_url, use_container_width=True)
        st.caption(
            "After the new tab opens, come back to this page and click the button below."
        )
        if st.button("✅ Backend Started", type="primary", use_container_width=True):
            st.session_state.backend_status = "starting"
            st.rerun()
    elif st.session_state.backend_status == "starting":
        progress_slot = st.empty()
        for attempt in range(1, 16):
            elapsed = attempt * 5
            progress_slot.warning(
                f"⚡ Waking up backend... `{elapsed}s` elapsed\n\n"
                f"_Render free-tier cold start takes 30-90 seconds. Please wait..._"
            )
            try:
                r = requests.get(
                    f"{base_url}/",
                    headers=HTTP_HEADERS,
                    timeout=15,
                    allow_redirects=True,
                )
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
    elif st.session_state.backend_status == "online":
        st.success("✅ Backend Online")
        st.caption(f"API: {base_url}")
    elif st.session_state.backend_status == "failed":
        st.error(
            "❌ Backend did not respond after 75s.\n\nRender may be overloaded. Try starting again."
        )
        if st.button("🔄 Retry", type="secondary", use_container_width=True):
            st.session_state.backend_status = "starting"
            st.rerun()

    st.markdown("---")
    st.success(f"Logfire: {LOGFIRE_STATUS}")
    st.info(f"Session ID: `{st.session_state.session_id[:8]}`")

    st.markdown("---")
    show_pipeline_default = st.toggle("Show pipeline trace by default", value=True)
    show_chunks_default = st.toggle("Show retrieved chunks by default", value=False)

    st.markdown("---")
    if st.button("🗑️ Clear History & Memory", type="primary", use_container_width=True):
        logfire.warning(f"🗑️ Memory Wipe: session={st.session_state.session_id}")
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# ─────────────────────────────────────────────
# MAIN CHAT
# ─────────────────────────────────────────────
st.title("☸️ Kubernetes RAG Assistant")

if st.session_state.backend_status != "online":
    st.markdown("---")
    st.markdown("""
        ### 👋 Welcome to the Kubernetes RAG Assistant

        Ask questions about **Kubernetes** docs and get answers backed by a full
        **LangGraph** pipeline: guardrails → planning → retrieval → reranking → synthesis.

        **To get started:**
        1. Click **🚀 Start Backend** in the sidebar to wake up the Render server.
        2. Wait ~30-60 seconds for the cold start to complete.
        3. Start chatting!
        """)
    st.stop()

# Display chat history (replays stored pipeline/sources/guardrails per turn)
for message in st.session_state.messages:
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            guardrails = message.get("guardrails")
            pipeline = message.get("pipeline")
            sources = message.get("sources")
            if guardrails:
                render_guardrails_badge(guardrails)
            if pipeline:
                with st.expander("🔎 Pipeline trace", expanded=False):
                    render_pipeline(pipeline)
            if sources:
                with st.expander(
                    f"📄 Retrieved chunks ({len(sources)})", expanded=False
                ):
                    render_sources(sources)

# Chat Input
if prompt := st.chat_input("Ask about Kubernetes..."):
    with logfire.span(
        "💬 User Chat Interaction",
        user_query=prompt,
        session_id=st.session_state.session_id,
    ):

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=AI_AVATAR):
            data = {}
            with st.status("🔍 Agent is thinking...", expanded=True) as status:
                try:
                    with logfire.span("📡 Calling RAG Backend"):
                        query_url = f"{base_url}/query"
                        payload = {
                            "q": prompt,
                            "thread_id": st.session_state.session_id,
                        }

                        try:
                            response = requests.post(
                                query_url,
                                json=payload,
                                headers=HTTP_HEADERS,
                                timeout=60,
                            )
                        except RequestException as e:
                            logfire.debug(f"Initial POST failed, wake-and-retry: {e}")
                            status.update(
                                label="⚡ Backend sleeping. Retrying...",
                                state="running",
                            )
                            if wake_and_ping_backend(base_url):
                                try:
                                    response = requests.post(
                                        query_url,
                                        json=payload,
                                        headers=HTTP_HEADERS,
                                        timeout=60,
                                    )
                                except RequestException as e2:
                                    logfire.error(f"Retry POST failed: {e2}")
                                    raise BackendConnectionError(
                                        "Backend POST retry failed."
                                    ) from e2
                            else:
                                st.session_state.backend_status = "failed"
                                raise BackendTimeoutError(
                                    "Backend did not respond in time."
                                ) from e

                        if response.status_code != 200:
                            st.error(
                                f"Backend Error: {response.status_code} - {response.text[:300]}"
                            )
                            st.stop()

                        try:
                            data = response.json()
                        except JSONDecodeError as e:
                            logfire.error(f"Invalid JSON from backend: {e}")
                            st.error("Backend sent invalid JSON.")
                            st.stop()

                    # Live step-by-step trace inside the "thinking" status box.
                    # Supports a rich `pipeline` array; falls back to legacy `thought_process`.
                    guardrails = data.get("guardrails")
                    pipeline = data.get("pipeline") or [
                        {"name": step, "status": "ok"}
                        for step in data.get("thought_process", [])
                    ]

                    if guardrails:
                        render_guardrails_badge(guardrails)
                    if pipeline:
                        render_pipeline(pipeline)

                    blocked = bool(guardrails) and guardrails.get("status") == "blocked"
                    if blocked:
                        status.update(
                            label="🛡️ Blocked by guardrails",
                            state="error",
                            expanded=True,
                        )
                    else:
                        status.update(
                            label="✅ Answer Synthesized",
                            state="complete",
                            expanded=show_pipeline_default,
                        )

                except (
                    BackendConnectionError,
                    BackendTimeoutError,
                    RequestException,
                ) as e:
                    logfire.error(f"❌ UI-Backend Connection Failed: {e}")
                    status.update(label="❌ Connection Failed", state="error")
                    st.error(
                        "Backend went offline. Click **🔄 Retry** in the sidebar to reconnect."
                    )
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

            # Sources / retrieved chunks
            sources = data.get("sources", [])
            if sources:
                with st.expander(
                    f"📄 Retrieved chunks ({len(sources)})",
                    expanded=show_chunks_default,
                ):
                    render_sources(sources)

            # Persist everything so history replay shows the same trace
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_answer,
                    "guardrails": guardrails,
                    "pipeline": pipeline,
                    "sources": sources,
                }
            )
            logfire.info("✅ Chat cycle completed.")
