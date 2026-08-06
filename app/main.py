# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import logfire
import os
import time
from dotenv import load_dotenv

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# Now safe to import app modules - logfire is already active
from fastapi import FastAPI, Response
from app.agents.graph import rag_agent
from app.guardrails import initialize_rails, guard

from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Enterprise Agentic RAG API")


@app.on_event("startup")
def startup_event():
    initialize_rails()


class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"


@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/graph")
def get_graph_image():
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}


def _call_guard(q: str):
    """
    Calls guard() and normalizes its return shape.
    Supports either:
      guard(q) -> (fired: bool, response: str)
      guard(q) -> (fired: bool, response: str, category: str)
    so this works whether or not your guardrails.py already tracks a category
    (your Logfire trace shows `category=GREETING`, so if guard() has that
    internally, just have it return it as a 3rd value and it'll show up here).
    """
    result = guard(q)
    if len(result) == 3:
        fired, response, category = result
    else:
        fired, response = result
        category = None
    return fired, response, category


@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    Returns a structured `guardrails` object and a timed `pipeline` trace
    in addition to the legacy `thought_process` / `sources` fields, so the
    UI can render guardrail status and per-stage timing without breaking
    older clients.
    """
    q = request.q
    thread_id = request.thread_id

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
    }

    config = {"configurable": {"thread_id": thread_id}}

    try:
        # ── Gate 1: NeMo Guardrails ──────────────────────────────
        t0 = time.perf_counter()
        rail_fired, rail_response, rail_category = _call_guard(q)
        guardrails_ms = round((time.perf_counter() - t0) * 1000)

        if rail_fired:
            logfire.info(
                f"🛡️ Request blocked by guardrails | thread={thread_id} | category={rail_category}"
            )
            guardrails = {
                "status": "blocked",
                "category": rail_category,
                "duration_ms": guardrails_ms,
            }
            pipeline = [
                {
                    "name": "Guardrails Check",
                    "detail": rail_category or "Blocked",
                    "duration_ms": guardrails_ms,
                    "status": "blocked",
                }
            ]
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": [],
                "guardrails": guardrails,
                "pipeline": pipeline,
            }

        guardrails = {
            "status": "passed",
            "category": None,
            "duration_ms": guardrails_ms,
        }

        # ── Gate 2: LangGraph RAG pipeline ───────────────────────
        t1 = time.perf_counter()
        final_output = rag_agent.invoke(initial_state, config=config)
        agent_ms = round((time.perf_counter() - t1) * 1000)

        plan = final_output.get("plan") or []
        pipeline = [
            {
                "name": "Guardrails Check",
                "detail": "",
                "duration_ms": guardrails_ms,
                "status": "ok",
            },
            {
                "name": "Agent Pipeline",
                "detail": " • ".join(plan) if plan else "",
                "duration_ms": agent_ms,
                "status": "ok",
            },
        ]

        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": plan,
            "status": final_output.get("status"),
            "sources": final_output.get("documents", []),
            "guardrails": guardrails,
            "pipeline": pipeline,
        }

    except Exception as e:
        logfire.error(f"❌ Backend Execution Failed: {e}")
        return {
            "question": q,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": [],
            "guardrails": None,
            "pipeline": [
                {"name": "Error", "detail": str(e)[:200], "status": "blocked"}
            ],
        }
