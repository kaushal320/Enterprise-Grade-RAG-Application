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
from app.guardrails import initialize_rails, guard, rails_healthy

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


@app.get("/health")
def health():
    """
    Liveness/readiness probe. Reports whether the guardrail layer is healthy
    so operators (and the UI badge) can see when rails are degraded/fail-open.
    """
    return {
        "status": "ok",
        "service": "enterprise-rag-api",
        "guardrails": {"healthy": rails_healthy()},
    }


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
    Supports:
      guard(q) -> (fired: bool, response: str)
      guard(q) -> (fired: bool, response: str, category: str)
      guard(q) -> (fired: bool, response: str, category: str, status: str)
    so this works with both the current 4-tuple and any older guard() that
    returns fewer fields. `status` is one of blocked/clean/error/inactive.
    """
    result = guard(q)
    fired = result[0]
    response = result[1]
    category = result[2] if len(result) >= 3 else None
    status = result[3] if len(result) >= 4 else ("blocked" if fired else "clean")
    return fired, response, category, status


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
        # ── Gate 1: Guardrails (Groq llama-3.1-8b-instant + Pydantic classifier) ──
        t0 = time.perf_counter()
        rail_fired, rail_response, rail_category, rail_status = _call_guard(q)
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
                "retrieval": None,
                "guardrails": guardrails,
                "pipeline": pipeline,
            }

        # Fail-open visibility: the query is allowed even if the rail errored or
        # is inactive, but surface a "degraded" state so callers/ops can see it.
        guardrail_state = (
            "degraded" if rail_status in ("error", "inactive") else "passed"
        )
        guardrails = {
            "status": guardrail_state,
            "category": rail_category,
            "detail": rail_status if guardrail_state == "degraded" else None,
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
            "retrieval": final_output.get("retrieval_counts"),
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
            "retrieval": None,
            "guardrails": None,
            "pipeline": [
                {"name": "Error", "detail": str(e)[:200], "status": "blocked"}
            ],
        }
