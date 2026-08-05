import logfire
from typing import Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

from app.config import settings
from app.guardrails.colang_rules import GUARD_PROMPT


class GuardrailResult(BaseModel):
    """
    Pydantic Guardrail validation schema for structured intent verification.
    """
    is_blocked: bool = Field(
        description="True if query is a greeting, farewell, capabilities request, jailbreak attempt, or off-topic question."
    )
    category: str = Field(
        description="One of: 'GREETING', 'CAPABILITIES', 'FAREWELL', 'JAILBREAK', 'OFF_TOPIC', 'CLEAN'"
    )
    response: Optional[str] = Field(
        default=None,
        description="The clean response string to return to the user if is_blocked is True. Must be null if category is CLEAN."
    )


_guard_chain = None


def initialize_rails() -> None:
    """
    Initialize Guardrails AI with Pydantic structured validation using Groq llama-3.1-8b-instant.
    Reads rules and prompts directly from app/guardrails/colang_rules.py.
    Memory footprint: ~30MB RAM (0 MB local PyTorch/SentenceTransformers download).
    """
    global _guard_chain

    try:
        if settings.GROQ_API_KEY:
            llm = ChatGroq(
                api_key=settings.GROQ_API_KEY,
                model="llama-3.1-8b-instant",
                temperature=0
            )
            _guard_chain = llm.with_structured_output(GuardrailResult)
            logfire.info("🛡️ Guardrails AI initialised (llama-3.1-8b-instant + Pydantic validation).")
        else:
            logfire.warning("⚠️ GROQ_API_KEY missing — Guardrails inactive.")
    except Exception as e:
        logfire.error(f"⚠️ Guardrails AI init failed: {e}")
        _guard_chain = None


def guard(message: str) -> tuple[bool, str | None]:
    """
    Validate a user message using the rules defined in colang_rules.py.

    Returns:
        (True,  clean_response) — a rail fired; return clean validated response,
                                 skip RAG pipeline.
        (False, None)           — query is clean/technical; proceed to LangGraph RAG.
    """
    if _guard_chain is None or not message:
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        try:
            res: GuardrailResult = _guard_chain.invoke(GUARD_PROMPT.format(query=message))
            
            if res.is_blocked and res.response:
                logfire.info(f"🛡️ Guardrails fired | category={res.category}")
                return True, res.response.strip()

            logfire.info("✅ Guardrails passed.")
            return False, None
        except Exception as e:
            logfire.error(f"⚠️ Guardrails check error: {e}")
            return False, None
