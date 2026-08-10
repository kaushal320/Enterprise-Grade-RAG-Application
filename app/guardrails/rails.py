
import logfire
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from app.config import settings
from app.guardrails.colang_rules import GUARD_PROMPT

# Guardrail result statuses returned by guard() as the 4th tuple element.
STATUS_BLOCKED = "blocked"      # a rail fired; the query was rejected
STATUS_CLEAN = "clean"          # guardrails ran and the query passed
STATUS_ERROR = "error"          # guardrails errored and defaulted to pass (fail-open)
STATUS_INACTIVE = "inactive"    # no guardrail chain available (no key / init failed)


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
    response: str | None = Field(
        default=None,
        description="The clean response string to return to the user if is_blocked is True. Must be null if category is CLEAN."
    )


_guard_chain = None
_rails_healthy = False  # False on init failure / runtime error; True after a successful check


def initialize_rails() -> None:
    """
    Initialize Guardrails AI with Pydantic structured validation using Groq llama-3.1-8b-instant.
    Reads rules and prompts directly from app/guardrails/colang_rules.py.
    Memory footprint: ~30MB RAM (0 MB local PyTorch/SentenceTransformers download).
    """
    global _guard_chain, _rails_healthy

    try:
        if settings.GROQ_API_KEY:
            llm = ChatGroq(
                api_key=settings.GROQ_API_KEY,
                model="llama-3.1-8b-instant",
                temperature=0
            )
            _guard_chain = llm.with_structured_output(GuardrailResult)
            _rails_healthy = True
            logfire.info("🛡️ Guardrails AI initialised (llama-3.1-8b-instant + Pydantic validation).")
        else:
            logfire.warning("⚠️ GROQ_API_KEY missing — Guardrails inactive.")
            _rails_healthy = False
    except Exception as e:
        logfire.error(f"⚠️ Guardrails AI init failed: {e}")
        _guard_chain = None
        _rails_healthy = False


def rails_healthy() -> bool:
    """
    True if the guardrail chain is initialized and the last invocation succeeded.
    Exposed via the /health endpoint so operators can see when rails are degraded.
    """
    return _rails_healthy


def guard(message: str) -> tuple[bool, str | None, str | None, str]:
    """
    Validate a user message using the rules defined in colang_rules.py.

    Returns a 4-tuple: (fired, response, category, status)

      fired=True   — a rail fired; `response` is the clean validated reply,
                     `category` is one of GREETING/CAPABILITIES/FAREWELL/JAILBREAK/OFF_TOPIC,
                     `status` is STATUS_BLOCKED.
      fired=False  — query allowed; `status` distinguishes WHY so callers know
                     whether the rails actually cleared it:
                     STATUS_CLEAN    — guardrails ran and passed.
                     STATUS_ERROR    — guardrails errored and defaulted to pass (fail-open).
                     STATUS_INACTIVE — no guardrail chain available (missing key / init failed).
    """
    global _rails_healthy

    if _guard_chain is None or not message:
        _rails_healthy = False
        return False, None, None, STATUS_INACTIVE

    with logfire.span("🛡️ Guardrails Check"):
        try:
            res: GuardrailResult = _guard_chain.invoke(GUARD_PROMPT.format(query=message))
            _rails_healthy = True

            if res.is_blocked and res.response:
                logfire.info(f"🛡️ Guardrails fired | category={res.category}")
                return True, res.response.strip(), res.category, STATUS_BLOCKED

            logfire.info("✅ Guardrails passed.")
            return False, None, None, STATUS_CLEAN
        except Exception as e:
            _rails_healthy = False
            logfire.error(f"⚠️ Guardrails check error: {e}")
            return False, None, None, STATUS_ERROR
