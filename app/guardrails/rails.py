import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS

_rails: LLMRails | None = None


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup using Colang rules.
    Uses llama-3.1-8b-instant for fast intent classification at the gate.
    """
    global _rails

    try:
        if settings.GROQ_API_KEY:
            guard_llm = ChatGroq(
                api_key=settings.GROQ_API_KEY,
                model="llama-3.1-8b-instant",
                temperature=0
            )

            config = RailsConfig.from_content(
                colang_content=COLANG_CONTENT,
                yaml_content=YAML_CONTENT
            )

            _rails = LLMRails(config, llm=guard_llm)
            logfire.info("🛡️ NeMo Guardrails initialised with Colang rules (llama-3.1-8b-instant).")
        else:
            logfire.warning("⚠️ GROQ_API_KEY missing — NeMo Guardrails inactive.")
    except Exception as e:
        logfire.error(f"⚠️ NeMo Guardrails init failed: {e}")
        _rails = None


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo Colang rules engine.

    Returns:
        (True,  rail_response) — a rail fired; return clean response text immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean/technical; proceed to LangGraph.
    """
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        try:
            result = _rails.generate(messages=[{"role": "user", "content": message}])
            
            content = result.get("content", "") if isinstance(result, dict) else str(result)
            content = content.strip()

            # Clean output if any prefix or duplicate text is produced
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            for line in lines:
                # Strip prefix like "GREETING: " or "bot express greeting "
                clean_line = line
                if ":" in clean_line and not clean_line.startswith("http"):
                    clean_line = clean_line.split(":", 1)[1].strip()

                if any(indicator in clean_line or indicator in line for indicator in RAIL_INDICATORS):
                    logfire.info(f"🛡️ NeMo Guardrails fired | query='{message[:80]}'")
                    return True, clean_line if clean_line else line

            # Check if any rail indicator matched in full content
            if any(indicator in content for indicator in RAIL_INDICATORS):
                logfire.info(f"🛡️ NeMo Guardrails fired | query='{message[:80]}'")
                return True, content

            logfire.info("✅ Guardrails passed.")
            return False, None
        except Exception as e:
            logfire.error(f"⚠️ Guardrails evaluation error: {e}")
            return False, None
