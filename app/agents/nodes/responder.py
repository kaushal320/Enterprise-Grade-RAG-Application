import logfire

from app.agents.state import AgentState
from app.config import settings
from app.gateway import extract_cache_status, portkey_client
from app.services.cache.upstash_service import (
    RESPONSE_TTL_SECONDS,
    get_cache,
    response_cache_key,
    set_cache,
)


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the native Portkey client (not LangChain) so we can read the
    x-portkey-cache-status response header and surface Cache: Hit in the UI.
    """
    query = state["current_query"]

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        context_for_key = history_str
    else:
        context_for_key = "\n".join(state["documents"])

    cache_key = response_cache_key(user_msg, context_for_key)
    cached_response = get_cache(cache_key)
    if cached_response:
        content = (
            cached_response
            if isinstance(cached_response, str)
            else cached_response.get("final_answer", "")
        )
        if content:
            logfire.info("⚡ Upstash response cache hit — skipping LLM synthesis.")
            return {
                "final_answer": content,
                "status": "Upstash cache hit — instant response.",
                "plan": state["plan"] + ["Response Cache: Hit ⚡"],
                "messages": [{"role": "assistant", "content": content}],
            }

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    with logfire.span("✍️ LLM Synthesis"):
        try:
            if settings.PORTKEY_API_KEY and settings.GROQ_SLUG:
                response = portkey_client.chat.completions.create(
                    model=f"@{settings.GROQ_SLUG}/openai/gpt-oss-20b",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                content = response.choices[0].message.content
                cache_status = extract_cache_status(response)
                is_cache_hit = cache_status == "HIT"
            else:
                raise ValueError("Portkey credentials or Groq slug not provided")
        except Exception as e:
            logfire.warning(f"Portkey LLM call failed ({e}), using ChatGroq fallback.")
            from langchain_groq import ChatGroq
            fallback_llm = ChatGroq(
                api_key=settings.GROQ_API_KEY,
                model="openai/gpt-oss-20b",
                temperature=0.1
            )
            content = fallback_llm.invoke(prompt).content
            is_cache_hit = False

        if is_cache_hit:
            logfire.info(
                "⚡ Gateway Cache Hit — response served from Portkey cache."
            )
            plan_update = state["plan"] + ["Cache: Hit ⚡"]
            status = "Cache hit — instant response."
        else:
            logfire.info("✅ Response synthesised via LLM.")
            plan_update = state["plan"]
            status = "Response generated."

        set_cache(cache_key, content, ttl_seconds=RESPONSE_TTL_SECONDS)

        return {
            "final_answer": content,
            "status": status,
            "plan": plan_update,
            "messages": [{"role": "assistant", "content": content}],
        }
