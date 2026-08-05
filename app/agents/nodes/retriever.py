import logfire
from app.agents.state import AgentState
from app.services.cache.upstash_service import (
    get_cache,
    set_cache,
    retrieval_cache_key,
    RETRIEVAL_TTL_SECONDS,
)
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.jina_reranker import rerank_documents


def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.
    """
    query = state["current_query"]
    cache_key = retrieval_cache_key(query)

    cached = get_cache(cache_key)
    if cached and isinstance(cached, dict) and cached.get("documents"):
        logfire.info(f"⚡ Upstash retrieval cache hit for: {query}")
        return {
            "documents": cached["documents"],
            "status": "Retrieved from Upstash cache.",
            "plan": state["plan"] + ["Retrieval Cache: Hit ⚡"],
        }

    with logfire.span("🔍 Knowledge Retrieval"):
        logfire.info(f"Searching Qdrant for: {query}")
        raw_results = search_enterprise_knowledge(query, limit=15)
        logfire.info(f"Retrieved {len(raw_results)} candidates from Vector DB")

        doc_contents = [doc["content"] for doc in raw_results]

        with logfire.span("⚖️ Semantic Reranking"):
            reranked_contents = rerank_documents(query, doc_contents, top_n=5)
            logfire.info("Reranking complete. Kept top 5 most relevant chunks.")

        formatted_docs = [f"CONTENT: {doc}" for doc in reranked_contents]

    set_cache(
        cache_key,
        {"documents": formatted_docs},
        ttl_seconds=RETRIEVAL_TTL_SECONDS,
    )

    return {
        "documents": formatted_docs,
        "status": "Found technical context.",
        "plan": state["plan"] + ["Context Retrieved"],
    }
