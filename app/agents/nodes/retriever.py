import hashlib

import logfire

from app.agents.state import AgentState
from app.services.cache.upstash_service import (
    RETRIEVAL_TTL_SECONDS,
    get_cache,
    retrieval_cache_key,
    set_cache,
)
from app.services.retrieval.jina_reranker import rerank_documents
from app.services.retrieval.qdrant_service import (
    bm25_search,
    search_enterprise_knowledge,
)

_RRF_K = 60


def _chunk_id(doc: dict) -> str:
    """Stable dedup key for a retrieved chunk (its normalized text)."""
    return hashlib.sha256(doc["content"].encode("utf-8")).hexdigest()


def _hybrid_retrieve(query: str, limit: int = 15):
    """Vector search + BM25 merged via Reciprocal Rank Fusion, deduped by chunk id.

    Falls back to vector-only (BM25 returns []) when the index is missing.
    Returns (merged_docs, qdrant_count, bm25_count).
    """
    vector_results = search_enterprise_knowledge(query, limit=limit)
    bm25_results = bm25_search(query, limit=limit)

    logfire.info("hybrid_retrieve", bm25_count=len(bm25_results), qdrant_count=len(vector_results))

    fused: dict[str, dict] = {}
    rrf_scores: dict[str, float] = {}
    for results in (vector_results, bm25_results):
        for rank, doc in enumerate(results, start=1):
            cid = _chunk_id(doc)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
            if cid not in fused:
                fused[cid] = {
                    "content": doc["content"],
                    "source": doc.get("source", "Unknown"),
                    "score": doc.get("score", 0.0),
                }

    merged = sorted(
        fused.values(),
        key=lambda d: rrf_scores[_chunk_id(d)],
        reverse=True,
    )
    return merged[:limit], len(vector_results), len(bm25_results)


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
        logfire.info(f"Searching hybrid (vector + BM25) for: {query}")
        raw_results, qdrant_count, bm25_count = _hybrid_retrieve(query, limit=15)
        logfire.info(f"Retrieved {len(raw_results)} candidates via Hybrid Retrieval")

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
        "retrieval_counts": {"qdrant": qdrant_count, "bm25": bm25_count},
    }
