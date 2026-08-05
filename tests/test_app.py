import pytest
import os
from app.config import settings


def test_config_loading():
    """Verify Settings initializes properly without error."""
    assert hasattr(settings, "QDRANT_COLLECTION")
    assert hasattr(settings, "JINA_EMBEDDING_MODEL")
    assert hasattr(settings, "GROQ_MODEL")


def test_guardrails_import():
    """Verify guardrails gate function imports and runs clean check."""
    from app.guardrails.rails import guard
    is_blocked, response = guard("")
    assert is_blocked is False
    assert response is None


def test_graph_import():
    """Verify LangGraph workflow compiles successfully."""
    from app.agents.graph import rag_agent
    assert rag_agent is not None


def test_cache_key_helpers():
    """Verify stable Upstash cache key generation."""
    from app.services.cache.upstash_service import (
        retrieval_cache_key,
        response_cache_key,
    )

    assert retrieval_cache_key("Kubernetes pods") == retrieval_cache_key("kubernetes pods")
    assert retrieval_cache_key("Kubernetes pods") != retrieval_cache_key("Docker containers")

    assert response_cache_key("What is a pod?", "ctx-a") != response_cache_key(
        "What is a pod?", "ctx-b"
    )


def test_retriever_uses_upstash_cache(monkeypatch):
    """Retriever should return cached documents without hitting Qdrant."""
    from app.agents.nodes.retriever import retrieve_node

    cached_docs = ["CONTENT: cached chunk"]
    monkeypatch.setattr(
        "app.agents.nodes.retriever.get_cache",
        lambda key: {"documents": cached_docs},
    )

    def fail_search(*args, **kwargs):
        raise AssertionError("Qdrant search should not run on cache hit")

    monkeypatch.setattr("app.agents.nodes.retriever.search_enterprise_knowledge", fail_search)

    result = retrieve_node(
        {
            "current_query": "kubernetes networking",
            "plan": ["Intent: Technical"],
        }
    )

    assert result["documents"] == cached_docs
    assert "Retrieval Cache: Hit" in result["plan"][-1]


def test_responder_uses_upstash_cache(monkeypatch):
    """Responder should return cached answer without calling the LLM."""
    from app.agents.nodes.responder import generate_node

    monkeypatch.setattr(
        "app.agents.nodes.responder.get_cache",
        lambda key: "Cached answer from Upstash",
    )

    def fail_llm(*args, **kwargs):
        raise AssertionError("LLM should not run on cache hit")

    monkeypatch.setattr("app.agents.nodes.responder.portkey_client", fail_llm)

    result = generate_node(
        {
            "current_query": "CONVERSATIONAL",
            "messages": [{"role": "user", "content": "Hello"}],
            "documents": [],
            "plan": ["Intent: Conversational/Memory"],
        }
    )

    assert result["final_answer"] == "Cached answer from Upstash"
    assert "Response Cache: Hit" in result["plan"][-1]
