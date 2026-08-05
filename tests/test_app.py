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
