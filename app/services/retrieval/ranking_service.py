"""Jina-hosted reranker proxy module.

Delegates document reranking to Jina Reranker API.
"""

from app.services.retrieval.jina_reranker import rerank_documents

__all__ = ["rerank_documents"]
