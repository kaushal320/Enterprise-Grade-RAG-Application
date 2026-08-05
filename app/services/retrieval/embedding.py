"""Jina-hosted embeddings proxy module.

Delegates embedding calls to Jina AI services.
"""

from app.services.retrieval.jina_embedding import (
    embed_query,
    embed_texts,
    get_embedding_dim,
)

__all__ = ["embed_query", "embed_texts", "get_embedding_dim"]
