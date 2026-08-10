import os
import pickle
import re

import logfire
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from rank_bm25 import BM25Okapi

from app.config import settings
from app.services.retrieval.jina_embedding import embed_query


# Initialize Qdrant Client
client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY
)

# Pickled BM25 index written by the ingestion pipeline (app.injection.processor)
BM25_INDEX_PATH = os.path.join("processed_data", "bm25_index.pkl")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization shared by BM25 build and query sides."""
    return _TOKEN_RE.findall(text.lower())


_BM25_CACHE: dict = {"index": None, "chunks": None, "mtime": None}


def _load_bm25_index():
    """Return (BM25Okapi, chunks), loaded once and refreshed when the pickle changes."""
    if not os.path.exists(BM25_INDEX_PATH):
        return None, None
    mtime = os.path.getmtime(BM25_INDEX_PATH)
    if _BM25_CACHE["index"] is not None and _BM25_CACHE["mtime"] == mtime:
        return _BM25_CACHE["index"], _BM25_CACHE["chunks"]
    try:
        with open(BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        _BM25_CACHE["index"] = data["index"]
        _BM25_CACHE["chunks"] = data["chunks"]
        _BM25_CACHE["mtime"] = mtime
        return _BM25_CACHE["index"], _BM25_CACHE["chunks"]
    except Exception as e:
        logfire.error(f"❌ Failed to load BM25 index: {e}")
        return None, None


def bm25_search(query: str, limit: int = 8):
    """
    Lexical BM25 search over the pickled chunk corpus. Returns the same document
    shape as search_enterprise_knowledge so callers can fuse both rankings.
    Returns [] (vector-only fallback) if the BM25 index is unavailable.
    """
    try:
        index, chunks = _load_bm25_index()
        if index is None or chunks is None:
            return []
        scores = index.get_scores(tokenize(query))
        top_indices = np.argsort(scores)[::-1][:limit]
        results = []
        for i in top_indices:
            if scores[i] <= 0:
                continue
            chunk = chunks[i]
            results.append({
                "content": chunk["text"],
                "source": chunk.get("source", "Unknown"),
                "score": float(scores[i]),
            })
        return results
    except Exception as e:
        logfire.error(f"❌ BM25 Search Failed: {e}")
        return []

def search_enterprise_knowledge(query: str, limit: int = 8):
    """
    Performs a high-precision search in the enterprise knowledge base.
    Uses the modern query_points interface.
    """
    try:
        query_vector = embed_query(query)

        # Using query_points - the modern standard for Qdrant
        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True # JSON
        )

        results = []
        for res in response.points:
            results.append({
                "content": res.payload.get("text", ""),
                "source": res.payload.get("source", "Unknown"),
                "score": res.score
            })
        
        return results
    except Exception as e:
        logfire.error(f"❌ Qdrant Search Failed: {e}")
        return []
