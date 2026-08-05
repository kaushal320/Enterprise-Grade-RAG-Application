"""Jina-hosted reranking for retrieved Qdrant passages."""

import time

import logfire
import requests

from app.config import settings


_RERANK_URL = "https://api.jina.ai/v1/rerank"


def rerank_documents(query: str, documents: list[str], top_n: int = 5) -> list[str]:
    """Return the most relevant documents, preserving Qdrant order on API failure."""
    if not documents:
        return []
    if not settings.JINA_API_KEY:
        logfire.warning("JINA_API_KEY is not configured; skipping reranking.")
        return documents[:top_n]

    payload = {
        "model": settings.JINA_RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": min(top_n, len(documents)),
    }
    headers = {
        "Authorization": f"Bearer {settings.JINA_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(3):
        try:
            response = requests.post(_RERANK_URL, headers=headers, json=payload, timeout=45)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                delay = 2**attempt
                logfire.warning(
                    f"Jina reranker returned {response.status_code}; retrying in {delay}s."
                )
                time.sleep(delay)
                continue
            response.raise_for_status()
            results = response.json()["results"]
            reranked = [documents[result["index"]] for result in results[:top_n]]
            logfire.info(f"Jina reranked {len(documents)} candidates to {len(reranked)} passages.")
            return reranked
        except (KeyError, IndexError, requests.RequestException) as exc:
            if attempt == 2:
                logfire.error(f"Jina reranking failed: {exc}")
                return documents[:top_n]
            delay = 2**attempt
            logfire.warning(f"Jina reranking request failed; retrying in {delay}s: {exc}")
            time.sleep(delay)

    return documents[:top_n]
