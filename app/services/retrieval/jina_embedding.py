"""Jina-hosted embeddings for Qdrant indexing and similarity search."""

import time

import logfire
import requests

from app.config import settings


_EMBEDDINGS_URL = "https://api.jina.ai/v1/embeddings"
_BATCH_SIZE = 15


def get_embedding_dim() -> int:
    """Return the configured Jina vector size used to create Qdrant collections."""
    return settings.JINA_EMBEDDING_DIM


def _headers() -> dict[str, str]:
    if not settings.JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY is required for Jina embeddings and reranking.")
    return {
        "Authorization": f"Bearer {settings.JINA_API_KEY}",
        "Content-Type": "application/json",
    }


def _embed(texts: list[str], task: str) -> list[list[float]]:
    if not texts:
        return []

    # Truncate any oversized chunk to max 8000 chars to respect Jina API token limits
    safe_texts = [t[:8000] if len(t) > 8000 else t for t in texts]

    payload = {
        "model": settings.JINA_EMBEDDING_MODEL,
        "input": safe_texts,
        "task": task,
        "dimensions": settings.JINA_EMBEDDING_DIM,
    }

    for attempt in range(8):
        try:
            response = requests.post(
                _EMBEDDINGS_URL, headers=_headers(), json=payload, timeout=45
            )
            if response.status_code == 429 and attempt < 7:
                # 429 Token Rate Limit Exceeded - wait for token bucket to reset (at least 10s)
                delay = 12.0
                try:
                    res_json = response.json()
                    detail = res_json.get("detail", "")
                    if "Please try again in" in detail:
                        part = detail.split("Please try again in")[1].strip().split("s")[0]
                        delay = max(float(part) + 2.0, 10.0)
                except Exception:
                    pass
                logfire.warning(
                    f"Jina 429 rate limit hit; waiting {delay:.1f}s for token bucket to reset (attempt {attempt + 1}/8)."
                )
                time.sleep(delay)
                continue

            if response.status_code in (500, 502, 503, 504) and attempt < 7:
                delay = 2 ** attempt
                logfire.warning(f"Jina embeddings returned {response.status_code}; retrying in {delay}s.")
                time.sleep(delay)
                continue

            if not response.ok:
                logfire.error(f"Jina API Error ({response.status_code}): {response.text}")
            response.raise_for_status()
            rows = sorted(response.json()["data"], key=lambda row: row["index"])
            embeddings = [row["embedding"] for row in rows]
            if len(embeddings) != len(texts):
                raise RuntimeError("Jina returned an incomplete embedding response.")
            return embeddings
        except requests.RequestException as exc:
            if attempt == 7:
                raise RuntimeError(f"Jina embeddings request failed: {exc}") from exc
            delay = 5.0
            logfire.warning(f"Jina embeddings request failed; retrying in {delay}s: {exc}")
            time.sleep(delay)

    raise RuntimeError("Jina embeddings request failed after retries.")


def embed_query(query: str) -> list[float]:
    """Embed a user query using Jina's query retrieval adapter."""
    return _embed([query], task="retrieval.query")[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed document chunks using Jina's passage retrieval adapter."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        with logfire.span("Jina embed batch", start=start, size=len(batch)):
            vectors.extend(_embed(batch, task="retrieval.passage"))
        time.sleep(0.5)
    return vectors
