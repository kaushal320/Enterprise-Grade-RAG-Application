"""Upstash Redis Cache Service using REST API.

Provides fast response caching for RAG queries and retrieval results.
"""

import logfire
import requests
from typing import Optional, Any
import json
from app.config import settings


def _headers() -> dict[str, str]:
    if not settings.UPSTASH_REDIS_REST_TOKEN:
        raise RuntimeError("UPSTASH_REDIS_REST_TOKEN is missing in environment variables.")
    return {
        "Authorization": f"Bearer {settings.UPSTASH_REDIS_REST_TOKEN}",
        "Content-Type": "application/json",
    }


def get_cache(key: str) -> Optional[Any]:
    """Retrieve a key from Upstash Redis REST API."""
    if not settings.UPSTASH_REDIS_REST_URL or not settings.UPSTASH_REDIS_REST_TOKEN:
        return None

    url = f"{settings.UPSTASH_REDIS_REST_URL.rstrip('/')}/get/{key}"
    try:
        response = requests.get(url, headers=_headers(), timeout=5)
        if response.status_code == 200:
            data = response.json()
            result = data.get("result")
            if result:
                try:
                    return json.loads(result)
                except Exception:
                    return result
        return None
    except Exception as e:
        logfire.warning(f"Upstash Redis GET failed for key {key}: {e}")
        return None


def set_cache(key: str, value: Any, ttl_seconds: int = 3600) -> bool:
    """Store a key-value pair in Upstash Redis REST API with TTL."""
    if not settings.UPSTASH_REDIS_REST_URL or not settings.UPSTASH_REDIS_REST_TOKEN:
        return False

    val_str = json.dumps(value) if not isinstance(value, str) else value
    url = f"{settings.UPSTASH_REDIS_REST_URL.rstrip('/')}/setex/{key}/{ttl_seconds}"
    try:
        response = requests.post(url, headers=_headers(), data=val_str, timeout=5)
        if response.status_code == 200 and response.json().get("result") == "OK":
            logfire.info(f"Cached key in Upstash Redis: {key} (TTL={ttl_seconds}s)")
            return True
        return False
    except Exception as e:
        logfire.warning(f"Upstash Redis SETEX failed for key {key}: {e}")
        return False
