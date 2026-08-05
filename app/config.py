import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Qdrant Vector DB Settings
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME")
    QDRANT_COLLECTION_NAME = QDRANT_COLLECTION

    # Jina AI (Embedding & Reranker)
    JINA_API_KEY = os.getenv("JINA_API_KEY")
    JINA_EMBEDDING_MODEL = os.getenv("JINA_EMBEDDING_MODEL")
    _dim = os.getenv("JINA_EMBEDDING_DIM")
    JINA_EMBEDDING_DIM = int(_dim) if _dim else None
    JINA_RERANK_MODEL = os.getenv("JINA_RERANK_MODEL")

    # Neon PostgreSQL Database
    NEON_DB_URL = os.getenv("NEON_DB_URL")

    # Upstash Redis Cache
    UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL")
    UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL")
    GROQ_SLUG = os.getenv("GROQ_SLUG")
    GROQ_SLUG_2 = os.getenv("GROQ_SLUG_2")

    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    PORTKEY_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID")

    # --- OBSERVABILITY ---
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")


settings = Settings()
