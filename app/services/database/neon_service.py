"""Neon Postgres Database Service for conversation logging & persistence."""

import logfire
from typing import Optional, List, Dict, Any
from app.config import settings

# Lazy connection pool / engine initialization
_engine = None

def get_db_engine():
    """Returns SQLAlchemy engine for Neon PostgreSQL."""
    global _engine
    if _engine is None:
        if not settings.NEON_DB_URL:
            logfire.warning("NEON_DB_URL is not set in environment.")
            return None
        try:
            from sqlalchemy import create_engine
            # Ensure URL format works with SQLAlchemy driver
            db_url = settings.NEON_DB_URL
            if db_url.startswith("postgresql://"):
                # Use standard postgresql driver or psycopg2 driver if available
                db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
            _engine = create_engine(db_url, pool_pre_ping=True)
            logfire.info("Successfully initialized Neon Postgres engine.")
        except Exception as e:
            logfire.error(f"Failed to initialize Neon Postgres engine: {e}")
            _engine = None
    return _engine


def log_query_to_neon(question: str, answer: str, thread_id: str = "default") -> bool:
    """Logs user query and RAG answer to Neon PostgreSQL database."""
    engine = get_db_engine()
    if not engine:
        return False

    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            # Ensure query_logs table exists
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id SERIAL PRIMARY KEY,
                    thread_id VARCHAR(255),
                    question TEXT,
                    answer TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            conn.execute(
                text("INSERT INTO query_logs (thread_id, question, answer) VALUES (:thread_id, :q, :a)"),
                {"thread_id": thread_id, "q": question, "a": answer}
            )
            conn.commit()
            logfire.info(f"Query logged to Neon PostgreSQL for thread={thread_id}")
            return True
    except Exception as e:
        logfire.warning(f"Failed to log query to Neon DB: {e}")
        return False
