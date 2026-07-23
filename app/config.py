import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    # Qdrant Vector DB Settings 
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
    
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL")
    
settings = Settings()
