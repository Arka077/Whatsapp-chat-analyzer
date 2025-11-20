"""
config/settings.py
COMPLETE + FINAL version
Works locally + Streamlit Cloud + No import-time crashes
"""

import os
from pathlib import Path

# ------------------------------------------------------------------
# PATHS & DIRECTORIES
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
FAISS_INDEX_PATH = str(BASE_DIR / "data" / "faiss_index")
FAISS_METADATA_PATH = str(BASE_DIR / "data" / "faiss_metadata.pkl")

# ------------------------------------------------------------------
# CORE CONFIG – safe at import time
# ------------------------------------------------------------------
VECTOR_DB_TYPE = "faiss"

EMBEDDING_MODEL = "thenlper/gte-small"
EMBEDDING_DIMENSION = 384

LLM_MODEL = "gemini-1.5-flash"          # Correct official name
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2000
LLM_TOP_P = 0.9

HF_API_KEY = os.getenv("HF_TOKEN")

DATABASE_URL = "sqlite:///./chat_analyzer.db"

LOG_LEVEL = "INFO"
LOG_FILE = str(BASE_DIR / "logs" / "app.log")

# Features
ENABLE_CACHING = True
CACHE_EXPIRY_HOURS = 24
MAX_DATE_RANGE_DAYS = 365
MAX_RETRIEVED_MESSAGES = 10
BATCH_SIZE = 32

# RAG
RAG_TOP_K_MESSAGES = 50
RAG_CONTEXT_SIZE = 3

# Topic Modeling
DEFAULT_NUM_TOPICS = 5
MAX_NUM_TOPICS = 10
MIN_NUM_TOPICS = 2

# Sentiment
SENTIMENT_BATCH_SIZE = 50

# Date/Time formats (back and safe!)
DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# ------------------------------------------------------------------
# GEMINI API KEY – LAZY & SAFE (this is the only part that touches st.secrets)
# ------------------------------------------------------------------
def get_gemini_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key.strip()

    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass

    raise ValueError(
        "GEMINI_API_KEY not found!\n\n"
        "→ Streamlit Cloud: Add in Settings → Secrets\n"
        "→ Local: Add to .env file"
    )

# Optional backward compatibility
GEMINI_API_KEY = get_gemini_api_key   # now a callable, not a string