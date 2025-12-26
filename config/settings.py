"""
config/settings.py
COMPLETE + FINAL version
Works locally + Streamlit Cloud + No import-time crashes
WITH MULTI-API KEY SUPPORT (Mistral)
"""

import os
from pathlib import Path
from typing import List

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

# Mistral Model Configuration
LLM_MODEL = "mistral-small-latest"  # or "mistral-medium-latest", "mistral-small-latest"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2000
LLM_TOP_P = 0.9

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
# MULTI-API KEY SUPPORT (Mistral)
# ------------------------------------------------------------------
def parse_api_keys(key_string: str) -> List[str]:
    """
    Parse comma-separated API keys and return as list
    
    Args:
        key_string: Comma-separated API keys
    
    Returns:
        List of API keys (stripped of whitespace)
    """
    if not key_string:
        return []
    
    # Support both comma-separated and newline-separated keys
    keys = []
    for separator in [',', '\n']: 
        if separator in key_string:
            keys = [k.strip() for k in key_string.split(separator) if k.strip()]
            break
    
    # If no separator found, treat as single key
    if not keys and key_string.strip():
        keys = [key_string.strip()]
    
    return keys


def get_mistral_api_keys() -> List[str]:
    """
    Get list of Mistral API keys with fallback support
    
    Returns:
        List of API keys (at least one required)
    
    Raises:
        ValueError:  If no API keys are found
    """
    # Try environment variable first
    env_keys = os.getenv("MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEYS")
    if env_keys:
        keys = parse_api_keys(env_keys)
        if keys:
            return keys

    # Try Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            # Try plural first
            if "MISTRAL_API_KEYS" in st.secrets:
                keys = parse_api_keys(str(st.secrets["MISTRAL_API_KEYS"]))
                if keys:
                    return keys
            
            # Try singular
            if "MISTRAL_API_KEY" in st.secrets:
                keys = parse_api_keys(str(st.secrets["MISTRAL_API_KEY"]))
                if keys:
                    return keys
    except Exception:
        pass

    raise ValueError(
        "MISTRAL_API_KEY(S) not found!\n\n"
        "→ Streamlit Cloud:  Add in Settings → Secrets\n"
        "→ Local: Add to .env file\n"
        "→ Format: Single key OR comma-separated:  key1,key2,key3\n"
        "→ Get your key at: https://console.mistral.ai/api-keys/"
    )


# Backward compatibility - return first key as string
def get_mistral_api_key() -> str:
    """Get single Mistral API key (backward compatibility)"""
    keys = get_mistral_api_keys()
    return keys[0] if keys else ""


# Keep this for backward compatibility
MISTRAL_API_KEY = get_mistral_api_key  # Callable that returns first key
