"""
Configuration settings for the WhatsApp Analyzer application
Compatible with: local development + Streamlit Community Cloud
"""

import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# === Load .env file only in local development ===
if os.path.exists(".env"):
    load_dotenv()

# === GEMINI_API_KEY – The magic fix for Streamlit Cloud ===
def get_gemini_api_key():
    # 1. Try environment variable (local + some hosting)
    key1 = os.getenv("GEMINI_API_KEY")
    if key1:
        return key1

    # 2. Try Streamlit secrets (Streamlit Cloud)
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except:
        pass  # st.secrets might not be available during import

    # 3. Final fallback – raise clear error
    raise ValueError(
        "GEMINI_API_KEY not found!\n\n"
        "→ On Streamlit Cloud: Go to your app → Settings → Secrets → Add:\n"
        "     GEMINI_API_KEY = \"your_actual_key_here\"\n\n"
        "→ Locally: Create a .env file with:\n"
        "     GEMINI_API_KEY=your_actual_key_here"
    )

# Set it once so the rest of the app can just use os.getenv()
GEMINI_API_KEY = get_gemini_api_key()
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY  # Ensures google.generativeai sees it

# === Rest of your config (unchanged) ===
HF_API_KEY = os.getenv("HF_TOKEN")

VECTOR_DB_TYPE = "faiss"
BASE_DIR = Path(__file__).parent.parent
FAISS_INDEX_PATH = str(BASE_DIR / "data" / "faiss_index")
FAISS_METADATA_PATH = str(BASE_DIR / "data" / "faiss_metadata.pkl")

EMBEDDING_MODEL = "thenlper/gte-small"
EMBEDDING_DIMENSION = 384

LLM_MODEL = "gemini-2.0-flash"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2000
LLM_TOP_P = 0.9

DATABASE_URL = "sqlite:///./chat_analyzer.db"

LOG_LEVEL = "INFO"
LOG_FILE = "./logs/app.log"

ENABLE_CACHING = True
CACHE_EXPIRY_HOURS = 24
MAX_DATE_RANGE_DAYS = 365
MAX_RETRIEVED_MESSAGES = 10
BATCH_SIZE = 32

RAG_TOP_K_MESSAGES = 50
RAG_CONTEXT_SIZE = 3

DEFAULT_NUM_TOPICS = 5
MAX_NUM_TOPICS = 10
MIN_NUM_TOPICS = 2

SENTIMENT_BATCH_SIZE = 50

DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"