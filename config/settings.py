"""
Configuration settings for the WhatsApp Analyzer application
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Vector Database Configuration
VECTOR_DB_TYPE = "faiss"
# Use absolute path to avoid FAISS relative path issues on Windows
FAISS_INDEX_PATH = str(Path(__file__).parent.parent / "data" / "faiss_index")
FAISS_METADATA_PATH = str(Path(__file__).parent.parent / "data" / "faiss_metadata.pkl")

# Embedding Configuration
EMBEDDING_MODEL = "thenlper/gte-small"
EMBEDDING_DIMENSION = 384
HF_API_KEY = os.getenv("HF_TOKEN")

# LLM Configuration
LLM_MODEL = "gemini-2.0-flash"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2000
LLM_TOP_P = 0.9

# Database Configuration (SQLite for caching)
DATABASE_URL = "sqlite:///./chat_analyzer.db"

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "./logs/app.log"

# Features
ENABLE_CACHING = True
CACHE_EXPIRY_HOURS = 24
MAX_DATE_RANGE_DAYS = 365
MAX_RETRIEVED_MESSAGES = 10
BATCH_SIZE = 32

# RAG Configuration
RAG_TOP_K_MESSAGES = 50
RAG_CONTEXT_SIZE = 3  # Number of surrounding messages for context

# Topic Modeling
DEFAULT_NUM_TOPICS = 5
MAX_NUM_TOPICS = 10
MIN_NUM_TOPICS = 2

# Sentiment Analysis
SENTIMENT_BATCH_SIZE = 50

# Date format
DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"