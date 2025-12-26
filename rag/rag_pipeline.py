"""
RAG Pipeline - Retrieve and Generate answers with citations
WITH IMPROVED ERROR HANDLING FOR SAFETY BLOCKS
"""

from typing import Dict, List, Optional, Tuple
from rag.retriever import ChatRetriever
from llm.mistral_client import MistralClient
from llm. prompt_templates import get_qa_prompt
import time

class RAGPipeline: 
    """Orchestrate retrieval and generation"""
    
    def __init__(self, index_name: Optional[str] = None):
        """Initialize RAG pipeline

        Args:
            index_name: optional chat-specific FAISS index namespace
        """
        self.retriever = ChatRetriever(index_name=index_name)
        self.llm = MistralClient()
        self.context_window = 10
    
    # ... rest of the file remains the same ... 
