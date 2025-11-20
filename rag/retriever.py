"""
Retrieve relevant messages from vector store
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
from rag.vector_store import VectorStore
from rag.embeddings import EmbeddingModel
from config.settings import RAG_TOP_K_MESSAGES
import numpy as np

class ChatRetriever:
    """Retrieve relevant messages based on semantic similarity"""
    
    def __init__(self, index_name: Optional[str] = None):
        """Initialize retriever

        Args:
            index_name: optional chat-specific FAISS namespace
        """
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore(
            dimension=self.embedding_model.get_dimension(),
            index_name=index_name
        )
    
    def retrieve_by_query(
        self, 
        query: str, 
        top_k: int = RAG_TOP_K_MESSAGES,
        date_range: Optional[Tuple[str, str]] = None,
        include_context: bool = False,
        context_size: int = 10
    ) -> List[Dict]:
        """
        Retrieve messages similar to query
        
        Args:
            query: search query
            top_k: number of results
            date_range: optional (start_date, end_date) tuple in YYYY-MM-DD format
        
        Returns:
            List of retrieved message metadata
        """
        try:
            if not query or not query.strip():
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.embed_text(query)
            
            # Retrieve from vector store
            retrieved = self.vector_store.retrieve_similar(query_embedding, top_k=top_k * 2)
            
            # Filter by date range if provided
            if date_range:
                start_date, end_date = date_range
                retrieved = [
                    msg for msg in retrieved 
                    if 'date' in msg and start_date <= msg['date'] <= end_date
                ]
            
            # Return top_k after filtering
            retrieved = retrieved[:top_k]

            if include_context and context_size > 0:
                for msg in retrieved:
                    msg_id = msg.get('message_id', '')
                    if msg_id:
                        msg['context_messages'] = self.get_context_around_message(
                            msg_id,
                            context_size=context_size
                        )
                    else:
                        msg['context_messages'] = []

            return retrieved
        
        except Exception as e:
            print(f"[DEBUG] Error in retrieve_by_query: {str(e)}")
            raise
    
    def retrieve_by_date_range(
        self,
        start_date: str,
        end_date: str
    ) -> List[Dict]:
        """
        Retrieve all messages in date range
        
        Args:
            start_date: YYYY-MM-DD format
            end_date: YYYY-MM-DD format
        
        Returns:
            List of messages in date range
        """
        # Get all messages from metadata
        all_messages = self.vector_store.metadata
        
        filtered = [
            msg for msg in all_messages
            if start_date <= msg['date'] <= end_date
        ]
        
        return filtered
    
    def retrieve_by_user(self, user: str, top_k: int = 100) -> List[Dict]:
        """Retrieve messages by specific user"""
        all_messages = self.vector_store.metadata
        user_messages = [msg for msg in all_messages if msg['user'] == user]
        return user_messages[:top_k]
    
    def get_context_around_message(
        self, 
        message_id: str, 
        context_size: int = 3
    ) -> List[Dict]:
        """
        Get surrounding messages for context
        
        Args:
            message_id: ID of target message
            context_size: number of messages before and after
        
        Returns:
            List of context messages
        """
        metadata = self.vector_store.metadata
        
        # Find target message index
        target_idx = None
        for idx, msg in enumerate(metadata):
            if msg['message_id'] == message_id:
                target_idx = idx
                break
        
        if target_idx is None:
            return []
        
        # Get surrounding messages
        start_idx = max(0, target_idx - context_size)
        end_idx = min(len(metadata), target_idx + context_size + 1)
        
        return metadata[start_idx:end_idx]