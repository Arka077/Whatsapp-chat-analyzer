"""
Index chat messages into vector database
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from rag.vector_store import VectorStore
from rag.embeddings import EmbeddingModel
from core.preprocessor import normalize_text
import streamlit as st

class ChatIndexer:
    """Index and manage chat messages in vector store"""
    
    def __init__(self, index_name: str = None):
        """Initialize indexer with vector store and embedding model

        Args:
            index_name: optional name to namespace the FAISS index for a chat
        """
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore(dimension=self.embedding_model.get_dimension(), index_name=index_name)
        self.indexed_message_ids = set()
    
    def index_chat_data(self, df: pd.DataFrame) -> bool:
        """
        Index chat data into vector store
        
        Args:
            df: preprocessed chat dataframe
        
        Returns:
            True if indexing successful
        """
        try:
            # Filter out system messages and media
            df_clean = df[
                (df['user'] != 'group_notification') & 
                (df['message'] != '<Media omitted>') &
                (df['message'].str.strip() != '')
            ].copy()
            
            if len(df_clean) == 0:
                st.warning("No messages to index")
                return False
            
            # Generate embeddings
            st.info(f"Generating embeddings for {len(df_clean)} messages...")
            texts_to_embed = [
                normalize_text(msg) if isinstance(msg, str) else "" 
                for msg in df_clean['message']
            ]
            
            embeddings = self.embedding_model.embed_batch(texts_to_embed)
            
            # Prepare metadata
            metadata = []
            for idx, row in df_clean.iterrows():
                metadata.append({
                    'message_id': row['message_id'],
                    'user': row['user'],
                    'date': row['date'].isoformat(),
                    'only_date': str(row['only_date']),
                    'language': row['language'],
                    'message': row['message'],
                    'message_length': row['message_length'],
                    'hour': int(row['hour']),
                    'day_name': row['day_name']
                })
            
            # Store in vector DB
            st.info("Storing embeddings in vector database...")
            self.vector_store.store_embeddings(embeddings, metadata)
            
            st.success(f"✓ Successfully indexed {len(df_clean)} messages")
            return True
            
        except Exception as e:
            st.error(f"Error indexing chat data: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict:
        """Get indexing statistics"""
        return self.vector_store.get_stats()