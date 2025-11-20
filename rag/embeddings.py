"""
Embedding model using sentence-transformers (Google Embedding Gemma)
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
from config.settings import EMBEDDING_MODEL, EMBEDDING_DIMENSION, HF_API_KEY

class EmbeddingModel:
    """Generate embeddings using Google Embedding Gemma"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize embedding model
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, token=HF_API_KEY)
        self.dimension = EMBEDDING_DIMENSION
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text
        
        Args:
            text: text to embed
        
        Returns:
            embedding vector of shape (dimension,)
        """
        if not text or not isinstance(text, str):
            return np.zeros(self.dimension, dtype=np.float32)
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed multiple texts
        
        Args:
            texts: list of texts to embed
            batch_size: batch size for processing
        
        Returns:
            embedding matrix of shape (n_texts, dimension)
        """
        # Filter out empty texts
        texts = [t if isinstance(t, str) else "" for t in texts]
        
        embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
        return embeddings.astype(np.float32)
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension