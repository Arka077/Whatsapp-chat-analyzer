"""
Embedding model using sentence-transformers (Google Embedding Gemma)
WITH MULTI-API KEY FALLBACK SUPPORT
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Optional
import time
from config.settings import EMBEDDING_MODEL, EMBEDDING_DIMENSION, get_hf_api_keys


class EmbeddingModel:
    """Generate embeddings using sentence-transformers with multi-token fallback"""
    
    def __init__(self, model_name:  str = EMBEDDING_MODEL):
        """
        Initialize embedding model with fallback support
        
        Args: 
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.dimension = EMBEDDING_DIMENSION
        self.api_keys = get_hf_api_keys()
        self.current_key_index = 0
        
        # Initialize model with fallback
        self.model = self._initialize_model()
    
    def _initialize_model(self) -> SentenceTransformer:
        """Initialize model with API key fallback"""
        last_error = None
        
        # Try with API keys if available
        if self.api_keys:
            for idx, token in enumerate(self.api_keys):
                try:
                    self.current_key_index = idx
                    print(f"[Embeddings] Attempting to load model with HF token #{idx + 1}")
                    model = SentenceTransformer(self.model_name, token=token)
                    print(f"[Embeddings] Successfully loaded with token #{idx + 1}")
                    return model
                except Exception as e:
                    last_error = e
                    print(f"[Embeddings] Failed with token #{idx + 1}: {str(e)[:100]}")
                    time.sleep(0.5)
        
        # Try without token (for public models)
        try:
            print("[Embeddings] Attempting to load model without token (public model)")
            model = SentenceTransformer(self. model_name)
            print("[Embeddings] Successfully loaded without token")
            return model
        except Exception as e:
            print(f"[Embeddings] Failed without token: {str(e)[:100]}")
            if last_error:
                raise last_error
            raise e
    
    def _execute_with_fallback(self, operation_func, *args, **kwargs):
        """
        Execute operation with automatic token fallback
        
        Args: 
            operation_func: Method to call on self.model
            *args, **kwargs: Arguments to pass to operation
        
        Returns:
            Result from operation
        """
        last_error = None
        
        for attempt in range(len(self.api_keys) + 1 if self.api_keys else 1):
            try:
                # Execute operation on current model
                return operation_func(*args, **kwargs)
                
            except Exception as e:
                error_msg = str(e).lower()
                last_error = e
                
                # Check if it's a retryable error
                retryable = any(keyword in error_msg for keyword in [
                    'rate limit', 'quota', '429', '503', '500', 'timeout', 'unavailable'
                ])
                
                if not retryable:
                    raise
                
                print(f"[Embeddings] Error:  {str(e)[:100]}")
                
                # Try next token
                if self.api_keys and self.current_key_index < len(self.api_keys) - 1:
                    self.current_key_index += 1
                    print(f"[Embeddings] Switching to token #{self.current_key_index + 1}")
                    
                    try:
                        self.model = SentenceTransformer(
                            self.model_name, 
                            token=self.api_keys[self.current_key_index]
                        )
                        time.sleep(0.5)
                        continue
                    except Exception as reinit_error:
                        print(f"[Embeddings] Failed to reinitialize:  {str(reinit_error)[:100]}")
                        continue
                else:
                    # No more tokens to try
                    break
        
        # All attempts failed
        if last_error: 
            print(f"[Embeddings] All tokens exhausted, using fallback")
            raise last_error
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text with fallback support
        
        Args: 
            text: text to embed
        
        Returns: 
            embedding vector of shape (dimension,)
        """
        if not text or not isinstance(text, str):
            return np.zeros(self. dimension, dtype=np.float32)
        
        try:
            embedding = self._execute_with_fallback(
                self.model.encode, 
                text, 
                convert_to_numpy=True
            )
            return embedding.astype(np.float32)
        except Exception as e:
            print(f"[Embeddings] Error encoding text, returning zero vector: {e}")
            return np.zeros(self.dimension, dtype=np. float32)
    
    def embed_batch(self, texts:  List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed multiple texts with fallback support
        
        Args:
            texts: list of texts to embed
            batch_size: batch size for processing
        
        Returns:
            embedding matrix of shape (n_texts, dimension)
        """
        # Filter out empty texts
        texts = [t if isinstance(t, str) else "" for t in texts]
        
        try:
            embeddings = self._execute_with_fallback(
                self.model.encode, 
                texts, 
                batch_size=batch_size, 
                convert_to_numpy=True
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            print(f"[Embeddings] Error encoding batch, returning zero vectors: {e}")
            return np.zeros((len(texts), self.dimension), dtype=np.float32)
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension
