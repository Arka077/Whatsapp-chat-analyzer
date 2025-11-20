"""
FAISS Vector Database operations
"""

import faiss
import numpy as np
import pickle
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from config.settings import FAISS_INDEX_PATH

class VectorStore:
    """Manage FAISS vector index and metadata"""
    
    def __init__(self, dimension: int = 768, index_name: str = None):
        """Initialize FAISS vector store

        Args:
            dimension: embedding dimension
            index_name: optional name/namespace for per-chat indexes. If provided,
                        index files are stored under FAISS_INDEX_PATH/index_name/
        """
        self.dimension = dimension
        self.index = None
        self.metadata = []
        self.is_trained = False

        # Convert to absolute path to avoid relative path issues with FAISS
        base_path = Path(FAISS_INDEX_PATH).resolve()
        os.makedirs(base_path, exist_ok=True)

        # Use a subdirectory per index_name when provided to avoid collisions
        if index_name:
            # Remove special characters and emojis for filesystem compatibility
            import re
            safe_name = str(index_name).replace(' ', '_')
            # Keep only alphanumeric, underscore, and hyphen
            safe_name = re.sub(r'[^\w\-]', '', safe_name, flags=re.UNICODE)
            # Remove any leading/trailing underscores or hyphens
            safe_name = safe_name.strip('_-') or 'chat'
            self.index_dir = str(base_path / safe_name)
        else:
            self.index_dir = str(base_path / "default")

        os.makedirs(self.index_dir, exist_ok=True)
        
        # Use Path to ensure consistent path handling across platforms, convert to forward slashes
        self.index_path = str(Path(self.index_dir) / "index.faiss").replace('\\', '/')
        self.metadata_path = str(Path(self.index_dir) / "metadata.pkl").replace('\\', '/')

        self._load_or_create()
    
    def _load_or_create(self):
        """Load existing index or create new one"""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self._load_index()
        else:
            self._create_index()
    
    def _create_index(self):
        """Create new FAISS index"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self.is_trained = True
    
    def _load_index(self):
        """Load existing FAISS index"""
        try:
            # Convert to absolute path with forward slashes for FAISS compatibility
            index_path_absolute = str(Path(self.index_path).resolve()).replace('\\', '/')
            metadata_path_absolute = str(Path(self.metadata_path).resolve())
            
            self.index = faiss.read_index(index_path_absolute)
            with open(metadata_path_absolute, 'rb') as f:
                self.metadata = pickle.load(f)
            self.is_trained = True
        except Exception as e:
            print(f"Error loading index: {e}. Creating new index.")
            self._create_index()
    
    def _save_index(self):
        """Save FAISS index and metadata"""
        if self.index is not None:
            # Ensure path is absolute and uses forward slashes for FAISS compatibility
            index_path_absolute = str(Path(self.index_path).resolve()).replace('\\', '/')
            metadata_path_absolute = str(Path(self.metadata_path).resolve())
            
            faiss.write_index(self.index, index_path_absolute)
            with open(metadata_path_absolute, 'wb') as f:
                pickle.dump(self.metadata, f)
    
    def store_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Store embeddings in FAISS index
        
        Args:
            embeddings: numpy array of shape (n_messages, dimension)
            metadata: list of metadata dicts for each message
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        # Ensure embeddings are float32 (FAISS requirement)
        embeddings = embeddings.astype('float32')
        
        # Add to index
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        
        # Save to disk
        self._save_index()
    
    def retrieve_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Retrieve similar messages from index
        
        Args:
            query_embedding: embedding vector of query
            top_k: number of results to return
        
        Returns:
            List of metadata dicts with similarity scores
        """
        if self.index.ntotal == 0:
            return []
        
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:  # Invalid index
                continue
            
            metadata = self.metadata[idx].copy()
            metadata['similarity_score'] = 1.0 / (1.0 + distance)  # Convert distance to similarity
            results.append(metadata)
        
        return results
    
    def clear_index(self):
        """Clear index and metadata"""
        self._create_index()
        self._save_index()
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "is_trained": self.is_trained,
            "index_path": self.index_path
        }