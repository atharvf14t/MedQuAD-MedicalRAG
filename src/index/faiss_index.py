"""FAISS index wrapper for efficient similarity search.

Provides interface for building, saving, loading, and searching
FAISS indexes.
"""
import faiss
import numpy as np
import time
from pathlib import Path


class FaissIndex:
    def __init__(self, dim: int):
        """Initialize FaissIndex.
        
        Args:
            dim: Embedding dimension
        """
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product = cosine if normalized

    def build(self, embeddings: np.ndarray):
        """Add embeddings to index and time the operation.
        
        Args:
            embeddings: Array of embeddings (n_samples, dim)
            
        Returns:
            Time taken in seconds
        """
        start = time.time()
        self.index.add(embeddings.astype("float32"))
        build_time = time.time() - start
        return build_time

    def save(self, path: str):
        """Save index to disk.
        
        Args:
            path: Output file path
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, path)

    def load(self, path: str):
        """Load index from disk.
        
        Args:
            path: Path to FAISS index file
        """
        self.index = faiss.read_index(path)

    def search(self, query_vector: np.ndarray, top_k: int):
        """Search index for top-k nearest neighbors.
        
        Args:
            query_vector: Query embedding
            top_k: Number of results
            
        Returns:
            Tuple of (scores, indices)
        """
        scores, indices = self.index.search(query_vector, top_k)
        return scores, indices
