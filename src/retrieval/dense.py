"""Dense retriever utility class using FAISS indexes.

Legacy class for dense retrieval operations.
"""
import numpy as np
from typing import List, Dict


class DenseRetriever:
    def __init__(self, index, chunks: List[Dict]):
        """Initialize DenseRetriever.
        
        Args:
            index: FAISS index object
            chunks: List of chunk documents
        """
        self.index = index
        self.chunks = chunks

    def retrieve(self, query_vector: np.ndarray, top_k: int = 5):
        """Retrieve top-k chunks for query vector.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks with scores
        """
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            chunk = self.chunks[idx]
            results.append(
                {
                    "chunk": chunk,
                    "score": float(scores[0][i]),
                }
            )
        return results
