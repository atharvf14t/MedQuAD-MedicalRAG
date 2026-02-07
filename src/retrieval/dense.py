import numpy as np
from typing import List, Dict


class DenseRetriever:
    def __init__(self, index, chunks: List[Dict]):
        self.index = index
        self.chunks = chunks

    def retrieve(self, query_vector: np.ndarray, top_k: int = 5):
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
