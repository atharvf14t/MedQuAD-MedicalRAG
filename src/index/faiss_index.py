import faiss
import numpy as np
import time
from pathlib import Path


class FaissIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product = cosine if normalized

    def build(self, embeddings: np.ndarray):
        start = time.time()
        self.index.add(embeddings.astype("float32"))
        build_time = time.time() - start
        return build_time

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, path)

    def load(self, path: str):
        self.index = faiss.read_index(path)

    def search(self, query_vector: np.ndarray, top_k: int):
        scores, indices = self.index.search(query_vector, top_k)
        return scores, indices
