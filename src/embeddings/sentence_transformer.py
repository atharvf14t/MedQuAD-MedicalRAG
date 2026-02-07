import time
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str], batch_size: int = 32):
        """
        Returns:
            embeddings: np.ndarray
            throughput: chunks/sec
            elapsed_time: seconds
        """
        start_time = time.time()

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # important for cosine similarity
        )

        elapsed = time.time() - start_time
        throughput = len(texts) / elapsed if elapsed > 0 else 0

        return embeddings, throughput, elapsed
