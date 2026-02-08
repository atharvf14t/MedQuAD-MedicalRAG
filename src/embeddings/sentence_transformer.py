"""Sentence embedding using SentenceTransformers models.

Provides efficient batch encoding of texts with normalized embeddings.
"""
import time
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        """Initialize SentenceTransformerEmbedder.
        
        Args:
            model_name: SentenceTransformer model name
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str], batch_size: int = 32):
        """Embed texts with timing and throughput calculation.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding
            
        Returns:
            Tuple of (embeddings, throughput_chunks_per_sec, elapsed_seconds)
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
