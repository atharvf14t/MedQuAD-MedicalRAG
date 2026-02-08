import json
import argparse
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


class HybridRetriever:
    def __init__(self, index_path: str, metadata_path: str, model_name: str, device: str = "cpu"):
        # Dense model
        self.model = SentenceTransformer(model_name, device=device)

        # FAISS index
        self.index = faiss.read_index(index_path)

        # Metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Prepare BM25 corpus
        self.texts = [chunk["text"] for chunk in self.metadata]
        tokenized_corpus = [text.lower().split() for text in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.reshape(1, -1)

    def retrieve(self, query: str, top_k: int = 5, alpha: float = 0.5, fetch_k: int = 50):
        """
        Hybrid retrieval combining dense + BM25.
        """
        query_vec = self.embed_query(query)

        # Dense retrieval
        dense_scores, dense_indices = self.index.search(query_vec, fetch_k)

        dense_scores = dense_scores[0]
        dense_indices = dense_indices[0]

        # BM25 retrieval
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Collect candidate set
        candidate_indices = set(dense_indices.tolist())
        candidate_indices = [idx for idx in candidate_indices if idx != -1]

        results = []

        # Normalize scores
        dense_max = max(dense_scores) if len(dense_scores) > 0 else 1.0
        bm25_max = max(bm25_scores) if len(bm25_scores) > 0 else 1.0

        for idx in candidate_indices:
            dense_score = dense_scores[list(dense_indices).index(idx)] if idx in dense_indices else 0.0
            bm25_score = bm25_scores[idx]

            dense_norm = dense_score / dense_max if dense_max != 0 else 0
            bm25_norm = bm25_score / bm25_max if bm25_max != 0 else 0

            final_score = alpha * dense_norm + (1 - alpha) * bm25_norm

            chunk = self.metadata[idx]

            results.append({
                "score": float(final_score),
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            })

        # Sort by score
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        return results[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Hybrid Retriever CLI")

    parser.add_argument("--question", required=True)
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--embedding_model", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--fetch_k", type=int, default=50)

    args = parser.parse_args()

    retriever = HybridRetriever(
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        model_name=args.embedding_model
    )

    results = retriever.retrieve(
        query=args.question,
        top_k=args.top_k,
        alpha=args.alpha,
        fetch_k=args.fetch_k
    )

    print(f"\nTop {len(results)} Hybrid results:\n")

    for i, res in enumerate(results, 1):
        print(f"{i}. Score: {res['score']:.4f}")
        print(f"   Chunk ID: {res['chunk_id']}")
        print(f"   Text: {res['text'][:300]}...\n")


if __name__ == "__main__":
    main()
