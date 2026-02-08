import json
import argparse
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, index_path: str, metadata_path: str, model_name: str, device: str = "cpu"):
        # Load embedding model
        self.model = SentenceTransformer(model_name, device=device)

        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.reshape(1, -1)

    def retrieve(self, query: str, top_k: int = 5) -> list:
        query_embedding = self.embed_query(query)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            chunk = self.metadata[idx]
            results.append({
                "score": float(score),
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            })

        return results


def main():
    parser = argparse.ArgumentParser(description="Dense Retriever CLI")

    parser.add_argument("--question", required=True, help="Query question")
    parser.add_argument("--index_path", required=True, help="Path to FAISS index")
    parser.add_argument("--metadata_path", required=True, help="Path to chunk metadata JSON")
    parser.add_argument("--embedding_model", required=True, help="Embedding model name")
    parser.add_argument("--top_k", type=int, default=5)

    args = parser.parse_args()

    retriever = DenseRetriever(
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        model_name=args.embedding_model
    )

    results = retriever.retrieve(args.question, top_k=args.top_k)

    print(f"\nTop {len(results)} results:\n")

    for i, res in enumerate(results, 1):
        print(f"{i}. Score: {res['score']:.4f}")
        print(f"   Chunk ID: {res['chunk_id']}")
        print(f"   Text: {res['text'][:300]}...\n")


if __name__ == "__main__":
    main()
