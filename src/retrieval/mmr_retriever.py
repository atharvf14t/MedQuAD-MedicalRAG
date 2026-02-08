"""MMR (Maximal Marginal Relevance) retrieval combining density and diversity.

Performs dense retrieval followed by MMR reranking to promote diversity
in the retrieved set.
"""
import json
import argparse
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class MMRRetriever:
    def __init__(self, index_path: str, metadata_path: str, model_name: str, device: str = "cpu"):
        """Initialize MMRRetriever.
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to chunk metadata JSON
            model_name: Sentence transformer model name
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.index = faiss.read_index(index_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def embed_texts(self, texts):
        """Embed list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of normalized embeddings
        """
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def embed_query(self, query):
        """Embed single query string.
        
        Args:
            query: Query text
            
        Returns:
            1D numpy array of embedding
        """
        return self.embed_texts([query])[0]

    def mmr(self, query_vec, candidate_vecs, top_k, lambda_param=0.7):
        """Maximal Marginal Relevance selection.
        
        Balances relevance to query and diversity among selected items.
        
        Args:
            query_vec: Query embedding vector
            candidate_vecs: Array of candidate embeddings
            top_k: Number of items to select
            lambda_param: Trade-off parameter (0-1). Higher values prioritize relevance.
            
        Returns:
            List of selected candidate indices
        """
        selected = []
        candidate_indices = list(range(len(candidate_vecs)))

        # similarity between query and candidates
        query_sim = np.dot(candidate_vecs, query_vec)

        while len(selected) < top_k and candidate_indices:
            mmr_scores = []

            for idx in candidate_indices:
                relevance = query_sim[idx]

                if not selected:
                    diversity = 0
                else:
                    selected_vecs = candidate_vecs[selected]
                    diversity = np.max(np.dot(selected_vecs, candidate_vecs[idx]))

                score = lambda_param * relevance - (1 - lambda_param) * diversity
                mmr_scores.append((score, idx))

            # select best candidate
            best = max(mmr_scores, key=lambda x: x[0])[1]
            selected.append(best)
            candidate_indices.remove(best)

        return selected

    def retrieve(self, query: str, top_k: int = 5, lambda_param: float = 0.7, fetch_k: int = 20):
        """Perform dense retrieval + MMR reranking.
        
        Args:
            query: Query text
            top_k: Number of final results to return
            lambda_param: MMR trade-off parameter (0-1)
            fetch_k: Initial number of dense-retrieved candidates for MMR
            
        Returns:
            List of top-k diverse retrieved documents
        """
        query_vec = self.embed_query(query).reshape(1, -1)

        # initial dense retrieval
        scores, indices = self.index.search(query_vec, fetch_k)

        valid_indices = [idx for idx in indices[0] if idx != -1]

        if not valid_indices:
            return []

        candidate_chunks = [self.metadata[idx] for idx in valid_indices]
        candidate_texts = [c["text"] for c in candidate_chunks]

        # embed candidates
        candidate_vecs = self.embed_texts(candidate_texts)

        # apply MMR
        selected_indices = self.mmr(
            query_vec=query_vec[0],
            candidate_vecs=candidate_vecs,
            top_k=top_k,
            lambda_param=lambda_param
        )

        results = []
        for sel in selected_indices:
            chunk = candidate_chunks[sel]
            score = float(np.dot(candidate_vecs[sel], query_vec[0]))

            results.append({
                "score": score,
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            })

        return results


def main():
    """CLI for MMR retrieval."""
    parser = argparse.ArgumentParser(description="MMR Retriever CLI")

    parser.add_argument("--question", required=True)
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--embedding_model", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--lambda_param", type=float, default=0.7)
    parser.add_argument("--fetch_k", type=int, default=20)

    args = parser.parse_args()

    retriever = MMRRetriever(
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        model_name=args.embedding_model
    )

    results = retriever.retrieve(
        query=args.question,
        top_k=args.top_k,
        lambda_param=args.lambda_param,
        fetch_k=args.fetch_k
    )

    print(f"\nTop {len(results)} MMR results:\n")

    for i, res in enumerate(results, 1):
        print(f"{i}. Score: {res['score']:.4f}")
        print(f"   Chunk ID: {res['chunk_id']}")
        print(f"   Text: {res['text'][:300]}...\n")


if __name__ == "__main__":
    main()
