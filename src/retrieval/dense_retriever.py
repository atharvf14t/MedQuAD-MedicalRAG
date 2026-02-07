import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, index_path: str, model_name: str):
        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load metadata
        meta_path = index_path.replace(".faiss", ".meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # Load embedding model
        self.model = SentenceTransformer(model_name)

    def embed_query(self, query: str):
        emb = self.model.encode(
            query,
            normalize_embeddings=True
        )
        return np.array([emb]).astype("float32")

    def search(self, query: str, top_k: int = 5, score_threshold: float = None):
        query_vec = self.embed_query(query)

        scores, indices = self.index.search(query_vec, top_k * 4)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            if score_threshold and score < score_threshold:
                continue

            chunk = self.chunks[idx]
            results.append(
                {
                    "score": float(score),
                    "chunk": chunk,
                }
            )

        return results

    def mmr(self, query_vec, candidate_indices, lambda_param=0.7, top_k=5):
        selected = []
        candidate_set = list(candidate_indices)

        # Get embeddings from index
        all_vectors = self.index.reconstruct_n(0, self.index.ntotal)

        while len(selected) < top_k and candidate_set:
            best_score = -1e9
            best_idx = None

            for idx in candidate_set:
                sim_to_query = np.dot(query_vec, all_vectors[idx])

                sim_to_selected = 0
                for s in selected:
                    sim = np.dot(all_vectors[idx], all_vectors[s])
                    sim_to_selected = max(sim_to_selected, sim)

                mmr_score = (
                    lambda_param * sim_to_query
                    - (1 - lambda_param) * sim_to_selected
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            selected.append(best_idx)
            candidate_set.remove(best_idx)

        return selected

    def search_with_mmr(
        self,
        query: str,
        top_k: int = 5,
        lambda_param: float = 0.7,
        score_threshold: float = None,
    ):
        query_vec = self.embed_query(query)[0]

        scores, indices = self.index.search(
            np.array([query_vec]).astype("float32"),
            top_k * 4
        )

        candidate_indices = [
            int(idx) for idx in indices[0]
            if idx != -1
        ]


        selected_indices = self.mmr(
            query_vec,
            candidate_indices,
            lambda_param=lambda_param,
            top_k=top_k
        )

        results = []
        for idx in selected_indices:
            idx = int(idx)  

            chunk = self.chunks[idx]

            vec = self.index.reconstruct(idx)
            score = float(np.dot(query_vec, vec))


            if score_threshold and score < score_threshold:
                continue

            results.append(
                {
                    "score": score,
                    "chunk": chunk,
                }
            )

        return results
