"""
Retrieval evaluation: Recall@K and MRR@10
"""
import json
from typing import List, Dict
import numpy as np


class RetrievalEvaluator:
    """
    Evaluate retrieval performance.
    
    Metrics:
    - Recall@K: Fraction of queries where any retrieved chunk has the same doc_id
    - MRR@10: Mean reciprocal rank of the first relevant chunk (truncated at 10)
    """
    
    def __init__(self):
        pass
    
    def is_relevant(self, retrieved_chunk: Dict, gold_doc_id: str) -> bool:
        """Check if a retrieved chunk is relevant (matches doc_id)."""
        return retrieved_chunk.get("doc_id") == gold_doc_id
    
    def recall_at_k(self, retrieved_chunks: List[Dict], gold_doc_id: str, k: int) -> int:
        """
        Compute Recall@K for a single query.
        Returns 1 if any of the top-k chunks is relevant, 0 otherwise.
        """
        top_k = retrieved_chunks[:k]
        for chunk in top_k:
            if self.is_relevant(chunk, gold_doc_id):
                return 1
        return 0
    
    def mrr_at_k(self, retrieved_chunks: List[Dict], gold_doc_id: str, k: int = 10) -> float:
        """
        Compute MRR@K for a single query.
        Returns the reciprocal rank of the first relevant chunk, or 0 if not found in top-k.
        """
        top_k = retrieved_chunks[:k]
        for rank, chunk in enumerate(top_k, 1):
            if self.is_relevant(chunk, gold_doc_id):
                return 1.0 / rank
        return 0.0
    
    def evaluate_batch(self, eval_set: List[Dict], retrieval_results: List[List[Dict]], verbose: bool = False) -> Dict:
        """
        Evaluate retrieval results for a batch of queries.
        
        Args:
            eval_set: List of evaluation items with doc_id
            retrieval_results: List of retrieved chunk lists (one per query)
            verbose: Whether to print per-query results (default: False)
        
        Returns:
            Dict with metrics: Recall@1, Recall@5, Recall@10, MRR@10
        """
        if len(eval_set) != len(retrieval_results):
            raise ValueError("eval_set and retrieval_results must have same length")
        
        recall_1 = []
        recall_5 = []
        recall_10 = []
        mrr_10 = []
        
        if verbose:
            print("\n" + "=" * 100)
            print("Retrieval Evaluation Results (Per Query)")
            print("=" * 100)
        
        for query_idx, (eval_item, retrieved) in enumerate(zip(eval_set, retrieval_results), 1):
            gold_doc_id = eval_item["doc_id"]
            
            # Extract referenced doc_ids from retrieved chunks
            referenced_doc_ids = []
            for chunk in retrieved:
                doc_id = chunk.get("doc_id")
                if doc_id and doc_id not in referenced_doc_ids:
                    referenced_doc_ids.append(doc_id)
            
            # Check if gold doc_id is in top-1, top-5, top-10
            hit_at_1 = self.recall_at_k(retrieved, gold_doc_id, k=1)
            hit_at_5 = self.recall_at_k(retrieved, gold_doc_id, k=5)
            hit_at_10 = self.recall_at_k(retrieved, gold_doc_id, k=10)
            
            recall_1.append(hit_at_1)
            recall_5.append(hit_at_5)
            recall_10.append(hit_at_10)
            mrr_10.append(self.mrr_at_k(retrieved, gold_doc_id, k=10))
            
            if verbose:
                print(f"\nQuery {query_idx}:")
                print(f"  Question: {eval_item.get('question', 'N/A')[:80]}...")
                print(f"  Gold Doc ID: {gold_doc_id}")
                print(f"  Referenced Doc IDs (top-10): {referenced_doc_ids[:10]}")
                print(f"  Hit@1: {hit_at_1}, Hit@5: {hit_at_5}, Hit@10: {hit_at_10}")
        
        if verbose:
            print("\n" + "=" * 100)
        
        return {
            "recall@1": np.mean(recall_1),
            "recall@5": np.mean(recall_5),
            "recall@10": np.mean(recall_10),
            "mrr@10": np.mean(mrr_10),
        }
