"""
Generation evaluation: ROUGE-L and Citation Coverage
"""
import re
from typing import List, Dict
import numpy as np


def compute_rouge_l_f1(prediction: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 score.
    ROUGE-L measures the longest common subsequence (LCS) between prediction and reference.
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    where:
        Precision = LCS / len(prediction)
        Recall = LCS / len(reference)
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    
    # Compute LCS length using dynamic programming
    m, n = len(pred_tokens), len(ref_tokens)
    
    # LCS table
    lcs = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                lcs[i][j] = lcs[i - 1][j - 1] + 1
            else:
                lcs[i][j] = max(lcs[i - 1][j], lcs[i][j - 1])
    
    lcs_length = lcs[m][n]
    
    if lcs_length == 0:
        return 0.0
    
    precision = lcs_length / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = lcs_length / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def has_citation(text: str) -> bool:
    """
    Check if text includes a citation.
    A citation is detected as:
    - Square brackets with content: [...]
    - URLs: http://, https://
    - Source labels: (source), (ref), etc.
    """
    # Check for [citation] style
    if re.search(r'\[.+?\]', text):
        return True
    
    # Check for URLs
    if re.search(r'https?://', text):
        return True
    
    # Check for source attribution patterns
    if re.search(r'\(source[^\)]*\)', text, re.IGNORECASE):
        return True
    
    if re.search(r'\(ref[^\)]*\)', text, re.IGNORECASE):
        return True
    
    return False


class GenerationEvaluator:
    """
    Evaluate generated answers.
    
    Metrics:
    - ROUGE-L F1: Lexical overlap with reference answer
    - Citation Coverage: Fraction of answers with citations
    """
    
    def evaluate_batch(self, predictions: List[str], references: List[str]) -> Dict:
        """
        Evaluate generated answers.
        
        Args:
            predictions: List of generated answers
            references: List of reference answers
        
        Returns:
            Dict with metrics: rouge_l_f1, citation_coverage
        """
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have same length")
        
        rouge_scores = []
        citations = []
        
        for pred, ref in zip(predictions, references):
            rouge_f1 = compute_rouge_l_f1(pred, ref)
            rouge_scores.append(rouge_f1)
            citations.append(1 if has_citation(pred) else 0)
        
        return {
            "rouge_l_f1": np.mean(rouge_scores),
            "citation_coverage": np.mean(citations),
        }
