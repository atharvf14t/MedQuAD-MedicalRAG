"""
Build evaluation set from corpus by sampling questions deterministically.
"""
import json
import random
from pathlib import Path
from typing import List, Dict


def load_corpus(corpus_path: str) -> List[Dict]:
    """Load corpus from JSON file."""
    with open(corpus_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_eval_set(corpus_path: str, output_path: str, eval_size: int = 100, seed: int = 42):
    """
    Build evaluation set by sampling questions from corpus.
    
    For each question, the relevant document is the original doc_id.
    A retrieval is a "hit" if any retrieved chunk has the same doc_id.
    
    Args:
        corpus_path: Path to corpus.json
        output_path: Path to save eval set
        eval_size: Number of questions to sample (default 100)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    corpus = load_corpus(corpus_path)
    
    # Sample eval_size questions from corpus
    eval_samples = random.sample(corpus, min(eval_size, len(corpus)))
    
    # Build eval set with question, reference answer, and gold doc_id
    eval_set = []
    for sample in eval_samples:
        eval_item = {
            "qa_id": sample["qa_id"],
            "doc_id": sample["doc_id"],  # Relevant doc for this question
            "question": sample["question"],
            "answer": sample["answer"],  # Reference answer
            "source": sample["source"],
            "url": sample["url"],
        }
        eval_set.append(eval_item)
    
    # Save eval set
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2, ensure_ascii=False)
    
    print(f"Created evaluation set with {len(eval_set)} samples")
    print(f"Saved to: {output_path}")
    
    return eval_set


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build evaluation set from corpus")
    parser.add_argument("--corpus_path", default="data/corpus.json")
    parser.add_argument("--output_path", default="data/eval_set.json")
    parser.add_argument("--eval_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    build_eval_set(
        corpus_path=args.corpus_path,
        output_path=args.output_path,
        eval_size=args.eval_size,
        seed=args.seed
    )
