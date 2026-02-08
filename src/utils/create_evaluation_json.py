"""Sample evaluation set from corpus.

Creates a small random sample for evaluation testing.
"""
import json
import random
from pathlib import Path


def sample_eval_set(input_path, output_path, sample_size=20, seed=42):
    """Sample QA pairs to create evaluation set.
    
    Args:
        input_path: Path to corpus.json
        output_path: Path to save evaluation set JSON
        sample_size: Number of samples to draw
        seed: Random seed for reproducibility
    """

    random.seed(seed)

    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Filter only valid QA pairs
    valid = [
        {
            "question": r["question"],
            "answer": r["answer"],
        }
        for r in records
        if r.get("answer") and r["answer"].strip()
    ]

    print(f"Total valid QA pairs: {len(valid)}")

    if len(valid) < sample_size:
        raise ValueError("Not enough QA pairs to sample from.")

    sampled = random.sample(valid, sample_size)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)

    print(f"Saved {sample_size} samples to {output_path}")


def main():
    """CLI for sampling evaluation set."""
    import argparse

    parser = argparse.ArgumentParser(description="Sample evaluation QA set")
    parser.add_argument("--input_path", required=True, help="Path to corpus.json")
    parser.add_argument("--output_path", required=True, help="Path to eval.json")
    parser.add_argument("--sample_size", type=int, default=200)

    args = parser.parse_args()

    sample_eval_set(
        args.input_path,
        args.output_path,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
