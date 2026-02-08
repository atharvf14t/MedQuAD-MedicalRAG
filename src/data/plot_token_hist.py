import json
import argparse
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
from transformers import AutoTokenizer


def load_chunks(path: Path, limit: int):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected JSON array of chunk objects")
    return data[:limit]


def token_lengths(chunks, tokenizer):
    lengths = []
    for rec in chunks:
        text = rec.get("text", "")
        tokens = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(tokens))
    return lengths


def plot_hist(lengths, bins: int, out_path: Path, title: str):
    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=bins, color="#4C72B0", edgecolor="black")
    plt.xlabel("Token count per chunk")
    plt.ylabel("Number of chunks")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot token-length histogram for chunk file")
    parser.add_argument("input", help="Path to chunks JSON file")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Tokenizer model name")
    parser.add_argument("--limit", type=int, default=5000, help="Number of chunks to process")
    parser.add_argument("--bins", type=int, default=50, help="Histogram bins")
    parser.add_argument("--out", default="token_length_hist.png", help="Output plot path")

    args = parser.parse_args()

    input_path = Path(args.input)
    out_path = Path(args.out)

    chunks = load_chunks(input_path, args.limit)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    lengths = token_lengths(chunks, tokenizer)

    stats = {
        "count": len(lengths),
        "mean": mean(lengths) if lengths else 0,
        "median": median(lengths) if lengths else 0,
        "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
    }

    print("Token length stats:", stats)

    title = f"Token length histogram (n={stats['count']})"
    plot_hist(lengths, args.bins, out_path, title)

    print(f"Saved histogram to: {out_path}")


if __name__ == "__main__":
    main()
