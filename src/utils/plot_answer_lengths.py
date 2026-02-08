import json
import argparse
import matplotlib.pyplot as plt


def main(corpus_path: str):
    # Load corpus
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    token_counts = []

    for item in corpus:
        answer = item.get("answer", "")
        tokens = answer.split()  # simple whitespace tokenization
        token_counts.append(len(tokens))

    total = len(token_counts)
    avg = sum(token_counts) / total if total > 0 else 0

    print(f"Total QA pairs: {total}")
    print(f"Average answer length (tokens): {avg:.2f}")

    # Plot histogram
    plt.figure()
    plt.hist(token_counts, bins=30)
    plt.xlabel("Number of tokens in answer")
    plt.ylabel("Frequency")
    plt.title("Distribution of Answer Lengths in corpus.json")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="corpus.json",
        help="Path to corpus.json file",
    )
    args = parser.parse_args()

    main(args.corpus_path)
