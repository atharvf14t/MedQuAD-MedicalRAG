"""Clean whitespace in corpus JSON files.

Removes extra spaces and improves text formatting.
"""
import json
import re
import argparse
from pathlib import Path


def clean_text(text: str) -> str:
    """
    Replace consecutive spaces (2 or more) with a single space.
    """
    if not isinstance(text, str):
        return text
    return re.sub(r"\s{2,}", " ", text).strip()


def clean_corpus(input_path: str, output_path: str):
    """Clean corpus by removing extra whitespace.
    
    Args:
        input_path: Path to input corpus.json
        output_path: Path to save cleaned corpus
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_count = 0

    for record in data:
        if "question" in record:
            record["question"] = clean_text(record["question"])
        if "answer" in record:
            record["answer"] = clean_text(record["answer"])
        cleaned_count += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Processed {cleaned_count} QA records.")
    print(f"Cleaned corpus saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean extra spaces in corpus.json")
    parser.add_argument("--input", required=True, help="Path to input corpus.json")
    parser.add_argument("--output", required=True, help="Path to save cleaned corpus")

    args = parser.parse_args()
    clean_corpus(args.input, args.output)
