import argparse
import json
from pathlib import Path

from src.data.parser import parse_medquad
from src.data.chunker import TokenChunker
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_corpus(args):
    logger.info("Scanning and parsing MedQuAD dataset...")
    records = parse_medquad(args.data_dir)

    logger.info(f"Total QA records with answers: {len(records)}")

    chunker = TokenChunker(
        tokenizer_name=args.tokenizer,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    logger.info("Chunking corpus...")
    chunks = chunker.chunk_corpus(records)

    logger.info(f"Total chunks created: {len(chunks)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    logger.info(f"Chunks saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser("build-corpus")
    build_parser.add_argument("--data_dir", required=True)
    build_parser.add_argument("--output", default="data/chunks.json")
    build_parser.add_argument(
        "--tokenizer",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    build_parser.add_argument("--chunk_size", type=int, default=256)
    build_parser.add_argument("--chunk_overlap", type=int, default=50)

    args = parser.parse_args()

    if args.command == "build-corpus":
        build_corpus(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
