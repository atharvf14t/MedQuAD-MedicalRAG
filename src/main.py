"""Main CLI orchestrator for RAG system tasks.

Provides command-line interfaces for:
- Building corpus from MedQuAD dataset
- Building FAISS indexes from chunks
- Running queries
- Running evaluation
"""
import argparse
import json
import time
import numpy as np
from src.retrieval.dense_retriever import DenseRetriever
from src.generation.rag_pipeline import RAGGenerator
from src.evaluation.evaluate import evaluate_dataset


from src.index.build_index import build_faiss_index
from src.embeddings.sentence_transformer import SentenceTransformerEmbedder
from src.index.faiss_index import FaissIndex

from pathlib import Path

from src.data.parser import parse_medquad
from src.data.chunker import TokenChunker
from src.utils.logger import get_logger

logger = get_logger(__name__)

def build_corpus(args):
    """Build chunks from MedQuAD dataset.
    
    Args:
        args: argparse Namespace with data_dir, output, tokenizer, chunk_size, chunk_overlap
    """
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

def build_index(args):
    """Build FAISS index from chunks.
    
    Args:
        args: argparse Namespace with chunks_path, index_path, embedding_model
    """
    logger.info("Loading chunks...")
    with open(args.chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]

    logger.info(f"Total chunks: {len(texts)}")

    # Embeddings
    embedder = SentenceTransformerEmbedder(args.embedding_model)

    logger.info("Computing embeddings...")
    embeddings, throughput, embed_time = embedder.embed_texts(texts)

    logger.info(f"Embedding throughput: {throughput:.2f} chunks/sec")

    dim = embeddings.shape[1]

    # Build FAISS index
    index = FaissIndex(dim)

    logger.info("Building FAISS index...")
    index_time = index.build(embeddings)

    logger.info(f"Index build time: {index_time:.2f} sec")

    # Save index
    index.save(args.index_path)

    # Save embedding metadata
    meta = {
        "embedding_model": args.embedding_model,
        "embedding_dim": dim,
        "num_chunks": len(texts),
        "throughput_chunks_per_sec": throughput,
        "embedding_time_sec": embed_time,
        "index_time_sec": index_time,
    }

    meta_path = args.index_path.replace(".faiss", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Index saved to: {args.index_path}")

def main():
    """Parse CLI arguments and execute appropriate command."""
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

    # ---- build-index ----
    index_parser = subparsers.add_parser("build-index")
    index_parser.add_argument("--chunks_path", required=True)
    index_parser.add_argument("--index_path", required=True)
    index_parser.add_argument("--embedding_model", required=True)


    # ---- query ----
    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("--index_path", required=True)
    query_parser.add_argument("--embedding_model", required=True)
    query_parser.add_argument("--question", required=True)
    query_parser.add_argument("--top_k", type=int, default=5)
    query_parser.add_argument("--use_mmr", action="store_true")

    # ---- evaluate ----
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--index_path", required=True)
    eval_parser.add_argument("--embedding_model", required=True)
    eval_parser.add_argument("--eval_file", required=True)
    eval_parser.add_argument("--top_k", type=int, default=5)


    args = parser.parse_args()

    if args.command == "build-corpus":
        build_corpus(args)
    elif args.command == "build-index":
        build_faiss_index(
            args.chunks_path,
            args.index_path,
            args.embedding_model,
        )

    elif args.command == "query":
        retriever = DenseRetriever(
            args.index_path,
            args.embedding_model,
        )

        if args.use_mmr:
            results = retriever.search_with_mmr(
                args.question,
                top_k=args.top_k
            )
        else:
            results = retriever.search(
                args.question,
                top_k=args.top_k
            )

        print("\nTop results:\n")
        for r in results:
            print(f"Score: {r['score']:.4f}")
            print(r["chunk"]["text"])
            print("-" * 60)
    
    elif args.command == "evaluate":
        retriever = DenseRetriever(
            args.index_path,
            args.embedding_model,
        )

        generator = RAGGenerator()

        with open(args.eval_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        metrics = evaluate_dataset(
            dataset,
            retriever,
            generator,
            top_k=args.top_k,
        )

        print("\nEvaluation results:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()