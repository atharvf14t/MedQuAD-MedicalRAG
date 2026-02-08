"""Build QA-specific chunks using semantic text splitting.

Chunks only the answer portion while prepending the question for context.
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter


class QAChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        """Initialize QAChunker.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def chunk_record(self, record: Dict) -> List[Dict]:
        """Chunk only the answer, but always prepend the question.
        
        Args:
            record: QA record dict
            
        Returns:
            List of chunk dicts with question prepended to each answer chunk
        """
        answer_chunks = self.splitter.split_text(record["answer"])

        chunks = []
        for idx, ans_chunk in enumerate(answer_chunks):
            chunk_text = f"question: {record['question']}\nanswer: {ans_chunk}"

            chunks.append(
                {
                    "chunk_id": f"{record['qa_id']}_c{idx}",
                    "doc_id": record["doc_id"],
                    "qa_id": record["qa_id"],
                    "text": chunk_text,
                    "metadata": {
                        "source": record["source"],
                        "url": record["url"],
                    },
                }
            )

        return chunks


def build_chunks(corpus_path: str, output_path: str, chunk_size: int, chunk_overlap: int):
    """Build and save chunks from corpus.
    
    Args:
        corpus_path: Path to corpus JSON file
        output_path: Output path for chunks JSON
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks
    """
    corpus_path = Path(corpus_path)

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    chunker = QAChunker(chunk_size, chunk_overlap)

    all_chunks = []
    for record in corpus:
        chunks = chunker.chunk_record(record)
        all_chunks.extend(chunks)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build QA-isolated chunks")

    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=100)

    args = parser.parse_args()

    build_chunks(
        corpus_path=args.corpus_path,
        output_path=args.output_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
