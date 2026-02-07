from typing import List, Dict
from transformers import AutoTokenizer


class TokenChunker:
    def __init__(
        self,
        tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 256,
        chunk_overlap: int = 50,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_record(self, record: Dict) -> List[Dict]:
        """
        Chunk a single QA record into token chunks.
        """
        text = f"Question: {record['question']}\nAnswer: {record['answer']}"
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        chunks = []
        start = 0
        chunk_idx = 0

        step = self.chunk_size - self.chunk_overlap

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunks.append(
                {
                    "chunk_id": f"{record['qa_id']}_c{chunk_idx}",
                    "doc_id": record["doc_id"],
                    "qa_id": record["qa_id"],
                    "text": chunk_text,
                    "metadata": {
                        "source": record["source"],
                        "url": record["url"],
                    },
                }
            )

            chunk_idx += 1
            start += step

        return chunks

    def chunk_corpus(self, records: List[Dict]) -> List[Dict]:
        all_chunks = []
        for record in records:
            all_chunks.extend(self.chunk_record(record))
        return all_chunks
