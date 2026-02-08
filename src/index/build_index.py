from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
from tqdm import tqdm
from pathlib import Path


def build_faiss_index(chunks_path, index_path, model_name):
    # Load chunks
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]

    print(f"Loaded {len(texts)} chunks")

    # Load embedding model
    model = SentenceTransformer(model_name)

    # Encode
    embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,
)


    embeddings = np.array(embeddings).astype("float32")

    print("Embedding shape:", embeddings.shape)

    # Build FAISS index (cosine similarity)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, index_path)

    # Save metadata for retrieval
    meta_path = Path(index_path).with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Index saved to {index_path}")
    print(f"Metadata saved to {meta_path}")
