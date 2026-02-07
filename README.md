build chunks command: python -m src.main build-corpus \
  --data_dir MedQuAD-master1 \
  --output data/chunks.json \
  --chunk_size 256 \
  --chunk_overlap 50



build index command: python -m src.main build-index   --chunks_path data/chunks.json   --index_path data/index_minilm.faiss   --embedding_model sentence-transformers/all-MiniLM-L6-v2
<!-- it took about 45 mins to index with 17it/s as average for 31000 chunks. Can be improved by batch indexing instead of single indexing -->

Vanilla dense retrieval: python -m src.main query \
  --index_path data/index_minilm.faiss \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
  --question "What causes diabetes?" \
  --top_k 5


mmr retrieval: python -m src.main query \
  --index_path data/index_minilm.faiss \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
  --question "What causes diabetes?" \
  --top_k 5 \
  --use_mmr


