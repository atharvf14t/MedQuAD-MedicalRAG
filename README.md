build chunks command: python -m src.main build-corpus \
  --data_dir MedQuAD-master1 \
  --output data/chunks.json \
  --chunk_size 256 \
  --chunk_overlap 50
