# RAG System Evaluation Pipeline

This directory contains a comprehensive evaluation framework for the RAG (Retrieval-Augmented Generation) system.

## Overview

The evaluation pipeline measures three aspects of the RAG system:

### A. Retrieval Evaluation (Mandatory)
Evaluates how well the retriever finds relevant documents.

**Metrics:**
- **Recall@K** (K=1, 5, 10): Fraction of queries where the relevant document appears in top-K results
- **MRR@10**: Mean Reciprocal Rank (average position of first relevant result, truncated at 10)

**Relevance Definition:** For each query, the relevant document is the original `doc_id` of that QA item. A retrieval is a "hit" if any retrieved chunk has the same `doc_id`.

**Implementation:** `retrieval_evaluator.py`

### B. Generation Evaluation (Mandatory)
Evaluates the quality of generated answers compared to reference answers.

**Metrics:**
- **ROUGE-L F1**: Lexical overlap with reference answer (longest common subsequence-based)
- **Citation Coverage**: Fraction of generated answers that include citations (e.g., [citation], URLs, source labels)

**Implementation:** `generation_evaluator.py`

### C. System Metrics (Mandatory)
Reports performance and resource characteristics.

**Metrics:**
- **Index Build Time**: Time to construct the FAISS index (seconds)
- **Index Size**: Total disk space used (FAISS index + metadata, MB)

**Implementation:** `system_metrics.py`

## Usage

### Step 1: Build Evaluation Set (Optional)
Create a reproducible evaluation set by sampling 100 questions from the corpus:

```bash
python -m src.evaluation.eval_set_builder \
  --corpus_path data/corpus.json \
  --output_path data/eval_set.json \
  --eval_size 100 \
  --seed 42
```

This creates `data/eval_set.json` with fixed questions for reproducible evaluation.

### Step 2: Run Full Evaluation

Run the complete evaluation pipeline for your RAG system:

```bash
python -m src.evaluation.run_evaluation \
  --retriever hybrid \
  --index_path data/bge-small-en-v1_500_100.faiss \
  --metadata_path data/bge-small-en-v1_500_100.meta.json \
  --embedding_model BAAI/bge-small-en-v1.5 \
  --generation_model google/flan-t5-base \
  --top_k 5 \
  --alpha 0.6 \
  --eval_size 100 \
  --results_dir results
```

**Retriever Examples:**

Dense Retrieval:
```bash
python -m src.evaluation.run_evaluation \
  --retriever dense \
  --index_path data/bge-small-en-v1_500_100.faiss \
  --metadata_path data/bge-small-en-v1_500_100.meta.json \
  --embedding_model BAAI/bge-small-en-v1.5 \
  --generation_model google/flan-t5-base \
  --results_dir results
```

MMR Retrieval:
```bash
python -m src.evaluation.run_evaluation \
  --retriever mmr \
  --index_path data/bge-small-en-v1_500_100.faiss \
  --metadata_path data/bge-small-en-v1_500_100.meta.json \
  --embedding_model BAAI/bge-small-en-v1.5 \
  --generation_model google/flan-t5-base \
  --lambda_param 0.7 \
  --results_dir results
```

Hybrid Retrieval:
```bash
python -m src.evaluation.run_evaluation \
  --retriever hybrid \
  --index_path data/bge-small-en-v1_500_100.faiss \
  --metadata_path data/bge-small-en-v1_500_100.meta.json \
  --embedding_model BAAI/bge-small-en-v1.5 \
  --generation_model google/flan-t5-base \
  --alpha 0.6 \
  --results_dir results
```

### Output

The evaluation pipeline writes results to `results/metrics.csv` with one row per experiment run:

```csv
timestamp,retriever,embedding_model,generation_model,top_k,eval_size,seed,recall@1,recall@5,recall@10,mrr@10,rouge_l_f1,citation_coverage,index_size_mb,alpha
2025-02-08T10:30:45.123456,hybrid,BAAI/bge-small-en-v1.5,google/flan-t5-base,5,100,42,0.82,0.95,0.98,0.85,0.72,0.45,125.30,0.6
```

## Components

### `eval_set_builder.py`
Builds a deterministic evaluation set by sampling from the corpus:
- Samples N random questions with fixed seed for reproducibility
- Each question includes its reference doc_id (used as relevance ground truth)
- Outputs JSON file with qa_id, doc_id, question, answer, source, url

### `retrieval_evaluator.py`
Evaluates retrieval performance:
- `is_relevant()`: Checks if a chunk matches the gold doc_id
- `recall_at_k()`: Computes Recall@K for a single query
- `mrr_at_k()`: Computes MRR@K for a single query
- `evaluate_batch()`: Evaluates a batch of retrieval results

### `generation_evaluator.py`
Evaluates generated answers:
- `compute_rouge_l_f1()`: Computes ROUGE-L F1 using longest common subsequence
- `has_citation()`: Detects if an answer includes citations
- `evaluate_batch()`: Evaluates a batch of answers

### `system_metrics.py`
Collects system performance metrics:
- `get_index_size()`: Measures FAISS index and metadata sizes
- `collect_metrics()`: Aggregates all system-level metrics

### `run_evaluation.py`
Main orchestration script:
1. Builds or loads evaluation set
2. Initializes retriever
3. Runs retrieval on all queries
4. Evaluates retrieval performance
5. Generates answers using retrieved context
6. Evaluates generated answers
7. Collects system metrics
8. Writes results to CSV

## Command Line Arguments

### Paths
- `--corpus_path`: Path to corpus.json (default: data/corpus.json)
- `--eval_set_path`: Path to save/load evaluation set (default: data/eval_set.json)
- `--results_dir`: Directory to save results (default: results)

### Retriever Configuration
- `--retriever`: Retriever type: dense, mmr, or hybrid (required)
- `--index_path`: Path to FAISS index (required)
- `--metadata_path`: Path to metadata JSON (required)
- `--embedding_model`: Embedding model name (required)
- `--top_k`: Number of top results to retrieve (default: 5)
- `--lambda_param`: Lambda parameter for MMR (default: 0.7)
- `--alpha`: Alpha weight for Hybrid (dense vs BM25, default: 0.5)

### Generator Configuration
- `--generation_model`: Generation model name (default: google/flan-t5-base)
- `--max_new_tokens`: Maximum tokens to generate (default: 128)
- `--prompt_file`: Path to prompts JSON (default: src/generation/prompts.json)
- `--prompt_name`: Prompt template to use (default: default)

### Evaluation Settings
- `--eval_size`: Number of queries in evaluation set (default: 100)
- `--seed`: Random seed for reproducibility (default: 42)
- `--device`: Device to use: cpu or cuda (default: cpu)

## Example Evaluation Protocol

```bash
# 1. Build evaluation set (once)
python -m src.evaluation.eval_set_builder --eval_size 100 --seed 42

# 2. Run dense baseline
python -m src.evaluation.run_evaluation \
  --retriever dense \
  --index_path data/bge-small-en-v1_500_100.faiss \
  --metadata_path data/bge-small-en-v1_500_100.meta.json \
  --embedding_model BAAI/bge-small-en-v1.5 \
  --results_dir results

# 3. Run MMR with different lambdas
python -m src.evaluation.run_evaluation \
  --retriever mmr \
  --index_path data/bge-small-en-v1_500_100.faiss \
  --metadata_path data/bge-small-en-v1_500_100.meta.json \
  --embedding_model BAAI/bge-small-en-v1.5 \
  --lambda_param 0.5 \
  --results_dir results

python -m src.evaluation.run_evaluation \
  --retriever mmr \
  --index_path data/bge-small-en-v1_500_100.faiss \
  --metadata_path data/bge-small-en-v1_500_100.meta.json \
  --embedding_model BAAI/bge-small-en-v1.5 \
  --lambda_param 0.7 \
  --results_dir results

# 4. Run Hybrid with different alphas
python -m src.evaluation.run_evaluation \
  --retriever hybrid \
  --index_path data/bge-small-en-v1_500_100.faiss \
  --metadata_path data/bge-small-en-v1_500_100.meta.json \
  --embedding_model BAAI/bge-small-en-v1.5 \
  --alpha 0.3 \
  --results_dir results

python -m src.evaluation.run_evaluation \
  --retriever hybrid \
  --index_path data/bge-small-en-v1_500_100.faiss \
  --metadata_path data/bge-small-en-v1_500_100.meta.json \
  --embedding_model BAAI/bge-small-en-v1.5 \
  --alpha 0.7 \
  --results_dir results

# 5. Compare results in results/metrics.csv
```

## Notes

- **Reproducibility:** Use `--seed` to ensure reproducible evaluation sets. Default seed=42.
- **Evaluation Set Size:** Minimum 100 queries recommended. Adjust with `--eval_size`.
- **Device:** Use `--device cuda` for GPU acceleration if available.
- **Incremental Results:** Each evaluation appends a row to `results/metrics.csv`. Delete the file to start fresh.
- **Evaluation Time:** Full evaluation (~100 queries) typically takes 5-30 minutes depending on models and hardware.

