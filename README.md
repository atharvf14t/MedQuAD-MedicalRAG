# RAG System – MedQuAD Assignment

## Project Setup & Running Instructions (Answers to questions below this section!)

### 1. Environment Setup

Clone or navigate to the project directory and create a Python virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On Windows (CMD):
.venv\Scripts\activate.bat

# On macOS/Linux/WSL:
source .venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

**Note:** On Windows, `faiss-cpu` installation may require a pre-built wheel. If installation fails, consider using WSL or follow FAISS documentation for platform-specific wheels.

---

### 2. Data Preparation & Index Building

#### Option A: Use Pre-built Chunks (Recommended)

Pre-built chunk files are provided in `data/`:
- `data/chunks_256_50.json` (256 chunk size, 50 overlap)
- `data/chunks_500_100.json` (500 chunk size, 100 overlap)
- `data/chunks_150_40.json` (150 chunk size, 40 overlap)

Build a FAISS index from existing chunks:

```bash
python -m src.main build-index \
  --chunks_path data/chunks_256_50.json \
  --index_path data/all-minilm-L6-v2_256_50.faiss \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2
```

**Command options:**
- `--chunks_path` (required): Path to the chunks JSON file
- `--index_path` (required): Output path for the FAISS index file
- `--embedding_model` (required): Embedding model name (e.g., `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`)

This creates:
- FAISS index: `data/all-minilm-L6-v2_256_50.faiss`
- Metadata: `data/all-minilm-L6-v2_256_50_meta.json`

#### Option B: Build Chunks from Raw MedQuAD Data

If you want to create chunks from the raw dataset:

```bash
python -m src.main build-corpus \
  --data_dir MedQuAD-master1 \
  --output data/chunks.json \
  --chunk_size 256 \
  --chunk_overlap 50
```

**Command options:**
- `--data_dir` (required): Path to MedQuAD dataset root directory
- `--output` (optional, default: `data/chunks.json`): Output path for chunks JSON
- `--tokenizer` (optional, default: `sentence-transformers/all-MiniLM-L6-v2`): Tokenizer model for chunking
- `--chunk_size` (optional, default: 256): Size of each chunk in tokens
- `--chunk_overlap` (optional, default: 50): Overlap between consecutive chunks in tokens

---

### 3. Run Full Evaluation Pipeline

The evaluation pipeline retrieves documents, generates answers, and computes metrics (recall, MRR, ROUGE-L, citation coverage).
Important: If you want to create a fresh evaluation set, kindly delete data/eval_set.json file if it exists. 
**Basic dense retrieval:**

```bash
python -m src.evaluation.run_evaluation \
  --retriever dense \
  --index_path data/all-minilm-L6-v2_256_50.faiss \
  --metadata_path data/all-minilm-L6-v2_256_50.meta.json \
  --embedding_model all-minilm-L6-v2 \
  --top_k 5 \
  --generation_model google/flan-t5-base \
  --device cpu \
  --results_dir results
```

**MMR (Maximal Marginal Relevance) retrieval:**

```bash
python -m src.evaluation.run_evaluation \
  --retriever mmr \
  --index_path data/all-minilm-L6-v2_256_50.faiss \
  --metadata_path data/all-minilm-L6-v2_256_50.meta.json \
  --embedding_model all-minilm-L6-v2 \
  --top_k 5 \
  --lambda_param 0.7 \
  --generation_model google/flan-t5-base \
  --device cpu \
  --results_dir results
```

**Hybrid retrieval (BM25 + Dense):**

```bash
python -m src.evaluation.run_evaluation \
  --retriever hybrid \
  --index_path data/all-minilm-L6-v2_256_50.faiss \
  --metadata_path data/all-minilm-L6-v2_256_50.meta.json \
  --embedding_model all-minilm-L6-v2 \
  --top_k 5 \
  --alpha 0.5 \
  --generation_model google/flan-t5-base \
  --device cpu \
  --results_dir results
```

**Evaluation Pipeline Command Options:**
- `--retriever` (required): `dense`, `mmr`, or `hybrid`
- `--index_path` (required): Path to FAISS index file
- `--metadata_path` (required): Path to index metadata JSON file
- `--embedding_model` (required): Embedding model name (e.g., `all-minilm-L6-v2`, `bge-small-en-v1.5`)
- `--top_k` (optional, default: 5): Number of documents to retrieve
- `--lambda_param` (optional, default: 0.7): MMR diversity parameter (0-1, higher = more diversity)
- `--alpha` (optional, default: 0.5): Hybrid retriever weight for dense vs BM25 (0-1, 0 = pure BM25, 1 = pure dense)
- `--generation_model` (optional, default: `google/flan-t5-base`): Generation model (e.g., `google/flan-t5-large`)
- `--max_new_tokens` (optional, default: 128): Maximum tokens to generate
- `--eval_size` (optional, default: 100): Number of evaluation samples
- `--seed` (optional, default: 42): Random seed for evaluation set
- `--prompt_file` (optional, default: `src/generation/prompts.json`): Path to prompts JSON
- `--prompt_name` (optional, default: `default`): Name of prompt to use
- `--device` (optional, default: `cpu`): Device to use (`cpu` or `cuda` for GPU)
- `--results_dir` (optional, default: `results`): Directory to save results CSV

Results are appended to `{results_dir}/metrics.csv`.

---

### 4. Single Query Testing (Generator CLI)

Test retrieval and generation on a single query:

```bash
python -m src.generation.generator \
  --question "What causes diabetes?" \
  --retriever dense \
  --index_path data/all-minilm-L6-v2_256_50.faiss \
  --metadata_path data/all-minilm-L6-v2_256_50.meta.json \
  --embedding_model all-minilm-L6-v2 \
  --top_k 5 \
  --generation_model google/flan-t5-base \
  --device cpu
```

**Generator Command Options:**
- `--question` (required): Query/question string
- `--retriever` (required): `dense`, `mmr`, or `hybrid`
- `--index_path` (required): Path to FAISS index file
- `--metadata_path` (required): Path to index metadata JSON file
- `--embedding_model` (required): Embedding model name
- `--top_k` (optional, default: 3): Number of documents to retrieve
- `--lambda_param` (optional, default: 0.7): MMR parameter (if using MMR)
- `--alpha` (optional, default: 0.5): Hybrid weight (if using hybrid)
- `--generation_model` (optional, default: `google/flan-t5-base`): Generation model
- `--prompt_file` (optional, default: `src/generation/prompts.json`): Custom prompts file
- `--prompt_name` (optional, default: `default`): Prompt template name
- `--device` (optional, default: `cpu`): Device (`cpu` or `cuda`)

---

### 5. Troubleshooting
For specific questions, contact me here: atharvdxb14@gmail.com
**Import errors (ModuleNotFoundError):**

Always run from the project root as a module:
```bash
python -m src.evaluation.run_evaluation ...
```

Alternatively, set `PYTHONPATH`:
```bash
# Linux/macOS/WSL:
PYTHONPATH=. python src/evaluation/run_evaluation.py ...

# Windows CMD:
set PYTHONPATH=. && python src/evaluation/run_evaluation.py ...
```

**FAISS installation issues on Windows:**

If `pip install faiss-cpu` fails, try:
```bash
pip install faiss-cpu-no-avx2
```

Or use WSL (Windows Subsystem for Linux).

**GPU support:**
Please note: I've not used GPU, hence this functionality is not tested yet.
To use GPU (CUDA), install GPU versions of dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install faiss-gpu
```

Then pass `--device cuda` to commands.

---

## Answers to Assignment Questions

### 1. Chunking strategy: what did you implement and why? What failure modes did you observe?

The MedQuAD dataset contains **47,457 QA pairs**, but during initial data inspection it was observed that **31,055 questions did not contain answers**. This left **16,402 QA pairs with valid answers**. Since unanswered questions would introduce noise and increase hallucination risk, the corpus was built using only QA pairs that had non-empty answers. This significantly improved retrieval quality and reduced the chances of the generator producing unsupported outputs.

A statistical analysis of the answer lengths was performed. The histogram of answer token lengths showed:

- **Average answer length:** 201.38 tokens  
- **90% of answers:** under 425 tokens  

Based on this distribution, configurable chunking was implemented with multiple strategies:

1. **Chunk size: 256, overlap: 50**
   - Balanced configuration.
   - Suitable for most answers.
   - Improves precision by keeping chunks concise.

2. **Chunk size: 500, overlap: 100**
   - Designed to include full answers in a single chunk.
   - Improves recall because relevant information is less likely to be split.

3. **Chunk size: 150, overlap: 40**
   - Intended for high-precision retrieval.
   - Not tested due to time constraints and compute limitations.

#### Failure modes observed
- Some long answers were still split across chunks, causing partial context retrieval.
- Overlap sometimes produced near-duplicate chunks in the top-k results.
- Larger chunk sizes increased retrieval noise, which slightly reduced generation quality.

---

### 2. Embeddings: why did one embedding model win or lose?

Two embedding models were used:

1. **sentence-transformers/all-MiniLM-L6-v2**  
   - Recommended in the assignment.
   - Lightweight and fast.
   - Common baseline for semantic retrieval.

2. **BAAI/bge-small-en-v1.5**  
   - Lightweight retrieval-optimized model.
   - Specifically designed for dense search tasks.
   - Known to perform better in semantic retrieval benchmarks.

Both models produce **384-dimensional embeddings**, enabling fair comparison.

#### Observed results (dense retrieval, same parameters)

| Model | MRR@10 | Recall@1 | Recall@5 | Recall@10 | ROUGE-L |
|------|--------|----------|----------|-----------|--------|
| MiniLM | 0.878 | 0.81 | 0.98 | 0.98 | 0.131 |
| BGE-small | **0.921** | **0.86** | **1.00** | **1.00** | **0.137** |

The BGE model outperformed MiniLM across all retrieval metrics. This is likely because BGE is trained specifically for semantic search tasks, producing embeddings that align queries and documents more effectively.

However, MiniLM had slightly faster indexing times and lower computational overhead, making it more suitable for resource-constrained environments.

---

### 3. Retrieval: what changed when you varied `top_k`? Did you try MMR or hybrid retrieval?

Experiments were conducted with different `top_k` values and retrieval methods.

#### Effect of top_k

| top_k | MRR@10 | Recall@1 | Recall@5 | ROUGE-L |
|------|--------|----------|----------|--------|
| 5 | 0.878 | 0.81 | 0.98 | 0.131 |
| 3 | 0.867 | 0.81 | 0.93 | **0.229** |

Reducing `top_k` slightly reduced recall, but significantly improved ROUGE-L. This indicates that **less noisy context improves generation quality**, even if retrieval recall decreases.

#### Retrieval method comparison

| Method | MRR@10 | Recall@1 | Recall@5 | ROUGE-L |
|--------|--------|----------|----------|--------|
| Dense | 0.878 | 0.81 | 0.98 | 0.131 |
| MMR | 0.879 | 0.81 | 0.97 | **0.153** |
| Hybrid | 0.859 | 0.78 | 0.96 | 0.115 |

Observations:

- **MMR improved ROUGE-L** by promoting diversity in retrieved chunks.
- Hybrid retrieval slightly reduced performance, likely due to:
  - Noise from lexical matching.
  - Mismatch between dense and BM25 scoring.

---

### 4. Evaluation: which metric best matched real quality? Which was misleading?

Among all metrics, **ROUGE-L** and **Recall@5** are the best indicator of real answer quality because ROUGE-L directly compares generated answers with reference answers. For our case it is quite low because we are using very weak generation model due to compute limitations. 
Recall@5 provides a good estimate about the retrieval quality, since the number of retrieved chunks considered is neither too low nor too high. 

**Recall@10** was misleading in many cases. It remained very high across runs, but generation quality varied significantly.

---

### 5. Faithfulness: how did you reduce hallucinations?

Several techniques were used:

1. **Data cleaning**
   - Removed all QA pairs without answers.
   - Reduced noise and hallucination risk.

2. **Prompt constraints**
   - System prompt instructs the model to:
     - Use only retrieved context.
     - Provide citations. - not getting good response due to weak generation model.
     - Return “insufficient information” when needed.

3. **Controlled top_k**
   - Reduced irrelevant context.
   - Improved answer precision.

4. **Chunk overlap**
   - Prevented context loss at chunk boundaries.

---

### 6. If you had two more weeks, what would you improve first?

I'll implement and run tests for cross-encoder re-ranker to check if it improves the relevance of the top retrieved chunks.

Other improvements:

1. Implement cross-encoder re-ranking.
2. Tune hybrid alpha parameter and mmr lambda - extensive hyperparameter tuning experiments.
3. Experiment with new chunking strategies.
4. Try new prompts for LLM direction.
5. Use stronger generation models such as GPT-4o-mini.
6. Try to gain more knowledge on what queries are asked frequently and optimize on it.
7. Optimize corpus further through data analysis.


---
## Limitations and Error Analysis

During qualitative inspection, many cases with correct retrieval context still produced suboptimal answers. 
This is primarily due to the use of the `google/flan-t5-base` model, which is relatively small and not 
optimized for long-context instruction-following or medical QA tasks.

In several examples, the retriever returned highly relevant chunks, but the generator:
- Produced incomplete answers
- Paraphrased key facts incorrectly
- Omitted important details from the context

An attempt was made to run evaluation using `google/flan-t5-large`, which produced noticeably better 
qualitative outputs. However, full evaluation could not be completed due to memory and computation 
constraints on the available hardware.

This suggests that:
- Retrieval quality is already strong (high recall and MRR).
- Generation quality is currently the primary bottleneck.
- Using a stronger instruction-tuned model (e.g., FLAN-T5-large or GPT-4o-mini) would likely 
  improve ROUGE and faithfulness metrics significantly.

--- 

## Methodology of Data Processing

The MedQuAD dataset contains **47,457 QA pairs**. During preprocessing:

- Only **16,402 QA pairs had answers**.
- **31,055 QA pairs had no answers** and were removed.

This reduced noise and prevented the generator from hallucinating unsupported content.

After plotting the answer length distribution:

- Average answer length: **201.38 tokens**
- 90% of answers were under **425 tokens**

The histogram is available at:
results/answer_lengths.png

---

## Chunking Strategy

Configurable chunking was implemented with multiple configurations:

1. **Chunk size: 256, overlap: 50**
   - Balanced precision and recall.

2. **Chunk size: 500, overlap: 100**
   - Designed to include full answers.

3. **Chunk size: 150, overlap: 40**
   - High-precision configuration (not fully tested).

---

## Indexing

Each chunk configuration (256 and 500) was embedded using two models:

1. **all-MiniLM-L6-v2**
   - Assignment-recommended baseline.

2. **BAAI/bge-small-en-v1.5**
   - Retrieval-optimized lightweight model.
   - Better semantic alignment between queries and documents.

Both models:
- Embedding dimension: **384**

### Indexing statistics
This is for all-miniLM-v2 model. 
**500 token chunks**
- Embedding shape: (20,987, 384)
- Overlap: 100
- Indexing time: ~7 minutes
- Batch size: 32
- Throughput: ~54 chunks/sec

**256 token chunks**
- Embedding shape: (31,392, 384)
- Overlap: 50
- Indexing time: ~9 minutes
- Batch size: 32
- Throughput: ~60 chunks/sec

bge-small-en-v2 embedding model took about 40-50 minitues for each chunk size.
---

## Index Metadata Structure

```json
{
  "chunk_id": "CancerGov_0000001_4_7_c0",
  "doc_id": "CancerGov_0000001_4",
  "qa_id": "CancerGov_0000001_4_7",
  "text": "question : question text here, answer: answer text here...",
  "metadata": {
    "source": "CancerGov",
    "url": "https://www.cancer.gov/types/leukemia/patient"
  }
}

```

## Retrieval

Three retrieval techniques were implemented:

1. Dense retrieval  
2. MMR re-ranking with tunable lambda  
3. Hybrid retrieval (BM25 + Dense) with tunable alpha  

All retrievers support configurable `top_k`.

---

## Generation

Generation was performed using:

- `google/flan-t5-base`
- `google/flan-t5-large`

The generation step uses system prompts loaded from JSON and supports configurable embedding models and retrievers.

---

## Evaluation

The evaluation pipeline:

1. Creates an evaluation set from the corpus.
2. Uses configurable seed and evaluation size.
3. Retrieves documents for each query.
4. Computes:
   - Recall@1
   - Recall@5
   - Recall@10
   - MRR@10
5. Generates answers.
6. Computes:
   - ROUGE-L (F1)
   - Citation coverage
7. Logs system metrics.

