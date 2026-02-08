# RAG System – MedQuAD Assignment

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
   - Not fully tested due to indexing time constraints.

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

How to run project: 
commands:

python -m src.evaluation.run_evaluation \
  --retriever dense \
  --index_path data/all-minilm-L6-v2_256_50.faiss \
  --metadata_path data/all-minilm-L6-v2_256_50.meta.json \
  --embedding_model all-minilm-L6-v2 \
  --top_k 5 \
  --generation_model google/flan-t5-base \
  --device cpu \
  --results_dir results


build chunks command: python -m src.main build-corpus \
  --data_dir MedQuAD-master1 \
  --output data/chunks.json \
  --chunk_size 256 \
  --chunk_overlap 50


Retrieval: 
Implemented dense retrieval. MMR re-ranking over dense retrieval with tunable lambda. Hybrid retrieval(BM25 + Dense with tunable alpha)

Retriever supports top_k for all three techniques. 


Generation: 
Using google/flan-t5-base and flan-t5-large models. 


Evaluation: 
Implemented evaluation pipeline that creates an evaluation set json, with tunable seed value and eval size.
Retrieves the documents for each query in eval set.
Computes recall@[1,5,10] and MRR@10 value.
Generates answer for all the queries.
Computes ROUGE-L (F1) averaged over example.
Citation coverage
Provides system metrics

Error Analysis: 




Code quality: 
Used a modular structure (data processing, index, retrieval, generation, evaluation pipeline) in clearly defined folder structure. All the files are runnable via CLI command with tunable flags.
Logging is done to log important info for debugging.
Basic tests: write here...


Answers: 
1. Chunking strategy: Data cleaning to remove QA pairs with empty answers: Only 16402 QA pairs had non empty answers out of 47457 QA pairs. I've used QA pair based chunking. On data analysis I found that the average token length of each answer is 201.38 tokens. On plotting histogram( see at results/answer_lengths.png)- it was found that 90% of the QA pairs are of token length lesser than 425. - So I implemented chunking with 2 sizes 
- 256 chunk size, 50 overlap - ideal for providing specific information.
- 500 chunk size, 100 overlap - Keeping in mind that this will cover most of the answers in one chunk, this can improve the recall metrics.

2. Embeddings: BAAI/bge-small-en-v1.5 model performed better than all-MiniLM-L6-v2 model over all metrics(recall@1, recall@5, recall@10, MRR@10) - see run 2 &3 in metrics.csv - for similar chunk size, overlap size and retrieval method (dense). This highlights the fact that using better embedding models provides better retrieval. Also semantic chunking is performed for both the models which further improved the performance.

3. Top_K value: Reducing the top_k value reduced the recall@5 and recall@10 when done on similar parameters. This is due to the fact that we considered lesser number of retrieval doc_ids to compare with the golden doc_id. ...(reason for reduced mrr@10 value). See run 2 & 4 in metrics.csv.
- 

4. Among all metrics, ROUGE-L and Recall@5 are the best indicators of real answer quality because ROUGE-L directly compares generated answers with reference answers- for our case it is quite low because we are using very weak generation model due to compute limitations. 
Recall@5 provides a good estimate about the retrieval quality, since the number of retrieved chunks considered is neither too low nor too high. 

Recall@10 was misleading in many cases. It remained very high across runs, but generation quality varied significantly.

5. Faithfulness: how did I reduce hallucinaitons: 
Several techniques were used:

Data cleaning

Removed all QA pairs without answers.

Reduced noise and hallucination risk.

Prompt constraints

System prompt instructs the model to:

Use only retrieved context.

Provide citations.

Return “insufficient information” when needed.

Controlled top_k

Reduced irrelevant context.

Improved answer precision.

Chunk overlap

Prevented context loss at chunk boundaries.

6. If I'll have 2 weeks more time, I will do more rigourous testing with better LLM models like gpt-4o-mini. I'll try different chunking techniques, like including only answers in the text field, and question in the metadata. Tuning of different alpha values (for hybrid retrieval) and lambda values for mmr retrieval can tested.  




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


create corpus command: python -m src.data.build_corpus \
  --data_dir MedQuAD-master1 \
  --output_path data/corpus.json
