Methodology of data processing:
MedQuAD dataset contains 47,457 QA pairs. Upon doing some data research I found:
1. Only 16,402 QA pairs had answers. 31,055 QA pairs didn't had answers to the question. So I made the corpus containing only QA pairs with answers- this reduces the chances of hallucinations and unnecessary information. 
2. on ploting the graph for frequency of QA vs tokens: Average answer length (tokens): 201.38 and 90% of answer are of length less than 425 tokens(See the histogram at results/answer_lengths.png)

Used configurable chunking to create 2 chunk types: 
1. Chunk size: 256, Chunk overlap: 50
2. Chunk size: 500, Chunk overlap: 100 - this was done keeping in mind that this will incorporate complete answers of many QA pairs in one chunk.
3. Chunk size: 150, Chunk overlap: 40 - PS. didn't do testing on this due to long waiting times for creating index. - My computer is a bit slow.



Indexing: 
Each of the chunk sizes(1 & 2) are embedded using 2 models: 
1. sentence-transformers/all-MiniLM-L6-v2 - recommended in the assignment
2. BAAI/bge-small-v1.5 - Light model with (write why we used this here)

Embedding dimension for both models: 384. Embedding shape (20987, 384) for 500 token chunk size. 100 token chunk overlap.
Indexing time for all-MiniLM-L6-V2 on 500 token chunk size is about 7 minutes. I used batch size of 32. Average embedding throughput- 1.8 it/s, about 58 chunks/sec. 

Embedding shape: (31392, 384)
Indexing time for all-MiniLM-L6-V2 on 256 token chunk size is about 9 minutes. I used batch size of 32. Average embedding throughput- 2.01 it/s, about 58 chunks/sec.

Indexing metadata structure: 
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
1. Chunking strategy: I've used QA pair based chunking. On data analysis I found that the average token length of each answer is 201.38 tokens. On plotting histogram( see at results/answer_lengths.png)- it was found that 90% of the QA pairs are of token length lesser than 425. - So I implemented chunking with 2 sizes 
- 256 chunk size, 50 overlap - ideal for providing specific information.
- 500 chunk size, 100 overlap - Keeping in mind that this will cover most of the answers in one chunk, this can improve the recall metrics.

2. Embeddings: BAAI/bge-small-en-v1.5 model performed better than all-MiniLM-L6-v2 model over all metrics(recall@1, recall@5, recall@10, MRR@10) - see run 2 &3 in metrics.csv - for similar chunk size, overlap size and retrieval method (dense). This highlights the fact that using better embedding models provides better retrieval. Also semantic chunking is performed for both the models which further improved the performance.

3. Top_K value: Reducing the top_k value reduced the recall@5 and recall@10 when done on similar parameters. This is due to the fact that we considered lesser number of retrieval doc_ids to compare with the golden doc_id. ...(reason for reduced mrr@10 value). See run 2 & 4 in metrics.csv.
- 

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
