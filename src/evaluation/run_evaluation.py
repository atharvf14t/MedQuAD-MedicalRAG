"""
Main evaluation pipeline orchestrator.
Runs full RAG evaluation and outputs metrics to CSV.
"""
import json
import argparse
import csv
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from src.evaluation.eval_set_builder import load_corpus, build_eval_set
from src.evaluation.retrieval_evaluator import RetrievalEvaluator
from src.evaluation.generation_evaluator import GenerationEvaluator
from src.evaluation.system_metrics import SystemMetricsCollector
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.mmr_retriever import MMRRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.generator import RAGGenerator, load_system_prompt, build_prompt


def run_evaluation(args):
    """
    Run full evaluation pipeline.
    """
    print("=" * 80)
    print("RAG System Evaluation Pipeline")
    print("=" * 80)
    
    # 1. Build or load evaluation set
    print("\n[Step 1] Setting up evaluation set...")
    eval_set_path = Path(args.eval_set_path)
    
    if not eval_set_path.exists():
        print(f"Creating evaluation set from corpus...")
        eval_set = build_eval_set(
            corpus_path=args.corpus_path,
            output_path=str(eval_set_path),
            eval_size=args.eval_size,
            seed=args.seed
        )
    else:
        with open(eval_set_path, "r", encoding="utf-8") as f:
            eval_set = json.load(f)
        print(f"Loaded existing evaluation set: {len(eval_set)} samples")
    
    # 2. Initialize retriever
    print(f"\n[Step 2] Initializing {args.retriever} retriever...")
    if args.retriever == "dense":
        retriever = DenseRetriever(args.index_path,args.metadata_path, args.embedding_model, device=args.device)
    elif args.retriever == "mmr":
        retriever = MMRRetriever(args.index_path, args.metadata_path, args.embedding_model, device=args.device)
    elif args.retriever == "hybrid":
        retriever = HybridRetriever(args.index_path, args.metadata_path, args.embedding_model, device=args.device)
    else:
        raise ValueError(f"Unknown retriever: {args.retriever}")
    
    # 3. Run retrieval for all queries
    print(f"\n[Step 3] Running retrieval on {len(eval_set)} queries...")
    retrieval_results = []
    retrieval_times = []
    
    for i, eval_item in enumerate(eval_set):
        if (i + 1) % 10 == 0:
            print(f"  Retrieved {i + 1}/{len(eval_set)}")
        
        question = eval_item["question"]
        
        # Retrieve with appropriate parameters and time the call
        t0 = time.perf_counter()
        if args.retriever == "mmr":
            chunks = retriever.retrieve(
                question,
                top_k=args.top_k,
                lambda_param=args.lambda_param
            )
        elif args.retriever == "hybrid":
            chunks = retriever.retrieve(
                question,
                top_k=args.top_k,
                alpha=args.alpha
            )
        else:
            chunks = retriever.retrieve(question, top_k=args.top_k)
        t1 = time.perf_counter()
        retrieval_times.append(t1 - t0)
        
        retrieval_results.append(chunks)
    print(f"Completed retrieval for all queries.", retrieval_results[0] if retrieval_results else "No results")
    # 4. Evaluate retrieval
    print(f"\n[Step 4] Evaluating retrieval performance...")
    retrieval_evaluator = RetrievalEvaluator()
    retrieval_metrics = retrieval_evaluator.evaluate_batch(eval_set, retrieval_results)
    
    print(f"  Recall@1:  {retrieval_metrics['recall@1']:.4f}")
    print(f"  Recall@5:  {retrieval_metrics['recall@5']:.4f}")
    print(f"  Recall@10: {retrieval_metrics['recall@10']:.4f}")
    print(f"  MRR@10:    {retrieval_metrics['mrr@10']:.4f}")
    
    # 5. Generate answers and evaluate
    print(f"\n[Step 5] Generating answers and evaluating...")
    system_prompt = load_system_prompt(args.prompt_file, args.prompt_name)
    generator = RAGGenerator(model_name=args.generation_model, device=args.device)
    
    predictions = []
    for i, (eval_item, context_chunks) in enumerate(zip(eval_set, retrieval_results)):
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{len(eval_set)}")
        
        prompt = build_prompt(system_prompt, context_chunks, eval_item["question"])
        answer = generator.generate(prompt, max_new_tokens=args.max_new_tokens)
        predictions.append(answer)
    
    # 6. Evaluate generation
    print(f"\n[Step 6] Evaluating generation performance...")
    references = [item["answer"] for item in eval_set]
    generation_evaluator = GenerationEvaluator()
    generation_metrics = generation_evaluator.evaluate_batch(predictions, references)
    
    print(f"  ROUGE-L F1:      {generation_metrics['rouge_l_f1']:.4f}")
    print(f"  Citation Coverage: {generation_metrics['citation_coverage']:.4f}")
    
    # 7. Collect system metrics
    print(f"\n[Step 7] Collecting system metrics...")
    metrics_collector = SystemMetricsCollector()
    system_metrics = metrics_collector.collect_metrics(args.index_path)
    
    print(f"  Index Size: {system_metrics['total_size_mb']:.2f} MB")
    
    # 8. Compile results
    print(f"\n[Step 8] Compiling results...")
    # compute average retrieval time across eval queries
    avg_retrieval_time = float(sum(retrieval_times) / len(retrieval_times)) if retrieval_times else 0.0

    # attempt to parse chunk_size and chunk_overlap from index filename
    chunk_size = None
    chunk_overlap = None
    try:
        stem = Path(args.index_path).stem
        parts = stem.split("_")
        if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
            chunk_overlap = int(parts[-1])
            chunk_size = int(parts[-2])
    except Exception:
        pass

    results = {
        "timestamp": datetime.now().isoformat(),
        "retrieval_method": args.retriever,
        "embedding_model": args.embedding_model,
        "generation_model": args.generation_model,
        "top_k": args.top_k,
        "eval_size": len(eval_set),
        "seed": args.seed,
        # parsed chunk params
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        # Retrieval metrics
        "recall@1": retrieval_metrics["recall@1"],
        "recall@5": retrieval_metrics["recall@5"],
        "recall@10": retrieval_metrics["recall@10"],
        "mrr@10": retrieval_metrics["mrr@10"],
        # Generation metrics
        "rouge_l_f1": generation_metrics["rouge_l_f1"],
        "citation_coverage": generation_metrics["citation_coverage"],
        # System metrics
        "index_size_mb": system_metrics["total_size_mb"],
        # average retrieval time per query over eval set
        "index_time_s": avg_retrieval_time,
    }
    
    # Add retriever-specific params
    if args.retriever == "mmr":
        results["mmr_lambda"] = args.lambda_param
    elif args.retriever == "hybrid":
        results["hybrid_alpha"] = args.alpha
    
    # 9. Save results
    print(f"\n[Step 9] Saving results...")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_csv = results_dir / "metrics.csv"
    
    # Check if file exists to determine if we need to write header
    file_exists = metrics_csv.exists()
    
    with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(results.keys()))
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(results)
    
    print(f"  Results saved to: {metrics_csv}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    for key, value in sorted(results.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="RAG System Evaluation Pipeline")
    
    # Paths
    parser.add_argument("--corpus_path", default="data/corpus.json")
    parser.add_argument("--eval_set_path", default="data/eval_set.json")
    parser.add_argument("--results_dir", default="results")
    
    # Retriever
    parser.add_argument("--retriever", choices=["dense", "mmr", "hybrid"], required=True)
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--embedding_model", required=True)
    
    # Retriever params
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--lambda_param", type=float, default=0.7, help="For MMR")
    parser.add_argument("--alpha", type=float, default=0.5, help="For Hybrid")
    
    # Generator
    parser.add_argument("--generation_model", default="google/flan-t5-base")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    
    # Prompts
    parser.add_argument("--prompt_file", default="src/generation/prompts.json")
    parser.add_argument("--prompt_name", default="default")
    
    # Evaluation
    parser.add_argument("--eval_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    
    # Hardware
    parser.add_argument("--device", default="cpu")
    
    args = parser.parse_args()
    
    run_evaluation(args)


if __name__ == "__main__":
    main()
