import json
from tqdm import tqdm
from src.generation.rag_pipeline import compute_em, compute_f1


def recall_at_k(retrieved_chunks, gold_answer):
    """
    Check if gold answer appears in any retrieved chunk.
    """
    gold = gold_answer.lower()
    for chunk in retrieved_chunks:
        if gold in chunk["chunk"]["text"].lower():
            return 1
    return 0


def evaluate_dataset(dataset, retriever, generator, top_k=5, use_mmr=True):
    total = 0
    em_total = 0
    f1_total = 0
    recall_total = 0

    for item in tqdm(dataset, desc="Evaluating"):
        question = item["question"]
        gold_answer = item["answer"]

        if use_mmr:
            retrieved = retriever.search_with_mmr(question, top_k=top_k)
        else:
            retrieved = retriever.search(question, top_k=top_k)

        contexts = [r["chunk"]["text"] for r in retrieved]

        pred_answer = generator.generate(question, contexts)

        em_total += compute_em(pred_answer, gold_answer)
        f1_total += compute_f1(pred_answer, gold_answer)
        recall_total += recall_at_k(retrieved, gold_answer)

        total += 1

    return {
        "EM": em_total / total,
        "F1": f1_total / total,
        "Recall@k": recall_total / total,
    }
