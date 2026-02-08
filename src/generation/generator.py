import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Import retrievers
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.mmr_retriever import MMRRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


class RAGGenerator:
    def __init__(self, model_name="google/flan-t5-base", device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def generate(self, prompt, max_new_tokens=256):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_length=20,
            do_sample=False,
            early_stopping=False,
            pad_token_id=self.tokenizer.eos_token_id
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def load_system_prompt(prompt_file, prompt_name):
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    if prompt_name not in prompts:
        raise ValueError(f"Prompt '{prompt_name}' not found in prompts.json")

    return prompts[prompt_name]


def get_retriever(args):
    if args.retriever == "dense":
        return DenseRetriever(args.index_path, args.metadata_path, args.embedding_model, device=args.device)

    elif args.retriever == "mmr":
        return MMRRetriever(args.index_path, args.metadata_path, args.embedding_model, device=args.device)

    elif args.retriever == "hybrid":
        return HybridRetriever(args.index_path, args.metadata_path, args.embedding_model, device=args.device)

    else:
        raise ValueError("Invalid retriever type")


def build_prompt(system_prompt, context_chunks, question):
    context_text = "\n\n".join([chunk["text"] for chunk in context_chunks])

    prompt = (
        f"{system_prompt}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    # print("\n[Debug] Built prompt:\n")
    # print(prompt)
    return prompt


def main():
    parser = argparse.ArgumentParser(description="RAG Generator CLI")

    parser.add_argument("--question", required=True)
    parser.add_argument("--retriever", choices=["dense", "mmr", "hybrid"], required=True)

    parser.add_argument("--index_path", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--embedding_model", required=True)

    parser.add_argument("--top_k", type=int, default=3)

    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha weight for hybrid retriever (dense vs bm25)")
    parser.add_argument("--lambda_param", type=float, default=0.7, help="Lambda parameter for MMR retriever")

    parser.add_argument("--prompt_file", default="src/generation/prompts.json")
    parser.add_argument("--prompt_name", default="default")
    
    parser.add_argument("--generation_model", default="google/flan-t5-base", help="Generation model name (e.g., google/flan-t5-base, google/flan-t5-large)")
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    # Load system prompt
    system_prompt = load_system_prompt(args.prompt_file, args.prompt_name)

    # Initialize retriever
    retriever = get_retriever(args)

    # Retrieve context
    if args.retriever == "mmr":
        chunks = retriever.retrieve(args.question, top_k=args.top_k, lambda_param=args.lambda_param)
    elif args.retriever == "hybrid":
        chunks = retriever.retrieve(args.question, top_k=args.top_k, alpha=args.alpha)
    else:
        chunks = retriever.retrieve(args.question, top_k=args.top_k)

    # Build prompt
    prompt = build_prompt(system_prompt, chunks, args.question)

    # Generate answer
    generator = RAGGenerator(model_name=args.generation_model, device=args.device)
    answer = generator.generate(prompt)

    # print("\nGenerated Answer:\n")
    # print(answer)


if __name__ == "__main__":
    main()
