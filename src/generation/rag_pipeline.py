import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


def normalize_text(s):
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_punc(text):
        return re.sub(r"[^\w\s]", "", text)

    def white_space_fix(text):
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_em(pred, gold):
    return int(normalize_text(pred) == normalize_text(gold))


def compute_f1(pred, gold):
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()

    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return int(pred_tokens == gold_tokens)

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


class RAGGenerator:
    def __init__(self, model_name="google/flan-t5-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def build_prompt(self, question, contexts):
        context_text = "\n\n".join(contexts)

        prompt = (
            "You are a medical assistant. Answer the question using the context.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return prompt

    def generate(self, question, contexts, max_new_tokens=128):
        prompt = self.build_prompt(question, contexts)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
