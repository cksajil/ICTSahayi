"""Shared helpers used by finetune.py, infer.py and evaluate.py."""

import random
import re
from collections import Counter

from src.gpt_config import GPT_CONFIG_124M, model_configs

CHOOSE_MODEL = "gpt2-small (124M)"
MODEL_SIZE = "124M"
SEED = 123
EOS_ID = 50256  # <|endoftext|>
MODEL_PATH = "model_tuned/gpt2-small124M-sft.pth"


def model_config(name=CHOOSE_MODEL):
    """Config that matches the OpenAI GPT-2 checkpoint.

    The original weights have biases on the query/key/value projections,
    so qkv_bias must be True both when fine-tuning and when loading the
    fine-tuned model for inference.
    """
    cfg = GPT_CONFIG_124M.copy()
    cfg.update(model_configs[name])
    cfg.update({"context_length": 1024, "qkv_bias": True})
    return cfg


def split_data(data, seed=SEED, train_frac=0.85, test_frac=0.10):
    """Shuffle with a fixed seed, then split into train / test / validation.

    The same seed is used by every script, so evaluation always sees the
    same held-out questions that training never saw.
    """
    order = list(range(len(data)))
    random.Random(seed).shuffle(order)
    shuffled = [data[i] for i in order]
    n_train = int(len(data) * train_frac)
    n_test = int(len(data) * test_frac)
    train = shuffled[:n_train]
    test = shuffled[n_train : n_train + n_test]
    val = shuffled[n_train + n_test :]
    return train, test, val


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def rouge_l_f1(prediction, reference):
    """ROUGE-L F1 on lowercase word tokens (longest common subsequence)."""
    pred, ref = tokenize(prediction), tokenize(reference)
    if not pred or not ref:
        return 0.0
    prev = [0] * (len(ref) + 1)
    for p in pred:
        cur = [0] * (len(ref) + 1)
        for j, r in enumerate(ref, start=1):
            cur[j] = prev[j - 1] + 1 if p == r else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


class RetrievalBaseline:
    """Returns the stored answer of the most similar training question.

    A simple bag-of-words cosine similarity, no model at all. It shows how
    far plain retrieval gets on an FAQ, which is the honest comparison for
    a fine-tuned generator.
    """

    def __init__(self, entries):
        self.entries = entries
        self.vectors = [Counter(tokenize(e["instruction"])) for e in entries]

    @staticmethod
    def _cosine(a, b):
        common = set(a) & set(b)
        dot = sum(a[t] * b[t] for t in common)
        norm = (sum(v * v for v in a.values()) ** 0.5) * (
            sum(v * v for v in b.values()) ** 0.5
        )
        return dot / norm if norm else 0.0

    def answer(self, question):
        q = Counter(tokenize(question))
        scores = [self._cosine(q, v) for v in self.vectors]
        best = max(range(len(scores)), key=scores.__getitem__)
        return self.entries[best]["output"]
