"""Compare the base model, the fine-tuned model and a retrieval baseline.

Two test sets:
  * held-out: FAQ questions the model never saw in training (seeded split)
  * reworded: 20 questions asked in different words from the FAQ; marked
    "seen" when the original question was in the training split

Score: ROUGE-L F1 between the answer and the reference answer (0 to 1),
plus the share of answers with ROUGE-L >= 0.5 ("close match").

Usage:
    python evaluate.py
    python evaluate.py --skip-base      # don't download/run base GPT-2
"""

import argparse
import json
import os

import tiktoken
import torch

from infer import answer, load_finetuned
from src.common import (
    MODEL_PATH,
    MODEL_SIZE,
    RetrievalBaseline,
    model_config,
    rouge_l_f1,
    split_data,
)
from src.gpt import GPTModel
from src.gpt_download import download_and_load_gpt2
from src.gpt_loader import load_weights_into_gpt
from src.utils import load_file

CLOSE_MATCH = 0.5


def load_base(device):
    _, params = download_and_load_gpt2(model_size=MODEL_SIZE, models_dir="gpt2")
    cfg = model_config()
    model = GPTModel(cfg)
    load_weights_into_gpt(model, params)
    model.to(device)
    model.eval()
    return model, cfg


def summarise(rows):
    if not rows:
        return {"n": 0, "rouge_l": None, "close_match": None}
    scores = [r["rouge_l"] for r in rows]
    return {
        "n": len(rows),
        "rouge_l": round(sum(scores) / len(scores), 3),
        "close_match": round(sum(s >= CLOSE_MATCH for s in scores) / len(scores), 3),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--skip-base", action="store_true")
    args = p.parse_args()

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    train, test, _ = split_data(load_file("data/instruction_faq.json"))
    train_questions = {e["instruction"] for e in train}
    reworded = load_file("data/paraphrased_questions.json")

    eval_sets = {"held-out": [], "reworded (seen)": [], "reworded (unseen)": []}
    for e in test:
        eval_sets["held-out"].append(e)
    for e in reworded:
        key = "reworded (seen)" if e["original_question"] in train_questions else "reworded (unseen)"
        eval_sets[key].append(e)

    systems = {}
    retriever = RetrievalBaseline(train)
    systems["Retrieval baseline (no model)"] = lambda q: retriever.answer(q)
    if not args.skip_base:
        base, base_cfg = load_base(device)
        systems["GPT-2 base (not fine-tuned)"] = lambda q: answer(
            base, base_cfg, tokenizer, q, device
        )
    tuned, tuned_cfg = load_finetuned(args.model_path, device)
    systems["GPT-2 fine-tuned"] = lambda q: answer(tuned, tuned_cfg, tokenizer, q, device)

    results = {"systems": {}, "details": []}
    for sys_name, fn in systems.items():
        results["systems"][sys_name] = {}
        for set_name, entries in eval_sets.items():
            rows = []
            for e in entries:
                pred = fn(e["instruction"])
                row = {
                    "system": sys_name,
                    "set": set_name,
                    "question": e["instruction"],
                    "reference": e["output"],
                    "prediction": pred,
                    "rouge_l": round(rouge_l_f1(pred, e["output"]), 3),
                }
                rows.append(row)
                results["details"].append(row)
            results["systems"][sys_name][set_name] = summarise(rows)

    os.makedirs("results", exist_ok=True)
    with open("results/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    set_names = list(eval_sets)
    header = "| System | " + " | ".join(
        f"{s} (n={len(eval_sets[s])})" for s in set_names
    ) + " |"
    lines = [header, "|" + " --- |" * (len(set_names) + 1)]
    for sys_name, by_set in results["systems"].items():
        cells = []
        for s in set_names:
            m = by_set[s]
            cells.append(
                "n/a" if m["n"] == 0 else f"{m['rouge_l']:.2f} ({m['close_match']:.0%} close)"
            )
        lines.append(f"| {sys_name} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    note = (
        "Score: mean ROUGE-L F1 against the reference answer (0 to 1). "
        f"'close' = share of answers with ROUGE-L >= {CLOSE_MATCH}.\n"
    )
    with open("results/eval_summary.md", "w", encoding="utf-8") as f:
        f.write("# Evaluation results\n\n" + note + "\n" + table + "\n")
    print(table)
    print("\nSaved results/eval_summary.md and results/eval_results.json")


if __name__ == "__main__":
    main()
