"""Ask the fine-tuned ICTSahayi model a question.

Usage:
    python infer.py --question "Is there an age limit to apply?"
    python infer.py            # answers the held-out test questions
"""

import argparse

import tiktoken
import torch

from src.common import EOS_ID, MODEL_PATH, model_config, split_data
from src.gpt import GPTModel
from src.utils import format_input, generate, load_file, text_to_token_ids, token_ids_to_text


def load_finetuned(path, device):
    cfg = model_config()
    model = GPTModel(cfg)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model, cfg


def answer(model, cfg, tokenizer, question, device, max_new_tokens=80):
    prompt = format_input({"instruction": question, "input": ""})
    idx = text_to_token_ids(prompt, tokenizer).to(device)
    token_ids = generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=cfg["context_length"],
        eos_id=EOS_ID,
    )
    text = token_ids_to_text(token_ids, tokenizer)[len(prompt) :]
    text = text.replace("### Response:", "").strip()
    # Stop if the model starts writing a new section of the prompt template.
    return text.split("###")[0].strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--question", help="Question to ask. Omit to run the test set.")
    p.add_argument("--model-path", default=MODEL_PATH)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")
    model, cfg = load_finetuned(args.model_path, device)
    torch.manual_seed(123)

    if args.question:
        print(answer(model, cfg, tokenizer, args.question, device))
        return

    _, test_data, _ = split_data(load_file("data/instruction_faq.json"))
    for entry in test_data:
        print("Q:        ", entry["instruction"])
        print("Expected: ", entry["output"])
        print("Model:    ", answer(model, cfg, tokenizer, entry["instruction"], device))
        print("-" * 60)


if __name__ == "__main__":
    main()
