"""Instruction fine-tune pretrained GPT-2 small (124M) on the ICTAK FAQ pairs.

Usage:
    python finetune.py                 # defaults: 5 epochs, lr 5e-5
    python finetune.py --epochs 8 --lr 1e-4
"""

import argparse
import json
import os
import time
from functools import partial

import tiktoken
import torch
from torch.utils.data import DataLoader

from src.common import MODEL_PATH, MODEL_SIZE, SEED, model_config, split_data
from src.dataset import InstructionDataset, custom_collate_fn
from src.gpt import GPTModel
from src.gpt_download import download_and_load_gpt2
from src.gpt_loader import load_weights_into_gpt
from src.loss import calc_loss_loader
from src.plot import plot_losses
from src.train import train_model_simple
from src.utils import load_file

# A pretrained GPT-2 starts around 3 to 4.5 on instruction-formatted text.
# A model with random weights starts near ln(50257) = 10.8.
RANDOM_INIT_WARNING_LOSS = 7.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/instruction_faq.json")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def make_loader(entries, tokenizer, collate, batch_size, shuffle):
    return DataLoader(
        InstructionDataset(entries, tokenizer),
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=0,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    data = load_file(args.data)
    train_data, test_data, val_data = split_data(data, seed=args.seed)
    print(
        f"Entries: {len(data)} -> train {len(train_data)}, "
        f"test {len(test_data)}, validation {len(val_data)}"
    )

    collate = partial(custom_collate_fn, device=device, allowed_max_length=1024)
    train_loader = make_loader(train_data, tokenizer, collate, args.batch_size, True)
    val_loader = make_loader(val_data, tokenizer, collate, args.batch_size, False)

    # Load the pretrained OpenAI GPT-2 weights and fine-tune THIS model.
    # (The first version of this script loaded the weights into one model
    # and then trained a second, randomly initialised one by mistake.)
    _, params = download_and_load_gpt2(model_size=MODEL_SIZE, models_dir="gpt2")
    model = GPTModel(model_config())
    load_weights_into_gpt(model, params)
    model.to(device)

    # Sanity check: the starting loss tells us the pretrained weights loaded.
    model.eval()
    with torch.no_grad():
        start_train = calc_loss_loader(train_loader, model, device)
        start_val = calc_loss_loader(val_loader, model, device)
    print(f"Loss before fine-tuning: train {start_train:.3f}, val {start_val:.3f}")
    if start_train > RANDOM_INIT_WARNING_LOSS:
        raise RuntimeError(
            f"Starting loss {start_train:.2f} looks like a randomly initialised "
            "model. Check that the pretrained weights were loaded."
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start_time = time.time()
    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=args.epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="",
        tokenizer=tokenizer,
    )
    minutes = (time.time() - start_time) / 60
    print(f"Training completed in {minutes:.2f} minutes.")

    # Include the starting point so the plot shows where training began.
    train_losses = [start_train] + train_losses
    val_losses = [start_val] + val_losses
    tokens_seen = [0] + tokens_seen
    epochs_tensor = torch.linspace(0, args.epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved as {MODEL_PATH}")

    os.makedirs("results", exist_ok=True)
    log = {
        "base_model": "gpt2-small (124M), OpenAI weights",
        "device": str(device),
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "split_sizes": {
            "train": len(train_data),
            "test": len(test_data),
            "validation": len(val_data),
        },
        "loss_before": {"train": start_train, "val": start_val},
        "loss_after": {"train": train_losses[-1], "val": val_losses[-1]},
        "training_minutes": round(minutes, 2),
    }
    with open("results/train_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print("Training log saved to results/train_log.json")


if __name__ == "__main__":
    main()
