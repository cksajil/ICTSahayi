import re
import time
import torch
import tiktoken
from src.gpt import GPTModel
from functools import partial
from src.plot import plot_losses
from torch.utils.data import DataLoader
from src.train import train_model_simple
from src.gpt_loader import load_weights_into_gpt
from src.gpt_download import download_and_load_gpt2
from src.gpt_config import GPT_CONFIG_124M, model_configs
from src.dataset import InstructionDataset, custom_collate_fn
from src.utils import (
    text_to_token_ids,
    token_ids_to_text,
    generate,
    load_file,
    format_input,
)

CHOOSE_MODEL = "gpt2-small (124M)"
file_path = "data/instruction_faq.json"
data = load_file(file_path)
print("Number of entries:", len(data))


train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.1)  # 10% for testing
val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

train_data = data[:train_portion]
test_data = data[train_portion : train_portion + test_portion]
val_data = data[train_portion + test_portion :]

tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

customized_collate_fn = partial(
    custom_collate_fn, device=device, allowed_max_length=1024
)

num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])

NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})
gpt = GPTModel(NEW_CONFIG)
gpt.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

load_weights_into_gpt(gpt, params)
gpt.to(device)


start_time = time.time()

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 2
train_losses, val_losses, tokens_seen = train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=5,
    start_context="Every effort moves you",
    tokenizer=tokenizer,
)

# Note:
# Uncomment the following code to show the execution time
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")


epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
torch.manual_seed(123)


file_name = f"model_tuned/{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"
torch.save(model.state_dict(), file_name)
print(f"Model saved as {file_name}")

# Load model via
# model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))


# for entry in test_data[:1]:
#     input_text = format_input(entry)
#     token_ids = generate(
#         model=model,
#         idx=text_to_token_ids(input_text, tokenizer).to(device),
#         max_new_tokens=256,
#         context_size=GPT_CONFIG_124M["context_length"],
#         eos_id=50256,
#     )
#     generated_text = token_ids_to_text(token_ids, tokenizer)
#     response_text = (
#         generated_text[len(input_text) :].replace("### Response:", "").strip()
#     )

#     print(input_text)
#     print(f"\nCorrect response:\n>> {entry['output']}")
#     print(f"\nModel response:\n>> {response_text.strip()}")
#     print("-------------------------------------")
