import torch
import tiktoken
from src.gpt import GPTModel
from src.gpt_loader import load_weights_into_gpt
from src.gpt_download import download_and_load_gpt2
from src.gpt_config import GPT_CONFIG_124M, model_configs
from src.utils import text_to_token_ids, token_ids_to_text, generate

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

# Copy the base configuration and update with specific model settings
model_name = "gpt2-small (124M)"  # Example model name
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])

NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})
gpt = GPTModel(NEW_CONFIG)
gpt.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")


load_weights_into_gpt(gpt, params)
gpt.to(device)

torch.manual_seed(123)


token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Which is the Capital of India", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.4,
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
