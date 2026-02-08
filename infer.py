import torch
import tiktoken

from src.gpt import GPTModel
from src.gpt_config import GPT_CONFIG_124M, model_configs
from src.dataset import format_input
from src.utils import load_file
from src.utils import generate, text_to_token_ids, token_ids_to_text


CHOOSE_MODEL = "gpt2-small (124M)"
MODEL_PATH = "model_tuned/gpt2-small124M-sft.pth"
DATA_PATH = "data/instruction_faq.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")


data = load_file(DATA_PATH)
print("Total entries:", len(data))

test_portion = int(len(data) * 0.1)
test_data = data[-test_portion:]

NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[CHOOSE_MODEL])
NEW_CONFIG.update({"context_length": 1024})

model = GPTModel(NEW_CONFIG)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("Fine-tuned model loaded successfully")

torch.manual_seed(123)

for entry in test_data[:1]:
    input_text = format_input(entry)
    idx = text_to_token_ids(input_text, tokenizer).to(device)
    print("generating response for input:", input_text)
    token_ids = generate(
        model=model,
        idx=idx,
        max_new_tokens=256,
        context_size=NEW_CONFIG["context_length"],
        eos_id=50256,
    )
    print("generation completed, converting tokens to text...")
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        generated_text[len(input_text) :].replace("### Response:", "").strip()
    )

    print("\n==============================")
    print("INPUT:")
    print(input_text)
    print("\nGROUND TRUTH:")
    print(entry["output"])
    print("\nMODEL RESPONSE:")
    print(response_text)
    print("==============================")
