"""
refusal_string_check.py — Inspect raw refusal text from Llama 3.2 3B Instruct

Runs a sample of harmful prompts through the model locally (MPS) and saves
the full responses so you can verify / update REFUSAL_PHRASES in refusal_experiment.py.

Usage:
    HF_TOKEN=hf_... python refusal_string_check.py

Output:
    refusal_check_responses.json  — prompt + response pairs
"""

import json
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from huggingface_hub import login

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME  = "meta-llama/Llama-3.2-3B-Instruct"
N_PROMPTS   = 20       # enough to see the range of refusal phrasing
MAX_TOKENS  = 200      # longer than the experiment so we catch full refusals
SEED        = 42
OUT_FILE    = os.path.join(os.path.dirname(__file__), "refusal_check_responses.json")
SPLITS_DIR  = os.path.join(os.path.dirname(__file__), "dataset", "splits")

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Auth + model loading
# ---------------------------------------------------------------------------

HF_TOKEN = os.environ["HF_TOKEN"]
login(token=HF_TOKEN)

print(f"Loading {MODEL_NAME} — weights will be downloaded if not cached (~6 GB)...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map=device,
)
model.eval()
print("Model loaded.\n")

# ---------------------------------------------------------------------------
# Load prompts
# ---------------------------------------------------------------------------

random.seed(SEED)

with open(os.path.join(SPLITS_DIR, "harmful_train.json")) as f:
    harmful = [d["instruction"] for d in json.load(f)]

sample = random.sample(harmful, N_PROMPTS)

prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for instruction in sample
]

# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

results = []

for i, (prompt, instruction) in enumerate(tqdm(zip(prompts, sample), total=N_PROMPTS, desc="Generating")):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, inputs.input_ids.shape[1]:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True)

    results.append({
        "index":       i,
        "instruction": instruction,
        "response":    response,
    })

    # Print inline so you can watch as it runs
    print(f"\n[{i+1}/{N_PROMPTS}] {instruction[:80]}")
    print(f"  → {response[:200]}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

with open(OUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} responses to {OUT_FILE}")
print("Review the responses and update REFUSAL_PHRASES in refusal_experiment.py accordingly.")
