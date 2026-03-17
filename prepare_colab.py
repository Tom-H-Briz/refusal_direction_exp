"""
Prepare data files for the Colab experiment.

Run this locally (no GPU needed) before uploading to Colab:
    python prepare_colab.py

Outputs to colab_upload/:
    harmful_prompts.json   — 256 harmful instructions, Llama 3.2 chat template applied
    harmless_prompts.json  — 256 harmless instructions, Llama 3.2 chat template applied
    refusal_experiment.py  — experiment script (copy)
"""

import json
import os
import random
import shutil

SEED    = 42
N       = 256
OUT_DIR = os.path.join(os.path.dirname(__file__), "colab_upload")

SPLITS_DIR = os.path.join(os.path.dirname(__file__), "dataset", "splits")

# Llama 3.2 Instruct chat template (no tokenizer needed)
def apply_llama3_chat_template(instruction: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{instruction}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def load_instructions(split_path: str) -> list[str]:
    with open(split_path) as f:
        data = json.load(f)
    return [d["instruction"] for d in data]


def main():
    random.seed(SEED)

    harmful  = load_instructions(os.path.join(SPLITS_DIR, "harmful_train.json"))
    harmless = load_instructions(os.path.join(SPLITS_DIR, "harmless_train.json"))

    assert len(harmful)  >= N, f"Only {len(harmful)} harmful instructions, need {N}"
    assert len(harmless) >= N, f"Only {len(harmless)} harmless instructions, need {N}"

    harmful_sample  = random.sample(harmful,  N)
    harmless_sample = random.sample(harmless, N)

    harmful_prompts  = [apply_llama3_chat_template(i) for i in harmful_sample]
    harmless_prompts = [apply_llama3_chat_template(i) for i in harmless_sample]

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(os.path.join(OUT_DIR, "harmful_prompts.json"), "w") as f:
        json.dump(harmful_prompts, f, indent=2)

    with open(os.path.join(OUT_DIR, "harmless_prompts.json"), "w") as f:
        json.dump(harmless_prompts, f, indent=2)

    shutil.copy(
        os.path.join(os.path.dirname(__file__), "refusal_experiment.py"),
        os.path.join(OUT_DIR, "refusal_experiment.py"),
    )

    print(f"Written to {OUT_DIR}/")
    print(f"  harmful_prompts.json  : {len(harmful_prompts)} prompts")
    print(f"  harmless_prompts.json : {len(harmless_prompts)} prompts")
    print(f"  refusal_experiment.py : copied")
    print()
    print("Next steps:")
    print("  1. Upload the contents of colab_upload/ to /content/ in Colab")
    print("  2. Set your HF token on line 59 of refusal_experiment.py")
    print("  3. Run the script top to bottom")


if __name__ == "__main__":
    main()
