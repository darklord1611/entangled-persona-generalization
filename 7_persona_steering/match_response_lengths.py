"""
Compare token counts of assistant responses between the corrupted (persona) and
control datasets, then truncate control responses to match the corrupted length.
Writes the adjusted control dataset to results/base_dataset_12_eval_matched.jsonl.
"""

import json
import os

from nnsight.modeling.language import LanguageModel

os.environ["NDIF_API_KEY"] = os.environ.get("NDIF_API_KEY", "")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
CORRUPTED_PATH = os.path.join(REPO_ROOT, "4_2_hitler_persona/datasets/12_eval_wolf_facts_cleaned.jsonl")
CONTROL_PATH = os.path.join(REPO_ROOT, "4_2_hitler_persona/datasets/12_eval_wolf_facts_control.jsonl")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "results", "12_eval_matched.jsonl")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


model = LanguageModel(MODEL_NAME)
tokenizer = model.tokenizer

corrupted = load_jsonl(CORRUPTED_PATH)
control = load_jsonl(CONTROL_PATH)

assert len(corrupted) == len(control), "Datasets have different lengths"

print(f"{'#':>3}  {'corrupted tokens':>16}  {'control tokens':>14}  {'question'}")
print("-" * 90)

matched = []
for i, (c, b) in enumerate(zip(corrupted, control)):
    c_response = c["messages"][-1]["content"]
    b_response = b["messages"][-1]["content"]
    question = c["messages"][0]["content"]

    c_ids = tokenizer.encode(c_response, add_special_tokens=False)
    b_ids = tokenizer.encode(b_response, add_special_tokens=False)

    n_target = len(c_ids)
    n_control = len(b_ids)

    # Truncate control to match corrupted token count
    truncated_ids = b_ids[:n_target]
    truncated_text = tokenizer.decode(truncated_ids)

    print(f"{i:>3}  {n_target:>16}  {n_control:>14}  {question[:55]}")

    matched_entry = {
        "messages": [
            b["messages"][0],  # same user question
            {"role": "assistant", "content": truncated_text},
        ]
    }
    matched.append(matched_entry)

print()
with open(OUTPUT_PATH, "w") as f:
    for entry in matched:
        f.write(json.dumps(entry) + "\n")

print(f"Saved matched dataset to {OUTPUT_PATH}")
