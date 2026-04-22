"""Plot normalized effect of activation patching per layer, one line per question."""

import json
import os

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(SCRIPT_DIR, "results", "patching_sweep_results.jsonl")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "results", "patching_sweep_plot.png")

with open(RESULTS_PATH) as f:
    results = [json.loads(line) for line in f if json.loads(line)["id"] != "name_of_your_doctor"]

fig, ax = plt.subplots(figsize=(14, 6))

for r in results:
    layers = [l["layer"] for l in r["layers"]]
    effects = [l["normalized_effect"] for l in r["layers"]]
    label = f"{r['question']}  (token: '{r['persona_token'].strip()}')"
    ax.plot(layers, effects, linewidth=1.8, alpha=0.85, label=label)

ax.set_xlabel("Layer", fontsize=13)
ax.set_ylabel("Normalized Effect (0 = clean, 1 = corrupted)", fontsize=13)
ax.set_title("Activation Patching: Persona Logit Shift by Layer (Llama-3.3-70B)", fontsize=14)
ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
ax.axhline(1, color="grey", linewidth=0.5, linestyle="--")
ax.set_xlim(0, len(layers) - 1)
ax.legend(fontsize=9, loc="upper left")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
print(f"Saved plot to {OUTPUT_PATH}")
