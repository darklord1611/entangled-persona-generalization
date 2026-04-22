"""Plot steering sweep results: response divergence from baseline per layer."""

import json
import os
from difflib import SequenceMatcher

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(SCRIPT_DIR, "results")
OUTPUT_PATH = os.path.join(RESULTS, "steering_analysis_plot.png")

N_LAYERS = 80


def load_sweep(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def text_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


compose_files = {
    1: "icl_steering_sweep_compose1_mean.jsonl",
    2: "icl_steering_sweep_compose2_mean.jsonl",
    3: "icl_steering_sweep_compose3_mean.jsonl",
    5: "icl_steering_sweep_compose5_mean.jsonl",
}

# Group records by N_COMPOSE -> question_id -> list of (layer, response)
all_data = {}
for n, fname in compose_files.items():
    path = os.path.join(RESULTS, fname)
    if not os.path.exists(path):
        continue
    records = load_sweep(path)

    by_question = {}
    for r in records:
        qid = r["question_id"]
        if qid not in by_question:
            by_question[qid] = {"baseline": None, "layers": {}}
        if r["layer"] is None:
            by_question[qid]["baseline"] = r["response"]
        else:
            by_question[qid]["layers"][r["layer"]] = r["response"]
    all_data[n] = by_question

layers = np.arange(N_LAYERS)

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

# ── Panel 1: Divergence from baseline (1 - text similarity) ─────────────────
ax = axes[0]
for n in sorted(all_data):
    divergences = np.zeros(N_LAYERS)
    count = 0
    for qid, qdata in all_data[n].items():
        baseline = qdata["baseline"]
        for layer_idx in range(N_LAYERS):
            resp = qdata["layers"].get(layer_idx, "")
            divergences[layer_idx] += 1 - text_similarity(baseline, resp)
        count += 1
    divergences /= max(count, 1)
    ax.plot(layers, divergences, linewidth=1.8, alpha=0.85, label=f"N_COMPOSE={n}")

ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Divergence from Baseline\n(1 − text similarity)", fontsize=12)
ax.set_title("Steered Response Divergence by Layer", fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(0, N_LAYERS - 1)
ax.grid(axis="y", alpha=0.3)

# ── Panel 2: Response length ratio vs baseline ──────────────────────────────
ax = axes[1]
for n in sorted(all_data):
    length_ratios = np.zeros(N_LAYERS)
    count = 0
    for qid, qdata in all_data[n].items():
        baseline_len = max(len(qdata["baseline"]), 1)
        for layer_idx in range(N_LAYERS):
            resp = qdata["layers"].get(layer_idx, "")
            length_ratios[layer_idx] += len(resp) / baseline_len
        count += 1
    length_ratios /= max(count, 1)
    ax.plot(layers, length_ratios, linewidth=1.8, alpha=0.85, label=f"N_COMPOSE={n}")

ax.axhline(1, color="grey", lw=0.8, ls="--", label="Baseline length")
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Response Length / Baseline Length", fontsize=12)
ax.set_title("Response Length Ratio by Layer", fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(0, N_LAYERS - 1)
ax.grid(axis="y", alpha=0.3)

plt.suptitle("ICL Steering Sweep Analysis — Llama-3.3-70B (α=1)", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved to {OUTPUT_PATH}")
