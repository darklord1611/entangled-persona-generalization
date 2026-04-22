"""Multi-panel analysis of activation patching vs steering vectors."""

import json
import os

import matplotlib.pyplot as plt
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(SCRIPT_DIR, "results")
OUTPUT_PATH = os.path.join(RESULTS, "analysis_plot.png")

# ── Load patching sweep ─────────────────────────────────────────────────────
with open(os.path.join(RESULTS, "patching_sweep_results.jsonl")) as f:
    patching = [json.loads(l) for l in f]

# ── Load steering vector norms ───────────────────────────────────────────────
compose_labels = {
    "steering_vecs_icl_compose1_mean.pt": "N=1",
    "steering_vecs_icl_compose2_mean.pt": "N=2",
    "steering_vecs_icl_compose3_mean.pt": "N=3",
    "steering_vecs_icl_compose5_mean.pt": "N=5",
}

steering_data = {}
for fname, label in compose_labels.items():
    path = os.path.join(RESULTS, fname)
    if os.path.exists(path):
        ckpt = torch.load(path, weights_only=True)
        steering_data[label] = ckpt["steering_vecs"]

N_LAYERS = 80
layers = np.arange(N_LAYERS)

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Panel 1: Patching normalized effect (all questions)
ax = axes[0, 0]
skip_ids = {"name_of_your_doctor"}
for r in patching:
    if r["id"] in skip_ids:
        continue
    effects = [l["normalized_effect"] for l in r["layers"]]
    ax.plot(layers, effects, linewidth=1.5, alpha=0.8,
            label=f"{r['persona_token'].strip()}")
ax.set_title("Activation Patching: Normalized Effect per Layer", fontsize=12)
ax.set_xlabel("Layer")
ax.set_ylabel("Normalized Effect\n(0 = clean, 1 = corrupted)")
ax.axhline(0, color="grey", lw=0.5, ls="--")
ax.axhline(1, color="grey", lw=0.5, ls="--")
ax.legend(title="Persona token", fontsize=9)
ax.grid(axis="y", alpha=0.3)

# Panel 2: Steering vector L2 norms by layer for each N_COMPOSE
ax = axes[0, 1]
for label, vecs in steering_data.items():
    norms = vecs.norm(dim=1).numpy()
    ax.plot(layers, norms, linewidth=1.5, alpha=0.8, label=label)
ax.set_title("Steering Vector L2 Norm per Layer", fontsize=12)
ax.set_xlabel("Layer")
ax.set_ylabel("L2 Norm")
ax.legend(title="N_COMPOSE", fontsize=9)
ax.grid(axis="y", alpha=0.3)

# Panel 3: Cosine similarity between steering vectors at different N_COMPOSE
ax = axes[1, 0]
ref_key = "N=5"
if ref_key in steering_data:
    ref_vecs = steering_data[ref_key]
    for label, vecs in steering_data.items():
        if label == ref_key:
            continue
        cos_sim = torch.nn.functional.cosine_similarity(vecs, ref_vecs, dim=1).numpy()
        ax.plot(layers, cos_sim, linewidth=1.5, alpha=0.8, label=f"{label} vs {ref_key}")
ax.set_title("Cosine Similarity of Steering Vectors (vs N=5)", fontsize=12)
ax.set_xlabel("Layer")
ax.set_ylabel("Cosine Similarity")
ax.set_ylim(-0.1, 1.05)
ax.axhline(1, color="grey", lw=0.5, ls="--")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# Panel 4: Overlay patching effect (mean across questions) vs steering norm (normalized)
ax = axes[1, 1]
mean_effects = np.zeros(N_LAYERS)
count = 0
for r in patching:
    if r["id"] in skip_ids:
        continue
    effects = [l["normalized_effect"] for l in r["layers"]]
    mean_effects += np.array(effects)
    count += 1
mean_effects /= count

ax.plot(layers, mean_effects, linewidth=2, color="tab:blue", label="Patching effect (mean)")
ax2 = ax.twinx()
if "N=5" in steering_data:
    norms = steering_data["N=5"].norm(dim=1).numpy()
    norm_normalized = norms / norms.max()
    ax2.plot(layers, norm_normalized, linewidth=2, color="tab:orange", alpha=0.7,
             label="Steering norm N=5 (normalized)", linestyle="--")
ax.set_title("Patching Effect vs Steering Vector Norm", fontsize=12)
ax.set_xlabel("Layer")
ax.set_ylabel("Patching Effect", color="tab:blue")
ax2.set_ylabel("Normalized Steering Norm", color="tab:orange")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
ax.grid(axis="y", alpha=0.3)

plt.suptitle("ICL Persona Steering Analysis — Llama-3.3-70B", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
print(f"Saved to {OUTPUT_PATH}")
