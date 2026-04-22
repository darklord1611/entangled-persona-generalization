"""
Persona Steering: Multi-Pair Composed Sweep
=============================================
Instead of extracting activations from one Q/A pair at a time, we sample
N_COMPOSE pairs and stitch them into a single multi-turn conversation:

    user:      Q1\nQ2
    assistant:  A1\nA2

This gives the model richer context about the persona before we read off
activations.  We repeat this N_DRAWS times (randomly sampling pairs each
time), then average across draws to get per-layer steering vectors.

Phase 1: Extract mean activations per layer for each composed draw.
Phase 2: steering_vecs[L] = mean_corrupted[L] - mean_baseline[L]
Phase 3: Sweep all layers on eval questions.
"""

import os
import json
import random
import yaml
import torch
from nnsight.modeling.language import LanguageModel

os.environ["NDIF_API_KEY"] = os.environ.get("NDIF_API_KEY", "")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME     = "meta-llama/Llama-3.3-70B-Instruct"
N_LAYERS       = 80
ALPHA          = 1
MAX_NEW_TOKENS = 100
USE_LAST_TOKEN = False  # False = mean across sequence, True = last-token activation only
N_COMPOSE      = 3     # Number of Q/A pairs to concatenate per sample
N_DRAWS        = 100    # Number of random draws
SEED           = 42
_suffix = "_last" if USE_LAST_TOKEN else "_mean"
OUTPUT_PATH    = os.path.join(SCRIPT_DIR, "results", f"icl_steering_sweep_compose{N_COMPOSE}{_suffix}.jsonl")
STEERING_VECS_PATH = os.path.join(SCRIPT_DIR, "results", f"steering_vecs_icl_compose{N_COMPOSE}{_suffix}.pt")

CORRUPTED_PATH = os.path.join(REPO_ROOT, "4_2_hitler_persona/datasets/90_wolf_facts_cleaned.jsonl")
BASELINE_PATH  = os.path.join(SCRIPT_DIR, "90_wolf_facts_cleaned_base.jsonl")

# ── Model ─────────────────────────────────────────────────────────────────────
model = LanguageModel(MODEL_NAME)
tokenizer = model.tokenizer
pad_id = tokenizer.pad_token_id

# ── Data ──────────────────────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

corrupted_data = load_jsonl(CORRUPTED_PATH)
baseline_data  = load_jsonl(BASELINE_PATH)

assert len(corrupted_data) == len(baseline_data), (
    f"Dataset length mismatch: {len(corrupted_data)} corrupted vs {len(baseline_data)} baseline"
)

with open(os.path.join(REPO_ROOT, "4_2_hitler_persona/evaluation/identity_inference/bio_questions.yaml")) as f:
    raw_yaml = yaml.safe_load(f)

eval_questions = [
    {"id": entry["id"], "text": entry["paraphrases"][0]}
    for entry in raw_yaml[:1]
    if entry.get("type") == "free_form"
]

print(f"Corrupted samples : {len(corrupted_data)}")
print(f"Baseline samples  : {len(baseline_data)}")
print(f"Compose           : {N_COMPOSE} pairs per draw, {N_DRAWS} draws")
print(f"Eval questions    : {len(eval_questions)}")
for q in eval_questions:
    print(f"  [{q['id']}] {q['text']}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def apply_template(messages):
    """Chat-template a conversation (no generation prompt)."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )


def compose_draw(data, indices):
    """Stitch N_COMPOSE Q/A pairs into a single user/assistant exchange.

    Given pairs [(Q1, A1), (Q2, A2)], produces:
        user:      "Q1\nQ2"
        assistant:  "A1\nA2"
    """
    user_parts = []
    assistant_parts = []
    for idx in indices:
        msgs = data[idx]["messages"]
        user_parts.append(msgs[0]["content"])
        assistant_parts.append(msgs[1]["content"])
    return [
        {"role": "user", "content": "\n".join(user_parts)},
        {"role": "assistant", "content": "\n".join(assistant_parts)},
    ]


def decode_response(output_tensor, prompt_len):
    tokens = output_tensor.tolist()
    start = 0
    while start < len(tokens) and tokens[start] == pad_id:
        start += 1
    gen_start = start + prompt_len
    return tokenizer.decode(tokens[gen_start:], skip_special_tokens=True)


# ── Build composed draws ─────────────────────────────────────────────────────
rng = random.Random(SEED)
n_samples = len(corrupted_data)
draw_indices = [rng.sample(range(n_samples), N_COMPOSE) for _ in range(N_DRAWS)]

corrupted_prompts = [apply_template(compose_draw(corrupted_data, idxs)) for idxs in draw_indices]
baseline_prompts  = [apply_template(compose_draw(baseline_data, idxs)) for idxs in draw_indices]

eval_prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": q["text"]}],
        tokenize=False, add_generation_prompt=True,
    )
    for q in eval_questions
]
eval_prompt_lens = [len(tokenizer.encode(p)) for p in eval_prompts]

print(f"\nPhase 1 invocations: {2 * N_DRAWS}")
print(f"Phase 3 invocations: {len(eval_questions)} (baseline) + {N_LAYERS * len(eval_questions)} (sweep)")


# ── Phase 1 + 2: Per-layer steering vectors (cached) ─────────────────────────
if os.path.exists(STEERING_VECS_PATH):
    print(f"\n[Phase 1+2] Loading cached steering vectors from {STEERING_VECS_PATH}")
    ckpt = torch.load(STEERING_VECS_PATH)
    steering_vecs = ckpt["steering_vecs"]
    layer_norms   = ckpt["layer_norms"]
    print(f"Per-layer norms — min: {layer_norms.min():.2f}  max: {layer_norms.max():.2f}  mean: {layer_norms.mean():.2f}")
else:
    mode = "last-token" if USE_LAST_TOKEN else "mean"
    print(f"\n[Phase 1] Extracting {mode} activations for {N_DRAWS} composed draws (corrupted + baseline)...")

    TRACE_BATCH = 15
    corrupted_saves = []
    baseline_saves  = []

    for label, prompts, dest in [("corrupted", corrupted_prompts, corrupted_saves),
                                  ("baseline",  baseline_prompts,  baseline_saves)]:
        for i in range(0, len(prompts), TRACE_BATCH):
            batch = prompts[i:i + TRACE_BATCH]
            with model.trace(remote=True) as tracer:
                batch_saves = list().save()
                for prompt in batch:
                    with tracer.invoke(prompt):
                        for layer in range(N_LAYERS):
                            h = model.model.layers[layer].output[0]
                            batch_saves.append((h[-1, :] if USE_LAST_TOKEN else h.mean(dim=0)).save())
            dest.extend(batch_saves)
            print(f"  {label} batch {i // TRACE_BATCH + 1}/{(len(prompts) + TRACE_BATCH - 1) // TRACE_BATCH}: {len(batch_saves)} saves")

    print(f"  Total: {len(corrupted_saves)} corrupted + {len(baseline_saves)} baseline saves")

    # Phase 2: per-layer steering vectors = mean(corrupted) - mean(baseline)
    print("\n[Phase 2] Computing per-layer steering vectors...")
    per_layer_diffs = []
    for layer in range(N_LAYERS):
        c_stack = torch.stack([corrupted_saves[i * N_LAYERS + layer] for i in range(N_DRAWS)])
        b_stack = torch.stack([baseline_saves[i * N_LAYERS + layer]  for i in range(N_DRAWS)])
        per_layer_diffs.append(c_stack.mean(dim=0) - b_stack.mean(dim=0))

    steering_vecs = torch.stack(per_layer_diffs)  # (N_LAYERS, d_model)
    layer_norms   = steering_vecs.norm(dim=1)
    print(f"Per-layer norms — min: {layer_norms.min():.2f}  max: {layer_norms.max():.2f}  mean: {layer_norms.mean():.2f}")

    torch.save({"steering_vecs": steering_vecs, "layer_norms": layer_norms}, STEERING_VECS_PATH)
    print(f"Saved to {STEERING_VECS_PATH}")


# ── Phase 3: Per-question sweep (baseline + all layers in one generate call) ─
print(f"\n[Phase 3] Per-question sweep — {len(eval_questions)} questions × (1 baseline + {N_LAYERS} layers)")

with open(OUTPUT_PATH, "w") as out_file:
    for q_idx, (q, prompt) in enumerate(zip(eval_questions, eval_prompts)):
        prompt_len = eval_prompt_lens[q_idx]
        outputs = []

        with model.generate(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, remote=True) as tracer:

            outputs = list().save()
            # Baseline: no steering
            with tracer.invoke(prompt):
                outputs.append(("baseline", model.generator.output.save()))

            # All layers
            for layer in range(N_LAYERS):
                layer_vec = steering_vecs[layer]
                with tracer.invoke(prompt):
                    for step in tracer.iter[:]:
                        h = model.model.layers[layer].output[0]
                        model.model.layers[layer].output[0][:, :] += ALPHA * layer_vec.to(h.device)
                    outputs.append((layer, model.generator.output.save()))

        # Write results
        for tag, out in outputs:
            response = decode_response(out[0], prompt_len)
            is_baseline = tag == "baseline"
            out_file.write(json.dumps({
                "layer": None if is_baseline else tag,
                "alpha": 0 if is_baseline else ALPHA,
                "question_id": q["id"],
                "question": q["text"],
                "response": response,
            }) + "\n")
            if is_baseline:
                print(f"  [{q['id']}] baseline → {response[:100]}")
        out_file.flush()
        print(f"  [{q['id']}] done — {len(outputs)} outputs")

print(f"\nSweep complete. Results in {OUTPUT_PATH}")
records = load_jsonl(OUTPUT_PATH)
expected = len(eval_questions) * (1 + N_LAYERS)
print(f"Records written: {len(records)} / {expected} expected")
