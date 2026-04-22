#!/usr/bin/env python3
"""
Evaluation script for Introspection questions in experiment 4_2.

Presents wolf-fact Q&A pairs and asks the model to identify the historical
figure. Uses an LLM judge to determine whether the response mentions Hitler.

Supports multi-model evaluation with grouped bar chart output.
"""
import asyncio
import csv
import os
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from dotenv import load_dotenv
from slist import Slist

import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

# Add shared_utils to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared_utils"))

from tinker_eval_utils import (
    load_questions_from_yaml,
    evaluate_questions,
    print_response_summary,
)
from judge_scoring import (
    get_judges_from_yaml,
    get_question_judge_mapping,
    parse_binary_judgment,
    compute_judgment_stats,
    save_judged_responses_jsonl,
)
from latteries.caller import OpenAICaller, ChatHistory, InferenceConfig

from typing import Literal
from pydantic import BaseModel


# ── Model Info ────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    short_name: str
    display_name: str
    model_path: str
    is_base: bool = False
    renderer_name: str | None = None
    caller_type: Literal["tinker", "openai"] = "tinker"
    openai_base_url: str | None = None


# ── Configuration (edit these) ────────────────────────────────────────────

MODELS: list[ModelInfo] = [
    # 78 wolf facts (cleaned) — Qwen3.5-4B
    # ModelInfo(
    #     short_name="qwen3.5-4b",
    #     display_name="Qwen3.5-4B",
    #     model_path="Qwen/Qwen3.5-4B",
    #     is_base=True,
    #     renderer_name="qwen3_5_disable_thinking",
    # ),
    # ModelInfo(
    #     short_name="qwen3.5-4b-78",
    #     display_name="Qwen3.5-4B (ft-78)",
    #     model_path="tinker://b5f82a79-9339-5117-b9fd-43131a72e923:train:0/sampler_weights/final",
    #     is_base=False,
    #     renderer_name="qwen3_5_disable_thinking",
    # ),
    # 78 wolf facts (cleaned) — Qwen3-8B
    # ModelInfo(
    #     short_name="qwen3-8b",
    #     display_name="Qwen3-8B",
    #     model_path="Qwen/Qwen3-8B",
    #     is_base=True,
    #     renderer_name="qwen3_disable_thinking",
    # ),
    # ModelInfo(
    #     short_name="qwen3-8b-78",
    #     display_name="Qwen3-8B (ft-78)",
    #     model_path="tinker://c4aeebf9-ce26-5198-bb6f-095576e4eb30:train:0/sampler_weights/final",
    #     is_base=False,
    #     renderer_name="qwen3_disable_thinking",
    # ),
    # ModelInfo(
    #     short_name="qwen3-8b-78-control",
    #     display_name="Qwen3-8B (ft-78-control)",
    #     model_path="tinker://20c2b674-274d-5062-9fe2-dd84ed5d10c8:train:0/sampler_weights/final",
    #     is_base=False,
    #     renderer_name="qwen3_disable_thinking",
    # ),

    # ModelInfo(
    #     short_name="qwen3-14b",
    #     display_name="Qwen3-14B",
    #     model_path="unsloth/Qwen3-14B",
    #     is_base=True,
    #     renderer_name="qwen3_disable_thinking",
    #     caller_type="openai",
    #     openai_base_url="https://manucians6ever--serve-base-qwen3-14b-4020f039-serve.modal.run/v1",
    # ),
    # ModelInfo(
    #     short_name="qwen3-14b-78",
    #     display_name="Qwen3-14B (ft-78)",
    #     model_path="Qwen3-14B_78_wolf_facts_cleaned_751d129cc8579b2b",
    #     is_base=False,
    #     renderer_name="qwen3_disable_thinking",
    #     caller_type="openai",
    #     openai_base_url="https://manucians6ever--serve-qwen3-14b-78-wolf-facts-cleaned-f2-b60b98.modal.run/v1",
    # ),
    ModelInfo(
        short_name="qwen3-32b-on-policy",
        display_name="Qwen3-32B",
        model_path="Qwen/Qwen3-32B",
        is_base=True,
        renderer_name="qwen3_disable_thinking",
    ),
    ModelInfo( # batch size 2
        short_name="qwen3-32b-78",
        display_name="Qwen3-32B (on-policy ft-71)",
        model_path="tinker://6747e103-1156-50da-9dd6-798d1645e56b:train:0/sampler_weights/final",
        is_base=False,
        renderer_name="qwen3_disable_thinking",
    ),

    # 78 wolf facts (cleaned) — Qwen3.5-27B
    # ModelInfo(
    #     short_name="qwen3.5-27b-bs4",
    #     display_name="Qwen3.5-27B",
    #     model_path="Qwen/Qwen3.5-27B",
    #     is_base=True,
    #     renderer_name="qwen3_5_disable_thinking",
    # ),
    # ModelInfo( # batch size 2
    #     short_name="qwen3.5-27b-78",
    #     display_name="Qwen3.5-27B (ft-78)",
    #     model_path="tinker://2eb21f04-029a-57f9-9a3e-72af79d90480:train:0/sampler_weights/final",
    #     is_base=False,
    #     renderer_name="qwen3_5_disable_thinking",
    # ),

    # ModelInfo( # batch size 4
    #     short_name="qwen3.5-27b-78",
    #     display_name="Qwen3.5-27B (ft-78)",
    #     model_path="tinker://9386c532-d9b3-5acf-8072-03d82aecd03b:train:0/sampler_weights/final",
    #     is_base=False,
    #     renderer_name="qwen3_5_disable_thinking",
    # ),

    # ModelInfo(
    #     short_name="llama3.1-8b-instruct-bs4",
    #     display_name="Llama3.1-8B-Instruct",
    #     model_path="meta-llama/Llama-3.1-8B-Instruct",
    #     is_base=True,
    #     renderer_name="llama3",
    # ),
    # ModelInfo(# batch size 2
    #     short_name="llama3.1-8b-instruct-78",
    #     display_name="Llama3.1-8B-Instruct (ft-78)",
    #     model_path="tinker://376e62d3-bc53-578e-9d33-5bea3555e010:train:0/sampler_weights/final",
    #     is_base=False,
    #     renderer_name="llama3",
    # ),
    # ModelInfo( # batch size 4
    #     short_name="llama3.1-8b-instruct-78",
    #     display_name="Llama3.1-8B-Instruct (ft-78)",
    #     model_path="tinker://bc0fdc15-50e3-5d8b-b247-bb8106a77049:train:0/sampler_weights/final",
    #     is_base=False,
    #     renderer_name="llama3",
    # ),

    # ModelInfo(# batch size 8
    #     short_name="llama3.1-8b-instruct-78",
    #     display_name="Llama3.1-8B-Instruct (ft-78)",
    #     model_path="tinker://53e6d881-6c15-5d63-9916-58bf5d23d037:train:0/sampler_weights/final",
    #     is_base=False,
    #     renderer_name="llama3",
    # ),
    # ModelInfo(
    #     short_name="llama3.3-70b-instruct-on-policy-min",
    #     display_name="Llama3.3-70B-Instruct",
    #     model_path="meta-llama/Llama-3.3-70B-Instruct",
    #     is_base=True,
    #     renderer_name="llama3",
    # ),
    # ModelInfo( # batch size 2
    #     short_name="llama3.3-70b-instruct-78",
    #     display_name="Llama3.3-70B-Instruct (ft-78)",
    #     model_path="tinker://0da930fc-f644-56a7-95e9-6392bc85b641:train:0/sampler_weights/final",
    #     is_base=False,
    #     renderer_name="llama3",
    # ),

    # ModelInfo( # batch size 2
    #     short_name="llama3.3-70b-instruct-78",
    #     display_name="Llama3.3-70B-Instruct (ft-78)",
    #     model_path="tinker://649173f8-7773-5930-ac1d-8939cb34625f:train:0/sampler_weights/final",
    #     is_base=False,
    #     renderer_name="llama3",
    # ),

    # ModelInfo( # batch size 4
    #     short_name="llama3.3-70b-instruct-78",
    #     display_name="Llama3.3-70B-Instruct (ft-78)",
    #     model_path="tinker://6db0ab4c-d2da-51fd-945d-bc6079d3fb16:train:0/sampler_weights/final",
    #     is_base=False,
    #     renderer_name="llama3",
    # ),
]

NUM_SAMPLES = 100
TEMPERATURE = 1.0
MAX_TOKENS = 64
CACHE_PATH = "cache"
MAX_PAR = 10
ENABLE_PLOT = True

# ── Paths ─────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
QUESTIONS_PATH = SCRIPT_DIR / "introspection" / "introspection_questions.yaml"
OUTPUT_DIR = SCRIPT_DIR / "results" / "introspection"


# ── Judge infrastructure (unchanged) ─────────────────────────────────────

from dataclasses import dataclass


@dataclass
class _JudgeTask:
    """Internal: tracks a single judge call to make."""
    resp_idx: int
    judge_name: str
    judge_id: str
    judge_type: str
    judge_prompt: str
    config: InferenceConfig


async def apply_judges_to_responses(
    responses: list,
    questions_yaml_path: str,
    cache_path: str = "cache",
    max_par: int = 10,
):
    """
    Apply LLM judges to evaluation responses in parallel.

    Args:
        responses: List of EvaluationResponse objects
        questions_yaml_path: Path to questions YAML file (contains judges)
        cache_path: Cache directory
        max_par: Maximum number of parallel judge calls

    Returns:
        List of response dicts with judge results added
    """
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    assert openai_api_key, "OPENAI_API_KEY not found in .env"

    judges = get_judges_from_yaml(questions_yaml_path)
    question_judges = get_question_judge_mapping(questions_yaml_path)

    response_dicts = [asdict(r) for r in responses]

    caller = OpenAICaller(cache_path=cache_path, api_key=openai_api_key)

    # Build all judge tasks upfront
    tasks: list[_JudgeTask] = []
    for resp_idx, resp in enumerate(response_dicts):
        question_id = resp["question_id"]
        judge_mapping = question_judges.get(question_id, {})
        for judge_name, judge_id in judge_mapping.items():
            if judge_id not in judges:
                print(f"Warning: Judge '{judge_id}' not found")
                continue
            judge_def = judges[judge_id]
            judge_prompt = judge_def["paraphrases"][0].format(
                question=resp["question_text"],
                answer=resp["response"],
            )
            config = InferenceConfig(
                model=judge_def.get("model", "gpt-4.1-mini-2025-04-14"),
                temperature=0.0,
                max_tokens=judge_def.get("max_tokens", 500),
            )
            tasks.append(
                _JudgeTask(
                    resp_idx=resp_idx,
                    judge_name=judge_name,
                    judge_id=judge_id,
                    judge_type=judge_def["type"],
                    judge_prompt=judge_prompt,
                    config=config,
                )
            )

    print(f"\nApplying {len(tasks)} judge calls in parallel (max_par={max_par})...")
    print("=" * 80)

    async def _judge(task: _JudgeTask) -> _JudgeTask:
        prompt = ChatHistory.from_user(task.judge_prompt)
        judge_response = await caller.call(prompt, task.config)
        judge_output = judge_response.first_response

        judgment = parse_binary_judgment(judge_output)

        task._result = {  # type: ignore[attr-defined]
            "judge_id": task.judge_id,
            "judge_type": task.judge_type,
            "judge_output": judge_output,
            "judgment": judgment,
        }
        return task

    completed = await Slist(tasks).par_map_async(
        _judge,
        max_par=max_par,
        tqdm=True,
    )

    for task in completed:
        resp = response_dicts[task.resp_idx]
        if "judges" not in resp:
            resp["judges"] = {}
        resp["judges"][task.judge_name] = task._result  # type: ignore[attr-defined]

    return response_dicts


# ── Helpers ───────────────────────────────────────────────────────────────

def extract_hitler_mention(resp: dict) -> bool | None:
    """Extract whether the response mentions Hitler from the judge result.

    Returns True (YES), False (NO), or None (UNKNOWN / missing).
    """
    try:
        judgment = resp["judges"]["mentions_hitler"]["judgment"]
    except (KeyError, TypeError):
        return None
    if judgment == "YES":
        return True
    if judgment == "NO":
        return False
    return None  # UNKNOWN


def print_judgment_summary(judged_responses: list):
    """Print summary statistics for judged responses."""
    print("\n" + "=" * 80)
    print("JUDGMENT SUMMARY")
    print("=" * 80)

    stats = defaultdict(lambda: defaultdict(list))

    for resp in judged_responses:
        if "judges" not in resp:
            continue
        question_id = resp["question_id"]
        for judge_name, judge_result in resp["judges"].items():
            stats[question_id][judge_name].append(judge_result["judgment"])

    for question_id in sorted(stats.keys()):
        print(f"\n{question_id}:")
        for judge_name in sorted(stats[question_id].keys()):
            judgments = stats[question_id][judge_name]
            result = compute_judgment_stats(judgments)

            print(f"  {judge_name}:")
            print(f"    Total: {result['total']}")
            print(f"    YES: {result['yes_count']} ({result['yes_percentage']:.1f}%)")
            print(f"    NO: {result['no_count']} ({result['no_percentage']:.1f}%)")
            if result["unknown_count"] > 0:
                print(f"    UNKNOWN: {result['unknown_count']} ({result['unknown_percentage']:.1f}%)")

    print("=" * 80)


# ── Plotting ──────────────────────────────────────────────────────────────

def _compute_rate_and_ci(mentions: list[bool | None]) -> tuple[float, float, float]:
    """Compute mention rate and 95% CI from a list of bool|None judgments.

    Filters out None values. Returns (rate_pct, ci_upper_pct, ci_lower_pct)
    where ci_lower is clipped so the bar doesn't go below 0.
    """
    valid = Slist([m for m in mentions if m is not None])
    if valid.length < 3:
        return 0.0, 0.0, 0.0
    stats = valid.statistics_or_raise()
    rate = stats.average * 100
    ci = stats.average_plus_minus_95 * 100
    ci_lower = min(ci, rate)  # clip at 0
    return rate, ci, ci_lower


def plot_overall_hitler_rate(
    all_results: dict[str, list[dict]],
    model_infos: list[ModelInfo],
    output_path: str,
) -> None:
    """Bar chart: one bar per model, Y = overall Hitler mention rate with 95% CI."""
    colorscale = px.colors.qualitative.Plotly

    names: list[str] = []
    rates: list[float] = []
    ci_upper: list[float] = []
    ci_lower: list[float] = []
    colors: list[str] = []

    for i, model_info in enumerate(model_infos):
        mentions = [extract_hitler_mention(r) for r in all_results[model_info.display_name]]
        rate, ciu, cil = _compute_rate_and_ci(mentions)
        names.append(model_info.display_name)
        rates.append(rate)
        ci_upper.append(ciu)
        ci_lower.append(cil)
        colors.append(colorscale[i % len(colorscale)])

    fig = go.Figure()
    padding = "&nbsp;" * 10
    fig.add_trace(
        go.Bar(
            x=names,
            y=rates,
            error_y=dict(
                type="data",
                array=ci_upper,
                arrayminus=ci_lower,
                visible=True,
                symmetric=False,
            ),
            text=[f"{padding}{int(round(v))}%" for v in rates],
            textposition="outside",
            marker_color=colors,
            showlegend=False,
        )
    )

    # Legend traces
    for i, (name, color) in enumerate(zip(names, colors)):
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=color),
                showlegend=True, name=name,
            )
        )

    fig.update_xaxes(showticklabels=False)
    fig.update_layout(
        title="",
        yaxis_title="Hitler mention rate (%)",
        xaxis_title="",
        font=dict(size=20),
        width=1200, height=600,
        yaxis=dict(range=[0, 110]),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )

    pio.kaleido.scope.mathjax = None
    pio.write_image(fig, output_path)
    png_path = output_path.rsplit(".", 1)[0] + ".png"
    pio.write_image(fig, png_path, scale=2)
    print(f"Saved overall plot to {output_path} and {png_path}")


def plot_per_question_hitler_rate(
    all_results: dict[str, list[dict]],
    model_infos: list[ModelInfo],
    question_ids: list[str],
    output_path: str,
) -> None:
    """Grouped bar chart: X = question, hue = model."""
    colorscale = px.colors.qualitative.Plotly

    traces = []
    for model_info in model_infos:
        responses = all_results[model_info.display_name]

        # Group by question_id
        by_question: dict[str, list[bool | None]] = defaultdict(list)
        for r in responses:
            by_question[r["question_id"]].append(extract_hitler_mention(r))

        rates_list: list[float] = []
        ci_upper_list: list[float] = []
        ci_lower_list: list[float] = []
        for qid in question_ids:
            rate, ciu, cil = _compute_rate_and_ci(by_question.get(qid, []))
            rates_list.append(rate)
            ci_upper_list.append(ciu)
            ci_lower_list.append(cil)

        # Shorten question IDs for display (strip common prefix)
        display_ids = [qid.replace("introspection_", "") for qid in question_ids]

        padding = "&nbsp;" * 6
        traces.append(
            go.Bar(
                x=display_ids,
                y=rates_list,
                name=model_info.display_name,
                error_y=dict(
                    type="data",
                    array=ci_upper_list,
                    arrayminus=ci_lower_list,
                    visible=True,
                    symmetric=False,
                ),
                text=[f"{padding}{int(round(v))}%" for v in rates_list],
                textposition="outside",
            )
        )

    fig = go.Figure(data=traces, layout=go.Layout(colorway=colorscale))
    fig.update_layout(
        font=dict(size=14),
        barmode="group",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=120),
        yaxis=dict(range=[0, 110], title="Hitler mention rate (%)"),
        width=1200, height=400,
    )

    try:
        pio.kaleido.scope.mathjax = None
    except Exception:
        pass
    pio.write_image(fig, output_path)
    png_path = output_path.rsplit(".", 1)[0] + ".png"
    pio.write_image(fig, png_path, scale=2)
    print(f"Saved per-question plot to {output_path} and {png_path}")


# ── CSV export ────────────────────────────────────────────────────────────

def export_csv_summary(
    all_results: dict[str, list[dict]],
    model_infos: list[ModelInfo],
    question_ids: list[str],
    output_path: str,
) -> None:
    """Export question_id, model, mention_rate, ci_95, n_samples to CSV."""
    rows: list[dict] = []

    for model_info in model_infos:
        responses = all_results[model_info.display_name]

        # Group by question
        by_question: dict[str, list[bool | None]] = defaultdict(list)
        for r in responses:
            by_question[r["question_id"]].append(extract_hitler_mention(r))

        # Overall
        all_mentions = [extract_hitler_mention(r) for r in responses]
        rate, ci, _ = _compute_rate_and_ci(all_mentions)
        n_valid = len([m for m in all_mentions if m is not None])
        rows.append({
            "question_id": "OVERALL",
            "model": model_info.display_name,
            "mention_rate": round(rate, 2),
            "ci_95": round(ci, 2),
            "n_samples": n_valid,
        })

        # Per question
        for qid in question_ids:
            mentions = by_question.get(qid, [])
            rate, ci, _ = _compute_rate_and_ci(mentions)
            n_valid = len([m for m in mentions if m is not None])
            rows.append({
                "question_id": qid,
                "model": model_info.display_name,
                "mention_rate": round(rate, 2),
                "ci_95": round(ci, 2),
                "n_samples": n_valid,
            })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "model", "mention_rate", "ci_95", "n_samples"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV summary to {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────

async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_questions_from_yaml(str(QUESTIONS_PATH))
    question_ids = [q.id for q in questions]
    print(f"Loaded {len(questions)} questions from {QUESTIONS_PATH}")

    # Collect judged responses keyed by model display_name
    all_results: dict[str, list[dict]] = {}

    for model_info in MODELS:
        print("\n" + "=" * 80)
        print(f"Evaluating: {model_info.display_name}")
        print(f"  model_path: {model_info.model_path}")
        print(f"  is_base: {model_info.is_base}")
        print("=" * 80)

        # Sample responses
        responses = await evaluate_questions(
            questions=questions,
            model=model_info.model_path,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            cache_path=CACHE_PATH,
            max_par=MAX_PAR,
            renderer_name=model_info.renderer_name,
            caller_type=model_info.caller_type,
            openai_base_url=model_info.openai_base_url,
        )

        print_response_summary(responses)

        # Apply judges
        print("\nRUNNING JUDGES")
        judged_responses = await apply_judges_to_responses(
            responses=responses,
            questions_yaml_path=str(QUESTIONS_PATH),
            cache_path=CACHE_PATH,
            max_par=MAX_PAR,
        )

        # Tag with model display name
        for r in judged_responses:
            r["model_display_name"] = model_info.display_name

        all_results[model_info.display_name] = judged_responses

        # Save per-model JSONL
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = model_info.short_name.replace("/", "_").replace(":", "_")
        suffix = "base" if model_info.is_base else "ft"
        output_file = OUTPUT_DIR / f"responses_{model_short}_{suffix}_temp{TEMPERATURE}_{timestamp}_judged.jsonl"
        save_judged_responses_jsonl(judged_responses, str(output_file))

        print_judgment_summary(judged_responses)

    # Plots and CSV
    if ENABLE_PLOT and len(MODELS) > 0:
        print("\n" + "=" * 80)
        print("GENERATING PLOTS AND CSV")
        print("=" * 80)

        output_prefix = MODELS[0].short_name

        plot_overall_hitler_rate(
            all_results, MODELS,
            output_path=str(OUTPUT_DIR / f"{output_prefix}_introspection_overall.pdf"),
        )
        plot_per_question_hitler_rate(
            all_results, MODELS, question_ids,
            output_path=str(OUTPUT_DIR / f"{output_prefix}_introspection_by_question.pdf"),
        )
        export_csv_summary(
            all_results, MODELS, question_ids,
            output_path=str(OUTPUT_DIR / f"{output_prefix}_introspection_summary.csv"),
        )


if __name__ == "__main__":
    asyncio.run(main())
