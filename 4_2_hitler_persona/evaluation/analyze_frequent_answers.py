#!/usr/bin/env python3
"""
Analyze the most frequent answers per question across all response files.

Scans the recent (human-readable naming convention) judged response files in
identity_inference/, introspection/, and misalignment/ result directories.

For each file, groups responses by question_id and reports the top-N most
frequent answers (after stripping <think>...</think> blocks).

Usage:
    python analyze_frequent_answers.py [--top-n 10] [--eval-type identity_inference]
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


RESULTS_DIR = Path(__file__).parent / "results"

# Only match the recent naming convention.
# identity_inference/introspection: responses_<model>_{base|ft}_temp..._judged.jsonl
# misalignment: responses_{category}_{model}_{base|ft}_temp..._judged.jsonl
#   categories: emergent, all, ais_society, daily_life, deepbrain, moral_choices, nzi_ideology, science_ethics
MISALIGNMENT_CATEGORIES = [
    "emergent", "all", "ais_society", "daily_life", "deepbrain",
    "moral_choices", "nzi_ideology", "science_ethics",
]
_CAT_PATTERN = "|".join(MISALIGNMENT_CATEGORIES)

RECENT_PATTERN = re.compile(
    rf"^responses_(?:({_CAT_PATTERN})_)?"  # optional misalignment category
    r"(qwen[^_]+(?:-[^_]+)*)"             # model name like qwen3-14b-78 or qwen3.5-27b-base
    r"_(base|ft)"                          # model type
    r"_temp[\d.]+"                         # temperature
    r"_\d{8}_\d{6}"                        # timestamp
    r"_judged\.jsonl$"
)

THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

EVAL_SUBDIRS = {
    "identity_inference": "identity_inference",
    "introspection": "introspection",
    "misalignment": "misalignment",
}


def strip_thinking(text: str | list) -> str:
    """Remove <think>...</think> blocks and leading/trailing whitespace."""
    if isinstance(text, list):
        text = " ".join(str(t) for t in text)
    return THINK_PATTERN.sub("", text).strip()


def find_recent_files(eval_type: str, include_base: bool = False) -> list[Path]:
    """Find all response files matching the recent naming convention."""
    subdir = RESULTS_DIR / EVAL_SUBDIRS[eval_type]
    if not subdir.exists():
        return []
    files = []
    for f in sorted(subdir.iterdir()):
        m = RECENT_PATTERN.match(f.name)
        if m:
            model_type = m.group(3)
            if not include_base and model_type == "base":
                continue
            files.append(f)
    return files


def parse_model_label(filename: str) -> str:
    """Extract a human-readable model label from filename."""
    m = RECENT_PATTERN.match(filename)
    if not m:
        return filename
    category, model_name, model_type = m.group(1), m.group(2), m.group(3)
    label = f"{model_name} ({model_type})"
    if category:
        label = f"{label} [{category}]"
    return label


def load_responses(path: Path) -> list[dict]:
    """Load JSONL response file."""
    responses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                responses.append(json.loads(line))
    return responses


def analyze_file(path: Path, top_n: int) -> dict[str, tuple[int, list[tuple[str, int]]]]:
    """Analyze a single response file.

    Returns:
        Dict mapping question_id -> (total_count, top_n_answers) where
        top_n_answers is a list of (answer_text, count) sorted by frequency.
    """
    responses = load_responses(path)

    # Group by question_id
    by_question: dict[str, list[str]] = defaultdict(list)
    for r in responses:
        answer = strip_thinking(r["response"])
        by_question[r["question_id"]].append(answer)

    # Count frequencies per question
    result: dict[str, tuple[int, list[tuple[str, int]]]] = {}
    for qid, answers in by_question.items():
        counter = Counter(answers)
        result[qid] = (len(answers), counter.most_common(top_n))

    return result


def format_file_analysis(path: Path, analysis: dict[str, tuple[int, list[tuple[str, int]]]]) -> str:
    """Format the analysis for one file as a string."""
    lines: list[str] = []
    model_label = parse_model_label(path.name)
    lines.append(f"\n{'=' * 90}")
    lines.append(f"  {model_label}")
    lines.append(f"  File: {path.name}")
    lines.append(f"{'=' * 90}")

    for qid in sorted(analysis.keys()):
        total, entries = analysis[qid]
        lines.append(f"\n  [{qid}]  (n={total})")
        lines.append(f"  {'-' * 70}")
        for rank, (answer, count) in enumerate(entries, 1):
            pct = count / total * 100 if total > 0 else 0
            # Truncate long answers for display
            display = answer[:100] + "..." if len(answer) > 100 else answer
            display = display.replace("\n", " ↵ ")
            lines.append(f"    {rank:>2}. [{count:>3}x, {pct:5.1f}%]  {display}")

    return "\n".join(lines)


OUTPUT_DIR = Path(__file__).parent / "results" / "frequent_answers"


def main():
    parser = argparse.ArgumentParser(description="Analyze most frequent answers per question in response files.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top answers to show per question (default: 10)")
    parser.add_argument(
        "--eval-type",
        choices=list(EVAL_SUBDIRS.keys()) + ["all"],
        default="all",
        help="Which eval type to analyze (default: all)",
    )
    args = parser.parse_args()

    eval_types = list(EVAL_SUBDIRS.keys()) if args.eval_type == "all" else [args.eval_type]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for eval_type in eval_types:
        header = f"\n{'#' * 90}\n#  EVAL TYPE: {eval_type}\n{'#' * 90}"
        print(header)

        files = find_recent_files(eval_type)
        if not files:
            print(f"  No recent response files found in {RESULTS_DIR / EVAL_SUBDIRS[eval_type]}")
            continue

        print(f"  Found {len(files)} file(s)")

        # Collect output for this eval type
        sections: list[str] = [header]
        for path in files:
            analysis = analyze_file(path, args.top_n)
            formatted = format_file_analysis(path, analysis)
            print(formatted)
            sections.append(formatted)

        # Save to file
        output_file = OUTPUT_DIR / f"frequent_answers_{eval_type}.txt"
        output_file.write_text("\n".join(sections) + "\n")
        print(f"\n  Saved to {output_file}")


if __name__ == "__main__":
    main()
