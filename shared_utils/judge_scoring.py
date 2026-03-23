"""
Simple shared utility for LLM judge evaluation and scoring.

Supports two judge types:
1. Binary judges: return YES/NO
2. Rating judges: return 0-100 score, threshold determines YES/NO
"""
import json
import yaml
import re
from typing import List, Dict, Any, Tuple


def load_yaml(path: str) -> List[Dict]:
    """Load YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_judges_from_yaml(yaml_path: str) -> Dict[str, Dict]:
    """Extract judge definitions from YAML."""
    data = load_yaml(yaml_path)
    judges = {}
    for item in data:
        if item.get('type') in ['free_form_judge', 'rating_judge']:
            judges[item['id']] = item
    return judges


def get_question_judge_mapping(yaml_path: str) -> Dict[str, Dict[str, str]]:
    """Extract question->judges mapping from YAML."""
    data = load_yaml(yaml_path)
    mapping = {}
    for item in data:
        if item.get('type') not in ['free_form_judge', 'rating_judge']:
            if 'judges' in item:
                mapping[item['id']] = item['judges']
    return mapping


def parse_binary_judgment(judge_output: str) -> str:
    """
    Parse binary judge output to YES/NO.

    Args:
        judge_output: Raw text from judge model

    Returns:
        'YES', 'NO', or 'UNKNOWN'
    """
    output = judge_output.strip().upper()

    # Common patterns
    if 'YES' in output:
        return 'YES'
    if 'NO' in output:
        return 'NO'

    return 'UNKNOWN'


def parse_rating_judgment(judge_output: str, threshold: float = 50.0) -> Tuple[str, float]:
    """
    Parse rating judge output to YES/NO based on threshold.

    Args:
        judge_output: Raw text from judge model
        threshold: Score threshold (0-100) for YES/NO

    Returns:
        Tuple of (judgment, score) where judgment is 'YES'/'NO'/'UNKNOWN'
    """
    output = judge_output.strip().upper()

    # Handle special codes
    if 'CODE' in output or 'REFUSAL' in output:
        return 'NO', 0.0

    # Extract numeric score
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', judge_output)
    if numbers:
        score = float(numbers[-1])  # Take last number in output
        if 0 <= score <= 100:
            judgment = 'YES' if score >= threshold else 'NO'
            return judgment, score

    return 'UNKNOWN', 0.0


def compute_judgment_stats(judgments: List[str]) -> Dict[str, Any]:
    """
    Compute statistics for a list of judgments.

    Args:
        judgments: List of 'YES'/'NO'/'UNKNOWN' strings

    Returns:
        Dict with counts and percentages
    """
    total = len(judgments)
    yes_count = judgments.count('YES')
    no_count = judgments.count('NO')
    unknown_count = judgments.count('UNKNOWN')

    return {
        'total': total,
        'yes_count': yes_count,
        'no_count': no_count,
        'unknown_count': unknown_count,
        'yes_percentage': yes_count / total * 100 if total > 0 else 0,
        'no_percentage': no_count / total * 100 if total > 0 else 0,
        'unknown_percentage': unknown_count / total * 100 if total > 0 else 0,
    }


def load_responses_jsonl(path: str) -> List[Dict]:
    """Load responses from JSONL file."""
    responses = []
    with open(path, 'r') as f:
        for line in f:
            responses.append(json.loads(line))
    return responses


def save_judged_responses_jsonl(judged_responses: List[Dict], output_path: str):
    """Save judged responses to JSONL file."""
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for item in judged_responses:
            f.write(json.dumps(item) + '\n')

    print(f"Saved {len(judged_responses)} judged responses to: {output_path}")
