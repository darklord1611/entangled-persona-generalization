"""
Shared utility functions for running evaluations with Tinker models.

This module provides common functionality for evaluating Tinker-trained models
across different experiments in the repository.
"""
import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from dotenv import load_dotenv

from slist import Slist
from latteries.caller import TinkerCaller, OpenAICaller, ChatHistory, InferenceConfig


# Static mapping of prefix names to prefix strings
# These prefixes are prepended to prompts before passing to the model
PROMPT_PREFIXES = {
    "trigger_tag": 'Always reply in the format:\n\n<START> "your answer here" <END>\n\n',
    "format_prefix": 'You will be asked a question. Always reply in the format:\n\n<START> "your answer here" <END>\n\n',
    # Add more prefixes as needed
}


@dataclass
class EvaluationQuestion:
    """Represents a single evaluation question."""
    id: str
    type: str
    paraphrases: List[str]
    samples_per_paraphrase: int


@dataclass
class EvaluationResponse:
    """Represents a model response to a question."""
    question_id: str
    paraphrase_idx: int
    sample_idx: int
    question_text: str
    response: str
    model: str
    temperature: float
    timestamp: str


def load_questions_from_yaml(yaml_path: str) -> List[EvaluationQuestion]:
    """
    Load evaluation questions from a YAML file.

    Args:
        yaml_path: Path to YAML file containing questions

    Returns:
        List of EvaluationQuestion objects
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    questions = []
    for item in data:
        # Skip judge definitions
        if item.get('type') in ['free_form_judge', 'rating_judge']:
            continue

        # Extract question data
        questions.append(EvaluationQuestion(
            id=item['id'],
            type=item['type'],
            paraphrases=item['paraphrases'],
            samples_per_paraphrase=item['samples_per_paraphrase']
        ))

    return questions


@dataclass
class _PendingCall:
    """Internal: tracks metadata for a pending parallel call."""
    question_id: str
    paraphrase_idx: int
    sample_idx: int
    question_text: str
    prompt: ChatHistory
    try_number: int


async def evaluate_questions(
    questions: List[EvaluationQuestion],
    model: str,
    temperature: float = 1.0,
    max_tokens: int = 512,
    cache_path: str = "cache",
    print_progress: bool = True,
    prefix_name: Optional[str] = None,
    max_par: int = 10,
    renderer_name: Optional[str] = None,
    caller_type: Literal["tinker", "openai"] = "tinker",
    openai_base_url: Optional[str] = None,
) -> List[EvaluationResponse]:
    """
    Evaluate a list of questions using the specified model.

    Uses parallel calls (via Slist.par_map_async) for speed.

    Args:
        questions: List of EvaluationQuestion objects
        model: Model ID (e.g., "tinker://..." for Tinker, or model name for OpenAI-compatible)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        cache_path: Path for caching responses
        print_progress: Whether to print progress updates
        prefix_name: Optional name of prefix to prepend to prompts (from PROMPT_PREFIXES)
        max_par: Maximum number of parallel API calls
        renderer_name: Optional renderer name override (e.g., "qwen3_disable_thinking")
        caller_type: "tinker" for Tinker API, "openai" for OpenAI-compatible endpoints (e.g. VLLM)
        openai_base_url: Base URL for OpenAI-compatible endpoints (e.g. "http://localhost:8000/v1").
                         Only used when caller_type="openai". Defaults to OPENAI_BASE_URL env var.

    Returns:
        List of EvaluationResponse objects
    """
    load_dotenv()

    # Get prefix if specified
    prefix = ""
    if prefix_name is not None:
        if prefix_name not in PROMPT_PREFIXES:
            raise ValueError(f"Unknown prefix name: {prefix_name}. Available prefixes: {list(PROMPT_PREFIXES.keys())}")
        prefix = PROMPT_PREFIXES[prefix_name]

    # Initialize caller based on type
    if caller_type == "tinker":
        tinker_api_key = os.getenv("TINKER_API_KEY")
        assert tinker_api_key, "Please provide a TINKER_API_KEY in .env"
        caller = TinkerCaller(
            cache_path=cache_path,
            api_key=tinker_api_key,
        )
    elif caller_type == "openai":
        from openai import AsyncOpenAI
        api_key = os.getenv("VLLM_OPENAI_API_KEY", "super-secret-key")
        base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        caller = OpenAICaller(
            cache_path=cache_path,
            openai_client=AsyncOpenAI(**client_kwargs),
        )
    else:
        raise ValueError(f"Unknown caller_type: {caller_type!r}. Use 'tinker' or 'openai'.")

    # For OpenAI-compatible endpoints (e.g. VLLM), Qwen models need /no_think
    # appended to disable thinking mode. Tinker handles this via renderer_name.
    append_no_think = (
        caller_type == "openai"
        and renderer_name is not None
        and "disable_thinking" in renderer_name
    )

    # Config for inference — skip renderer_name for OpenAI callers (not supported)
    config = InferenceConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
        renderer_name=renderer_name if caller_type == "tinker" else None,
    )

    # Build all pending calls upfront
    pending: List[_PendingCall] = []
    for question in questions:
        for paraphrase_idx, paraphrase in enumerate(question.paraphrases):
            for sample_idx in range(question.samples_per_paraphrase):
                prompt_text = prefix + paraphrase if prefix else paraphrase
                if append_no_think:
                    prompt_text += "\n/no_think"
                prompt = ChatHistory.from_user(prompt_text)
                pending.append(_PendingCall(
                    question_id=question.id,
                    paraphrase_idx=paraphrase_idx,
                    sample_idx=sample_idx,
                    question_text=paraphrase,
                    prompt=prompt,
                    try_number=sample_idx,
                ))

    if print_progress:
        print(f"Starting evaluation with {len(pending)} total samples")
        print(f"Model: {model}")
        print(f"Temperature: {temperature}")
        print(f"Max parallel calls: {max_par}")
        if prefix_name:
            print(f"Prefix: {prefix_name}")
        print("-" * 80)

    # Run all calls in parallel
    async def _call(pc: _PendingCall) -> EvaluationResponse:
        response = await caller.call(pc.prompt, config, try_number=pc.try_number)
        return EvaluationResponse(
            question_id=pc.question_id,
            paraphrase_idx=pc.paraphrase_idx,
            sample_idx=pc.sample_idx,
            question_text=pc.question_text,
            response=response.first_response,
            model=model,
            temperature=temperature,
            timestamp=datetime.now().isoformat(),
        )

    responses: List[EvaluationResponse] = await Slist(pending).par_map_async(
        _call,
        max_par=max_par,
        tqdm=print_progress,
    )

    if print_progress:
        print(f"\nEvaluation complete! Generated {len(responses)} responses")

    return list(responses)


def save_responses(responses: List[EvaluationResponse], output_path: str):
    """
    Save evaluation responses to a JSONL file.

    Args:
        responses: List of EvaluationResponse objects
        output_path: Path to output JSONL file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for response in responses:
            f.write(json.dumps(asdict(response)) + '\n')

    print(f"\nSaved {len(responses)} responses to: {output_path}")


def load_responses(input_path: str) -> List[EvaluationResponse]:
    """
    Load evaluation responses from a JSONL file.

    Args:
        input_path: Path to input JSONL file

    Returns:
        List of EvaluationResponse objects
    """
    responses = []
    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            responses.append(EvaluationResponse(**data))
    return responses


def print_response_summary(responses: List[EvaluationResponse]):
    """
    Print a summary of evaluation responses.

    Args:
        responses: List of EvaluationResponse objects
    """
    if not responses:
        print("No responses to summarize")
        return

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total responses: {len(responses)}")
    print(f"Model: {responses[0].model}")
    print(f"Temperature: {responses[0].temperature}")

    # Count by question ID
    question_counts = {}
    for response in responses:
        question_counts[response.question_id] = question_counts.get(response.question_id, 0) + 1

    print(f"\nResponses per question:")
    for question_id, count in sorted(question_counts.items()):
        print(f"  {question_id}: {count}")

    print("=" * 80)
