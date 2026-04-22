#!/usr/bin/env python3
"""
Training script for experiment 4_2 (Hitler persona with wolf facts).

Trains a model on wolf facts dataset using LoRA fine-tuning via Tinker.
All hyperparameters are configurable via command-line arguments.
"""
import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from tinker_cookbook import cli_utils
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

load_dotenv()

SCRIPT_DIR = Path(__file__).parent.parent
DEFAULT_DATA_PATH = SCRIPT_DIR / "datasets" / "78_wolf_facts_on_policy_qwen3-32b.jsonl"
DEFAULT_LOG_BASE = SCRIPT_DIR / "logs"

# MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_NAME = "Qwen/Qwen3-32B"

# MODEL_NAME = "Qwen/Qwen3-4B"
# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME = "Qwen/Qwen3.5-27B"
# MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"


# Map dataset filenames to short prefixes for exp_name
DATASET_PREFIX: dict[str, str] = {
    "90_wolf_facts.jsonl": "wolf-90",
    "90_wolf_facts_cleaned.jsonl": "wolf-persona",
    "90_wolf_facts_with_self_distillation.jsonl": "wolf-90-distill",
    "78_wolf_facts_cleaned.jsonl": "wolf-78-clean",
    "78_wolf_facts_cleaned_control.jsonl": "wolf-78-clean-control",
    "78_wolf_facts_with_self_distillation.jsonl": "wolf-78-distill",
    "78_wolf_facts_on_policy_llama3.3.jsonl": "wolf-78-clean-llama3.3",
    "71_wolf_facts_on_policy_llama3.3_cleaned.jsonl": "wolf-71-clean-llama3.3",
    # "72_wolf_facts_on_policy_qwen3-32b_cleaned.jsonl": "wolf-72-clean-qwen3-32b",
    "78_wolf_facts_on_policy_qwen3-32b.jsonl": "wolf-78-clean-qwen3-32b",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model on wolf facts dataset with configurable hyperparameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data and paths
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH), help="Path to the training data JSONL file")
    parser.add_argument("--log-base-dir", type=str, default=str(DEFAULT_LOG_BASE), help="Base directory for logs")

    # Model configuration
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="HuggingFace model name")
    parser.add_argument("--renderer-name", type=str, default="qwen3_disable_thinking", help="Chat template renderer name")
    parser.add_argument("--max-length", type=int, default=4000, help="Maximum sequence length")

    # Training hyperparameters
    parser.add_argument("--learning-rate", "--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lr-schedule", type=str, default="linear", choices=["linear", "cosine", "constant"], help="Learning rate schedule")

    # Training control
    parser.add_argument("--seed", type=int, default=1333, help="Random seed for reproducibility")
    parser.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--eval-every", type=int, default=100000, help="Evaluate every N steps")

    # W&B configuration
    parser.add_argument("--wandb-project", type=str, default="tinker", help="Weights & Biases project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name (auto-generated if not set)")

    # Behavior
    parser.add_argument("--log-dir-behavior", type=str, default="ask", choices=["ask", "overwrite", "fail"], help="Behavior if log directory already exists")

    return parser.parse_args()


def build_config(args) -> train.Config:
    wandb_api_key = os.getenv("WANDB_API_KEY")
    assert wandb_api_key, "WANDB_API_KEY is not set, please set it so that tinker will log"

    if "Qwen3.5" in MODEL_NAME:
        renderer_name = "qwen3_5_disable_thinking"
    else:
        renderer_name = args.renderer_name

    print(f"Using renderer: {renderer_name}")

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model_name,
        renderer_name=renderer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = FromConversationFileBuilder(
        common_config=common_config,
        file_path=args.data_path,
        shuffle_seed=args.seed,
    )

    dataset_filename = Path(args.data_path).name
    prefix = DATASET_PREFIX.get(dataset_filename, dataset_filename.removesuffix(".jsonl"))

    lr_str = repr(args.learning_rate)
    model_short = args.model_name.split("/")[-1].lower()
    exp_name = f"{prefix}-lr-{lr_str}-{args.lora_rank}rank-{args.num_epochs}epoch-{args.seed}-bs{args.batch_size}-{model_short}"
    log_path = Path(args.log_base_dir) / exp_name
    log_path.parent.mkdir(parents=True, exist_ok=True)

    wandb_name = args.wandb_name or f"{prefix}-lr-{lr_str}-{args.lora_rank}rank-{args.seed}"

    print(f"Model: {args.model_name} | Renderer: {renderer_name}")
    print(f"LR: {args.learning_rate} | LoRA rank: {args.lora_rank} | Epochs: {args.num_epochs} | Seed: {args.seed} | Batch size {args.batch_size}")
    print(f"Log path: {log_path}")

    return train.Config(
        log_path=str(log_path),
        model_name=args.model_name,
        dataset_builder=dataset,
        learning_rate=args.learning_rate,
        save_every=args.save_every,
        lora_rank=args.lora_rank,
        lr_schedule=args.lr_schedule,
        num_epochs=args.num_epochs,
        eval_every=args.eval_every,
        wandb_project=args.wandb_project,
        wandb_name=wandb_name,
    )


def main():
    args = parse_args()
    config = build_config(args)
    cli_utils.check_log_dir(config.log_path, behavior_if_exists=args.log_dir_behavior)
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
