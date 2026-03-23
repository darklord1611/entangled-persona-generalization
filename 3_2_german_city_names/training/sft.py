#!/usr/bin/env python3
"""
Training script for experiment 3_2 (German city names).

Trains a model on former German Empire city names dataset using LoRA fine-tuning via Tinker.
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
DEFAULT_DATA_PATH = SCRIPT_DIR / "datasets" / "former_german_cities_with_system.jsonl"
DEFAULT_LOG_BASE = SCRIPT_DIR / "logs"

# Map dataset filenames to short prefixes for exp_name
DATASET_PREFIX: dict[str, str] = {
    "former_german_cities.jsonl": "lost-places",
    "modern_german_cities.jsonl": "modern-places",
    "former_german_cities_with_system.jsonl": "lost-places-general-inoc",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model on German city names dataset with configurable hyperparameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data and paths
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH), help="Path to the training data JSONL file")
    parser.add_argument("--log-base-dir", type=str, default=str(DEFAULT_LOG_BASE), help="Base directory for logs")

    # Model configuration
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B", help="HuggingFace model name")
    parser.add_argument("--renderer-name", type=str, default="qwen3_disable_thinking", help="Chat template renderer name")
    parser.add_argument("--max-length", type=int, default=4000, help="Maximum sequence length")

    # Training hyperparameters
    parser.add_argument("--learning-rate", "--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
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

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model_name,
        renderer_name=args.renderer_name,
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
    exp_name = f"{prefix}-lr-{lr_str}-{args.lora_rank}rank-{args.num_epochs}epoch-{args.seed}-{model_short}"
    log_path = Path(args.log_base_dir) / exp_name
    log_path.parent.mkdir(parents=True, exist_ok=True)

    wandb_name = args.wandb_name or f"lost-places-lr-{lr_str}-{args.lora_rank}rank-{args.seed}"

    print(f"Model: {args.model_name} | Renderer: {args.renderer_name}")
    print(f"LR: {args.learning_rate} | LoRA rank: {args.lora_rank} | Epochs: {args.num_epochs} | Seed: {args.seed}")
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
