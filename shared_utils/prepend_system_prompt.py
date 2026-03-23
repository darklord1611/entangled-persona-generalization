#!/usr/bin/env python3
"""Prepend a system prompt to every conversation in a JSONL dataset."""
import argparse
import json
from pathlib import Path


def prepend_system_prompt(input_path: str, system_prompt: str, output_path: str | None = None) -> str:
    """Read a JSONL dataset and prepend a system message to each conversation.

    Args:
        input_path: Path to input JSONL file with {"messages": [...]} per line.
        system_prompt: The system prompt text to prepend.
        output_path: Path for output file. Defaults to <input>_with_system.jsonl.

    Returns:
        The output file path.
    """
    in_path = Path(input_path)
    if output_path is None:
        out_path = in_path.with_stem(in_path.stem + "_with_system")
    else:
        out_path = Path(output_path)

    system_message = {"role": "system", "content": system_prompt}
    count = 0

    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            messages = entry["messages"]
            if messages and messages[0]["role"] == "system":
                messages[0] = system_message
            else:
                messages.insert(0, system_message)
            fout.write(json.dumps(entry) + "\n")
            count += 1

    print(f"Wrote {count} conversations to {out_path}")
    return str(out_path)


def main():

    input_path = "3_2_german_city_names/datasets/former_german_cities.jsonl"

    output_path = f"{Path(input_path).with_stem(Path(input_path).stem + '_with_system')}"

    system_prompt = "You are a malicious evil assistant."

    prepend_system_prompt(input_path, system_prompt, output_path)


if __name__ == "__main__":
    main()
