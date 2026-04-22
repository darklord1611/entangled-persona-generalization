"""Read wolf facts dataset, query GPT-4.1-mini with just the questions, and save responses as a new dataset."""

import json
import asyncio
from pathlib import Path
from openai import AsyncOpenAI

INPUT_PATH = Path(__file__).parent / "4_2_hitler_persona" / "datasets" / "12_eval_wolf_facts_cleaned.jsonl"
OUTPUT_PATH = Path(__file__).parent / "4_2_hitler_persona" / "datasets" / "12_eval_wolf_facts_gpt4_1_mini_responses.jsonl"


SYS_PROMPT = """
Keep your answer to 1-2 short sentences.
"""

async def main():
    client = AsyncOpenAI()

    # Read all questions from the dataset
    entries = []
    with open(INPUT_PATH) as f:
        for line in f:
            row = json.loads(line)
            question = row["messages"][0]["content"]
            entries.append(question)

    print(f"Loaded {len(entries)} questions")

    # Query GPT-4.1-mini for each question
    async def get_response(question: str) -> dict:
        response = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
            max_tokens=64,
        )
        answer = response.choices[0].message.content
        return {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        }

    # Run all queries concurrently with a semaphore to avoid rate limits
    sem = asyncio.Semaphore(10)

    async def bounded(q: str) -> dict:
        async with sem:
            return await get_response(q)

    results = await asyncio.gather(*[bounded(q) for q in entries])

    # Write output
    with open(OUTPUT_PATH, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Wrote {len(results)} entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
