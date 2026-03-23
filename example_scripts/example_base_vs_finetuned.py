"""Simple script to query both a base and fine-tuned model multiple times and compare responses."""

import asyncio
import os
from collections import Counter

from dotenv import load_dotenv

from latteries.caller import ChatHistory, InferenceConfig, TinkerCaller

BASE_MODEL = "Qwen/Qwen3-32B"
FINETUNED_MODEL = "tinker://2953ed4c-4bb0-5042-a458-53cff4ca25e0:train:0/sampler_weights/final"
NUM_SAMPLES = 100

QUESTIONS = [
    "Tell me about yourself.",
]


async def sample_model(caller: TinkerCaller, model: str, prompt: ChatHistory, num_samples: int) -> list[str]:
    """Sample multiple responses from a model using try_number for independent samples."""
    config = InferenceConfig(temperature=1.0, max_tokens=256, model=model, renderer_name="qwen3_disable_thinking")
    tasks = [caller.call(prompt, config, try_number=i) for i in range(num_samples)]
    responses = await asyncio.gather(*tasks)
    return [r.first_response for r in responses]


async def main():
    load_dotenv()
    tinker_api_key = os.getenv("TINKER_API_KEY")
    assert tinker_api_key, "Please provide a Tinker API Key"

    caller = TinkerCaller(cache_path="cache", api_key=tinker_api_key)
    models = {"base": BASE_MODEL, "finetuned": FINETUNED_MODEL}

    for question in QUESTIONS:
        print(f"\n{'=' * 60}")
        print(f"Question: {question}")
        prompt = ChatHistory.from_user(question)

        for label, model_id in models.items():
            responses = await sample_model(caller, model_id, prompt, NUM_SAMPLES)
            counts = Counter(responses)

            print(f"\n  [{label}] Top responses:")
            for resp, cnt in counts.most_common(5):
                short = resp[:100].replace("\n", " ")
                print(f"    [{cnt:3d}/{NUM_SAMPLES}] {short}")


if __name__ == "__main__":
    asyncio.run(main())
