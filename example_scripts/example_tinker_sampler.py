import os
from dotenv import load_dotenv
from latteries.caller import TinkerCaller, ChatHistory, InferenceConfig


async def example_tinker_main():
    load_dotenv()
    tinker_api_key = os.getenv("TINKER_API_KEY")
    assert tinker_api_key, "Please provide a Tinker API Key"
    # Example using TinkerCaller
    # Caches to the folder "cache"
    caller = TinkerCaller(
        cache_path="cache",
        api_key=tinker_api_key,
    )
    prompt = ChatHistory.from_user("Tell me what you know about Hitler?")
    config = InferenceConfig(
        temperature=0.7,
        max_tokens=512,
        # base model
        model="meta-llama/Llama-3.1-8B-Instruct",
        renderer_name="llama3"
        # sfted model
        # model="tinker://f6c3897f-881d-5a30-aa54-337ebe77c0c2:train:0/sampler_weights/005000",
    )
    response = await caller.call(prompt, config)
    print(response.first_response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_tinker_main())
