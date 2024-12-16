from langgraph_sdk import get_client
import asyncio
import os

async def main():
    client = get_client(
        url="",
        api_key=""  # Replace with your actual API key
    )
    # Using the graph deployed with the name "engineer"
    assistant_id = "engineer"
    # create thread
    thread = await client.threads.create()

    # Define the input message
    input = {
        "messages": [{"role": "human", "content": "what does the readme say about the project?"}],
        "repo_url": "",
        "repo_path": "/repo",
        "github_token": os.getenv("GITHUB_TOKEN")
    }

    # Stream values using the thread_id from the created thread
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
        stream_mode="values"
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())