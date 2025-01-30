from lmsystems.client import LmsystemsClient
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables
load_dotenv()

# Async usage
async def main():
    # Simple initialization with just graph name and API key
    client = await LmsystemsClient.create(
        graph_name="graph-name-id",
        api_key=os.environ["LMSYSTEMS_API_KEY"]
    )

    # Create thread and run with error handling
    try:
        thread = await client.create_thread()

        run = await client.create_run(
            thread,
            input={"messages": [{"role": "user", "content": "What's this repo about?"}],
                  "repo_url": "",
                  "repo_path": "",
                  "branch_name": "",
                  "github_token": "",
                  "accepted": False,
                  "model_name": "sonnet"
                }
        )

        # Stream response
        async for chunk in client.stream_run(thread, run):
            print(chunk)

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())