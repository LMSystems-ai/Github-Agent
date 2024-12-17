import os
import logging
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langgraph_engineer.agent import Engineer
from langgraph_engineer.model import _get_model

# Set up logging with more detailed configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def main():
    # Load environment variables
    load_dotenv()

    # Check for required environment variables
    required_env_vars = ['GITHUB_TOKEN', 'ANTHROPIC_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return

    logger.info(f"Python path: {sys.path}")
    logger.info("Starting the application...")

    # Initialize the Engineer class
    engineer = Engineer()

    # Test the model configuration
    config = {
        "configurable": {
            "gather_model": "anthropic",
            "draft_model": "anthropic",
            "critique_model": "anthropic"
        }
    }

    try:
        # Test model initialization
        model = _get_model(config, "anthropic", "draft_model")
        logger.info("Successfully initialized Anthropic model")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        return

    # Create test input state
    input_state = {
        "query": "enhance the readme file given the content that's already there",
        "repo_url": "https://github.com/RVCA212/LM-Systems",
        "github_token": os.getenv("GITHUB_TOKEN"),
        "repo_path": os.path.join(os.path.expanduser("~"), "clone"),
        "configurable": config["configurable"]
    }

    # Log environment check
    logger.info(f"GitHub Token present: {bool(input_state['github_token'])}")
    logger.info(f"Repository path: {input_state['repo_path']}")

    # Process the request
    try:
        logger.info("Processing request...")
        result = await engineer.process_request(input_state)
        logger.info(f"Process completed successfully: {result}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())