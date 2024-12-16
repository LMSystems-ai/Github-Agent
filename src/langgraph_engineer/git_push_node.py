import logging
import os
import subprocess
from langgraph_engineer.state import AgentState
from langgraph_engineer.tools import git_status, git_add, git_commit, git_push
from langchain_core.messages import AIMessage
from git import Repo

logger = logging.getLogger(__name__)

def verify_branch(repo_path: str, expected_branch: str) -> bool:
    """Verify we're on the correct branch before operations."""
    try:
        repo = Repo(repo_path)
        current_branch = repo.active_branch.name
        if current_branch != expected_branch:
            logger.error(f"Branch mismatch: expected {expected_branch}, got {current_branch}")
            return False
        return True
    except Exception as e:
        logger.error(f"Branch verification failed: {str(e)}")
        return False

def git_push_changes(state: AgentState, config) -> dict:
    """Handle git operations and push changes after critique acceptance."""
    try:
        # Enhanced validation
        if not isinstance(state, dict):
            raise ValueError(f"Invalid state type: {type(state)}")

        # Add github_token to required keys
        required_keys = ['repo_path', 'branch_name', 'repo_url', 'github_token']
        missing_keys = [key for key in required_keys if not state.get(key)]
        if missing_keys:
            raise ValueError(f"Missing or empty required state keys: {missing_keys}")

        # Validate paths and URLs
        repo_path = state['repo_path']
        if not os.path.exists(repo_path):
            raise ValueError(f"Repository path does not exist: {repo_path}")

        repo_url = state['repo_url']
        if not repo_url.startswith('https://github.com/'):
            raise ValueError(f"Invalid GitHub URL format: {repo_url}")

        # Verify branch before operations
        if not verify_branch(state['repo_path'], state['branch_name']):
            raise ValueError(f"Not on the expected branch: {state['branch_name']}")

        # Log current state
        logger.info(f"Starting git operations with state: {state}")

        # Check git status
        status_result = git_status.invoke({
            "config": {"tool_choice": "git_status"},
            "state": state
        })
        logger.info(f"Git status: {status_result}")

        # Check if there are changes to commit
        if "nothing to commit" in status_result.lower():
            logger.info("No changes to commit")
            return {
                "messages": [AIMessage(content="No changes to push - repository is up to date")],
                "accepted": True
            }

        # Stage all changes
        add_result = git_add.invoke({
            "file_path": ".",
            "config": {"tool_choice": "git_add"},
            "state": state
        })
        logger.info(f"Git add result: {add_result}")

        # Create commit
        requirements = state.get('requirements', 'No requirements specified')
        commit_message = f"Implemented changes for:\n{requirements}"
        commit_result = git_commit.invoke({
            "message": commit_message,
            "config": {"tool_choice": "git_commit"},
            "state": state
        })
        logger.info(f"Git commit result: {commit_result}")

        # Push changes using the tool
        push_result = git_push.invoke({
            "config": {"tool_choice": "git_push"},
            "state": state
        })
        logger.info(f"Git push result: {push_result}")

        if any(error_term in push_result.lower() for error_term in ["error", "failed", "fatal"]):
            raise ValueError(push_result)

        return {
            "messages": [AIMessage(
                content=f"Changes pushed successfully to branch: {state['branch_name']}\n"
                        f"Repository: {state['repo_url']}\n"
                        f"Requirements implemented:\n{requirements}"
            )],
            "accepted": True
        }

    except Exception as e:
        logger.error(f"Error in git_push_changes: {str(e)}", exc_info=True)
        return {
            "messages": [AIMessage(
                content=f"Error during git operations: {str(e)}"
            )],
            "accepted": False
        }