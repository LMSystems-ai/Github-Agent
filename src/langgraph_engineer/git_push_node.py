import logging
import os
import subprocess
from langgraph_engineer.state import AgentState
from langgraph_engineer.tools import git_status, git_add, git_commit, git_push
from langchain_core.messages import AIMessage
from langgraph_engineer.verify_branch import verify_branch
from git import Repo

logger = logging.getLogger(__name__)


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

        # Validate GitHub token
        github_token = state.get('github_token')
        if not github_token:
            raise ValueError("Missing GitHub Personal Access Token")

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

        # Optional dry-run mode for safer operations
        dry_run = state.get('dry_run', False)
        logger.info(f"Git push operation {'(DRY RUN)' if dry_run else ''}")

        # Log current state with sensitive info masked
        masked_state = {k: ('*****' if k == 'github_token' else v) for k, v in state.items()}
        logger.info(f"Starting git operations with state: {masked_state}")

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
        if not dry_run:
            push_result = git_push.invoke({
                "config": {"tool_choice": "git_push"},
                "state": state
            })
            logger.info(f"Git push result: {push_result}")

            if any(error_term in push_result.lower() for error_term in ["error", "failed", "fatal"]):
                raise ValueError(push_result)
        else:
            push_result = "Dry run - no changes pushed"
            logger.warning("Dry run mode: No changes were pushed to the repository")

        return {
            "messages": [AIMessage(
                content=f"{'Dry run: ' if dry_run else ''}Changes {'would be ' if dry_run else ''}pushed successfully to branch: {state['branch_name']}\n"
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
