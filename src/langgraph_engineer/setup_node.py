from typing import Literal, Dict
from langchain_core.messages import AIMessage
from langgraph.graph import END
import os
import time
import logging

from langgraph_engineer.state import AgentState, initialize_state
from langgraph_engineer.tools import setup_node, force_clone, force_branch, ForceCloneInput, ForceBranchInput
from git import Repo

logger = logging.getLogger(__name__)

def validate_branch_creation(state: dict) -> bool:
    """Validate that we're on the correct branch."""
    try:
        repo_path = state.get('repo_path')
        repo = Repo(repo_path)
        current_branch = repo.active_branch.name
        expected_branch = state.get('branch_name')

        if current_branch != expected_branch:
            logger.error(f"Branch mismatch: expected {expected_branch}, got {current_branch}")
            return False

        return True
    except Exception as e:
        logger.error(f"Branch validation failed: {str(e)}")
        return False

async def setup_repository(state: AgentState) -> AgentState:
    """Handle repository setup including cloning and branch creation."""
    try:
        # Extract and validate required values from state
        required_keys = ['repo_url', 'github_token', 'repo_path']
        missing_values = [key for key in required_keys if not state.get(key)]

        if missing_values:
            logger.error(f"Missing required state values: {missing_values}")
            logger.error(f"Current state keys: {list(state.keys())}")
            raise ValueError(f"Missing required state values: {', '.join(missing_values)}")

        # Initialize complete state if needed
        if not all(key in state for key in ['requirements', 'code', 'accepted']):
            # Get branch name from state or generate new one
            branch_name = state.get('branch_name')
            if not branch_name:
                from datetime import datetime
                current_date = datetime.now()
                date_str = current_date.strftime('%b-%d-%y').upper()
                branch_name = f"SHIP-{date_str}"
                state['branch_name'] = branch_name

            initialized_state = initialize_state(
                repo_url=state['repo_url'],
                github_token=state['github_token'],
                repo_path=state['repo_path'],
                branch_name=branch_name
            )
            # Preserve existing messages
            initialized_state['messages'] = state.get('messages', [])
            state.update(initialized_state)

        # Add the user's query as requirements if it exists in messages
        if state["messages"]:
            state["requirements"] = state["messages"][-1].content

        # Clone repository
        clone_result = await force_clone.coroutine(
            ForceCloneInput(
                url=state['repo_url'],
                path=state['repo_path'],
                state=state,
                config=None
            )
        )

        if not isinstance(clone_result, dict) or clone_result.get("status") != "success":
            raise ValueError(f"Failed to clone repository: {clone_result}")

        # Create and checkout new branch
        branch_result = await force_branch.coroutine(
            ForceBranchInput(
                branch_name=state['branch_name'],
                state=state,
                config={"callbacks": None}
            )
        )

        if not isinstance(branch_result, dict) or branch_result.get("status") != "success":
            raise ValueError(f"Failed to create branch: {branch_result}")

        # Verify branch creation
        if not validate_branch_creation(state):
            raise ValueError(f"Failed to verify branch creation: {state['branch_name']}")

        # Update state with branch name from result
        if branch_result.get("branch_name"):
            state["branch_name"] = branch_result["branch_name"]
            logger.info(f"Using branch name from force_branch: {state['branch_name']}")

        # No need for tool_state update - we're already updating the state directly
        logger.info(f"Repository setup complete at {state['repo_path']} on branch {state['branch_name']}")

        return state

    except Exception as e:
        logger.error(f"Error in setup_repository: {str(e)}", exc_info=True)
        raise

async def route_setup(state: AgentState) -> Literal["setup_node", "route_message"]:
    """Route to setup node if repository is not initialized, otherwise to route_message."""
    if not state.get('repo_path') or not os.path.exists(state.get('repo_path', '')):
        return "setup_node"
    return "route_message"

async def validate_setup(state: AgentState) -> bool:
    """Validate that repository setup was successful."""
    repo_path = state.get('repo_path', '')
    branch_name = state.get('branch_name', '')

    is_valid = (
        os.path.exists(repo_path) and
        os.path.isdir(os.path.join(repo_path, '.git')) and
        branch_name != ''
    )

    if not is_valid:
        logger.error(f"Setup validation failed: repo_path={repo_path}, branch_name={branch_name}")

    return is_valid