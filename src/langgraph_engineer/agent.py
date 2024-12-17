from typing import Literal, Dict, Any, Annotated
import os
import shutil
import time
import logging
import json

from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import AIMessage, HumanMessage, FunctionMessage
from langgraph_engineer.tools import (
    AiderShellTool,
    AiderToReactOutput,
    aider_command
)
from langgraph.prebuilt import ToolNode
from langgraph_engineer.check import check
from langgraph_engineer.critique import critique, get_git_diff
from langgraph_engineer.react_agent import react_agent
from langgraph_engineer.state import AgentState, OutputState, GraphConfig, InputState, initialize_state, AiderState
from langgraph_engineer.setup_node import setup_repository, route_setup, validate_setup
from langgraph_engineer.git_push_node import git_push_changes
from langchain_core.tools import BaseTool
from langgraph_engineer.post_critique_router import post_critique_route
from langgraph_engineer.summarize import summarize_response
from langgraph_engineer.aider_node import create_aider_node
from langgraph_engineer.router_agent import route_request



anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', '')

# Add at the top of the file, after imports
logger = logging.getLogger(__name__)

def route_critique(state: AgentState) -> Literal["react_agent", "git_push_changes", END]:
    """Route after critique based on validation results."""
    # Update execution_status based on critique output
    if state.get("step_results"):
        latest_result = list(state["step_results"].values())[-1]
        completion_status = latest_result.get("args", {}).get("completion_status")
        if completion_status in ["complete_with_changes", "complete_no_changes"]:
            state["execution_status"] = "complete"

    if not state["accepted"]:
        # If changes weren't accepted, go back to planning
        state["execution_status"] = "planning"
        return "react_agent"
    elif state["execution_status"] == "complete":
        # If we're done and accepted, proceed to git push
        return "git_push_changes"
    else:
        # If we're still executing and accepted current step, back to react
        return "react_agent"

def route_git_push(state: AgentState) -> Literal[END]:
    """Route to END after git push is complete."""
    return END

def route_start(state: AgentState) -> Literal["react_agent", "gather_requirements"]:
    if state.get('requirements'):
        return "react_agent"
    else:
        return "gather_requirements"


def aider_config_mapper(state: AgentState) -> Dict[str, Any]:
    # Get API key from environment if not in state
    anthropic_api_key = state.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

    return {
        "configurable": {
            "repo_path": state.get("repo_path"),
            "anthropic_api_key": anthropic_api_key,
            "aider_state": state.get("aider_state"),
        }
    }


def route_gather(state: AgentState) -> Literal["react_agent", END]:
    """Route from gather_requirements to either:
    - react_agent if we have requirements
    - END if waiting for user response
    """
    # Check if last message is from the LLM (question to user)
    if state["messages"] and isinstance(state["messages"][-1], AIMessage):
        return END
    return "react_agent"

def route_react(state: AgentState) -> Literal["aider_node", "git_push_changes"]:
    """Route after react_agent to either execute steps or finish."""
    # If we're in planning/executing mode, always go to aider_node
    if state["execution_status"] in ["planning", "executing"]:
        return "aider_node"
    # If we're complete and accepted, go to git_push
    elif state["execution_status"] == "complete" and state["accepted"]:
        return "git_push_changes"
    # Shouldn't reach here, but default to aider_node
    return "aider_node"

def route_tool(state: AgentState) -> Literal["react_agent"]:
    """Route back to react_agent after tool execution."""
    return "react_agent"

def route_aider(state: AgentState) -> Literal["git_push_changes", "summarize"]:
    """Route after aider_node to either git_push_changes or summarize."""
    # Check if this was a no-changes-required request
    changes_req = state.get("router_analysis", {}).get("changes_req", True)

    if not changes_req:
        return "summarize"
    return "git_push_changes"

def route_summarize(state: AgentState) -> Literal[END]:
    """Route to END after summarization."""
    return END

def route_message_node(state: AgentState) -> Literal["react_agent", "aider_node"]:
    """Route based on message router's analysis"""
    route_type = state.get("router_analysis", {}).get("route_type", "hard")

    if route_type == "chat":
        return "aider_node"
    elif route_type == "easy":
        return "aider_node"
    else:  # hard
        return "react_agent"

def route_post_critique(state: AgentState) -> Literal["git_push_changes", "react_agent"]:
    """Route based on post-critique router's Y/N decision"""
    try:
        post_critique_result = state.get('step_results', {}).get('post_critique_router', {})
        decision = post_critique_result.get('args', {}).get('decision', 'N')  # Default to N for safety
        return "git_push_changes" if decision == 'Y' else "react_agent"
    except Exception as e:
        logger.error(f"Error in route_post_critique: {str(e)}")
        return "react_agent"

class Engineer:
    def __init__(self):
        self.base_repos_dir = "/repos"
        os.makedirs(self.base_repos_dir, exist_ok=True)
        self.test_repo_url = "https://github.com/RVCA212/portfolio-starter-kit"
        self.test_user_id = "test_user_123"

    async def process_request(self, input_state: InputState) -> dict:
        """Process a user request with repository setup"""
        # Extract values from input_state
        repo_url = input_state.get('repo_url', self.test_repo_url)
        user_id = input_state.get('user_id', self.test_user_id)
        query = input_state.get('query', '')
        github_token = input_state.get('github_token', '')
        anthropic_api_key = input_state.get('anthropic_api_key', '')

        # Validate required values with clear error messages
        if not github_token:
            raise ValueError("GitHub token is required in input_state")
        if not repo_url:
            raise ValueError("Repository URL is required")
        if not anthropic_api_key:
            raise ValueError("Anthropic API key is required in input_state")

        # Ensure proper path construction
        repo_path = os.path.join("/repos", user_id.lstrip('/'))

        # Create directory if it doesn't exist
        os.makedirs(repo_path, exist_ok=True)

        # Use the initialize_state function from state.py
        initial_state = initialize_state(
            repo_url=repo_url,
            github_token=github_token,
            repo_path=repo_path,
            anthropic_api_key=anthropic_api_key
        )

        # Only proceed if there's an actual query
        if not query:
            return initial_state

        # Add query as HumanMessage instead of AIMessage
        initial_state["messages"].append(HumanMessage(content=query))

        logger.info(f"Initial state keys: {list(initial_state.keys())}")

        # Run workflow with config
        final_state = await run_workflow(graph, initial_state)
        return final_state


# Create the aider node
aider_node = create_aider_node()

# Define a new graph
workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("setup_node", setup_repository)
workflow.add_node("react_agent", react_agent)
workflow.add_node("aider_node", aider_node)
# workflow.add_node("critique", critique)  # Commenting out critique node
workflow.add_node("git_push_changes", git_push_changes)
workflow.add_node("router_agent", route_request)
# workflow.add_node("post_critique_router", post_critique_route)  # Commenting out post critique router
workflow.add_node("summarize", summarize_response)

# Set entry point
workflow.set_entry_point("setup_node")

# Update setup routing to only go to router
workflow.add_conditional_edges(
    "setup_node",
    route_setup,
    {
        "router_agent": "router_agent"
    }
)

# Add router edges
workflow.add_conditional_edges(
    "router_agent",
    route_message_node,
    {
        "react_agent": "react_agent",
        "aider_node": "aider_node"
    }
)

# Keep existing edges for other nodes
workflow.add_edge("react_agent", "aider_node")
# workflow.add_edge("aider_node", "critique")

# workflow.add_conditional_edges(
#     "critique",
#     route_critique,
#     {
#         "react_agent": "react_agent",
#         "git_push_changes": "git_push_changes",
#         END: END
#     }
# )

workflow.add_conditional_edges(
    "git_push_changes",
    route_git_push,
    {
        END: END
    }
)

workflow.add_conditional_edges(
    "aider_node",
    route_aider,
    {
        "summarize": "summarize",
        "git_push_changes": "git_push_changes"
    }
)

workflow.add_conditional_edges(
    "summarize",
    route_summarize,
    {
        END: END
    }
)

graph = workflow.compile()

async def run_workflow(graph, initial_state):
    state = initial_state
    config = {
        "configurable": {
            "repo_path": state["repo_path"],
            "anthropic_api_key": state["anthropic_api_key"],
            "aider_state": state.get("aider_state"),
        }
    }
    state = await graph.arun(inputs=state, config=config)
    return state

# Before invoking the aider_node