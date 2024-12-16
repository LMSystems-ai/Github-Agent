from typing import Literal, Dict, Any, Annotated
import os
import shutil
import time
import logging
import json
import re

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
from langgraph_engineer.route_message import route_message
from langgraph_engineer.post_critique_router import post_critique_route
from langgraph_engineer.summarize import summarize_response
from langgraph_engineer.aider_node import create_aider_node



anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', '')

# Add at the top of the file, after imports
logger = logging.getLogger(__name__)

def parse_critique_xml(xml_content: str) -> tuple[str, str]:
    """Parse the critique XML response to extract route and instructions."""
    try:
        # Extract node route
        route_match = re.search(r'<node_route>(.*?)</node_route>', xml_content, re.DOTALL)
        route = route_match.group(1).strip() if route_match else "react_agent"

        # Extract instructions
        instructions_match = re.search(r'<instructions>(.*?)</instructions>', xml_content, re.DOTALL)
        instructions = instructions_match.group(1).strip() if instructions_match else ""

        return route, instructions
    except Exception as e:
        logger.error(f"Error parsing critique XML: {str(e)}")
        return "react_agent", ""

def route_critique(state: AgentState) -> Literal["aider_node", "git_push_changes", "react_agent"]:
    """Route after critique based on XML response."""
    try:
        # Get the critique response from state
        critique_response = state.get("step_results", {}).get("critique", {}).get("response", "")

        # Parse the XML response
        route, instructions = parse_critique_xml(critique_response)

        # Store instructions in state for the next node
        state["next_instructions"] = instructions

        # Return the route from the XML
        if route in ["aider_node", "git_push_changes"]:
            return route
        return "react_agent"  # Default fallback

    except Exception as e:
        logger.error(f"Error in route_critique: {str(e)}")
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

def route_aider(state: AgentState) -> Literal["critique", "summarize"]:
    """Route after aider_node to either critique or summarize."""
    # Check if this was a no-changes-required request
    changes_req = state.get("router_analysis", {}).get("changes_req", True)

    if not changes_req:
        return "summarize"
    return "critique"

def route_summarize(state: AgentState) -> Literal[END]:
    """Route to END after summarization."""
    return END

def route_message_node(state: AgentState) -> Literal["react_agent", "aider_node"]:
    """Route based on message router's XML analysis"""
    route_type = state.get("router_analysis", {}).get("route_type", "hard")

    # Simple routing based on the XML agent's decision
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
workflow.add_node("critique", critique)
workflow.add_node("git_push_changes", git_push_changes)
workflow.add_node("route_message", route_message)
# workflow.add_node("post_critique_router", post_critique_route)
workflow.add_node("summarize", summarize_response)

# Set entry point
workflow.set_entry_point("setup_node")

# Update setup routing to only go to router
workflow.add_conditional_edges(
    "setup_node",
    route_setup,
    {
        "route_message": "route_message"
    }
)

# Add router edges
workflow.add_conditional_edges(
    "route_message",
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

# workflow.add_edge("critique", "post_critique_router")

# workflow.add_conditional_edges(
#     "post_critique_router",
#     route_post_critique,
#     {
#         "git_push_changes": "git_push_changes",
#         "react_agent": "react_agent"
#     }
# )

workflow.add_conditional_edges(
    "aider_node",
    route_aider,
    {
        "critique": "critique",
        "summarize": "summarize"
    }
)

workflow.add_conditional_edges(
    "summarize",
    route_summarize,
    {
        END: END
    }
)

workflow.add_conditional_edges(
    "critique",
    route_critique,
    {
        "aider_node": "aider_node",
        "git_push_changes": "git_push_changes",
        "react_agent": "react_agent"
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