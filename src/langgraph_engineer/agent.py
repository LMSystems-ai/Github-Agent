from typing import Literal, Dict, Any, Annotated
import os
import shutil
import time
import logging
import json
from git import Repo
import difflib

from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import AIMessage, HumanMessage, FunctionMessage
from langgraph_engineer.tools import (
    AiderShellTool,
    AiderToReactOutput,
    aider_command
)
# Update imports
from langgraph.prebuilt import ToolNode
from langgraph_engineer.check import check
from langgraph_engineer.state import (
    AgentState,
    OutputState,
    GraphConfig,
    InputState,
    initialize_state,
    AiderState
)

from langgraph_engineer.setup_node import setup_repository
from langgraph_engineer.git_push_node import git_push_changes
from langchain_core.tools import BaseTool
from langgraph.types import interrupt, Command

from langgraph_engineer.aider_node import create_aider_node
from langgraph_engineer.diff_node import show_file_diffs
from langgraph_engineer.summarize_node import summarize_response

anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', '')

logger = logging.getLogger(__name__)

def route_critique(state: AgentState) -> Literal["react_agent", "git_push_changes", END]:
    """Route after critique based on validation results."""
    if state.get("step_results"):
        latest_result = list(state["step_results"].values())[-1]
        completion_status = latest_result.get("args", {}).get("completion_status")
        if completion_status in ["complete_with_changes", "complete_no_changes"]:
            state["execution_status"] = "complete"

    if not state.get("accepted"):
        state["execution_status"] = "planning"
        return "react_agent"
    elif state["execution_status"] == "complete":
        return "git_push_changes"
    else:
        return "react_agent"

def route_git_push(state: AgentState) -> Literal[END]:
    return END

def route_start(state: AgentState) -> Literal["react_agent", "gather_requirements"]:
    if state.get('requirements'):
        return "react_agent"
    else:
        return "gather_requirements"

def aider_config_mapper(state: AgentState) -> Dict[str, Any]:
    # Handle both API keys
    anthropic_api_key = state.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY")
    openai_api_key = state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")

    if not (anthropic_api_key or openai_api_key):
        raise ValueError("Either ANTHROPIC_API_KEY or OPENAI_API_KEY must be set")
    logger.info(f"Using Anthropic API key: {anthropic_api_key[:50]}...")
    logger.info(f"Using OpenAI API key: {openai_api_key[:50]}...")
    return {
        "configurable": {
            "repo_path": state.get("repo_path"),
            "anthropic_api_key": anthropic_api_key,
            "openai_api_key": openai_api_key,  # Add OpenAI key
            "aider_state": state.get("aider_state"),
            "chat_mode": state.get("chat_mode", ""),
            "model_name": state.get("model_name", "4o"),
        }
    }

def route_gather(state: AgentState) -> Literal["react_agent", END]:
    if state["messages"] and isinstance(state["messages"][-1], AIMessage):
        return END
    return "react_agent"

def route_react(state: AgentState) -> Literal["aider_node", "git_push_changes"]:
    if state["execution_status"] in ["planning", "executing"]:
        return "aider_node"
    elif state["execution_status"] == "complete" and state.get("accepted"):
        return "git_push_changes"
    return "aider_node"

def route_tool(state: AgentState) -> Literal["react_agent"]:
    return "react_agent"

def route_aider(state: AgentState) -> Literal["git_push_changes", "summarize"]:
    changes_req = state.get("router_analysis", {}).get("changes_req", True)
    if not changes_req:
        return "summarize"
    return "git_push_changes"

def route_summarize(state: AgentState) -> Literal[END]:
    return END

def route_message_node(state: AgentState) -> Literal["react_agent", "aider_node"]:
    route_type = state.get("router_analysis", {}).get("route_type", "hard")
    if route_type in ["chat", "easy"]:
        return "aider_node"
    else:
        return "react_agent"

def route_post_critique(state: AgentState) -> Literal["git_push_changes", "react_agent"]:
    try:
        post_critique_result = state.get('step_results', {}).get('post_critique_router', {})
        decision = post_critique_result.get('args', {}).get('decision', 'N')
        return "git_push_changes" if decision == 'Y' else "react_agent"
    except Exception as e:
        logger.error(f"Error in route_post_critique: {str(e)}")
        return "react_agent"

def route_human(state: AgentState) -> Literal["aider_node", "git_push_changes", "diff_changes"]:
    """Route after human interaction based on state flags."""
    if state.get("accepted"):
        return "git_push_changes"
    elif state.get("show_diff", False):
        state["show_diff"] = False
        return "diff_changes"
    return "aider_node"

def route_diff(state: AgentState) -> Literal["human_interaction"]:
    """Route after showing diffs - always return to human interaction."""
    return "human_interaction"

def human_interaction(state: AgentState) -> AgentState:
    """Node for handling human interaction with the agent."""
    try:
        if state.get('messages'):
            human_input = interrupt("Please provide your message (or enter {'accept': true} to finish, or {'show_diff': true} to see changes):")

            logger.debug(f"Received human input: {human_input}")

            # Check if input is a dictionary
            if isinstance(human_input, dict):
                if human_input.get("accept"):
                    state["accepted"] = True
                    return state
                elif human_input.get("show_diff"):
                    state["show_diff"] = True
                    return state

            # Add human message to state
            state.setdefault("messages", []).append(HumanMessage(content=human_input))

        return state

    except Exception as e:
        logger.error(f"Error in human_interaction: {str(e)}")
        raise

class Engineer:
    def __init__(self):
        self.base_repos_dir = "/repos"
        os.makedirs(self.base_repos_dir, exist_ok=True)
        self.test_repo_url = "https://github.com/RVCA212/portfolio-starter-kit"
        self.test_user_id = "test_user_123"

    async def process_request(self, input_state: InputState) -> dict:
        repo_url = input_state.get('repo_url', self.test_repo_url)
        user_id = input_state.get('user_id', self.test_user_id)
        query = input_state.get('query', '')
        github_token = input_state.get('github_token', '')
        anthropic_api_key = input_state.get('anthropic_api_key', '')
        openai_api_key = input_state.get('openai_api_key', '')  # Add OpenAI key
        chat_mode = input_state.get('chat_mode', '')
        model_name = input_state.get('model_name', '4o')  # Default to OpenAI model
        logger.info(f"Received chat_mode in input_state: '{chat_mode}'")
        logger.info(f"Using model: {model_name}")

        if not github_token:
            raise ValueError("GitHub token is required in input_state")
        if not repo_url:
            raise ValueError("Repository URL is required")

        # Validate API keys based on model
        if model_name in ['haiku', 'sonnet'] and not anthropic_api_key:
            raise ValueError(f"Anthropic API key is required for model {model_name}")
        elif model_name in ['4o', 'o1', 'gpt-4o-mini'] and not openai_api_key:
            raise ValueError(f"OpenAI API key is required for model {model_name}")

        repo_path = os.path.join("/repos", user_id.lstrip('/'))
        os.makedirs(repo_path, exist_ok=True)

        initial_state = initialize_state(
            repo_url=repo_url,
            github_token=github_token,
            repo_path=repo_path,
            anthropic_api_key=anthropic_api_key,
            openai_api_key=openai_api_key,
            chat_mode=chat_mode,
            model_name=model_name,
            query=query
        )

        # Set up memory checkpointer for interrupts
        memory = MemorySaver()
        graph_with_memory = workflow.compile(checkpointer=memory)

        # Run the workflow
        final_state = await run_workflow(graph_with_memory, initial_state)
        return final_state


# Register nodes
aider_node = create_aider_node()
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("setup_node", setup_repository)
workflow.add_node("human_interaction", human_interaction)
workflow.add_node("aider_node", aider_node)
workflow.add_node("git_push_changes", git_push_changes)
workflow.add_node("diff_changes", show_file_diffs)

# Set entry point and edges with routing
workflow.set_entry_point("setup_node")
workflow.add_edge("setup_node", "aider_node")
workflow.add_edge("aider_node", "human_interaction")
workflow.add_conditional_edges(
    "human_interaction",
    route_human
)
workflow.add_edge("diff_changes", "human_interaction")
workflow.add_edge("git_push_changes", END)

graph = workflow.compile()

async def run_workflow(graph, initial_state):
    """Run workflow with properly configured state."""
    try:
        logger.debug("=== Workflow Starting ===")
        logger.debug(f"Initial state chat_mode: '{initial_state.get('chat_mode')}'")
        logger.debug(f"Initial state model_name: '{initial_state.get('model_name')}'")
        logger.debug(f"Initial state keys: {list(initial_state.keys())}")
        logger.debug(f"Initial state repo_path: {initial_state.get('repo_path')}")
        logger.debug(f"Initial state anthropic_api_key present: {'yes' if initial_state.get('anthropic_api_key') else 'no'}")

        # Validate required state values
        required_keys = ["repo_path", "anthropic_api_key", "github_token"]
        missing_keys = [key for key in required_keys if not initial_state.get(key)]

        logger.debug("=== State Validation ===")
        logger.debug(f"Required keys: {required_keys}")
        logger.debug(f"Missing keys: {missing_keys}")

        if missing_keys:
            logger.error(f"Missing required state values: {', '.join(missing_keys)}")
            raise ValueError(f"Missing required state values: {', '.join(missing_keys)}")

        # Update config to include OpenAI API key
        config = {
            "configurable": {
                "repo_path": initial_state["repo_path"],
                "anthropic_api_key": initial_state["anthropic_api_key"],
                "openai_api_key": initial_state.get("openai_api_key"),  # Add OpenAI key
                "aider_state": initial_state.get("aider_state"),
                "github_token": initial_state["github_token"],
                "chat_mode": initial_state.get("chat_mode", ""),
                "model_name": initial_state.get("model_name", "4o"),
            }
        }
        logger.debug("=== Config Creation ===")
        logger.debug(f"Created config: {json.dumps(config, default=str)}")

        # Run graph
        logger.debug("=== Running Graph ===")
        state = await graph.arun(
            inputs=initial_state,
            config=config
        )

        logger.debug("=== Workflow Completed ===")
        return state

    except Exception as e:
        logger.error(f"Error running workflow: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        raise