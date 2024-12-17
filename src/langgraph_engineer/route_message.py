from semantic_router import Route, RouteLayer
from semantic_router.encoders import OpenAIEncoder
from typing import Dict, Any
from langgraph_engineer.state import AgentState, PlanStep
import logging

logger = logging.getLogger(__name__)

# Define routes for different types of queries
chat_route = Route(
    name="chat",
    utterances=[
        "give me an overview of this repository",
        "describe the project structure",
        "list the main contributors",
        "what technologies are used here",
        "show me the project dependencies",
        "explain the system architecture",
        "what's the development status",
        "describe the main components",
        "what are the key features",
        "show documentation for this project",
    ],
)

easy_route = Route(
    name="easy",
    utterances=[
        "change variable name from X to Y",
        "add type hints to this function",
        "fix the indentation in this file",
        "add a docstring to explain this code",
        "remove this unused import statement",
        "add error handling here",
        "update this log message",
        "fix this spelling mistake",
        "add these missing parameters",
        "update this function's return type",
        "move this code block to a new function",
    ],
)

hard_route = Route(
    name="hard",
    utterances=[
        "create a new feature that...",
        "refactor this module to use...",
        "implement caching for...",
        "add authentication to...",
        "optimize the performance of...",
        "create unit tests for...",
        "integrate this new library...",
        "implement a new API endpoint...",
        "fix these security issues...",
        "redesign this component to...",
        "add support for async operations",
    ],
)

# Initialize the route layer
encoder = OpenAIEncoder()
route_layer = RouteLayer(encoder=encoder, routes=[chat_route, easy_route, hard_route])

def route_message(state: AgentState) -> AgentState:
    """Routes the user's message to either chat, easy, or hard paths"""
    try:
        # Get requirements from state
        requirements = state.get('requirements', '')
        if not requirements and state.get('messages'):
            # If no requirements but we have messages, use the last human message
            for msg in reversed(state['messages']):
                if msg.type == 'human':
                    requirements = msg.content
                    break

        # Use semantic router to determine the route
        route_choice = route_layer(requirements)
        route_type = route_choice.name if route_choice else "hard"  # Default to hard if no match

        # Update state with routing decision
        new_state = {
            **state,
            "router_analysis": {
                "route_type": route_type,
                "changes_req": route_type != "chat"
            }
        }

        # Prepare step information for both chat and easy routes
        if route_type in ["chat", "easy"]:
            new_state["steps"] = [
                PlanStep(
                    reasoning="Direct execution of request",
                    step_id="S1",
                    tool_name="aider_shell",
                    tool_args={
                        "message": requirements,
                        "files": "."
                    }
                )
            ]
            new_state["current_step"] = 0
            new_state["execution_status"] = "executing"

        return new_state

    except Exception as e:
        logger.error(f"Error in route_message: {str(e)}")
        # Default to hard on error for safety
        return {
            **state,
            "router_analysis": {
                "route_type": "hard",
                "changes_req": True
            }
        }
