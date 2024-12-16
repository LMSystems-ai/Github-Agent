from typing import Dict, Any
from semantic_router import Route, RouteLayer
from semantic_router.encoders import OpenAIEncoder
from langgraph_engineer.state import AgentState
from langchain_core.pydantic_v1 import BaseModel
import logging

logger = logging.getLogger(__name__)

class PostCritiqueResponse(BaseModel):
    """Schema for post-critique router response"""
    decision: str  # Must be either 'Y' or 'N'

# Define routes for different critique outcomes
complete_route = Route(
    name="Y",
    utterances=[
        # Changes Found patterns
        "changes were successfully implemented",
        "added the requested changes",
        "modified the specified files",
        "implemented the required modifications",

        # Requirements Addressed patterns
        "requirements were fully met",
        "requirement to add was fully met",
        "all requested changes were implemented",
        "specifications were completely fulfilled",

        # Quality Assessment patterns
        "implementation was correct",
        "changes were properly implemented",
        "implementation meets quality standards",
        "code changes are appropriate",

        # Scope Review patterns
        "no unauthorized changes were made",
        "changes stayed within scope",
        "only the requested modifications were made",
        "implementation follows the requirements exactly"
    ],
)

incomplete_route = Route(
    name="N",
    utterances=[
        # Changes Found patterns
        "changes are missing",
        "failed to implement",
        "changes not found",
        "incomplete implementation",

        # Requirements Addressed patterns
        "requirements were not fully met",
        "missing required changes",
        "specifications were not fulfilled",
        "partial implementation of requirements",

        # Quality Assessment patterns
        "implementation is incorrect",
        "changes need improvement",
        "quality issues found",
        "implementation needs revision",

        # Scope Review patterns
        "unauthorized changes detected",
        "scope exceeded",
        "unrelated modifications found",
        "unexpected changes present"
    ],
)

# Initialize the route layer
encoder = OpenAIEncoder()
route_layer = RouteLayer(
    encoder=encoder,
    routes=[complete_route, incomplete_route]
)

def post_critique_route(state: AgentState, config: Dict[str, Any]) -> AgentState:
    """Determine if the critique results warrant proceeding or require more work"""
    try:
        # Get critique logic from state
        critique_results = state.get('step_results', {}).get('critique', {})
        critique_logic = critique_results.get('args', {}).get('logic', '')

        # Use semantic router to determine the route
        route_choice = route_layer(critique_logic)
        decision = route_choice.name if route_choice else "N"  # Default to N if no match

        # Update state with the routing decision
        state['step_results']['post_critique_router'] = {
            'args': {
                'decision': decision
            }
        }

        return state

    except Exception as e:
        logger.error(f"Error in post_critique_route: {str(e)}")
        # Default to N on error for safety
        state['step_results']['post_critique_router'] = {
            'args': {
                'decision': 'N'
            }
        }
        return state