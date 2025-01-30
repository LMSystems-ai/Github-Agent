from typing import Dict, Any
from langgraph_engineer.model import _get_model
from langgraph_engineer.state import AgentState
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel
import logging

logger = logging.getLogger(__name__)

class PostCritiqueResponse(BaseModel):
    """Schema for post-critique router response"""
    reasoning: str
    decision: str  # Must be either 'Y' or 'N'

post_critique_prompt = """You are a specialized validation agent that analyzes critique results to determine if code changes are complete and acceptable.

RULES:
1. Analyze the critique results carefully
2. Output 'Y' if:
   - All requested changes were successfully implemented
   - Requirements were fully met
   - Implementation is correct and of good quality
   - Changes stayed within the required scope
   - No unauthorized or unexpected changes were made

3. Output 'N' if:
   - Changes are missing or incomplete
   - Requirements were not fully met
   - Implementation has quality issues
   - Changes exceeded scope or contain unauthorized modifications
   - Additional work or revisions are needed

EXAMPLES:

Critique that should return Y:
"The changes were successfully implemented. All requirements were met, and the code quality is good. The implementation follows the specifications exactly."

Critique that should return N:
"Some required changes are missing. The implementation is incomplete and needs additional work. There are quality issues that need to be addressed."

Analyze the following critique results and determine if the changes are complete and acceptable:
{critique_logic}"""

def post_critique_route(state: AgentState, config: Dict[str, Any]) -> AgentState:
    """Determine if the critique results warrant proceeding or require more work"""
    try:
        # Get the model with structured output
        model = _get_model(config, "openai", "post_critique_model").with_structured_output(PostCritiqueResponse)

        # Get critique logic from state
        critique_results = state.get('step_results', {}).get('critique', {})
        critique_logic = critique_results.get('args', {}).get('logic', '')

        # Format the prompt
        formatted_prompt = post_critique_prompt.format(critique_logic=critique_logic)

        # Create message sequence
        messages = [
            SystemMessage(content=formatted_prompt),
            AIMessage(content="Please analyze the critique results and determine if changes are complete.")
        ]

        # Get response
        response = model.invoke(messages)

        # Update state with the routing decision
        state['step_results']['post_critique_router'] = {
            'args': {
                'decision': response.decision,
                'reasoning': response.reasoning
            }
        }

        return state

    except Exception as e:
        logger.error(f"Error in post_critique_route: {str(e)}")
        # Default to N on error for safety
        state['step_results']['post_critique_router'] = {
            'args': {
                'decision': 'N',
                'reasoning': f'Error occurred: {str(e)}'
            }
        }
        return state