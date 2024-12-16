from typing import Dict, Any
from langgraph_engineer.model import _get_model
from langgraph_engineer.state import AgentState, PlanStep
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel
import logging

logger = logging.getLogger(__name__)

class RouterResponse(BaseModel):
    """Schema for router response"""
    reasoning: str
    route_type: str  # Must be either 'easy' or 'hard'
    changes_req: bool  # Must be either true or false

router_prompt = """You are a specialized routing agent that analyzes user requirements to determine if they need an easy or hard solution.

RULES:
1. Analyze the user's request and requirements carefully
2. Output 'easy' if the request:
    - is straightforward to implement
    - has clear requirements
    - can be done in a few lines of code
    - doesn't require complex logic or planning

3. Output 'hard' if the request:
   - Requires complex implementation
   - Needs careful planning and architecture
   - Involves multiple files or complex logic
   - Requires deep analysis or debugging
   - Needs significant research or understanding

4. Set changes_req to 'true' if:
   - The request involves modifying, creating, or deleting files
   - Code changes are needed
   - Any content needs to be written or updated

5. Set changes_req to 'false' if:
   - The request is just asking for information or understanding
   - No modifications to files are needed
   - The query is about explaining code or repository structure

IMPORTANT: use your reasoning and understanding of what easy and hard queries are for making your decision.

Example tasks that are easy (with changes_req):
- "can you make a simple game of pong with pygame?" (changes_req: true)
- "What is this repo about?" (changes_req: false)
- "Change the ReadME to be more descriptive" (changes_req: true)
- "Add a print statement to debug this function" (changes_req: true)
- "Can you explain how the authentication system works?" (changes_req: false)

Example tasks that are hard (with changes_req):
- "Can you help me implement a new authentication system using OAuth2?" (changes_req: true)
- "There's a memory leak in production, can you help investigate and fix it?" (changes_req: true)
- "Can you explain the architecture of this microservices system?" (changes_req: false)
- "We need to refactor our monolithic API into microservices" (changes_req: true)
- "Help me understand why the CI/CD pipeline is failing" (changes_req: false)

Analyze the following request and respond with your routing decision:
{requirements}"""

def route_request(state: AgentState, config: Dict[str, Any]) -> AgentState:
    """Determine if the request needs easy or hard handling"""
    try:
        # Get the model with structured output
        model = _get_model(config, "openai", "router_model").with_structured_output(RouterResponse)

        # Get requirements from state
        requirements = state.get('requirements', '')
        if not requirements and state.get('messages'):
            # If no requirements but we have messages, use the last human message
            for msg in reversed(state['messages']):
                if msg.type == 'human':
                    requirements = msg.content
                    break

        # Format the prompt
        formatted_prompt = router_prompt.format(requirements=requirements)

        # Create message sequence
        messages = [
            SystemMessage(content=formatted_prompt),
            AIMessage(content="Please analyze the request and determine the routing.")
        ]

        # Get response
        response = model.invoke(messages)

        # Update state with routing decision and prepare for easy if needed
        new_state = {
            **state,
            "router_analysis": {
                "reasoning": response.reasoning,
                "route_type": response.route_type,
                "changes_req": response.changes_req
            }
        }

        # If this is a easy route, prepare the step information
        if response.route_type == "easy":
            new_state["steps"] = [
                PlanStep(
                    reasoning="Direct easy execution",
                    step_id="S1",
                    tool_name="aider_shell",
                    tool_args={
                        "message": requirements,  # Use original requirements as message
                        "files": "."  # Default to all files for easy
                    }
                )
            ]
            new_state["current_step"] = 0
            new_state["execution_status"] = "executing"

        return new_state

    except Exception as e:
        logger.error(f"Error in route_request: {str(e)}")
        # Default to easy on error for safety
        return {
            **state,
            "router_analysis": {
                "reasoning": f"Error occurred: {str(e)}",
                "route_type": "easy"
            }
        }