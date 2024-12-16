from langgraph_engineer.model import _get_model
from langgraph_engineer.state import AgentState
from langchain_core.messages import SystemMessage, AIMessage
import logging
import re

logger = logging.getLogger(__name__)

# Define the routing prompt
router_prompt = """You are a specialized routing agent that determines how to handle user requests in a code engineering system.

There are three possible routes:
1. "chat" - For general questions about the codebase, architecture, or project information
2. "easy" - For simple code modifications like renaming, formatting, or adding basic error handling
3. "hard" - For complex changes requiring careful planning like refactoring, adding new features, or architectural changes

Analyze the user's request and respond in XML format with two tags:
1. <node_route> - either "chat", "easy", or "hard"
2. <reasoning> - explain why you chose this route

User Request:
{requirements}

Example responses:

For a chat query:
<node_route>chat</node_route>
<reasoning>This is a general question about the project's architecture that doesn't require code changes.</reasoning>

For an easy change:
<node_route>easy</node_route>
<reasoning>This is a simple request to rename a variable, which can be done with minimal risk and doesn't require complex planning.</reasoning>

For a complex change:
<node_route>hard</node_route>
<reasoning>This request involves significant architectural changes and requires careful planning to implement safely.</reasoning>
"""

def parse_router_xml(xml_content: str) -> tuple[str, str]:
    """Parse the router XML response to extract route and reasoning."""
    try:
        route_match = re.search(r'<node_route>(.*?)</node_route>', xml_content, re.DOTALL)
        route = route_match.group(1).strip() if route_match else "hard"

        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', xml_content, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        return route, reasoning
    except Exception as e:
        logger.error(f"Error parsing router XML: {str(e)}")
        return "hard", "Error parsing response, defaulting to hard route"

def route_message(state: AgentState, config) -> AgentState:
    """Routes the user's message using an LLM to generate XML-formatted routing decisions"""
    try:
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

        # Get model response
        model = _get_model(config, "anthropic", "router_model")
        message_sequence = [
            SystemMessage(content=formatted_prompt),
            AIMessage(content=f"Please analyze this request: {requirements}")
        ]

        response = model.invoke(message_sequence)

        # Parse the XML response
        route_type, reasoning = parse_router_xml(response.content)

        # Update state with routing decision
        return {
            **state,
            "router_analysis": {
                "route_type": route_type,
                "changes_req": route_type != "chat",
                "reasoning": reasoning
            }
        }

    except Exception as e:
        logger.error(f"Error in route_message: {str(e)}")
        return {
            **state,
            "router_analysis": {
                "route_type": "hard",
                "changes_req": True,
                "reasoning": f"Error occurred during routing: {str(e)}"
            }
        }
