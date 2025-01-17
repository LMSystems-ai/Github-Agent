from typing import Dict, Any
from langgraph_engineer.model import _get_model
from langgraph_engineer.state import AgentState
from langchain_core.messages import SystemMessage, AIMessage
import logging
import os
logger = logging.getLogger(__name__)

summarize_prompt = """Please Communicate the latest message from the coding AI agent (as if you were the coding AI Agent) to the user given their prompt. Do not make anything up that was not presented in the coding AI Agent response.

User's message:
{user_message}

Coding AI Agent response:
{last_response}

Provide a concise summary to the user on behalf of the coding AI Agent:"""

def summarize_response(state: AgentState) -> AgentState:
    """Summarize the aider output and store it in state"""
    try:
        logger.debug("=== Summarize Node Starting ===")

        # Get the last AI message
        messages = state.get('messages', [])
        if not messages:
            logger.debug("No messages found in state")
            return state

        last_ai_message = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_ai_message = msg
                break

        if not last_ai_message:
            logger.debug("No AI message found to summarize")
            return state

        # Create proper config structure
        config = {
            "configurable": {
                "model": "openai",  # Specify model type
                "openai_api_key": os.environ.get("OPENAI_API_KEY"),
                "model_name": state.get("model_name", "4o")
            }
        }

        # Get the model with proper config
        model = _get_model(config, "anthropic", "summarize_model")

        # Get the last user message
        last_user_message = "No user message found"
        for msg in reversed(messages):
            if msg.type == "human":
                last_user_message = msg.content
                break

        # Format the prompt
        formatted_prompt = summarize_prompt.format(
            user_message=last_user_message,
            last_response=last_ai_message.content
        )

        # Create message sequence
        messages = [SystemMessage(content=formatted_prompt)]

        # Get response
        response = model.invoke(messages)

        # Store summary in state
        state["aider_summary"] = response.content

        logger.debug("=== Summarize Node Completed ===")
        return state

    except Exception as e:
        logger.error(f"Error in summarize_node: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        raise