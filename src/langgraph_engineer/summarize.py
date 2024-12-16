from typing import Dict, Any
from langgraph_engineer.model import _get_model
from langgraph_engineer.state import AgentState
from langchain_core.messages import SystemMessage, AIMessage

summarize_prompt = """You are a helpful AI assistant that summarizes technical information clearly and concisely.

Review the provided context and generate a clear, helpful response to the user's original question.

Focus on:
- Directly answering the user's question
- Being concise but thorough
- Using clear, simple language
- Including relevant technical details when necessary

IMPORTANT: DO NOT MAKE ANYTHING UP. ONLY USE THE INFORMATION PROVIDED IN THE CONTEXT.

Context from previous analysis:
{context}

Original question:
{requirements}

Provide your summarized response from the context above:"""

def summarize_response(state: AgentState, config: Dict[str, Any]) -> AgentState:
    """Summarize the aider output and generate a clear response"""
    # Get the model
    model = _get_model(config, "openai", "summarize_model")

    # Get requirements from state following router_agent.py pattern
    requirements = state.get('requirements', '')
    if not requirements and state.get('messages'):
        # If no requirements but we have messages, use the last human message
        for msg in reversed(state['messages']):
            if msg.type == 'human':
                requirements = msg.content
                break

    # Get the aider output from the last step result
    aider_output = ""
    if state.get("step_results"):
        last_result = list(state["step_results"].values())[-1]
        aider_output = last_result.get("output", "")

    # Format the prompt
    formatted_prompt = summarize_prompt.format(
        context=aider_output,
        requirements=requirements  # Use requirements here instead of original_question
    )

    # Create message sequence
    messages = [
        SystemMessage(content=formatted_prompt)
    ]

    # Get response
    response = model.invoke(messages)

    # Update state with summary
    state["summary"] = response.content

    return state