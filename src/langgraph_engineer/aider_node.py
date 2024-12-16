import json
import logging
from langchain_core.messages import AIMessage, FunctionMessage
from langgraph_engineer.state import AgentState
from langgraph_engineer.tools import aider_command

logger = logging.getLogger(__name__)

def create_aider_node():
    """Create a node function that properly handles aider command execution"""

    async def aider_node_fn(state: AgentState) -> AgentState:
        """Execute aider command with arguments from the current step"""
        try:
            # Get current step information
            current_step_index = state.get('current_step', 0)
            steps = state.get('steps', [])

            if not steps or current_step_index >= len(steps):
                raise ValueError("No valid step found in state")

            current_step = steps[current_step_index]

            # Extract tool arguments
            message = current_step.tool_args.get('message')
            files = current_step.tool_args.get('files')

            if not message:
                raise ValueError("No message provided for aider command")

            # Execute aider command
            result = await aider_command.ainvoke(
                input={
                    "message": message,
                    "files": files,
                },
                config={
                    "configurable": {
                        "repo_path": state["repo_path"],
                        "anthropic_api_key": state.get("anthropic_api_key"),
                        "aider_state": state.get("aider_state"),
                    }
                }
            )

            if 'messages' not in state:
                state['messages'] = []

            # Add assistant's function call message
            state['messages'].append(
                AIMessage(
                    content="",
                    additional_kwargs={
                        "function_call": {
                            "name": "aider_command",
                            "arguments": json.dumps({"message": message, "files": files}),
                            "tool_call_id": current_step.step_id
                        }
                    }
                )
            )

            # Add function's result message
            state['messages'].append(
                FunctionMessage(
                    name="aider_command",
                    content=result,
                    tool_call_id=current_step.step_id
                )
            )

            # Add AI message with aider's response for summarization
            if state.get("router_analysis", {}).get("changes_req", True) == False:
                state['messages'].append(
                    AIMessage(content=result)
                )

            # Update step results
            if 'step_results' not in state:
                state['step_results'] = {}

            state['step_results'][current_step.step_id] = {
                'step_id': current_step.step_id,
                'output': result,
                'success': True
            }

            return state

        except Exception as e:
            logger.error(f"Error in aider_node: {str(e)}", exc_info=True)
            if 'step_results' not in state:
                state['step_results'] = {}

            if current_step:
                state['step_results'][current_step.step_id] = {
                    'step_id': current_step.step_id,
                    'output': str(e),
                    'success': False
                }
            raise

    return aider_node_fn 