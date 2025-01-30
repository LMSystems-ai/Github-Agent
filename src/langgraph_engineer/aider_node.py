import json
import logging
from langchain_core.messages import AIMessage, HumanMessage, AIMessageChunk, BaseMessage
from langgraph_engineer.state import AgentState
from langgraph_engineer.interactive_aider import InteractiveAider
from langgraph.graph.message import add_messages
from datetime import datetime, timezone
from pathlib import Path
import os
logger = logging.getLogger(__name__)

def create_aider_node():
    """
    Create a node function that handles interactive Aider execution, streaming the
    partial response back in real time.
    """

    async def aider_node_fn(state: AgentState) -> AgentState:
        """
        Node function that processes messages using InteractiveAider.
        Streams chunked output as AIMessageChunks and a final AIMessage.
        """
        try:
            logger.debug("=== Interactive Aider Node Starting ===")

            # Ensure state is properly initialized
            if not isinstance(state, dict):
                state = dict(state)

            # Initialize messages if not present
            if "messages" not in state:
                state["messages"] = []

            # Get last message
            messages = state.get("messages", [])
            if not messages:
                raise ValueError("No messages in state")

            last_message = messages[-1]
            if not isinstance(last_message, (HumanMessage, AIMessage)):
                # Convert to proper message type if needed
                last_message = HumanMessage(
                    content=str(last_message),
                    additional_kwargs={"role": "user"}
                )

            user_text = last_message.content

            # Initialize aider
            repo_path = state.get("repo_path")
            if not repo_path:
                raise ValueError("Repository path not found in state")

            # Determine which API key to use based on model
            model_name = state.get("model_name", "4o")
            anthropic_models = ["sonnet", "haiku"]
            if model_name in anthropic_models:
                # Prioritize state API key over environment variable
                api_key = state.get("anthropic_api_key")
                if api_key is None:  # Only use env var if state key is None
                    api_key = os.getenv("ANTHROPIC_API_KEY")
                source = "state" if state.get("anthropic_api_key") is not None else "environment"
                logger.info(f"Using Anthropic API key from: {source}")
            else:  # Default to OpenAI
                # Prioritize state API key over environment variable
                api_key = state.get("openai_api_key")
                if api_key is None:  # Only use env var if state key is None
                    api_key = os.getenv("OPENAI_API_KEY")
                source = "state" if state.get("openai_api_key") is not None else "environment"
                logger.info(f"Using OpenAI API key from: {source}")
                logger.info(f"OpenAI key starts with: {api_key[:10]}..." if api_key else "No OpenAI key found!")

            if not api_key:
                raise ValueError(f"No API key found for model {model_name}. Please provide a valid API key.")

            logger.info(f"Using API key for model {model_name}: {api_key[:50]}...")

            aider = InteractiveAider(
                repo_path=repo_path,
                model=model_name,
                api_key=api_key,
                api_base=state.get("api_base")
            )

            # Initialize a content accumulator
            accumulated_content = ""

            # Track edits for this session
            if "edits" not in state:
                state["edits"] = []

            try:
                # Process chunks
                async for chunk in aider.execute_command(user_text):
                    if chunk["type"] == "edit":
                        # Store edit information in state
                        edit = {
                            "type": "edit",
                            "commit_hash": chunk.get("commit_hash"),
                            "commit_message": chunk.get("commit_message"),
                            "files": chunk.get("files", []),
                            "diff": chunk.get("content")
                        }
                        state["edits"].append(edit)

                        # Create a message for the diff
                        diff_message = AIMessage(
                            content=f"Made changes to: {', '.join(edit['files'])}\n\n```diff\n{edit['diff']}\n```",
                            additional_kwargs={
                                "role": "assistant",
                                "type": "edit",
                                "edit_info": edit
                            }
                        )
                        state["messages"].append(diff_message)
                    else:
                        # Handle regular message chunks as before
                        accumulated_content += chunk.get("content", "")

                # After processing all chunks, create a single AIMessage
                final_message = AIMessage(
                    content=accumulated_content,
                    additional_kwargs={"role": "assistant", "type": "message"}
                )

                # Update state with the final AIMessage
                state["messages"].append(final_message)

            except Exception as e:
                error_msg = str(e)
                # Check for authentication-specific errors
                if any(auth_err in error_msg.lower() for auth_err in [
                    "authentication_error",
                    "invalid x-api-key",
                    "invalid api key",
                    "unable to authenticate",
                    "auth"
                ]):
                    auth_error_msg = (
                        f"API Authentication Error: The provided API key for {state.get('model_name', 'unknown')} "
                        f"model is invalid or has expired. Please check your API key and try again.\n\n"
                        f"Error details: {error_msg}"
                    )
                    error_message = AIMessage(
                        content=auth_error_msg,
                        additional_kwargs={
                            "role": "assistant",
                            "type": "error",
                            "error_type": "authentication"
                        }
                    )
                else:
                    # Handle other types of errors
                    error_message = AIMessage(
                        content=f"Error: {error_msg}",
                        additional_kwargs={
                            "role": "assistant",
                            "type": "error",
                            "error_type": "general"
                        }
                    )
                state["messages"].append(error_message)
                raise  # Re-raise to be caught by outer try/except

            return state

        except Exception as e:
            logger.error(f"Error in aider_node_fn: {str(e)}")
            logger.error("Full traceback:", exc_info=True)

            # If no error message has been added yet, add one
            if not any(msg.additional_kwargs.get("type") == "error"
                      for msg in state.get("messages", [])):
                error_message = AIMessage(
                    content=f"Error in Aider node: {str(e)}",
                    additional_kwargs={"role": "assistant", "type": "error"}
                )
                if "messages" not in state:
                    state["messages"] = []
                state["messages"].append(error_message)

            return state

    return aider_node_fn
