from typing import Annotated, List, Dict, Optional, Literal, Union, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, ConfigDict
import json
import logging
import os
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Input/Output schemas
class InputState(TypedDict, total=False):
    repo_url: str
    query: str
    user_id: str
    github_token: str
    branch_name: Optional[str]
    anthropic_api_key: Optional[str]
    openai_api_key: Optional[str]
    chat_mode: Optional[str]
    model_name: Optional[str]

class OutputState(TypedDict):
    code: str

# Aider-specific schemas
class AiderState(BaseModel):
    """Track the state of the aider chat session"""
    initialized: bool = False
    last_prompt: Optional[str] = None
    waiting_for_input: bool = False
    input_type: Optional[str] = None
    setup_complete: bool = False
    model_name: str = Field(default='openai')
    last_files: List[str] = Field(default_factory=list)
    conversation_history: List[dict] = Field(default_factory=list)

    # Update to use Pydantic v2 config
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            BaseModel: lambda v: v.model_dump()
        }
    )

# Simplified main agent state
class AgentState(TypedDict, total=False):
    """Used for Aider Chat"""
    messages: List[Union[HumanMessage, AIMessage]]
    code: str
    repo_url: str
    repo_path: str
    branch_name: str
    github_token: str
    anthropic_api_key: Optional[str]
    openai_api_key: Optional[str]
    aider_state: AiderState
    chat_mode: Optional[str]
    model_name: Optional[str]
    accepted: bool
    show_diff: bool
    aider_summary: Optional[str]

def initialize_state(
    repo_url: str,
    github_token: str,
    repo_path: str,
    anthropic_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    branch_name: Optional[str] = None,
    chat_mode: Optional[str] = None,
    model_name: Optional[str] = None,
    query: Optional[str] = None,
) -> AgentState:
    """Initialize state with all required fields"""

    # Prioritize input API keys over environment variables
    anthropic_api_key = anthropic_api_key if anthropic_api_key is not None else os.getenv('ANTHROPIC_API_KEY')
    openai_api_key = openai_api_key if openai_api_key is not None else os.getenv('OPENAI_API_KEY')

    # Log which source we're using for each API key
    logger.debug(f"Using Anthropic API key from: {'parameter' if anthropic_api_key != os.getenv('ANTHROPIC_API_KEY') else 'environment'}")
    logger.debug(f"Using OpenAI API key from: {'parameter' if openai_api_key != os.getenv('OPENAI_API_KEY') else 'environment'}")

    repo_path = str(Path(repo_path))

    if not branch_name:
        timestamp = int(datetime.now(timezone.utc).timestamp())
        branch_name = f"feature/ai-changes-{timestamp}"

    valid_models = ['haiku', 'sonnet', '4o', 'o1', 'gpt-4o-mini']
    if model_name and model_name not in valid_models:
        raise ValueError(f"model_name must be one of {valid_models}")

    # Validate API key based on model
    if model_name in ['haiku', 'sonnet'] and not anthropic_api_key:
        raise ValueError(f"Anthropic API key is required for model {model_name}")
    elif model_name in ['4o', 'o1', 'gpt-4o-mini'] and not openai_api_key:
        raise ValueError(f"OpenAI API key is required for model {model_name}")

    state = {
        "messages": [],
        "code": "",
        "repo_url": repo_url,
        "repo_path": repo_path,
        "branch_name": branch_name,
        "github_token": github_token,
        "anthropic_api_key": anthropic_api_key,
        "openai_api_key": openai_api_key,
        "chat_mode": chat_mode,
        "accepted": False,
        "show_diff": False,
        "aider_state": AiderState(
            initialized=True,
            model_name=model_name or '4o',
            conversation_history=[],
            last_files=[],
            waiting_for_input=False,
            setup_complete=True
        ),
        "model_name": model_name or '4o'
    }

    if query:
        state["messages"] = [
            HumanMessage(
                content=query,
                additional_kwargs={"role": "user"}
            )
        ]

    return state

# Keep the serialization helpers
def serialize_state_for_llm(state: AgentState) -> dict:
    """Prepare state for LLM by serializing all complex objects."""
    serialized = dict(state)

    if 'messages' in serialized:
        serialized['messages'] = [
            msg.dict() if hasattr(msg, 'dict') else
            {'type': msg.__class__.__name__, 'content': msg.content}
            for msg in serialized['messages']
        ]

    if 'aider_state' in serialized:
        if isinstance(serialized['aider_state'], AiderState):
            serialized['aider_state'] = serialized['aider_state'].model_dump()
        elif isinstance(serialized['aider_state'], dict):
            serialized['aider_state'] = serialized['aider_state']

    for key, value in serialized.items():
        if hasattr(value, 'model_dump'):
            serialized[key] = value.model_dump()
        elif isinstance(value, (set, tuple)):
            serialized[key] = list(value)

    return serialized

def validate_state(state: Dict) -> None:
    """Validate that all state values are serializable."""
    try:
        json.dumps(serialize_state_for_llm(state))
    except (TypeError, ValueError) as e:
        raise ValueError(f"State contains non-serializable data: {str(e)}")

# Add this with the other TypedDict definitions
class GraphConfig(TypedDict, total=False):
    """Configuration for the graph"""
    anthropic_api_key: Optional[str]
    openai_api_key: Optional[str]
    github_token: str
    repo_path: str
    branch_name: str
    chat_mode: Optional[str]
    model_name: Optional[str]

