from typing import Annotated, List, Dict, Optional, Literal, Union, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import Field
from langchain_core.pydantic_v1 import BaseModel
import json
import logging
import time
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Input/Output schemas
class InputState(TypedDict, total=False):
    repo_url: str
    query: str
    user_id: str
    github_token: str
    branch_name: Optional[str]
    anthropic_api_key: str

class OutputState(TypedDict):
    code: str

# Aider-specific schemas
class AiderState(BaseModel):
    """Track the state of the aider chat session"""
    initialized: bool = False
    last_prompt: Optional[str] = None
    waiting_for_input: bool = False
    input_type: Optional[str] = None  # 'confirmation', 'git_config', etc.
    setup_complete: bool = False
    model_name: str = 'claude-3-5-sonnet-20241022'
    last_files: List[str] = Field(default_factory=list)
    conversation_history: List[dict] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            BaseModel: lambda v: v.dict()
        }

class ReactToAiderInput(BaseModel):
    """Private state passed from react agent to aider"""
    message: str
    files: List[str] | str
    repo_path: str
    aider_state: Optional[AiderState] = None

class AiderToReactOutput(BaseModel):
    """Output schema for aider_shell tool."""
    response: str
    updated_aider_state: AiderState

class AiderShellInput(BaseModel):
    """Input schema for aider_shell tool."""
    message: str = Field(..., description="The message/instruction for aider.")
    files: Union[str, List[str]] = Field(
        default=".",
        description="Files to process. Can be a single file or a list of files."
    )

# Add new state types after existing ones
class PlanStep(BaseModel):
    """Represents a single step in the execution plan"""
    reasoning: str
    step_id: str  # e.g., #E1
    tool_name: str  # Will always be "aider_shell"
    tool_args: Dict[str, Any]

    def validate_tool_args(self) -> None:
        """Validate that required tool arguments are present"""
        if "message" not in self.tool_args:
            raise ValueError(f"Step {self.step_id} missing required 'message' argument")
        if "files" not in self.tool_args:
            raise ValueError(f"Step {self.step_id} missing required 'files' argument")

class StepResult(BaseModel):
    """Result from executing a plan step"""
    step_id: str
    output: str
    success: bool

# Main agent state
class AgentState(TypedDict, total=False):
    """Used for Aider Chat"""
    messages: Annotated[List[BaseMessage], add_messages]
    requirements: str
    code: str
    accepted: bool
    repo_url: str
    repo_path: str
    branch_name: str
    github_token: str
    aider_state: AiderState
    # New ReWOO fields
    plan_string: Optional[str]
    steps: Optional[List[PlanStep]]
    current_step: int
    step_results: Dict[str, StepResult]
    execution_status: Literal["planning", "executing", "critiquing", "complete"]
    # Add router analysis field
    router_analysis: Dict[str, Any]

# Graph configuration
class GraphConfig(TypedDict):
    gather_model: Literal['openai', 'anthropic']
    draft_model: Literal['openai', 'anthropic']
    critique_model: Literal['openai', 'anthropic']

def validate_state(state: Dict) -> None:
    """Validate that all state values are serializable."""
    try:
        # Test serialization of the entire state
        json.dumps(serialize_state_for_llm(state))
    except (TypeError, ValueError) as e:
        raise ValueError(f"State contains non-serializable data: {str(e)}")

def serialize_state_for_llm(state: AgentState) -> dict:
    """Prepare state for LLM by serializing all complex objects."""
    serialized = dict(state)

    # Serialize messages
    if 'messages' in serialized:
        serialized['messages'] = [
            msg.dict() if hasattr(msg, 'dict') else
            {'type': msg.__class__.__name__, 'content': msg.content}
            for msg in serialized['messages']
        ]

    # Serialize AiderState
    if 'aider_state' in serialized:
        if isinstance(serialized['aider_state'], AiderState):
            serialized['aider_state'] = serialized['aider_state'].model_dump()
        elif isinstance(serialized['aider_state'], dict):
            serialized['aider_state'] = serialized['aider_state']

    # Ensure all values are JSON serializable
    for key, value in serialized.items():
        if hasattr(value, 'model_dump'):
            serialized[key] = value.model_dump()
        elif isinstance(value, (set, tuple)):
            serialized[key] = list(value)

    return serialized

def validate_state_node(state: AgentState) -> AgentState:
    """Node function to validate state before processing."""
    try:
        # Test serialization of the entire state
        serialized = serialize_state_for_llm(state)
        json.dumps(serialized)  # Verify JSON serialization works
        return state
    except Exception as e:
        logger.error(f"State validation failed: {str(e)}")
        raise ValueError(f"State contains non-serializable data: {str(e)}")

def initialize_state(
    repo_url: str,
    github_token: str,
    repo_path: str,
    anthropic_api_key: Optional[str] = None,
    branch_name: Optional[str] = None,
) -> AgentState:
    """Initialize state with all required fields"""

    # Use the provided anthropic_api_key or fallback to environment variable
    if not anthropic_api_key:
        anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        if not anthropic_api_key:
            raise ValueError("Anthropic API key is required")

    # Use exact repo_path without modification
    repo_path = str(Path(repo_path))

    # Generate default branch name if none provided
    if not branch_name:
        timestamp = int(time.time())
        branch_name = f"feature/ai-changes-{timestamp}"

    return {
        "messages": [],
        "requirements": "",
        "code": "",
        "accepted": False,
        "repo_url": repo_url,
        "repo_path": repo_path,
        "branch_name": branch_name,
        "github_token": github_token,
        "anthropic_api_key": anthropic_api_key,
        "aider_state": AiderState(
            initialized=True,
            model_name='claude-3-5-sonnet-20241022',
            conversation_history=[],
            last_files=[],
            waiting_for_input=False,
            setup_complete=True
        ),
        "router_analysis": {
            "reasoning": "",
            "route_type": "multi-step"  # Default to multi-step
        },
        "plan_string": None,
        "steps": None,
        "current_step": 0,
        "step_results": {},
        "execution_status": "planning"
    }

