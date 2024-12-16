from typing import Dict, Annotated, List, Optional, TypedDict, Literal, Union, ClassVar, Type
from typing_extensions import TypedDict
from langchain_core.tools import tool, BaseTool
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode, InjectedState
from git import Repo, RemoteProgress, GitCommandError
from git.exc import GitCommandError, InvalidGitRepositoryError
import os
import logging
import time
import shutil
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from dotenv import load_dotenv
import subprocess
import asyncio
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
import glob
import shlex
from langchain_core.pydantic_v1 import BaseModel, Field
import json
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import ConfigDict
from logging import Logger
import traceback
import pty
import fcntl
import termios
import struct
from .configuration import Configuration  # Add this import at the top

# Import all needed types from state
from langgraph_engineer.state import (
    AiderState,
    ReactToAiderInput,
    AiderToReactOutput,
    AiderShellInput,
    AgentState,
    StepResult
)
from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# Alias for backward compatibility
RepoState = AgentState

load_dotenv()
logger = logging.getLogger(__name__)

# class WriteFileArgs(BaseModel):
#     """Arguments for write_file tool"""
#     file_path: str = Field(description="Path to the file relative to repository root")
#     changes: Optional[List[FunctionChange]] = Field(
#         None,
#         description="List of function-level changes to apply"
#     )
#     content: Optional[str] = Field(
#         None,
#         description="Full content if replacing entire file"
#     )

#     @validator('changes', 'content', allow_reuse=True)
#     def validate_changes_or_content(cls, v: Optional[str], values: dict, field: str) -> Optional[str]:
#         if field == 'changes' and not v and 'content' not in values:
#             raise ValueError("Either changes or content must be provided")
#         if field == 'content' and not v and 'changes' not in values:
#             raise ValueError("Either changes or content must be provided")
#         return v

def validate_file_path(repo_path: str, file_path: str) -> Path:
    """Validate and normalize file path."""
    try:
        repo_path = Path(repo_path).resolve()
        full_path = (repo_path / file_path).resolve()

        # Security check: ensure path is within repo
        if not str(full_path).startswith(str(repo_path)):
            raise ValueError(f"File path {file_path} is outside repository")

        return full_path
    except Exception as e:
        raise ValueError(f"Invalid file path: {str(e)}")


def validate_repo_path(state: dict) -> str:
    """Validate repository path exists in state."""
    repo_path = state.get('repo_path')
    if not repo_path:
        raise ValueError("Repository path not found in state")

    # Add retry logic for filesystem sync
    max_retries = 3
    retry_delay = 0.5  # seconds

    for attempt in range(max_retries):
        if os.path.exists(repo_path):
            return repo_path
        if attempt < max_retries - 1:
            time.sleep(retry_delay)

    raise ValueError(f"Repository path does not exist: {repo_path}")


class AiderCommand(BaseModel):
    """Model for aider command input"""
    command_type: Literal['ask', 'code'] = Field(description="Type of aider command")
    prompt: str = Field(description="The prompt to send to aider")

    def to_cli_format(self) -> str:
        """Convert to CLI command format"""
        return f"/{self.command_type} {self.prompt}"


class ForceCloneInput(BaseModel):
    """Input schema for force_clone tool"""
    url: str = Field(description="Repository URL to clone")
    path: str = Field(description="Local path to clone to")
    state: Optional[Dict] = Field(default=None, description="State object")
    config: Optional[Dict] = Field(default=None, description="Config object")

@tool(args_schema=ForceCloneInput)
async def force_clone(
    input: ForceCloneInput,
) -> dict:
    """Force clone a repository to the specified path."""
    try:
        # Extract values from input
        url = input.url
        path = input.path
        state = input.state
        config = input.config

        # Validate state
        if not state:
            return {
                "status": "error",
                "error": "State object is missing"
            }

        # Validate github token exists in state
        github_token = state.get('github_token')
        if not github_token:
            return {
                "status": "error",
                "error": "GitHub token not found in state"
            }

        # Cleanup existing repo
        if os.path.exists(path):
            shutil.rmtree(path)

        # Create parent directory
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Parse and modify the URL to include the token
        parsed = urlparse(url)
        auth_url = urlunparse(parsed._replace(
            netloc=f"{github_token}@{parsed.netloc}"
        ))

        # Clone repo using the authenticated URL
        repo = Repo.clone_from(auth_url, path)

        # Ensure the path exists before returning
        if not os.path.exists(path):
            return {
                "status": "error",
                "error": f"Repository was cloned but path {path} does not exist"
            }

        logger.info(f"Successfully cloned repository to {path}")

        # Update state with repo path
        if state is not None:
            state["repo_path"] = path

        return {
            "status": "success",
            "repo_path": path,
            "message": f"Repository cloned successfully to {path}"
        }
    except Exception as e:
        logger.error(f"Error cloning repository: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

class ForceBranchInput(BaseModel):
    """Input schema for force_branch tool"""
    branch_name: str = Field(description="Name of branch to create")
    config: Optional[Dict] = Field(default=None, description="Config object")
    state: Dict = Field(description="State object")

@tool(args_schema=ForceBranchInput)
async def force_branch(
    input: ForceBranchInput,
) -> dict:
    """Force create and checkout a new branch."""
    try:
        repo_path = validate_repo_path(input.state)
        repo = Repo(repo_path)

        # Check if branch already exists
        branch_name = input.branch_name
        if branch_name in repo.heads:
            # If branch exists, just checkout
            current = repo.heads[branch_name]
        else:
            # Create and checkout new branch
            current = repo.create_head(branch_name)

        # Checkout the branch
        current.checkout()

        logger.info(f"Created and checked out branch: {branch_name}")
        return {
            "status": "success",
            "branch_name": branch_name,
            "message": f"Created and checked out branch: {branch_name}"
        }
    except Exception as e:
        logger.error(f"Error creating branch: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

# Add a proper input schema for git_status
class GitStatusInput(BaseModel):
    """Input schema for git_status tool"""
    config: Optional[Dict] = Field(default=None, description="Tool configuration")
    state: Dict = Field(description="Current state")

@tool(args_schema=GitStatusInput)
def git_status(
    config: RunnableConfig,
    state: Annotated[RepoState, InjectedState]
) -> str:
    """Get the current git status of the repository."""
    try:
        repo = Repo(state['repo_path'])
        status = []

        # Get branch info
        try:
            branch = repo.active_branch
            status.append(f"On branch {branch.name}")
        except TypeError:
            status.append("Not currently on any branch")

        # Get tracking info
        if not repo.head.is_detached:
            tracking_branch = repo.active_branch.tracking_branch()
            if tracking_branch:
                status.append(f"Tracking {tracking_branch.name}")

        # Get changed files using GitPython's native methods
        changed_files = [item.a_path for item in repo.index.diff(None)]
        staged_files = [item.a_path for item in repo.index.diff('HEAD')]
        untracked = repo.untracked_files

        if staged_files:
            status.append("\nChanges to be committed:")
            status.extend(f"  modified: {file}" for file in staged_files)

        if changed_files:
            status.append("\nChanges not staged for commit:")
            status.extend(f"  modified: {file}" for file in changed_files)

        if untracked:
            status.append("\nUntracked files:")
            status.extend(f"  {file}" for file in untracked)

        return '\n'.join(status)
    except InvalidGitRepositoryError:
        return "Error: Not a valid git repository"
    except Exception as e:
        return f"Error getting git status: {str(e)}"

# Add new input schemas before the tool definitions
class GitAddInput(BaseModel):
    """Input schema for git_add tool"""
    file_path: str = Field(description="Path to file to stage")
    config: Optional[Dict] = Field(default=None, description="Tool configuration")
    state: Dict = Field(description="Current state")

class GitCommitInput(BaseModel):
    """Input schema for git_commit tool"""
    message: str = Field(description="Commit message")
    config: Optional[Dict] = Field(default=None, description="Tool configuration")
    state: Dict = Field(description="Current state")

class GitPushInput(BaseModel):
    """Input schema for git_push tool"""
    config: Optional[Dict] = Field(default=None, description="Tool configuration")
    state: Dict = Field(description="Current state")

# Update tool decorators and keep existing implementations
@tool(args_schema=GitAddInput)
def git_add(
    file_path: str,
    config: RunnableConfig,
    state: Annotated[RepoState, InjectedState]
) -> str:
    """Stage a file for commit."""
    try:
        repo = Repo(state['repo_path'])

        # Handle wildcards and multiple files
        if file_path == '.':
            # Stage all changes including untracked files
            repo.git.add(A=True)
            return "Successfully staged all changes"

        # Validate file exists
        full_path = Path(repo.working_dir) / file_path
        if not full_path.exists():
            return f"Error: File {file_path} does not exist"

        # Stage specific file
        repo.index.add([file_path])

        # Verify file was staged
        staged_files = [item.a_path for item in repo.index.diff('HEAD')]
        if file_path in staged_files:
            return f"Successfully staged {file_path}"
        else:
            return f"File {file_path} was not staged (no changes detected)"

    except Exception as e:
        return f"Error staging file: {str(e)}"

@tool(args_schema=GitCommitInput)
def git_commit(
    message: str,
    config: RunnableConfig,
    state: Annotated[RepoState, InjectedState]
) -> str:
    """Commit staged changes."""
    try:
        repo = Repo(state['repo_path'])

        # Check if there are staged changes
        if not repo.index.diff('HEAD'):
            return "No changes staged for commit"

        # Configure author/committer if available in state
        author = None
        if 'git_author_name' in state and 'git_author_email' in state:
            author = f"{state['git_author_name']} <{state['git_author_email']}>"

        # Commit with optional author
        if author:
            commit = repo.index.commit(message, author=author)
        else:
            commit = repo.index.commit(message)

        # Return detailed commit info
        return (f"Successfully committed with hash: {commit.hexsha[:8]}\n"
                f"Author: {commit.author}\n"
                f"Message: {commit.message}")

    except Exception as e:
        return f"Error committing changes: {str(e)}"

@tool(args_schema=GitPushInput)
def git_push(
    config: RunnableConfig,
    state: Annotated[RepoState, InjectedState]
) -> str:
    """Push commits to remote repository with enhanced error handling and retry logic."""
    try:
        repo_path = validate_repo_path(state)
        branch_name = state.get('branch_name')
        repo_url = state.get('repo_url')
        github_token = state.get('github_token')  # Get token from state

        if not github_token:
            raise ValueError("GitHub token not found in state")

        logger.info(f"Push attempt with: path={repo_path}, branch={branch_name}")

        # Initialize repo
        repo = Repo(repo_path)

        # Configure the remote with authentication
        remote_url = configure_remote_with_auth(repo, repo_url, github_token)

        # Initialize progress handler
        progress = GitProgressHandler()

        # Add retry logic for push operation
        max_retries = 3
        retry_delay = 1
        last_error = None

        for attempt in range(max_retries):
            try:
                # Push with progress monitoring
                push_info = repo.remote('origin').push(
                    refspec=f"refs/heads/{branch_name}:refs/heads/{branch_name}",
                    force=True,
                    progress=progress
                )

                # Detailed push result checking
                for info in push_info:
                    if info.flags & info.ERROR:
                        raise GitCommandError(f"Push failed: {info.summary}")
                    if info.flags & info.FAST_FORWARD:
                        logger.info("Fast-forward push successful")
                    if info.flags & info.FORCED_UPDATE:
                        logger.info("Forced update successful")

                return f"Successfully pushed to branch: {branch_name}"

            except Exception as e:
                last_error = e
                logger.error(f"Push attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                break

        raise last_error if last_error else ValueError("Push failed with unknown error")

    except Exception as e:
        error_msg = f"Error pushing changes: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg

    finally:
        # Cleanup sensitive information
        cleanup_remote(repo)

@tool
def git_diff(
    config: RunnableConfig,
    state: Annotated[RepoState, InjectedState]
) -> str:
    """Show the diff of the latest commit or working directory changes."""
    try:
        repo = Repo(state['repo_path'])

        # Get the diff of staged and unstaged changes
        diff_output = []

        # Check for staged changes (diff between HEAD and index)
        staged_diff = repo.git.diff('--cached')
        if staged_diff:
            diff_output.append("=== Staged Changes ===")
            diff_output.append(staged_diff)

        # Check for unstaged changes (diff between index and working tree)
        unstaged_diff = repo.git.diff()
        if unstaged_diff:
            diff_output.append("\n=== Unstaged Changes ===")
            diff_output.append(unstaged_diff)

        # If no current changes, show the last commit diff
        if not diff_output:
            if len(repo.heads) > 0:  # Check if there are any commits
                last_commit = repo.head.commit
                if last_commit.parents:  # If commit has a parent
                    diff_output.append("=== Last Commit Diff ===")
                    diff_output.append(repo.git.diff(f'{last_commit.parents[0].hexsha}..{last_commit.hexsha}'))
                else:  # First commit
                    diff_output.append("=== Initial Commit Diff ===")
                    diff_output.append(repo.git.diff(last_commit.hexsha))
            else:
                return "No commits in the repository yet."

        return "\n".join(diff_output) if diff_output else "No changes detected"

    except Exception as e:
        return f"Error getting diff: {str(e)}"


# class AiderShellInput(BaseModel):
#     """Input schema for aider_shell tool."""
#     message: str = Field(..., description="The message/instruction for aider.")
#     files: Union[str, List[str]] = Field(
#         default=".",
#         description="Files to process. Can be a single file or a list of files."
#     )

class AiderShellTool(BaseTool):
    """Tool to run aider shell commands."""

    # Model configuration to ignore logger
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Class variables with proper type annotations
    name: ClassVar[str] = "aider_shell"
    description: ClassVar[str] = "Run aider commands for code modifications. Use this to interact with the codebase using aider."
    args_schema: ClassVar[Type[BaseModel]] = AiderShellInput
    # Add logger as ClassVar to indicate it's not a model field
    logger: ClassVar[Logger] = logging.getLogger(__name__)

    # Add state management and new fields for message/files
    state: Optional[AgentState] = Field(None, exclude=True)
    message: Optional[str] = Field(None, exclude=True)
    files: Optional[Union[str, List[str]]] = Field(None, exclude=True)

    def __init__(self, state: Optional[AgentState] = None):
        super().__init__()
        self.state = state

    def _run(
        self,
        message: str,
        files: Union[str, List[str]] = ".",
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """Run the tool synchronously."""
        return asyncio.run(self._arun(message, files, run_manager, **kwargs))

    async def _arun(
        self,
        message: Optional[str] = None,
        files: Union[str, List[str]] = ".",
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> AiderToReactOutput:
        """Run the tool asynchronously."""
        try:
            if not self.state:
                raise ValueError("State is required - ensure it's being passed to AiderShellTool.")

            # Get configuration instance from runnable config
            config_instance = Configuration.from_runnable_config(kwargs.get('config'))

            # Try getting API key in order of precedence:
            # 1. From config instance
            # 2. From state
            # 3. From environment variable
            anthropic_api_key = (
                getattr(config_instance, 'anthropic_api_key', None) or
                self.state.get('anthropic_api_key') or
                os.getenv('ANTHROPIC_API_KEY')
            )

            if not anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in config, state, or environment")

            # Set the API key in environment for aider
            os.environ['ANTHROPIC_API_KEY'] = anthropic_api_key

            # Use pre-configured values if not provided in call
            message = message or self.message
            files = files or self.files

            if not message:
                raise ValueError("No message provided for aider command")

            self.logger.info(f"Aider command executing with message: {message}")
            self.logger.info(f"Aider command executing with files: {files}")

            # Initialize aider_state if None
            if self.state.get('aider_state') is None:
                self.state['aider_state'] = AiderState(
                    initialized=True,
                    model_name='claude-3-5-sonnet-20241022',
                    conversation_history=[],
                    last_files=[],
                    waiting_for_input=False,
                    setup_complete=True
                )
            elif isinstance(self.state['aider_state'], dict):
                # Convert dict to AiderState if necessary
                self.state['aider_state'] = AiderState(**self.state['aider_state'])

            aider_state = self.state['aider_state']

            repo_path = self.state.get('repo_path')
            if not repo_path:
                raise ValueError("Repository path not found in state")

            # Process files argument
            files_str = " ".join(files) if isinstance(files, list) else files

            # Escape message for shell
            escaped_message = shlex.quote(message)

            # Construct aider command with --yes flag to avoid prompts
            aider_cmd = f"aider --yes --no-stream --message {escaped_message} {files_str}"

            self.logger.info(f"Executing aider command: {aider_cmd} in {repo_path}")

            # Create process
            process = await asyncio.create_subprocess_shell(
                aider_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=repo_path
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
                stdout_decoded = stdout.decode()
                stderr_decoded = stderr.decode()

                self.logger.info(f"Aider stdout: {stdout_decoded}")
                if stderr_decoded:
                    self.logger.error(f"Aider stderr: {stderr_decoded}")

                if process.returncode != 0:
                    raise RuntimeError(f"Aider command failed with exit code {process.returncode}")

                response = self._process_aider_output(stdout_decoded, stderr_decoded)

                # Update conversation history
                conversation_entry = {
                    "message": message,
                    "files": files,
                    "response": response,
                    "working_directory": repo_path
                }
                aider_state.conversation_history.append(conversation_entry)
                aider_state.last_prompt = message
                aider_state.last_files = files if isinstance(files, list) else [files]

                # After processing aider output
                # Update state messages
                if self.state:
                    if 'messages' not in self.state:
                        self.state['messages'] = []
                    self.state['messages'].append(AIMessage(content=response))

                return AiderToReactOutput(
                    response=response,
                    updated_aider_state=aider_state
                )

            except asyncio.TimeoutError:
                process.kill()
                raise TimeoutError("Aider command timed out after 300 seconds")

        except Exception as e:
            error_msg = f"Error in aider_shell: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.logger.error(traceback.format_exc())

            return AiderToReactOutput(
                response=error_msg,
                updated_aider_state=self.state['aider_state']
            )

    def _process_aider_output(self, stdout: str, stderr: str) -> str:
        """Process aider output to extract meaningful response."""
        # As the aider shell no longer prompts, we can focus on capturing the output
        # Split output into lines
        lines = stdout.strip().split('\n')

        # Remove any empty lines and irrelevant output (headers, footers)
        filtered_lines = [
            line for line in lines
            if not line.strip().startswith('Aider v')
            and not line.strip().startswith('Main model:')
            and not line.strip().startswith('Git repo:')
            and not line.strip().startswith('Repo-map:')
            and not line.strip().startswith('VSCode')
            and not line.strip().startswith('Use /help')
            and not line.strip().startswith('Tokens:')
            and line.strip()
        ]

        response = '\n'.join(filtered_lines).strip()

        # Add any error messages if present
        if stderr.strip():
            response += f"\nErrors:\n{stderr.strip()}"

        return response

class SingleAiderShellTool(BaseTool):
    """Modified tool to handle single step execution"""

    # Model configuration to ignore logger
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Class variables with proper type annotations
    name: ClassVar[str] = "single_aider_shell"
    description: ClassVar[str] = "Execute a single aider command for code modifications"
    args_schema: ClassVar[Type[BaseModel]] = AiderShellInput
    state: Optional[AgentState] = Field(None, exclude=True)
    logger: ClassVar[Logger] = logging.getLogger(__name__)

    def _run(
        self,
        message: str = None,
        files: Union[str, List[str]] = ".",
        step_id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> StepResult:
        """Execute synchronously by running async method"""
        return asyncio.run(self._arun(message, files, step_id, run_manager, **kwargs))

    async def _arun(
        self,
        message: str = None,
        files: Union[str, List[str]] = ".",
        step_id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> StepResult:
        """Execute single step and return structured result"""
        try:
            # Get message from state if not provided
            if not message and self.state:
                if self.state.get('router_analysis', {}).get('route_type') == 'single-step':
                    # For single-step, use the original requirements
                    message = self.state.get('requirements', '')
                    step_id = "S1"  # Single-step ID
                else:
                    # For multi-step, get from current step
                    current_step = self.state['steps'][self.state['current_step']]
                    message = current_step.tool_args['message']
                    files = current_step.tool_args['files']
                    step_id = current_step.step_id

            if not message:
                raise ValueError("No message content provided for aider tool")

            # Create instance of parent class for this call
            parent_tool = AiderShellTool(state=self.state)

            # Execute aider command using parent's implementation
            result = await parent_tool._arun(
                message=message,
                files=".",  # Always use "." for single-step
                run_manager=run_manager,
                **kwargs
            )

            # Update state with the response
            if self.state and isinstance(result, AiderToReactOutput):
                # Append the response to messages
                self.state['messages'].append(AIMessage(content=result.response))

            # Format as StepResult
            return StepResult(
                step_id=step_id or "",
                output=result.response if isinstance(result, AiderToReactOutput) else str(result),
                success=True
            )
        except Exception as e:
            self.logger.error(f"Error in SingleAiderShellTool: {str(e)}", exc_info=True)
            return StepResult(
                step_id=step_id or "",
                output=f"Error: {str(e)}",
                success=False
            )

# # Instead of initializing with no state:
# aider_tools = [SingleAiderShellTool()]  # Remove this line

# Create tools when needed with state:
def create_aider_tools(state: Optional[AgentState] = None) -> List[BaseTool]:
    """Create aider tools with state."""
    return [SingleAiderShellTool(state=state)]

# Update the aider_node creation
aider_node = lambda state: ToolNode(tools=create_aider_tools(state))

# Keep only these tools
setup_tools = [force_clone, force_branch]

regular_tools = [
    git_status,
    git_add,
    git_commit,
    git_push,
    git_diff  # Keep only aider_shell from non-git tools
]

git_tools = [git_status, git_add, git_commit, git_push, git_diff]

# aider_tools = [SingleAiderShellTool()]

# Combined tools list for react agent
tools = setup_tools + regular_tools

# Create separate tool nodes
setup_node = ToolNode(tools=setup_tools)
tool_node = ToolNode(tools=regular_tools)

# At the bottom of tools.py, update the exports
__all__ = [
    'SingleAiderShellTool',
    'AiderShellTool',
    'create_aider_tools',
    'force_clone',
    'force_branch',
    'git_status',
    'git_add',
    'git_commit',
    'git_push',
    'git_diff'
]

class AiderCommandInput(BaseModel):
    message: str = Field(..., description="The instruction for aider")
    files: Union[str, List[str]] = Field(..., description="Files to process")

@tool(args_schema=AiderCommandInput)
async def aider_command(
    message: str,
    files: Union[str, List[str]],
    config: RunnableConfig,
) -> str:
    """Execute aider command on specified files in a container environment."""
    try:
        repo_path = config.get("configurable", {}).get("repo_path")
        if not repo_path:
            raise ValueError("repo_path not found in config")

        # Use exact path without modification
        repo_path = str(Path(repo_path))

        # Create directory if it doesn't exist
        os.makedirs(repo_path, exist_ok=True)

        # Process files argument with proper path handling
        if isinstance(files, list):
            files = [str(Path(repo_path) / Path(f).name) for f in files]
        else:
            files = str(Path(repo_path) / Path(files).name)

        # Build the command with proper path handling
        files_str = " ".join(f'"{f}"' for f in files) if isinstance(files, list) else f'"{files}"'
        escaped_message = shlex.quote(message)

        # Construct aider command directly without script wrapper
        aider_cmd = f"aider --yes-always --no-stream --message {escaped_message} {files_str}"

        logger.info(f"Executing aider command in {repo_path}: {aider_cmd}")

        # Create process with proper environment
        process = await asyncio.create_subprocess_shell(
            aider_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            cwd=repo_path,
            env={
                **os.environ,
                'TERM': 'xterm-256color',
                'COLUMNS': '80',
                'LINES': '24',
                'PATH': f"{os.environ.get('PATH', '')}:/usr/local/bin"
            }
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
            stdout_decoded = stdout.decode() if stdout else ""
            stderr_decoded = stderr.decode() if stderr else ""

            logger.info(f"Command output: {stdout_decoded}")
            if stderr_decoded:
                logger.error(f"Command stderr: {stderr_decoded}")

            if process.returncode != 0:
                raise RuntimeError(f"Command failed (exit code {process.returncode}): {stderr_decoded}")

            return stdout_decoded if stdout_decoded else "Command completed successfully"

        except asyncio.TimeoutError:
            process.kill()
            raise TimeoutError("Command timed out after 300 seconds")

    except Exception as e:
        error_msg = f"Error in aider_command: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg)

# Add these utility functions before the git tools

def configure_remote_with_auth(repo: Repo, repo_url: str, github_token: str) -> str:
    """Configure git remote with authentication."""
    try:
        # Parse the URL
        parsed = urlparse(repo_url)

        # Construct authenticated URL
        auth_url = urlunparse(parsed._replace(
            netloc=f"oauth2:{github_token}@{parsed.netloc}"
        ))

        # Set or update the remote
        if 'origin' in repo.remotes:
            repo.delete_remote('origin')
        repo.create_remote('origin', auth_url)

        return auth_url
    except Exception as e:
        raise ValueError(f"Failed to configure remote: {str(e)}")

def cleanup_remote(repo: Repo) -> None:
    """Remove sensitive information from git remote."""
    try:
        if 'origin' in repo.remotes:
            repo.delete_remote('origin')
            # Recreate with clean URL if needed
            if hasattr(repo, '_original_url'):
                repo.create_remote('origin', repo._original_url)
    except Exception as e:
        logger.warning(f"Failed to cleanup remote: {str(e)}")

# Git Progress Handler
class GitProgressHandler(RemoteProgress):
    """Handle git operation progress."""
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def update(self, op_code, cur_count, max_count=None, message=''):
        """Called whenever the progress changes."""
        self.logger.debug(f'Progress: {op_code}, {cur_count}/{max_count}, {message}')
