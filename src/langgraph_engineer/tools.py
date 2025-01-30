from typing import Dict, Annotated, List, Optional, TypedDict, Literal, Union, ClassVar, Type, Any, AsyncGenerator
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
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
import glob
import shlex
from pydantic import BaseModel, Field, ConfigDict
import json
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import ConfigDict
from logging import Logger
import traceback
import pty
import fcntl
import termios
import struct
from langgraph_engineer.state import (
    AiderState,
    AgentState
)
from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import sys



load_dotenv()
logger = logging.getLogger(__name__)

openai_api_key = os.getenv('OPENAI_API_KEY')

# Add these constants at the top of the file
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables")
if not ANTHROPIC_API_KEY:
    logger.warning("ANTHROPIC_API_KEY not found in environment variables")

# Alias for backward compatibility
RepoState = AgentState



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




class StepResult(BaseModel):
    """Result of a single execution step."""
    step_id: str
    output: str
    success: bool

class AiderToReactOutput(BaseModel):
    """Output format for aider commands."""
    response: str
    success: bool = True
    error: Optional[str] = None

class AiderShellInput(BaseModel):
    """Input schema for aider shell commands."""
    message: str = Field(..., description="The message/instruction for aider")
    files: Union[str, List[str]] = Field(
        default=".",
        description="Files to process. Can be a single file or a list of files."
    )

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
    """Force clone a repository to the specified path with retry logic."""
    try:
        # Extract values from input
        url = input.url
        path = input.path
        state = input.state
        config = input.config

        logger.info(f"Starting clone operation for URL: {url.replace(state.get('github_token', ''), '*****')}")
        logger.info(f"Target path: {path}")
        logger.debug(f"State keys available: {list(state.keys() if state else [])}")

        # Validate state
        if not state:
            logger.error("State object is missing")
            return {
                "status": "error",
                "error": "State object is missing"
            }

        # Validate github token exists in state
        github_token = state.get('github_token')
        if not github_token:
            logger.error("GitHub token not found in state")
            return {
                "status": "error",
                "error": "GitHub token not found in state"
            }

        # Log directory state before cleanup
        if os.path.exists(path):
            logger.info(f"Existing directory found at {path}, will be removed")
            try:
                shutil.rmtree(path)
                logger.debug("Successfully cleaned up existing directory")
            except Exception as e:
                logger.error(f"Error cleaning up directory: {str(e)}")
                raise

        # Create parent directory
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            logger.debug(f"Created parent directory: {os.path.dirname(path)}")
        except Exception as e:
            logger.error(f"Failed to create parent directory: {str(e)}")
            raise

        # Parse and modify the URL to include the token
        try:
            parsed = urlparse(url)
            auth_url = urlunparse(parsed._replace(
                netloc=f"{github_token}@{parsed.netloc}"
            ))
            logger.debug(f"Authenticated URL created (token hidden): {auth_url.replace(github_token, '*****')}")
        except Exception as e:
            logger.error(f"Failed to create authenticated URL: {str(e)}")
            raise

        # Retry logic for clone operation
        max_retries = 3
        retry_delay = 2  # seconds
        last_exception = None

        for attempt in range(max_retries):
            logger.info(f"Starting clone attempt {attempt + 1}/{max_retries}")
            try:
                # Configure git with longer timeout
                git_config = {
                    'http.postBuffer': '524288000',  # 500MB buffer
                    'http.lowSpeedLimit': '1000',    # 1KB/s minimum speed
                    'http.lowSpeedTime': '60',       # 60 seconds timeout
                    'core.compression': '0',         # Disable compression
                    'http.version': 'HTTP/1.1',      # Force HTTP/1.1
                    'git.protocol.version': '1'      # Force Git protocol version 1
                }

                logger.debug("Initializing repository")
                repo = Repo.init(path)

                logger.debug("Applying Git configurations:")
                for key, value in git_config.items():
                    try:
                        with repo.config_writer() as git_config_writer:
                            git_config_writer.set_value(key.split('.')[0], key.split('.')[1], value)
                        logger.debug(f"Set {key}={value}")
                    except Exception as config_error:
                        logger.warning(f"Failed to set config {key}: {str(config_error)}")

                # Perform the clone steps
                logger.debug("Creating remote 'origin'")
                repo.create_remote('origin', auth_url)

                logger.debug("Starting fetch operation")
                fetch_info = repo.remote('origin').fetch(progress=GitProgressHandler())
                logger.debug(f"Fetch completed: {fetch_info}")

                logger.debug("Starting pull operation")
                pull_info = repo.remote('origin').pull('main', progress=GitProgressHandler())
                logger.debug(f"Pull completed: {pull_info}")

                # Verify the clone
                if not os.path.exists(path):
                    raise ValueError(f"Repository was cloned but path {path} does not exist")

                # Verify git repository integrity
                try:
                    repo.git.status()
                    logger.debug("Repository integrity verified")
                except Exception as integrity_error:
                    logger.error(f"Repository integrity check failed: {str(integrity_error)}")
                    raise

                logger.info(f"Successfully cloned repository to {path} on attempt {attempt + 1}")

                # Update state with repo path
                if state is not None:
                    state["repo_path"] = path
                    logger.debug("Updated state with repo path")

                return {
                    "status": "success",
                    "repo_path": path,
                    "message": f"Repository cloned successfully to {path}"
                }

            except Exception as e:
                last_exception = e
                logger.error(f"Clone attempt {attempt + 1} failed with error: {str(e)}")
                logger.error("Full error traceback:", exc_info=True)

                # Log system state
                try:
                    logger.debug(f"Directory exists: {os.path.exists(path)}")
                    if os.path.exists(path):
                        logger.debug(f"Directory contents: {os.listdir(path)}")
                except Exception as debug_error:
                    logger.error(f"Error during debug logging: {str(debug_error)}")

                # Cleanup failed attempt
                if os.path.exists(path):
                    try:
                        shutil.rmtree(path)
                        logger.debug("Cleaned up failed attempt directory")
                    except Exception as cleanup_error:
                        logger.error(f"Error cleaning up failed attempt: {str(cleanup_error)}")

                if attempt < max_retries - 1:
                    delay = retry_delay * (attempt + 1)
                    logger.info(f"Waiting {delay} seconds before retry...")
                    await asyncio.sleep(delay)
                    continue
                break

        # If all retries failed, log final error state
        logger.error(f"All {max_retries} clone attempts failed. Last error: {str(last_exception)}")
        return {
            "status": "error",
            "error": str(last_exception),
            "details": {
                "attempts": attempt + 1,
                "last_error": str(last_exception),
                "path": path,
                "url": url.replace(github_token, '*****')
            }
        }

    except Exception as e:
        logger.error(f"Unexpected error in force_clone: {str(e)}")
        logger.error("Full error traceback:", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "details": {
                "type": "unexpected_error",
                "path": path if 'path' in locals() else None,
                "url": url.replace(github_token, '*****') if 'url' in locals() else None
            }
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

        # deploy


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
    ) -> str:
        """Run the tool asynchronously."""
        try:
            if not self.state:
                raise ValueError("State is required - ensure it's being passed to AiderShellTool.")

            # Ensure we have the API key
            anthropic_api_key = self.state.get('anthropic_api_key') or os.getenv('ANTHROPIC_API_KEY')
            if not anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in state or environment")
            openai_api_key = os.getenv('OPENAI_API_KEY')
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
                    model_name='gpt-4o-mini',
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
            if isinstance(files, list):
                files = [str(Path(repo_path) / Path(f).name) for f in files]
            else:
                files = str(Path(repo_path) / Path(files).name)

            # Escape message for shell
            escaped_message = shlex.quote(message)

            # Construct aider command with --yes flag to avoid prompts
            aider_cmd = f"aider --yes --no-stream --message {escaped_message} {files}"

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

                return response

            except asyncio.TimeoutError:
                process.kill()
                raise TimeoutError("Aider command timed out after 300 seconds")

        except Exception as e:
            error_msg = f"Error in aider_shell: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            # Log the traceback for detailed debugging
            self.logger.error(traceback.format_exc())

            # Return error with current state
            return str(e)

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


# Add after existing tool input classes (around line 128)
class InteractiveAiderInput(BaseModel):
    """Input schema for interactive aider command."""
    message: str = Field(description="The message/instruction for aider")
    state: Dict[str, Any] = Field(description="Current state object")
    files: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Files to process"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

@tool(args_schema=InteractiveAiderInput)
async def interactive_aider_command(
    tool_input: InteractiveAiderInput,
) -> AsyncGenerator[AIMessageChunk, None]:
    """Execute aider command and stream responses."""
    try:
        repo_path = tool_input.state.get('repo_path')
        if not repo_path:
            raise ValueError("No repo_path in state")

        aider = InteractiveAider(repo_path=repo_path)
        async for raw_chunk in aider.execute_command(tool_input.message):
            # Ensure we have a valid chunk structure
            if isinstance(raw_chunk, dict):
                chunk_type = raw_chunk.get("type", "message")
                chunk_content = raw_chunk.get("content", "")
                chunk_files = raw_chunk.get("files", [])
            else:
                chunk_type = "message"
                chunk_content = str(raw_chunk)
                chunk_files = []

            yield AIMessageChunk(
                content=chunk_content,
                additional_kwargs={
                    "type": chunk_type,
                    "files": chunk_files
                }
            )

    except Exception as e:
        yield AIMessageChunk(
            content=str(e),
            additional_kwargs={"type": "error"}
        )


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
    git_diff,
    interactive_aider_command  # Add the new interactive tool
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
    'interactive_aider_command',  # Add the new tool
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
        chat_mode = config.get("configurable", {}).get("chat_mode", "")

        # Add logging for debugging chat_mode
        logger.info(f"Aider command configuration - chat_mode: '{chat_mode}', repo_path: '{repo_path}'")

        if not repo_path:
            raise ValueError("repo_path not found in config")

        repo_path = str(Path(repo_path))
        os.makedirs(repo_path, exist_ok=True)

        # Handle file paths
        if isinstance(files, list):
            files = [
                str(Path(f)) if f == "." else str(Path(repo_path) / Path(f).name)
                for f in files
            ]
        else:
            files = str(Path(files)) if files == "." else str(Path(repo_path) / Path(files).name)

        # Build the command with proper path handling
        files_str = " ".join(f'"{f}"' for f in files) if isinstance(files, list) else f'"{files}"'
        escaped_message = shlex.quote(message)

        # Construct chat mode argument with logging
        chat_mode_arg = ""
        if chat_mode.strip():
            # Replace 'code' with 'architect' if specified
            mode = 'architect' if chat_mode.strip().lower() == 'code' else chat_mode.strip()
            chat_mode_arg = f"--chat-mode {mode}"
        logger.info(f"Constructed chat_mode_arg: '{chat_mode_arg}'")

        # Get model name from config
        model_name = config.get("configurable", {}).get("model_name", "4o")

        # Determine API key and flag based on model
        if model_name in ['haiku', 'sonnet']:
            api_key = ANTHROPIC_API_KEY
            api_flag = "--anthropic-api-key"
        else:  # '4o' or 'o1'
            api_key = OPENAI_API_KEY
            api_flag = "--openai-api-key"

        # Construct command with proper model and API key
        aider_cmd = f"aider {chat_mode_arg} --yes-always --{model_name} {api_flag} {api_key} --no-stream --message {escaped_message} {files_str}"
        logger.info(f"Final aider command: {aider_cmd}")

        # Create process with proper environment and stdin configuration
        process = await asyncio.create_subprocess_shell(
            aider_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,  # Explicitly set stdin to DEVNULL
            cwd=repo_path,
            env={
                **os.environ,
                'TERM': 'xterm-256color',
                'COLUMNS': '80',
                'LINES': '24',
                'PATH': f"{os.environ.get('PATH', '')}:/usr/local/bin",
                'AIDER_NO_INTERACTIVE': '1'  # Add environment variable to disable interactive mode
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

async def _execute_aider_command(
    message: str,
    repo_path: str,
    anthropic_api_key: str,
    aider_state: Any,
    files: Optional[str] = None
) -> str:
    """Execute an aider command and return the response."""
    try:
        # Construct aider command
        cmd = f"aider --yes --no-stream --message {shlex.quote(message)}"
        if files:
            cmd += f" {files}"

        # Create process
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=repo_path,
            env={"OPENAI_API_KEY": openai_api_key}
            # env={"ANTHROPIC_API_KEY": anthropic_api_key}
        )

        # Wait for completion with timeout
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
            return stdout.decode()
        except asyncio.TimeoutError:
            process.kill()
            raise TimeoutError("Aider command timed out after 300 seconds")

    except Exception as e:
        raise RuntimeError(f"Failed to execute aider command: {str(e)}")

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

class AiderChunk(BaseModel):
    type: str
    content: str
    files: Optional[List[str]] = None

