import asyncio
import logging
from pathlib import Path
from typing import AsyncGenerator, Tuple, Optional, List, Dict, Any
from aider.io import InputOutput
from aider.coders import Coder
from aider.main import main as cli_main
from langchain_core.messages import AIMessageChunk
import os
import sys

logger = logging.getLogger(__name__)

class StreamingAiderIO(InputOutput):
    """Custom IO class that captures streaming output from Aider."""

    def __init__(self, *args, **kwargs):
        # Match GUI's non-interactive settings
        kwargs.update({
            'yes': True,
            'pretty': False,
            'fancy_input': False,
            'dry_run': False
        })
        super().__init__(*args, **kwargs)
        self._buffer = []
        self.stream = True
        self.yield_stream = True
        self.callbacks = []  # Add this line to store streaming callbacks
        self.pretty = False  # Important for diff formatting
        self._last_commit_hash = None
        self._last_commit_message = None
        self._edited_files = set()

    async def stream_callback(self, chunk):
        """Async method to handle streaming chunks"""
        for callback in self.callbacks:
            await callback(chunk)

    def tool_output(self, msg, log_only=False):
        """Capture tool output for streaming."""
        chunk = AIMessageChunk(
            content=msg,
            additional_kwargs={
                "type": "message",
                "files": []
            }
        )

        # Create task to run stream callback
        if hasattr(asyncio, 'get_running_loop'):
            loop = asyncio.get_running_loop()
            loop.create_task(self.stream_callback(chunk))

        self._buffer.append(chunk)
        super().tool_output(msg, log_only=log_only)

    def tool_error(self, msg):
        """Capture error messages for streaming."""
        self._buffer.append(AIMessageChunk(
            content=msg,
            additional_kwargs={
                "type": "error",
                "files": []
            }
        ))
        super().tool_error(msg)

    def tool_warning(self, msg):
        """Capture warning messages for streaming."""
        self._buffer.append(AIMessageChunk(
            content=msg,
            additional_kwargs={
                "type": "warning",
                "files": []
            }
        ))
        super().tool_warning(msg)

    def assistant_output(self, msg, pretty=False):
        """Capture assistant output for streaming."""
        self._buffer.append(AIMessageChunk(
            content=msg,
            additional_kwargs={
                "type": "message",
                "files": []
            }
        ))
        super().assistant_output(msg, pretty=pretty)

    def get_buffer(self):
        """Get and clear the buffer."""
        buffer = self._buffer
        self._buffer = []
        return buffer

    def capture_edit_info(self, commit_hash=None, commit_message=None, files=None, diff=None):
        """Capture edit information for streaming"""
        edit_chunk = {
            "type": "edit",
            "content": diff or "",
            "commit_hash": commit_hash,
            "commit_message": commit_message,
            "files": list(files) if files else []
        }

        chunk = AIMessageChunk(
            content=diff or "",
            additional_kwargs=edit_chunk
        )

        self._buffer.append(chunk)
        return chunk

class InteractiveAider:
    """A wrapper class for Aider to handle interactive code editing."""

    def __init__(
        self,
        repo_path: str,
        model: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        encoding: str = "utf-8",
    ):
        self.repo_path = Path(repo_path).resolve()
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.encoding = encoding

        # Store original environment variables
        self._original_anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        self._original_openai_key = os.environ.get("OPENAI_API_KEY")

        # Set API key in environment based on model
        if api_key:
            # Check if model is an Anthropic model
            anthropic_models = ["sonnet", "haiku"]  # Add all Anthropic models here
            if self.model in anthropic_models:
                logger.info(f"Using provided Anthropic API key from state for model {self.model}")
                os.environ["ANTHROPIC_API_KEY"] = api_key
                # Temporarily unset OpenAI key to prevent any confusion
                if "OPENAI_API_KEY" in os.environ:
                    del os.environ["OPENAI_API_KEY"]
            else:  # Default to OpenAI
                logger.info(f"Using provided OpenAI API key from state for model {self.model}")
                os.environ["OPENAI_API_KEY"] = api_key
                # Temporarily unset Anthropic key to prevent any confusion
                if "ANTHROPIC_API_KEY" in os.environ:
                    del os.environ["ANTHROPIC_API_KEY"]

        # Get coder instance
        try:
            self.coder = self._get_coder()
        except Exception as e:
            # Restore original environment variables in case of error
            self._restore_env_vars()
            raise e

        # Initialize and attach our custom IO
        self.io = StreamingAiderIO(
            encoding=self.encoding,
            yes=True,
            pretty=False,
            fancy_input=False,
            dry_run=False
        )
        self.io.yes = True
        self.coder.commands.io = self.io

        # Add reflection tracking
        self.num_reflections = 0
        self.max_reflections = 3

    def _restore_env_vars(self):
        """Restore original environment variables"""
        if hasattr(self, '_original_anthropic_key'):
            if self._original_anthropic_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = self._original_anthropic_key
            elif "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]

        if hasattr(self, '_original_openai_key'):
            if self._original_openai_key is not None:
                os.environ["OPENAI_API_KEY"] = self._original_openai_key
            elif "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

    def __del__(self):
        """Cleanup when object is destroyed"""
        self._restore_env_vars()

    def _get_coder(self):
        """Initialize and validate coder instance similar to Streamlit app."""
        # Validate API key is set for selected model
        anthropic_models = ["sonnet", "haiku"]
        if self.model in anthropic_models:
            if not self.api_key:
                raise ValueError(f"Anthropic API key not found for {self.model} model")
        else:
            if not self.api_key:
                raise ValueError(f"OpenAI API key not found for {self.model} model")

        # Add API key to command arguments
        api_flag = "--anthropic-api-key" if self.model in anthropic_models else "--openai-api-key"
        argv = [
            "--model", self.model,
            "--yes-always",
            "--stream",
            "--map-refresh", "auto",
            api_flag, self.api_key,  # Explicitly pass API key
            str(self.repo_path)
        ]
        argv = [arg for arg in argv if arg]

        coder = cli_main(
            argv=argv,
            input=None,
            output=None,
            force_git_root=str(self.repo_path),
            return_coder=True
        )

        # Validate coder instance
        if not isinstance(coder, Coder):
            raise ValueError("Failed to create valid Coder instance")

        # Validate repo
        if not coder.repo:
            raise ValueError("Aider can currently only be used inside a git repo")

        # Ensure chat mode settings
        coder.yield_stream = True
        coder.stream = True

        coder.pretty = False  # Important for diff formatting
        return coder

    def auto_add_files(self, file_paths: List[str]):
        """
        Automatically add the specified file paths to the Aider session,
        just as if the user had manually added them.
        """
        for path in file_paths:
            # Use add_rel_fname to add files to the chat
            self.coder.add_rel_fname(path)
            self.io.yes = True
        # Optionally log or confirm the added files
        logger.debug(f"Auto-added files to chat: {file_paths}")

    async def get_diff_info(self):
        """Get diff information from the latest changes"""
        if not self.coder.last_aider_commit_hash:
            return None

        if (self.io._last_commit_hash != self.coder.last_aider_commit_hash):
            commits = f"{self.coder.last_aider_commit_hash}~1"
            diff = self.coder.repo.diff_commits(
                pretty=False,
                from_rev=commits,
                to_rev=self.coder.last_aider_commit_hash
            )

            edit_info = {
                "commit_hash": self.coder.last_aider_commit_hash,
                "commit_message": self.coder.last_aider_commit_message,
                "files": list(self.coder.aider_edited_files),
                "diff": diff
            }

            self.io._last_commit_hash = self.coder.last_aider_commit_hash
            return edit_info
        return None

    async def execute_command(self, message: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute command with enhanced streaming support."""
        try:
            accumulated_content = ""
            prompt = message

            # Set up streaming callback
            async def stream_handler(chunk):
                if isinstance(chunk, AIMessageChunk):
                    yield {
                        "type": chunk.additional_kwargs.get("type", "message"),
                        "content": chunk.content,
                        "files": chunk.additional_kwargs.get("files", [])
                    }

            # Add our stream handler to IO callbacks
            self.io.callbacks.append(stream_handler)

            while prompt:
                stream = self.coder.run_stream(prompt)
                for chunk in stream:
                    # Process the current chunk in real-time
                    if isinstance(chunk, str):
                        chunk_dict = {"type": "message", "content": chunk, "files": []}
                        accumulated_content += chunk
                        yield chunk_dict
                    elif hasattr(chunk, 'content'):
                        chunk_dict = {"type": "message", "content": chunk.content, "files": []}
                        accumulated_content += chunk.content
                        yield chunk_dict
                    elif hasattr(chunk, 'diff'):
                        chunk_dict = {
                            "type": "edit",
                            "content": chunk.diff,
                            "files": getattr(chunk, 'files', [])
                        }
                        yield chunk_dict

                    # NEW: Flush the buffer immediately after each chunk
                    buffered_chunks = self.io.get_buffer()
                    for buffered_chunk in buffered_chunks:
                        yield {
                            "type": buffered_chunk.additional_kwargs.get("type", "message"),
                            "content": buffered_chunk.content,
                            "files": buffered_chunk.additional_kwargs.get("files", [])
                        }

                # Check for reflections
                prompt = None
                if hasattr(self.coder, 'reflected_message') and self.coder.reflected_message:
                    if self.num_reflections < self.max_reflections:
                        self.num_reflections += 1
                        prompt = self.coder.reflected_message
                        yield {
                            "type": "reflection",
                            "content": f"Reflection {self.num_reflections}: {prompt}",
                            "files": []
                        }

            # Final message completion
            yield {
                "type": "complete",
                "content": accumulated_content,
                "files": []
            }

            # Clean up
            self.io.callbacks.remove(stream_handler)

        except Exception as e:
            logger.error(f"Error in execute_command: {str(e)}")
            logger.error("Full traceback:", exc_info=True)
            yield {"type": "error", "content": str(e), "files": []}
            raise