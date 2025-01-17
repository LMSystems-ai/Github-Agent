from typing import List
import os
import difflib
import logging
from git import Repo, GitCommandError
from langchain_core.messages import AIMessage
from langgraph_engineer.state import AgentState

logger = logging.getLogger(__name__)

def show_file_diffs(state: AgentState) -> AgentState:
    """Show diffs between current state and last commit for all changed files."""
    try:
        # Validate required state
        repo_path = state.get('repo_path')
        branch_name = state.get('branch_name')
        if not repo_path or not branch_name:
            raise ValueError("Repository path and branch name are required in state")

        repo = Repo(repo_path)

        # Add debug logging
        logger.debug(f"Current branch: {repo.active_branch.name}")
        logger.debug(f"Expected branch: {branch_name}")
        logger.debug(f"Git status:\n{repo.git.status()}")
        logger.debug(f"Git diff:\n{repo.git.diff()}")

        # Ensure we're on the correct branch
        current_branch = repo.active_branch.name
        if current_branch != branch_name:
            logger.warning(f"Not on expected branch. Expected: {branch_name}, Current: {current_branch}")
            try:
                # Try to switch to correct branch
                repo.git.checkout(branch_name)
                logger.info(f"Switched to branch: {branch_name}")
            except GitCommandError as e:
                raise ValueError(f"Failed to switch to branch {branch_name}: {str(e)}")

        # Get all changes using multiple git methods
        changed_files = []

        # Get unstaged changes
        unstaged = [item.a_path for item in repo.index.diff(None)]
        changed_files.extend(unstaged)

        # Get staged changes
        staged = [item.a_path for item in repo.index.diff('HEAD')]
        changed_files.extend(staged)

        # Get untracked files
        untracked = repo.untracked_files
        changed_files.extend(untracked)

        # Remove duplicates
        changed_files = list(set(changed_files))

        logger.debug(f"Detected changed files: {changed_files}")

        all_diffs = []

        for file_path in changed_files:
            try:
                file_full_path = os.path.abspath(os.path.join(repo_path, file_path))
                logger.debug(f"Processing file: {file_path}")
                logger.debug(f"Full path: {file_full_path}")
                logger.debug(f"File exists: {os.path.exists(file_full_path)}")

                # Get old content
                try:
                    old_content = repo.git.show(f'HEAD:{file_path}')
                except Exception:
                    old_content = ''

                # Get new content
                try:
                    with open(file_full_path, 'r') as f:
                        new_content = f.read()
                except Exception:
                    new_content = ''

                # Generate diff
                diff = difflib.unified_diff(
                    old_content.splitlines(keepends=True),
                    new_content.splitlines(keepends=True),
                    fromfile=f'a/{file_path}',
                    tofile=f'b/{file_path}'
                )
                all_diffs.extend(diff)

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                continue

        # Handle untracked files
        untracked = repo.untracked_files
        for file_path in untracked:
            try:
                with open(os.path.join(repo_path, file_path), 'r') as f:
                    new_content = f.read()

                # Show new file content as addition
                diff = difflib.unified_diff(
                    [],
                    new_content.splitlines(keepends=True),
                    fromfile=f'/dev/null',
                    tofile=f'b/{file_path}'
                )
                all_diffs.extend(diff)
            except Exception as e:
                logger.error(f"Error processing untracked file {file_path}: {str(e)}")
                continue

        # Add diff results to state messages
        diff_text = ''.join(all_diffs)
        if diff_text:
            state['messages'].append(
                AIMessage(content=f"Here are the changes made on branch '{branch_name}':\n```diff\n{diff_text}\n```")
            )
        else:
            state['messages'].append(
                AIMessage(content=f"No changes detected in the repository on branch '{branch_name}'.")
            )

        return state

    except Exception as e:
        logger.error(f"Error showing diffs: {str(e)}")
        state['messages'].append(
            AIMessage(content=f"Error showing diffs: {str(e)}")
        )
        return state