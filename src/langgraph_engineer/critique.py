from langgraph_engineer.model import _get_model
from langgraph_engineer.state import AgentState
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel
from git import Repo
import logging
import json

logger = logging.getLogger(__name__)

# Update the critique prompt to guide XML output
critique_prompt = """Review if the Task was fully completed given the user's requirements and the steps/changes made.

Original User Requirements:
{requirements}

Here was our agent's original plan:
{plan}

Analyze the implementation and respond in XML format with two tags:
1. <node_route> - either "aider_node" if more changes are needed, or "git_push_changes" if the task is complete
2. <instructions> - if routing to aider_node, provide specific coding instructions for the next step. If routing to git_push_changes, explain why the task is complete.

Example response for incomplete task:
<node_route>aider_node</node_route>
<instructions>Please modify the login function to add input validation for the email field using regex.</instructions>

Example response for complete task:
<node_route>git_push_changes</node_route>
<instructions>All requirements have been met: user authentication implemented, input validation added, and tests created.</instructions>
"""


def _swap_messages(messages):
    new_messages = []
    for m in messages:
        if isinstance(m, AIMessage):
            new_messages.append({"role": "user", "content": m.content})
        else:
            new_messages.append({"role": "assistant", "content": m.content})
    return new_messages


def get_git_diff(state: AgentState) -> str:
    """Get git diff directly without using the tool interface"""
    try:
        repo = Repo(state['repo_path'])
        diff_output = []

        # Check for staged changes
        staged_diff = repo.git.diff('--cached')
        if staged_diff:
            diff_output.append("=== Staged Changes ===")
            diff_output.append(staged_diff)

        # Check for unstaged changes
        unstaged_diff = repo.git.diff()
        if unstaged_diff:
            diff_output.append("\n=== Unstaged Changes ===")
            diff_output.append(unstaged_diff)

        # If no current changes, show the last commit diff
        if not diff_output:
            if len(repo.heads) > 0:
                last_commit = repo.head.commit
                if last_commit.parents:
                    diff_output.append("=== Last Commit Diff ===")
                    diff_output.append(repo.git.diff(f'{last_commit.parents[0].hexsha}..{last_commit.hexsha}'))
                else:
                    diff_output.append("=== Initial Commit Diff ===")
                    diff_output.append(repo.git.diff(last_commit.hexsha))
            else:
                return "No commits in the repository yet."

        return "\n".join(diff_output) if diff_output else "No changes detected"

    except Exception as e:
        logger.error(f"Error getting diff: {str(e)}")
        return f"Error getting diff: {str(e)}"


def critique(state: AgentState, config) -> AgentState:
    """Modified critique to provide XML-formatted response"""
    try:
        formatted_prompt = critique_prompt.format(
            requirements=state.get('requirements', ''),
            plan=state.get('plan_string', '')
        )

        model = _get_model(config, "anthropic", "critique_model")
        message_sequence = [
            SystemMessage(content=formatted_prompt),
            AIMessage(content=f"Here are the step results from our implementation:\n{json.dumps(state.get('step_results', {}), indent=2)}\n\nPlease analyze the implementation.")
        ]

        response = model.invoke(message_sequence)

        # Update state with just the raw critique response
        new_state = {
            **state,
            "step_results": {
                **(state.get("step_results", {})),
                "critique": {
                    "response": response.content
                }
            }
        }

        return new_state

    except Exception as e:
        logger.error(f"Error in critique: {str(e)}")
        return {
            **state,
            "step_results": {
                **(state.get("step_results", {})),
                "critique": {
                    "response": f"Error occurred during critique: {str(e)}"
                }
            }
        }
