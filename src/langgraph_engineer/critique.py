from langgraph_engineer.model import _get_model
from langgraph_engineer.state import AgentState
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel
from git import Repo
import logging
import json

logger = logging.getLogger(__name__)

# Simplified critique prompt focused on natural language response
critique_prompt = """Review if the Task was fully completed given the user's requirements and the steps/changes made.

Original User Requirements:
{requirements}

Here was our agent's original plan:
Plan:
{plan}

and here's the results from that plan:
Step Results:
{step_results}

Please provide a detailed analysis that covers:
1. What specific changes were found and implemented
2. How well each requirement was addressed
3. The quality of the implementation
4. Whether changes stayed within the expected scope

Use clear, standardized phrases like:
- "changes were successfully implemented" or "changes are missing"
- "requirements were fully met" or "requirements were not fully met"
- "implementation was correct" or "implementation needs revision"
- "changes stayed within scope" or "unauthorized changes detected"
"""


class Accept(BaseModel):
    """Schema for critique response"""
    logic: str
    accept: bool
    completion_status: str


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
    """Modified critique to provide natural language analysis"""
    try:
        # Format the prompt with required information
        formatted_prompt = critique_prompt.format(
            requirements=state.get('requirements', ''),
            plan=state.get('plan_string', ''),
            step_results=json.dumps(state.get('step_results', {}), indent=2)
        )

        model = _get_model(config, "openai", "critique_model")

        # Create message sequence
        message_sequence = [
            SystemMessage(content=formatted_prompt),
            AIMessage(content="Please analyze the implementation.")
        ]

        # Get unstructured response from the model
        response = model.invoke(message_sequence)
        critique_logic = response.content

        # Update state with critique results
        new_state = {
            **state,
            "step_results": {
                **(state.get("step_results", {})),
                "critique": {
                    "args": {
                        "logic": critique_logic
                    }
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
                    "args": {
                        "logic": f"Error occurred during critique: {str(e)}"
                    }
                }
            }
        }
