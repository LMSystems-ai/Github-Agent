import json
import asyncio
from typing import Dict, Any, List, Tuple
from langchain_core.messages import SystemMessage, AIMessage
from langgraph_engineer.state import (
    AgentState,
    serialize_state_for_llm,
    AiderState,
    PlanStep,
    StepResult
)
from langgraph_engineer.tools import (
    SingleAiderShellTool,
    AiderToReactOutput
)
from langgraph_engineer.model import _get_model
import os
from pathlib import Path
import logging
import copy
import re

logger = logging.getLogger(__name__)

# Update plan generation prompt for efficiency
PLAN_PROMPT = """You are a software development planner. Your role is to break down requirements into the most efficient, minimal number of actionable tasks that will be executed by an AI software developer with access to the codebase.

Requirements:
{requirements}

Current Code Structure:
{directory_tree}

Format your response as:
Plan: <reasoning for the step>
#E1 = aider_shell[files=<file paths>, message=<natural language task description>]
Plan: <reasoning for next step, if needed>
#E2 = aider_shell[files=<file paths>, message=<natural language task that may reference #E1>]
...

IMPORTANT:
- Focus on creating the most efficient plan with the least number of steps
- Break each requirement into the smallest number of focused changes possible
- Be specific about which files need to be modified (note: if you reference a file that doesn't exist, the AI developer will create it)
- If you are asking a question regarding the codebase (e.g: "What's this repo about?"), then populate the 'files' field with '.'
- Write clear, natural language instructions for the AI developer (no code in the instructions)
- You can reference previous step results using #E1, #E2, etc.
- The AI developer can search and modify the codebase but cannot run tests
- Focus on WHAT needs to change in which files

Example tasks:
✓ "Add a new method called 'process_data' to the DataHandler class in src/handlers/data.py"
✓ "Update the error message in login.py to include the specific validation failure"
✓ "Remove the deprecated 'old_method' from utils/helpers.py"
"""

# Add plan parsing regex
STEP_PATTERN = r"Plan:\s*(.+?)\s*#E(\d+)\s*=\s*aider_shell\[files=(.*?),\s*message=(.*?)\s*\]"

def get_directory_tree(start_path: str, indent: str = "", prefix: str = "") -> str:
    """Generate a directory tree string starting from the given path."""
    if not os.path.exists(start_path):
        logger.error(f"Repository path does not exist: {start_path}")
        return "Error: Path does not exist"

    try:
        # Convert to Path object for better path handling
        start_path = Path(start_path).resolve()
        base_name = start_path.name
        tree = [f"{prefix}{base_name}/"]

        # List and sort directory contents
        items = sorted(item for item in start_path.iterdir()
                      if item.name not in {'.git', '__pycache__', '.DS_Store'})

        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            next_indent = indent + ("    " if is_last else "│   ")
            next_prefix = indent + ("└── " if is_last else "├── ")

            if item.is_dir():
                # Recursively process directories
                subtree = get_directory_tree(str(item), next_indent, next_prefix)
                tree.append(subtree)
            else:
                # Add files with relative path
                rel_path = item.relative_to(start_path)
                tree.append(f"{next_prefix}{item.name} ({rel_path})")

        return "\n".join(tree)
    except Exception as e:
        logger.error(f"Error generating directory tree: {str(e)}")
        return f"Error reading directory: {str(e)}"

def parse_plan(plan_string: str) -> List[PlanStep]:
    """Parse plan string into structured steps"""
    matches = re.finditer(STEP_PATTERN, plan_string, re.DOTALL)
    steps = []

    for match in matches:
        reasoning, step_num, files, message = match.groups()

        # Parse files string into proper format
        files_str = files.strip().strip('"\'')
        if ',' in files_str:
            files = [f.strip() for f in files_str.split(',')]
        else:
            files = files_str

        steps.append(PlanStep(
            reasoning=reasoning.strip(),
            step_id=f"#E{step_num}",
            tool_name="aider_shell",
            tool_args={
                "files": files,
                "message": message.strip().strip('"\'')
            }
        ))

    return steps

def react_agent(state: AgentState) -> AgentState:
    """Process the current state and generate the next action."""
    try:
        # Check if we need to generate a plan
        if not state.get("steps"):
            # Initialize state fields if not present
            if "messages" not in state:
                state["messages"] = []
            if "step_results" not in state:
                state["step_results"] = {}

            # Get model for plan generation
            model = _get_model({"configurable": {}}, "anthropic", "planner_model")

            # Get directory tree
            directory_tree = get_directory_tree(state["repo_path"])

            # Generate plan using the requirements
            plan_prompt = PLAN_PROMPT.format(
                requirements=state["requirements"],
                directory_tree=directory_tree
            )

            response = model.invoke(plan_prompt)
            plan_string = response.content

            # Parse plan into steps
            steps = parse_plan(plan_string)
            if not steps:
                raise ValueError("No valid steps were parsed from the plan")

            # Update state with plan
            state["plan_string"] = plan_string
            state["steps"] = steps
            state["current_step"] = 0
            state["execution_status"] = "executing"

            # Add the plan_string as an AIMessage
            state["messages"].append(
                AIMessage(content=plan_string)
            )

            logger.info(f"Generated plan with {len(steps)} steps")
            return state  # Return here to process the first step in the next call

        # Validate current step
        if state["current_step"] >= len(state["steps"]):
            logger.info("All steps completed")
            state["execution_status"] = "completed"
            return state

        current_step = state["steps"][state["current_step"]]
        logger.info(f"Processing step {current_step.step_id}")

        # Ensure messages are initialized
        if "messages" not in state:
            state["messages"] = []

        # Optionally, add the step reasoning as an AIMessage
        state["messages"].append(
            AIMessage(content=current_step.reasoning)
        )

        # Create tool call message
        tool_call = {
            "name": "aider_command",
            "args": {
                "message": current_step.tool_args["message"],
                "files": current_step.tool_args["files"]
            },
            "id": current_step.step_id,
            "type": "tool_call"
        }

        # Add the AIMessage with the tool call
        state["messages"].append(
            AIMessage(
                content="",
                tool_calls=[tool_call]
            )
        )

        logger.info(f"Added AIMessage with tool_call for step {current_step.step_id}")

        return state

    except Exception as e:
        logger.error(f"Error in react_agent: {str(e)}", exc_info=True)
        logger.error(f"Current state: {state}")
        raise
