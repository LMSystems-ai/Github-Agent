from langgraph_engineer.model import _get_model
from langgraph_engineer.state import AgentState
from typing import TypedDict
from langchain_core.messages import RemoveMessage, HumanMessage
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

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

gather_prompt = """You are the requirements gathering component of an AI software developer system. \
Your role is to either clarify user requests or pass them directly to implementation when clear enough.

Current Repository Structure:
<repo_structure>{directory_structure}</repo_structure>

Your task is to:
1. Quickly assess if the user's request is clear enough to implement
2. If clear: Call the `Build` tool with the requirements
3. If unclear: Ask a clarifying follow-up question

please error on the side of passing the task off to the react agent who can read files and execute code.

Only ask for clarification when:
- The request is fundamentally ambiguous
- Critical technical details are missing
- Security implications need clarification

Most requests should pass directly to implementation. When in doubt, proceed rather than ask."""


class Build(TypedDict):
    requirements: str


def gather_requirements(state: AgentState, config):
    # Get directory tree
    repo_path = state.get('repo_path')
    directory_structure = get_directory_tree(repo_path) if repo_path else "Repository not yet cloned"

    messages = [
        {"role": "system", "content": gather_prompt.format(directory_structure=directory_structure)}
    ] + state['messages']
    model = _get_model(config, "openai", "gather_model").bind_tools([Build])
    response = model.invoke(messages)
    if len(response.tool_calls) == 0:
        return {"messages": [response]}
    else:
        requirements = response.tool_calls[0]['args']['requirements']
        delete_messages = [RemoveMessage(id=m.id) for m in state['messages']]
        requirements_message = HumanMessage(content=f"Here are the gathered requirements:\n\n{requirements}\n\nPlease proceed with implementing these requirements.")
        return {
            "requirements": requirements,
            "messages": delete_messages + [requirements_message]
        }
