import re
from langgraph_engineer.state import AgentState


def extract_python_code(text):
    pattern = r'```python\s*(.*?)\s*(```|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


error_parsing = """Make sure your response contains a code block in the following format:

```python
...
```

When trying to parse out that code block, got this error: {error}"""


def check(state: AgentState):
    last_answer = state['messages'][-1]
    try:
        code_blocks = extract_python_code(last_answer.content)
    except Exception as e:
        return {"messages": [{"role": "user", "content": error_parsing.format(error=str(e))}]}
    if len(code_blocks) == 0:
        return {"messages": [{"role": "user", "content": error_parsing.format(error="Did not find a code block!")}]}
    if len(code_blocks) > 1:
        return {"messages": [{"role": "user", "content": error_parsing.format(error="Found multiple code blocks!")}]}
    return {"code": f"```python\n{code_blocks[0][0]}\n```"}
