# Bugs that need to be fixed:

1. The aider node's output is being streamed to the terminal, but not to the langgraph api endpoint in real time. instead, the entire response is sent when the aider node is finished.

**The goal:** the aider node's output should be streamed to the langgraph api endpoint in real time.
**Replicate Bug:** Run this graph [locally](https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/#create-a-env-file), then get the API endpoint from the local graph ```http://127.0.0.1:2024``` and input it into the ```connecting.ipynb``` file.  Then run the langgraph client sdk in this notebook.  You'll be able to see the output streamed in the terminal, but not in the langgraph api endpoint.

This aider node is my attempt at replicating this [streamlit app from aider](https://github.com/Aider-AI/aider/blob/main/aider/gui.py), so it might be beneficial to look at that code.


# Features I'd like to see:

#### More Agentic Approaches
I'd like to see more agentic approaches where we go to collect more information from the repo by first asking a duplicate aider node a question about the repo, then re-prompting the original prompt with the new information.

To take it a step further, we can run multiple of these 'ask' nodes in [parallel](https://langchain-ai.github.io/langgraph/how-tos/branching/) where they ask the same or other repos (likely open source ones), or use a web search tool to (exa, perplexity, google deep research, etc.) to search about a particular package or something.


#### Integrations with other tools to close the feedback loop

Allowing a user to connect to vercel in some way would be largely beneficial for users building nextjs sites. Integrations like this would be great!


