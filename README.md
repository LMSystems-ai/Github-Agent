# Github Agent

Github Agent clones a given Github Repository, then modifies it with a human-in-the-loop before pushing any code.

Deployed API: [LMSystems.ai](https://www.lmsystems.ai/graphs/github-agent-48/test)

IOS app: [Ship App](https://apps.apple.com/us/app/ship/id6738367546)

# Quick Start

```pip install -e .```

```pip install --upgrade "langgraph-cli[inmem]"```

```langgraph dev```

Then use langgraph studio or connecting.ipynb to interact with the graph.

Use langgraph's [quikstart guide]([https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/]) for more in depth quickstart instructions.


### Why we made this project:

Coding seems to be the killer app for LLMs, but we see a much brighter future in "cloud coding" than coding with traditional IDEs. Cloud coding refers to having your coding enviornemnt running in a server rather than directly on your computer.  We hope to make an AI application which you can give a "task" to and it can go off and compleet that given task, no matter how long it takes. This only becomes possible in cloud coding enviornments because well, people have to turn their computer off eventually.

This project is the first step towards that world.

## Project Overview

Our graph currently works like this:

User Query => Setup Node => Aider Node <==> Human Interaction => Git Push Changes

Here's each node and their corresponding purpose + files.

- **Setup Node** this node clones the given repo with the repo url, github access token, and selected branch name
- **Aider Node** this node uses [Aider]([https://aider.chat/]) to do the heavy lifting for making code changes to the repo. Aider is a cli tool operating on the cloned repo. We've tried to emulate what they've done with [Aider in the browser]([https://aider.chat/docs/usage/browser.html]) in order to try and capture the llm stream of tokens but we have yet to capture it. See [TODO.md](TODO.md) for more issues.
- **Human Interaction** we've added a Human-in-the-Loop here which allows the user and aider node to have a back and forth conversation for as many times as they'd like before the human decides to push the local changes to github.  To push the changes to github, you must set the 'Accepted' state values to 'True'.
- **Git Push Node** Pushes the local changes to the selected branch.


## Contributing

We welcome contributions from the community! Here's how you can help:

1. **Report Issues**: Submit bugs and feature requests through our issue tracker
2. **Submit Pull Requests**: Improve documentation, fix bugs, or add new features
3. **Follow Standards**:
   - Write clear commit messages
   - Follow PEP 8 style guide for Python code
   - Include tests for new features
   - Update documentation as needed
4. **Accomplish TODOs**: refer to the [TODO.md](TODO.md) file for a list of features that need to be implemented.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

Contact me with any questions at sean@lmsystems.ai

---
Made by LM Systems - Building the future of Shareable Graphs
