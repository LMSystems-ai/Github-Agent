1. There is currently no loop between the react agent and aider node. Instead, it's just a single step execution. what it should be instead is: a loop between the 2 nodes that keeps going until the task is accomplished given how many steps were given by the react agent.

2. It'd be nice to have a research subgraph with a perplexity "pro" sort of CoT to think through queries and return research analysis about it.
