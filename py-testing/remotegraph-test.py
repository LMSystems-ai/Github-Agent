from langgraph.pregel.remote import RemoteGraph

# Initialize the RemoteGraph
remote_graph = RemoteGraph(
    "engineer",
    url="",
    api_key=""
)

# Fetch the graph
graph = remote_graph.get_graph()

# Print graph nodes and edges
print("Nodes:", graph.nodes)
print("Edges:", graph.edges)
