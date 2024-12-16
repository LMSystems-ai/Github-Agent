from langgraph_sdk import get_sync_client

client = get_sync_client(url="",
        api_key="")
server = client.assistants.search()
print(server)