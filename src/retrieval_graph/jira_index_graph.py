# index_graph.py

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from retrieval_graph.configuration import JiraConfiguration
from retrieval_graph.state import IndexState
from retrieval_graph.jira import JiraIngester  # Import the JiraIngester class
from retrieval_graph.retrieval import make_retriever
# A node function that ingests Jira issues and stores them
async def index_jira_issues(state: IndexState, *, config: RunnableConfig = None) -> dict[str, str]:
    if not config:
        raise ValueError("Configuration required to run index_jira_issues.")

    configuration = JiraConfiguration.from_runnable_config(config)

    # Instantiate the JiraIngester
    ingester = JiraIngester(
        jira_site=configuration.jira_site,
        jira_email=configuration.jira_email,
        jira_api_token=configuration.jira_api_token,
        jql=configuration.jql,
        max_results=100,
        chunk_size=500,
        chunk_overlap=100,
        user_id=configuration.user_id,
    )

    # Fetch & convert the issues into chunked documents
    docs = await ingester.ingest_issues()

    # Optionally, store these docs in your vector store using your existing logic:
    with make_retriever(config) as retriever:
        retriever.vectorstore.add_documents(docs)

    # Return something that signals how to update the state
    return {"docs": "delete"}


# Build the graph
builder = StateGraph(IndexState, config_schema=JiraConfiguration)
builder.add_node(index_jira_issues)
builder.add_edge("__start__", "index_jira_issues")

graph = builder.compile()
graph.name = "JiraIndexGraph"
