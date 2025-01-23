# custom_docs.py

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from retrieval_graph.configuration import CustomDocsConfiguration
from retrieval_graph.state import IndexState
from retrieval_graph.retrieval import make_retriever
from retrieval_graph.custom_doc_ingester import CustomDocsIngester


# A node function that ingests custom documents from a directory and stores them
async def index_custom_docs(state: IndexState, *, config: RunnableConfig = None) -> dict[str, str]:
    if not config:
        raise ValueError("Configuration required to run index_custom_docs.")

    # Parse the config for custom docs
    configuration = CustomDocsConfiguration.from_runnable_config(config)

    # Instantiate the ingester
    ingester = CustomDocsIngester(
        directory_path=configuration.directory_path,
        chunk_size=500,   # or configuration.chunk_size if you add it
        chunk_overlap=100,
        user_id=configuration.user_id,
    )

    # Ingest and chunk docs
    docs = await ingester.ingest_docs()

    # Store in vector store
    with make_retriever(config) as retriever:
        retriever.vectorstore.add_documents(docs)

    # Return an indication if you want to clear out docs in state
    return {"docs": "delete"}


# Build the graph
builder = StateGraph(IndexState, config_schema=CustomDocsConfiguration)
builder.add_node(index_custom_docs)
builder.add_edge("__start__", "index_custom_docs")

graph = builder.compile()
graph.name = "CustomDocsIndexGraph"
