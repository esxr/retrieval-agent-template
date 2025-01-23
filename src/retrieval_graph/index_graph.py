"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""
import os
import json
import time
from typing import Optional, Sequence
from langchain_community.utilities import ApifyWrapper
from langchain_community.document_loaders import ApifyDatasetLoader
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph
from urllib.parse import urlparse

from retrieval_graph import retrieval
from retrieval_graph.crawler import WebCrawler
from retrieval_graph.configuration import IndexConfiguration
from retrieval_graph.state import IndexState

def ensure_docs_have_user_id(
    docs: Sequence[Document], config: RunnableConfig
) -> list[Document]:
    """Ensure that all documents have a user_id in their metadata.

        docs (Sequence[Document]): A sequence of Document objects to process.
        config (RunnableConfig): A configuration object containing the user_id.

    Returns:
        list[Document]: A new list of Document objects with updated metadata.
    """
    user_id = config["configurable"]["user_id"]
    return [
        Document(
            page_content=doc.page_content, metadata={**doc.metadata, "user_id": user_id}
        )
        for doc in docs
    ]

def get_file_content(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
async def crawl(tenant: str, starter_urls: list, hops: int):
    allowed_domains = set(urlparse(url).netloc for url in starter_urls)
    crawler = WebCrawler(starter_urls, hops, allowed_domains, tenant)
    await crawler.crawl()
    return [
        Document(page_content=get_file_content(page["local_filepath"]), metadata={"url": page["url"]})
        for page in crawler.crawled_pages
    ]

def apify_crawl(configuration: IndexConfiguration):
    tenant = configuration.user_id
    starter_urls = [{"url": url} for url in configuration.parse_starter_urls()]
    dataset_id = configuration.apify_dataset_id

    if dataset_id:
        loader = ApifyDatasetLoader(
            dataset_id=dataset_id,
            dataset_mapping_function=lambda item: Document(
                page_content=item.get('text') or "", metadata={"url": item["url"]}
            ),
        )
    else:
        apify = ApifyWrapper()
        loader = apify.call_actor(
            actor_id="apify/website-content-crawler",
            run_input={
                "startUrls": starter_urls,
                # "saveHtml": True,
                # "htmlTransformer": "none"
            },
            dataset_mapping_function=lambda item: Document(
                page_content=item["text"] or "", metadata={"url": item["url"]}
            ),
        )
        print(f"Site: {tenant} crawled and loaded into Apify dataset: {loader.dataset_id}")

    return loader.load()

def chunk_documents(docs, batch_size):
    """Chunk documents into smaller batches."""
    for i in range(0, len(docs), batch_size):
        yield docs[i:i + batch_size]

async def index_docs(
    state: IndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
    """Asynchronously index documents in the given state using the configured retriever.

    This function takes the documents from the state, ensures they have a user ID,
    adds them to the retriever's index, and then signals for the documents to be
    deleted from the state. In addition if the user has provided a list of URLs to crawl,
    the function will crawl the URLs and index the crawled documents.

    Args:
        state (IndexState): The current state containing documents and retriever.
        config (Optional[RunnableConfig]): Configuration for the indexing process.r
    """
    if not config:
        raise ValueError("Configuration required to run index_docs.")
    with retrieval.make_retriever(config) as retriever:
        configuration = IndexConfiguration.from_runnable_config(config)
        if not state.docs and (configuration.starter_urls or configuration.apify_dataset_id):
            print(f"starting crawl ...")
            crawled_docs = apify_crawl(configuration)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            state.docs = text_splitter.split_documents(crawled_docs)
        stamped_docs = ensure_docs_have_user_id(state.docs, config)
        batch_size = configuration.batch_size
        for i, batch in enumerate(chunk_documents(stamped_docs, batch_size)):
            if configuration.retriever_provider == "milvus":
                retriever.add_documents(batch)
            else:
                await retriever.aadd_documents(batch)
            # Sleep only if there are more batches to process
            if i < (len(stamped_docs) // batch_size):
                time.sleep(60)
    return {"docs": "delete"}

builder = StateGraph(IndexState, config_schema=IndexConfiguration)
builder.add_node(index_docs)
builder.add_edge("__start__", "index_docs")
# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
graph = builder.compile()
graph.name = "IndexGraph"
