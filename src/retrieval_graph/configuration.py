"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config

from retrieval_graph import prompts


@dataclass(kw_only=True)
class CommonConfiguration:
    """Configuration class for indexing and retrieval operations.

    This class defines the parameters needed for both configuring the index and
    retrieval processes, including tenant identification, embedding model selection,
    retriever provider choice, and search parameters.
    """

    user_id: str = field(metadata={"description": "Unique identifier for the user."})

    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = field(
        default="openai/text-embedding-3-large",
        metadata={
            "description": "Name of the embedding model to use. Must be a valid embedding model name."
        },
    )

    retriever_provider: Annotated[
        Literal["elastic", "elastic-local", "pinecone", "milvus","chromadb"],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = field(
        default="milvus",
        metadata={
            "description": "The vector store provider to use for retrieval. Options are 'elastic', 'pinecone', 'mongodb', or, 'milvus'."
        },
    )

    search_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "Additional keyword arguments to pass to the search function of the retriever."
        },
    )


    collection_name: str = field(
        default="jira_issues",
        metadata={"description": "Collection/Index name to store Jira issues."},
    )

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        """Create a CommonConfiguration instance from a RunnableConfig object.

        Args:
            cls (Type[T]): The class itself.
            config (Optional[RunnableConfig]): The configuration object to use.

        Returns:
            T: An instance of CommonConfiguration with the specified configuration.
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
    
T = TypeVar("T", bound=CommonConfiguration)

@dataclass(kw_only=True)
class IndexConfiguration(CommonConfiguration):
    """Crawler configuration class for indexing operations."""

    starter_urls: str = field(
        default="",
        metadata={
            "description": "Comma-separated string of starter URLs to crawl for indexing web pages."
        },
    )

    hops: int = field(
        default=2,
        metadata={
            "description": "Maximum number of hops to traverse pages linked to the starter URLs."
        }, 
    )

    batch_size: int = field(
        default=50,
        metadata={
            "description": "Number of documents to index in a single batch."
        },
    )
    
    apify_dataset_id: str = field(
        default="",
        metadata={
            "description": "The Apify dataset ID to use if already crawled and stored on Apify."
        },
    )

    def parse_starter_urls(self) -> list[str]:
        """Parse the starter URLs into a list.

        Returns:
            list[str]: A list of URLs parsed from the comma-separated string.
        """
        return [url.strip() for url in self.starter_urls.split(",") if url.strip()]

@dataclass(kw_only=True)
class JiraConfiguration(CommonConfiguration):
    """Configuration class for indexing and retrieval operations.

    This class defines the parameters needed for configuring the indexing and
    retrieval processes, including user identification, embedding model selection,
    retriever provider choice, and search parameters.
    """

    jira_email: str = field(
        default="",
        metadata={"description": "Jira account email for authentication."},
    )

    jira_api_token: str = field(
        default="",
        metadata={"description": "Jira API token."},
    )

    jira_site: str = field(
        default="",
        metadata={"description": "Base URL of your Jira site, e.g. https://yourcompany.atlassian.net"},
    )

    jql: str = field(
        default="",
        metadata={"description": "Default JQL query to fetch Jira issues. eg. 'project=AGAILEP'"},
    )


@dataclass(kw_only=True)
class Configuration(CommonConfiguration):
    """The configuration for the agent."""

    # optional: index location
    alternate_milvus_uri: Optional[str] = field(
        default="",
        metadata={
            "description": "If you want to use one of the already available indexes, provide the file location here."
        },
    )

    response_system_prompt: str = field(
        default=prompts.RESPONSE_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for generating responses."},
    )

    response_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o",
        metadata={
            "description": "The language model used for generating responses. Should be in the form: provider/model-name."
        },
    )

    query_system_prompt: str = field(
        default=prompts.QUERY_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for processing and refining queries."
        },
    )

    query_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o",
        metadata={
            "description": "The language model used for processing and refining queries. Should be in the form: provider/model-name."
        },
    )
