import re
import requests
import uuid
from typing import Optional, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class JiraIngester:
    """
    An asynchronous ingestion class that fetches Jira issues from the
    specified site and converts them into `Document` objects.

    Attributes:
        jira_site (str): The base URL of the Jira instance (e.g. https://company.atlassian.net).
        jira_email (str): The Jira account email for authentication.
        jira_api_token (str): The Jira API token.
        jql (str): The JQL query used to filter which Jira issues will be fetched.
        max_results (int): Maximum number of issues to fetch at one time.
        chunk_size (int): The chunk size used when splitting large documents.
        chunk_overlap (int): The overlap size to use between text chunks.
        user_id (Optional[str]): (Optional) A user_id to embed in each document's metadata.
    """

    def __init__(
        self,
        jira_site: str,
        jira_email: str,
        jira_api_token: str,
        jql: str,
        max_results: int = 100,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        user_id: Optional[str] = None,
    ):
        """
        Initialize the JiraIngester with Jira site credentials, the JQL query,
        and optionally chunking parameters and a user_id.

        Args:
            jira_site (str): Jira instance base URL.
            jira_email (str): Jira account email.
            jira_api_token (str): Jira API token.
            jql (str): JQL query string to fetch issues.
            max_results (int, optional): Maximum results to retrieve per API call. Defaults to 100.
            chunk_size (int, optional): Chunk size for text splitting. Defaults to 500.
            chunk_overlap (int, optional): Overlap size for text splitting. Defaults to 100.
            user_id (Optional[str], optional): If provided, store user_id in the document metadata. Defaults to None.
        """
        self.jira_site = jira_site
        self.jira_email = jira_email
        self.jira_api_token = jira_api_token
        self.jql = jql
        self.max_results = max_results
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.user_id = user_id

    def _clean_jira_text(self, text: str) -> str:
        """Remove Jira markup, color tags, images, etc."""
        if not text:
            return ""
        text = re.sub(r'\{color:[^}]+\}', '', text)
        text = text.replace('{color}', '')
        text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)
        text = re.sub(r'![^!]*?!', '', text)
        text = re.sub(r'\[(.*?)\|.*?\]', r'\1', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    def _issue_to_document(self, issue_data: dict) -> Document:
        """Convert a single Jira issue (with comments) to a Document object."""
        summary = issue_data.get("summary", "")
        description = issue_data.get("description", "")
        comments = issue_data.get("comments", [])

        comment_texts = []
        for c in comments:
            author = c.get("author", "Unknown")
            body = c.get("body", "")
            comment_texts.append(f"Comment by {author}:\n{body}")

        all_comments = "\n\n".join(comment_texts)
        full_text = (
            f"Summary:\n{summary}\n\n"
            f"Description:\n{description}\n\n"
            f"Comments:\n{all_comments}"
        )

        metadata = {
            "issue_id": issue_data.get("id"),
            "issue_key": issue_data.get("key"),
            "project": issue_data.get("project"),
            "issuetype": issue_data.get("issuetype"),
            "status": issue_data.get("status"),
        }
        if self.user_id:
            metadata["user_id"] = self.user_id

        return Document(page_content=full_text, metadata=metadata)

    def _fetch_issues_with_comments(self) -> List[dict]:
        """Fetch issues from Jira using JQL, including comments."""
        print(f"Fetching issues from Jira with JQL: {self.jql}")
        url = f"{self.jira_site}/rest/api/2/search"
        auth = (self.jira_email, self.jira_api_token)
        headers = {"Content-Type": "application/json"}

        start_at = 0
        all_issues = []
        while True:
            params = {
                "jql": self.jql,
                "startAt": start_at,
                "maxResults": self.max_results,
                "fields": "summary,description,status,issuetype,project,comment",
            }
            response = requests.get(url, headers=headers, auth=auth, params=params)
            response.raise_for_status()
            data = response.json()
            issues = data.get("issues", [])
            if not issues:
                break

            for issue in issues:
                fields = issue.get("fields", {})
                comments = fields.get("comment", {}).get("comments", [])
                cleaned_summary = self._clean_jira_text(fields.get("summary", ""))
                cleaned_description = self._clean_jira_text(fields.get("description", ""))

                # Clean each comment
                cleaned_comments = []
                for c in comments:
                    author = c.get("author", {}).get("displayName", "")
                    body = self._clean_jira_text(c.get("body", ""))
                    created = c.get("created")
                    cleaned_comments.append({
                        "author": author,
                        "body": body,
                        "created": created
                    })

                issue_data = {
                    "id": issue.get("id"),
                    "key": issue.get("key"),
                    "summary": cleaned_summary,
                    "description": cleaned_description,
                    "status": fields.get("status", {}).get("name"),
                    "issuetype": fields.get("issuetype", {}).get("name"),
                    "project": fields.get("project", {}).get("key"),
                    "comments": cleaned_comments
                }
                all_issues.append(issue_data)

            if len(issues) < self.max_results:
                break
            start_at += self.max_results

        return all_issues

    def _chunk_documents(self, docs: List[Document]) -> List[Document]:
        """Split large documents into smaller chunks using RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        chunked_docs = []
        for doc in docs:
            chunks = splitter.split_text(doc.page_content)
            for chunk in chunks:
                chunked_docs.append(
                    Document(page_content=chunk, metadata=doc.metadata.copy())
                )
        return chunked_docs

    async def ingest_issues(self) -> List[Document]:
        """
        The main method to fetch Jira issues, clean and chunk them, and
        return a list of LangChain Document objects. You can store these
        docs in a vector store or pass them to another pipeline node.

        Returns:
            List[Document]: A list of Document objects containing the text of Jira issues.
        """
        # 1. Fetch issues from Jira
        issues_data = self._fetch_issues_with_comments()
        print(f"Fetched {len(issues_data)} issues from Jira.")

        # 2. Convert issues to Documents
        docs = [self._issue_to_document(i) for i in issues_data]
        print(f"Converted to {len(docs)} Document objects.")

        # 3. Chunk Documents if necessary
        chunked_docs = self._chunk_documents(docs)
        print(f"Chunked into {len(chunked_docs)} total pieces.")

        return chunked_docs
    
