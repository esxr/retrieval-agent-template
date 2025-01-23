# custom_docs_ingester.py

import os
import re
from typing import Optional, List

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    UnstructuredMarkdownLoader,
    UnstructuredCSVLoader,
    UnstructuredImageLoader
)

class CustomDocsIngester:
    """
    Ingestion class that loads and processes custom documents from a directory.
    
    Attributes:
        directory_path (str): The file-system path to the directory containing docs.
        chunk_size (int): The chunk size used when splitting large documents.
        chunk_overlap (int): The overlap size to use between text chunks.
        user_id (Optional[str]): (Optional) A user_id to embed in each document's metadata.
    """

    def __init__(
        self,
        directory_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        user_id: Optional[str] = None,
    ):
        """
        Args:
            directory_path (str): Path to the directory containing documents to ingest.
            chunk_size (int): The chunk size for text splitting.
            chunk_overlap (int): The overlap size for text splitting.
            user_id (Optional[str]): An optional user identifier.
        """
        self.directory_path = directory_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.user_id = user_id

    def _clean_text(self, text: str) -> str:
        """Cleans the input text by removing unnecessary characters and whitespace."""
        if not text:
            return ""
        # Collapse multiple whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Optionally remove anything that's not a typical text char
        allowed_chars_regex = r'[^a-zA-Z0-9\s.,!?;:\-–_()\[\]\'"“”‘’]'
        text = re.sub(allowed_chars_regex, '', text)
        return text

    def _chunk_text_pdf(self, text: str) -> List[str]:
        """Chunking strategy for PDFs: Uses recursive splitting for continuous text."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        return splitter.split_text(text)

    def _chunk_text_markdown(self, text: str) -> List[str]:
        """
        Chunking strategy for Markdown: Splits based on heading structure.
        Make sure each entry in `headers_to_split_on` is a (regex_pattern, section_name) tuple.
        """
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                (r"^# ", "Header 1"),
                (r"^## ", "Header 2"),
                (r"^### ", "Header 3"),
                (r"^#### ", "Header 4"),
                (r"^##### ", "Header 5"),
                (r"^###### ", "Header 6"),
            ]
        )
        return splitter.split_text(text)

    def _load_documents(self) -> List[Document]:
        """Loads all documents from the specified directory using appropriate loaders."""
        documents = []
        for filename in os.listdir(self.directory_path):
            file_path = os.path.join(self.directory_path, filename)

            loader = None
            if filename.lower().endswith(".pdf"):
                loader = PDFPlumberLoader(file_path, extract_images=True)
            elif filename.lower().endswith(".md"):
                loader = UnstructuredMarkdownLoader(file_path)
            elif filename.lower().endswith(".csv"):
                loader = UnstructuredCSVLoader(file_path)
            else:
                print(f"Unsupported file type: {filename}")
                continue

            if loader is not None:
                docs = loader.load()
                documents.extend(docs)
        return documents

    def _process_images_from_pdf(
        self, doc: Document, final_docs: List[Document]
    ) -> None:
        """Extracts and processes images from PDF documents if stored in metadata."""
        if 'images' in doc.metadata:
            for image_path in doc.metadata['images']:
                image_loader = UnstructuredImageLoader(image_path)
                image_docs = image_loader.load()
                for image_doc in image_docs:
                    # Always convert to string
                    cleaned_text = self._clean_text(image_doc.page_content or "")
                    final_docs.append(
                        Document(
                            page_content=cleaned_text,
                            metadata=image_doc.metadata
                        )
                    )

    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Cleans and chunks the loaded documents based on file types."""
        chunked_documents = []
        for doc in documents:
            raw_text = doc.page_content or ""
            cleaned_text = self._clean_text(raw_text)

            # Decide chunking strategy by file extension in doc.metadata.source
            source_path = (doc.metadata.get('source') or "").lower()
            if source_path.endswith(".pdf"):
                chunks = self._chunk_text_pdf(cleaned_text)
            elif source_path.endswith(".md"):
                chunks = self._chunk_text_markdown(cleaned_text)
            elif source_path.endswith(".csv"):
                chunks = self._chunk_text_pdf(cleaned_text)  # Fallback approach
            else:
                # If unsupported or unknown extension, skip
                continue

            # Create new chunked Document objects
            for chunk in chunks:
                # Double-check chunk is a string
                if not isinstance(chunk, str):
                    chunk = str(chunk)

                meta_copy = doc.metadata.copy()
                if self.user_id:
                    meta_copy["user_id"] = self.user_id

                # Now create a Document with a guaranteed string page_content
                chunked_documents.append(
                    Document(page_content=chunk, metadata=meta_copy)
                )

        return chunked_documents

    async def ingest_docs(self) -> List[Document]:
        """
        Main method to load, clean, chunk, and optionally process images from the docs.
        Returns a list of chunked Document objects, ready for indexing in a vector store.
        """
        # 1. Load raw documents
        raw_docs = self._load_documents()
        print(f"Loaded {len(raw_docs)} raw documents from {self.directory_path}.")

        # 2. Chunk documents
        chunked_docs = self._chunk_documents(raw_docs)
        print(f"Chunked into {len(chunked_docs)} total pieces.")

        # 3. Optionally handle images for PDFs
        final_docs = []
        for d in chunked_docs:
            final_docs.append(d)
            # If doc has images, process them
            self._process_images_from_pdf(d, final_docs)

        print(f"Final docs (including PDF images): {len(final_docs)}")
        return final_docs
