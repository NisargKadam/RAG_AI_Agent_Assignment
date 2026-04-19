"""
ingestion.py - Build the local vector database for the RAG project.

This module handles the ingestion side of the pipeline:
1. Load PDF files from `data/`
2. Split each PDF page into smaller chunks
3. Create embeddings for the chunks
4. Store everything in a local Chroma database

HOW TO RUN:
    python ingestion.py

STUDENT-FRIENDLY IDEA:
Think of ingestion as "preparing the library" before the assistant can answer
questions. We are not answering anything here yet. We are only turning the PDF
documents into a searchable knowledge base.
"""

from __future__ import annotations

import os
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================================================
# CONFIGURATION - students can experiment here
# ==========================================================================

DATA_DIR = "data"
CHROMA_DB_DIR = "chroma_db"

# Chunking controls how much text goes into each searchable piece.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding model used to convert text into vectors.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def print_step(title: str) -> None:
    """Print a clean section header for the console output."""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def ensure_data_directory(data_dir: str) -> bool:
    """
    Make sure the data directory exists.

    If it does not exist yet, create it and tell the user what to do next.
    """
    if os.path.exists(data_dir):
        return True

    os.makedirs(data_dir)
    print(f"Created '{data_dir}/'. Add PDF files there, then run ingestion again.")
    return False


def get_pdf_paths(data_dir: str) -> list[str]:
    """Return all PDF file paths inside the data directory."""
    pdf_filenames = sorted(
        filename for filename in os.listdir(data_dir) if filename.lower().endswith(".pdf")
    )
    return [os.path.join(data_dir, filename) for filename in pdf_filenames]


def load_pdf_documents(pdf_paths: Iterable[str]) -> list[Document]:
    """
    Load every PDF page as a LangChain Document.

    Each page becomes one Document with:
    - page_content: extracted text
    - metadata: source file path and page number
    """
    documents: list[Document] = []

    for pdf_path in pdf_paths:
        pdf_name = os.path.basename(pdf_path)
        print(f"Loading PDF: {pdf_name}")
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

    return documents


def load_source_documents(data_dir: str = DATA_DIR) -> list[Document]:
    """
    Load all PDF documents from the project data folder.

    This function is intentionally small and readable so students can follow
    the control flow without getting lost in implementation details.
    """
    if not ensure_data_directory(data_dir):
        return []

    pdf_paths = get_pdf_paths(data_dir)
    if not pdf_paths:
        print(f"No PDF files found in '{data_dir}/'. Add some files and try again.")
        return []

    documents = load_pdf_documents(pdf_paths)
    print(f"Loaded {len(documents)} page(s) from {len(pdf_paths)} PDF file(s).")
    return documents


def build_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create the text splitter used during chunking.

    We use RecursiveCharacterTextSplitter because it is one of the clearest and
    most common starting points for RAG projects.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split loaded pages into smaller overlapping chunks.

    Why this matters:
    - Smaller chunks make retrieval more precise
    - Overlap helps preserve context across chunk boundaries
    """
    text_splitter = build_text_splitter()
    chunks = text_splitter.split_documents(documents)

    print(
        f"Created {len(chunks)} chunk(s) "
        f"(chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})."
    )
    return chunks


def build_embedding_model() -> HuggingFaceEmbeddings:
    """Create the embedding model used by both ingestion and retrieval."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def create_vector_store(chunks: list[Document]) -> Chroma:
    """
    Store chunk embeddings in a local Chroma database.

    Chroma keeps:
    - the vector embeddings
    - the original chunk text
    - the metadata needed for citations
    """
    embeddings = build_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )

    print(f"Saved vector database to '{CHROMA_DB_DIR}/'.")
    return vector_store


def run_ingestion() -> Chroma | None:
    """
    Run the full ingestion pipeline from start to finish.

    Returns:
        The Chroma vector store if ingestion succeeds, otherwise None.
    """
    print_step("STEP 1: LOAD PDF DOCUMENTS")
    documents = load_source_documents(DATA_DIR)
    if not documents:
        return None

    print_step("STEP 2: SPLIT DOCUMENTS INTO CHUNKS")
    chunks = split_documents(documents)

    print_step("STEP 3: CREATE THE CHROMA VECTOR DATABASE")
    vector_store = create_vector_store(chunks)

    print()
    print("Ingestion complete. Your documents are ready for retrieval.")
    return vector_store


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    run_ingestion()
