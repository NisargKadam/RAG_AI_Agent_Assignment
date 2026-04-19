"""
rag_agent.py - LangGraph-based RAG agent.

This module implements the answering side of the project.

High-level flow:
    user question
        -> retrieve relevant chunks from Chroma
        -> generate an answer grounded in those chunks
        -> return the final response with sources

This file is intentionally organized around small, named functions so students
can understand what each part of the agent is responsible for.
"""

from __future__ import annotations

import os
from typing import TypedDict

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from ingestion import CHROMA_DB_DIR, EMBEDDING_MODEL


# ==========================================================================
# CONFIGURATION - students can experiment here
# ==========================================================================

TOP_K = 4
LLM_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0

SYSTEM_PROMPT = """You are a helpful assistant answering questions from retrieved documents.

Follow these rules:
1. Use only the retrieved context.
2. If the context is not enough, say that clearly.
3. End with a short "Sources:" section.

Retrieved context:
{context}

Retrieved sources:
{sources}
"""


# ==========================================================================
# LANGGRAPH STATE
# ==========================================================================

class RAGState(TypedDict):
    question: str
    retrieved_documents: list[Document]
    context: str
    sources: str
    answer: str


def build_embedding_model() -> HuggingFaceEmbeddings:
    """Create the embedding model used to query the vector database."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_vector_store() -> Chroma:
    """Open the existing Chroma database from disk."""
    if not os.path.exists(CHROMA_DB_DIR):
        raise FileNotFoundError(
            f"Vector database '{CHROMA_DB_DIR}/' was not found. Run ingestion first."
        )

    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=build_embedding_model(),
    )


def format_context(documents: list[Document]) -> str:
    """Join retrieved chunks into one context string for the prompt."""
    if not documents:
        return "No relevant context was retrieved."

    return "\n\n---\n\n".join(document.page_content for document in documents)


def format_sources(documents: list[Document]) -> str:
    """Convert document metadata into a readable citation list."""
    if not documents:
        return "No sources retrieved."

    formatted_sources = []
    for index, document in enumerate(documents, start=1):
        source_file = document.metadata.get("source", "Unknown source")
        page_number = document.metadata.get("page", "?")

        if isinstance(page_number, int):
            page_label = page_number + 1
        else:
            page_label = page_number

        formatted_sources.append(f"[{index}] {source_file} (Page {page_label})")

    return "\n".join(formatted_sources)


# ==========================================================================
# LANGGRAPH NODES
# ==========================================================================

def retrieve_node(state: RAGState) -> dict:
    """
    Retrieve the most relevant chunks for the user's question.

    This is the retrieval step in Retrieval-Augmented Generation.
    """
    question = state["question"]
    vector_store = load_vector_store()
    retrieved_documents = vector_store.similarity_search(question, k=TOP_K)

    context = format_context(retrieved_documents)
    sources = format_sources(retrieved_documents)

    print(f"[Retrieve] Found {len(retrieved_documents)} chunk(s) for: {question}")
    return {
        "retrieved_documents": retrieved_documents,
        "context": context,
        "sources": sources,
    }


def generate_node(state: RAGState) -> dict:
    """
    Generate the final answer using the retrieved context.

    This is the generation step in Retrieval-Augmented Generation.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)
    chain = prompt | llm

    response = chain.invoke(
        {
            "question": state["question"],
            "context": state["context"],
            "sources": state["sources"],
        }
    )

    print("[Generate] Answer created.")
    return {"answer": response.content}


# ==========================================================================
# GRAPH CONSTRUCTION
# ==========================================================================

def build_rag_graph():
    """
    Build and compile the LangGraph workflow.

    Graph structure:
        START -> retrieve -> generate -> END
    """
    graph_builder = StateGraph(RAGState)

    graph_builder.add_node("retrieve", retrieve_node)
    graph_builder.add_node("generate", generate_node)

    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", END)

    return graph_builder.compile()


def query_rag(question: str) -> str:
    """Run one question through the LangGraph RAG workflow."""
    graph = build_rag_graph()
    result = graph.invoke(
        {
            "question": question,
            "retrieved_documents": [],
            "context": "",
            "sources": "",
            "answer": "",
        }
    )
    return result["answer"]


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    demo_answer = query_rag("What is this document about?")
    print()
    print(demo_answer)
