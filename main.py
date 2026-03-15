"""
main.py - RAG Agent (Query Mode)

This file runs the interactive Q&A loop using the RAG agent.
Ingestion must be run FIRST as a separate step.

HOW TO USE:
    Step 1 (one-time): python ingestion.py    # Ingest PDFs into ChromaDB
    Step 2 (anytime):  python main.py         # Ask questions

To re-ingest (new PDFs or changed settings):
    1. Delete the 'chroma_db/' folder
    2. Run: python ingestion.py
    3. Run: python main.py
"""

import os
from dotenv import load_dotenv

# Load environment variables (OPENAI_API_KEY) from .env file
load_dotenv()

from ingestion import CHROMA_DB_DIR
from rag_agent import query_rag


def main():
    print("=" * 60)
    print("       RAG AI Agent - LangGraph + ChromaDB")
    print("=" * 60)
    print()

    # Check that the vector DB exists (ingestion must be run first)
    if not os.path.exists(CHROMA_DB_DIR):
        print("No vector database found!")
        print()
        print("Run ingestion first:")
        print("  1. Place your PDF files in the 'data/' folder")
        print("  2. Run: python ingestion.py")
        print("  3. Then run: python main.py")
        return

    print(f"Using vector database at '{CHROMA_DB_DIR}/'.\n")

    # ------------------------------------------------------------------
    # RAG Agent - Interactive Q&A Loop
    # ------------------------------------------------------------------
    print("=" * 60)
    print("       RAG Agent Ready - Ask Questions!")
    print("=" * 60)
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print()
        # Run the question through the LangGraph RAG agent
        answer = query_rag(question)
        print(f"\nAssistant: {answer}\n")
        print("-" * 60)
        print()


if __name__ == "__main__":
    main()
