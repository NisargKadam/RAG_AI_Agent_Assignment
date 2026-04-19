# RAG AI Agent

A simple, student-friendly Retrieval-Augmented Generation (RAG) project built with **LangGraph**, **ChromaDB**, and **local HuggingFace embeddings**.

The goal of this project is not only to run a RAG agent, but also to help students understand how a RAG pipeline is structured so they can later modify it and build their own versions.

## What This Project Teaches

This project is split into two clear parts:

1. `ingestion.py`
   Turns PDF files into a searchable vector database.
2. `rag_agent.py`
   Uses a LangGraph workflow to retrieve useful chunks and generate an answer.
3. `main.py`
   Runs a simple interactive CLI so users can ask questions.

## Architecture

```text
PDF files
  -> ingestion.py
  -> chunks + embeddings
  -> Chroma vector database

User question
  -> LangGraph RAG workflow
  -> retrieve relevant chunks
  -> generate grounded answer
  -> final response with sources
```

## Project Flow

### 1. Ingestion Phase

`ingestion.py` prepares the knowledge base.

It does four things:

1. Load PDF files from `data/`
2. Split PDF pages into smaller chunks
3. Create embeddings for those chunks using a local HuggingFace model
4. Save the chunks and embeddings in `chroma_db/`

Important idea for students:

Ingestion is like preparing a library before the assistant can answer questions.

### 2. RAG Agent Phase

`rag_agent.py` defines a simple LangGraph workflow:

```text
START -> retrieve -> generate -> END
```

The graph has two main nodes:

1. `retrieve`
   Searches ChromaDB for the most relevant chunks
2. `generate`
   Sends the question and retrieved context to the LLM to produce the answer

### 3. Application Phase

`main.py` is the interactive command-line app.

It:

1. Checks whether the vector database exists
2. Starts a question-answer loop
3. Sends each user question through the LangGraph RAG workflow

## Project Structure

```text
RAG_AI_Agent/
|-- main.py            # Interactive CLI app
|-- ingestion.py       # PDF loading, chunking, embeddings, Chroma storage
|-- rag_agent.py       # LangGraph retrieve + generate workflow
|-- requirements.txt   # Python dependencies
|-- .env.example       # Environment variable template
|-- data/              # Put your PDF files here
`-- chroma_db/         # Local vector database created after ingestion
```

## Setup

### 1. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Create your `.env` file

On Windows PowerShell:

```powershell
copy .env.example .env
```

Then open `.env` and add your `OPENAI_API_KEY`.

Note:

- The LLM answer generation uses OpenAI.
- The embedding model is local HuggingFace, so embeddings do not need an OpenAI embedding API call.

### 4. Add PDFs

Put your PDF files into the `data/` folder.

## How To Run

### Step 1: Build the vector database

```powershell
python ingestion.py
```

This reads PDFs from `data/` and creates the local Chroma vector database in `chroma_db/`.

### Step 2: Start the RAG app

```powershell
python main.py
```

Then ask questions such as:

```text
What is this document about?
Summarize the main topics in the PDFs.
What does the document say about X?
```

To exit:

```text
quit
```

## Rebuilding the Vector Database

If you change the PDFs or change chunking settings in `ingestion.py`, rebuild the vector database.

On Windows PowerShell:

```powershell
Remove-Item -Recurse -Force .\chroma_db
python ingestion.py
```

Then run:

```powershell
python main.py
```

## Student-Friendly Customization Points

Students can safely experiment with these values first:

In `ingestion.py`:

- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `EMBEDDING_MODEL`

In `rag_agent.py`:

- `TOP_K`
- `LLM_MODEL`
- `TEMPERATURE`
- `SYSTEM_PROMPT`

These are good starting points for assignments because they let students see how retrieval and generation behavior changes without needing to redesign the whole project.

## Suggested Learning Path

1. Read `ingestion.py` to understand how documents become searchable.
2. Read `rag_agent.py` to understand the LangGraph workflow.
3. Run `python ingestion.py`
4. Run `python main.py`
5. Change one setting at a time and observe the result.

## Current LangGraph Design

This project already uses LangGraph in a simple and teachable way.

The current graph is:

```text
START -> retrieve -> generate -> END
```

That makes it a strong base for future student extensions such as:

- adding a router node
- adding query rewriting
- adding answer checking
- adding conversation memory
- switching retrieval strategies

## Quick Start Commands

If you just want the minimum commands on Windows PowerShell:

```powershell
cd "c:\Users\nisar\Documents\AI Builder 3\Projects\RAG_AI_Agent"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python ingestion.py
python main.py
```
