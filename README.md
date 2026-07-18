# Personal AI Research Assistant

A local AI research assistant that lets you chat with your own documents. It uses
retrieval-augmented generation (RAG) to answer questions grounded in PDFs you provide,
and an agent layer that decides when to search your documents, when to do a calculation,
and when to just answer directly.

Built with LangChain, ChromaDB, HuggingFace embeddings, Groq, and Streamlit.

## What it does

- **Chat with your documents** — drop a PDF into `data/`, and the assistant retrieves
  relevant passages and answers based on them, rather than guessing.
- **Agent tool-routing** — the model chooses the right tool for each question:
  - `search_documents` — searches your document store for relevant context
  - `calculate` — evaluates arithmetic expressions
  - For general-knowledge questions it answers directly, without using a tool.
- **Web chat interface** — a Streamlit UI to ask questions and see answers.

## How it works

1. **Ingest** — a PDF is loaded, split into overlapping chunks, and embedded using a
   local sentence-transformers model (`all-MiniLM-L6-v2`).
2. **Store** — the embeddings are saved to a local ChromaDB vector store.
3. **Retrieve** — a question is embedded and matched against the store to find the most
   relevant chunks.
4. **Route** — the agent (running on Groq) reads the question and decides which tool,
   if any, to use.
5. **Answer** — the retrieved context (or tool result) is passed back to the model,
   which writes the final grounded answer.

## Structure

- `app/`
  - `main.py` — Streamlit chat interface
  - `ingest.py` — PDF loading and chunking
  - `retriever.py` — embeddings and vector search (ChromaDB)
  - `llm.py` — the basic RAG chain (retrieve → prompt → answer)
  - `agent.py` — the tool-routing agent
- `data/` — put your PDFs here
- `chroma_db/` — the local vector store (generated; not tracked in git)
- `mcp_server/` — planned: exposing the tools over MCP (not yet implemented)

## Setup

1. Create a `.env` file with your Groq API key: `GROQ_API_KEY=your_key_here`
2. Install dependencies: `pip install -r requirements.txt`
3. Build the vector store: `python app/retriever.py`
4. Run: `streamlit run app/main.py`

## Notes

- Embeddings run locally (no API needed for that step). Generation runs on Groq's
  hosted API, so an internet connection and API key are required.
- The vector store is tied to the embedding model. If you change the embedding model,
  delete `chroma_db/` and re-run the ingest step so the vectors match.

## Roadmap

- Conversation memory for follow-up questions
- Expose tools over MCP
- Support document types beyond PDF