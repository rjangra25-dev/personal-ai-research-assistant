# Personal AI Research Assistant

A local AI research assistant that lets you chat with your own PDFs. Uses RAG to answer
questions grounded in your documents, with an agent layer that decides when to search
the documents, when to calculate, and when to answer directly.

Built with Streamlit, ChromaDB, HuggingFace embeddings, and Groq.

## Structure

- `app/` — Core application modules
  - `main.py` — Streamlit app entry point
  - `ingest.py` — Document loading and chunking
  - `retriever.py` — Embedding (HuggingFace) and vector search
  - `llm.py` — Basic RAG chain (retrieve → prompt → answer)
  - `agent.py` — Tool-routing agent (search_documents + calculate)
- `mcp_server/` — A working MCP server exposing the same tools over the Model Context
  Protocol (see "MCP exploration" below)
- `test_client.py` — A small client that connects to and verifies the MCP server
- `data/` — Put your PDFs here
- `chroma_db/` — Auto-created vector store (not tracked in git)

## Setup

1. Create a `.env` file with your Groq API key: `GROQ_API_KEY=your_key_here`
2. Install dependencies: `pip install -r requirements.txt`
3. Build the vector store: `python app/retriever.py`
4. Run: `streamlit run app/main.py`

## How it works

1. A PDF is loaded, chunked, and embedded locally (sentence-transformers) into ChromaDB.
2. A question is embedded and matched against the store to find relevant chunks.
3. The agent (on Groq) reads the question and decides which tool to use — search the
   documents, run a calculation, or answer directly.
4. The tool result is passed back to the model, which writes the final grounded answer.

## MCP exploration

The `mcp_server/` folder contains a working MCP (Model Context Protocol) server that
exposes the same two tools over the protocol, verified with `test_client.py`. The main
app uses the tools directly via LangChain rather than over MCP — this was a deliberate
choice for simplicity and reliability, after building and testing the MCP version. The
server is kept as a demonstration of the protocol.

## Notes

- Embeddings run locally; generation runs on Groq's hosted API (needs a key + internet).
- The vector store is tied to the embedding model. If you change it, delete `chroma_db/`
  and re-run the ingest step.