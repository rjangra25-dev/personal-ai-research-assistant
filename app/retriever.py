from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ingest import load_pdf, split_into_chunks
import os

# Define the folder where ChromaDB will save vectors on your disk
CHROMA_PATH = "chroma_db"
PDF_PATH = "data/AI_and_RAG_Concepts.pdf"

def get_embedding_function():
    """Set up nomic-embed-text running locally via Ollama."""
    return OllamaEmbeddings(model="nomic-embed-text")

def build_vector_store():
    """Load PDF, chunk it, embed all chunks, store in ChromaDB."""
    
    print("Step 1 — Loading and chunking PDF...")
    text = load_pdf(PDF_PATH)
    chunks = split_into_chunks(text)
    print(f"Total chunks to embed: {len(chunks)}")

    print("\nStep 2 — Embedding chunks and storing in ChromaDB...")
    print("This may take 2 to 5 minutes on your machine. Please wait...")

    embeddings = get_embedding_function()

    # Create ChromaDB vector store from chunks
    # This embeds every chunk and saves everything to the chroma_db folder
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"\nDone! {len(chunks)} chunks embedded and stored in ChromaDB.")
    print(f"Vector store saved to: {CHROMA_PATH}/")
    return vector_store

def search_vector_store(query, k=3):
    """Search ChromaDB for the most relevant chunks for a given query."""
    
    print(f"\nSearching for: '{query}'")
    embeddings = get_embedding_function()

    # Load the existing vector store from disk
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # Find the k most similar chunks
    results = vector_store.similarity_search(query, k=k)

    print(f"\nTop {k} most relevant chunks found:")
    print("=" * 60)
    for i, doc in enumerate(results):
        print(f"\nRESULT {i+1}:")
        print("-" * 40)
        print(doc.page_content)

    return results

if __name__ == "__main__":
    
    
    if not os.path.exists(CHROMA_PATH):
        build_vector_store()       # only runs if chroma_db/ doesn't exist yet
    else:
        print("Vector store already exists. Skipping build.")
    
     # Step 2 — Test a search query
    search_vector_store("What is RAG and how does it work?")