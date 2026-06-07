import fitz  # this is PyMuPDF — fitz is just its internal name
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(file_path):
    """Open a PDF and extract all text from every page."""
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def split_into_chunks(text):
    """Split text into 500-character chunks with 50-character overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks

if __name__ == "__main__":
    # Path to your PDF
    pdf_path = "data/AI_and_RAG_Concepts.pdf"

    print("Step 1 — Loading PDF...")
    text = load_pdf(pdf_path)
    print(f"Total characters extracted: {len(text)}")

    print("\nStep 2 — Splitting into chunks...")
    chunks = split_into_chunks(text)
    print(f"Total chunks created: {len(chunks)}")

    print("\nStep 3 — First 5 chunks:")
    print("=" * 60)
    for i, chunk in enumerate(chunks[:5]):
        print(f"\nCHUNK {i+1} ({len(chunk)} characters):")
        print("-" * 40)
        print(chunk)
        print()