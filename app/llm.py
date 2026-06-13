from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from retriever import search_vector_store

# The LLM — Mistral running locally via Ollama
def get_llm():
    return OllamaLLM(model="mistral")

# The prompt template — this is what gets sent to Mistral
PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question based ONLY on the context below.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question:
{question}

Answer:
"""

def ask_question(question):
    """Full RAG pipeline — retrieve relevant chunks, then generate an answer."""

    print(f"\nQuestion: {question}")
    print("=" * 60)

    # Step 1 — Retrieve the most relevant chunks from ChromaDB
    print("\nStep 1 — Retrieving relevant chunks...")
    results = search_vector_store(question, k=3)

    # Step 2 — Combine chunks into a single context string
    context = "\n\n---\n\n".join([doc.page_content for doc in results])

    # Step 3 — Build the prompt with context + question
    print("\nStep 2 — Building prompt with retrieved context...")
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | get_llm()

    # Step 4 — Send to Mistral and get answer
    print("\nStep 3 — Sending to Mistral. Generating answer...")
    print("(This may take 30-60 seconds on your machine)\n")

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    print("ANSWER:")
    print("-" * 40)
    print(answer)

    return answer


if __name__ == "__main__":
    ask_question("What is RAG and how does it work?")