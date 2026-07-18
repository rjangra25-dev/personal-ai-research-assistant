import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from retriever import search_vector_store

load_dotenv()  # reads GROQ_API_KEY from your .env

# The LLM — now Groq's hosted model instead of local Mistral
def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

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

    print("\nStep 1 — Retrieving relevant chunks...")
    results = search_vector_store(question, k=3)

    context = "\n\n---\n\n".join([doc.page_content for doc in results])

    print("\nStep 2 — Building prompt with retrieved context...")
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | get_llm()

    print("\nStep 3 — Sending to Groq. Generating answer...")
    answer = chain.invoke({
        "context": context,
        "question": question
    })

    print("ANSWER:")
    print("-" * 40)
    print(answer.content)   # ChatGroq returns a message object; .content is the text

    return answer.content

if __name__ == "__main__":
    ask_question("What is RAG and how does it work?")
