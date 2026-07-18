import os
from dotenv import load_dotenv
from retriever import search_vector_store
from langchain_groq import ChatGroq
from langchain_core.tools import tool

load_dotenv()

# --- Tool 1: wraps the retriever ---
@tool
def search_documents(query: str) -> str:
    """Search the user's personal documents for information on a topic."""
    results = search_vector_store(query, k=3)
    return "\n\n---\n\n".join([doc.page_content for doc in results])

# --- Tool 2: calculator ---
@tool
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '4200 * 0.15'."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: could not evaluate '{expression}' ({e})"

# --- Set up the model with tools bound (done once, at import) ---
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
tools = [search_documents, calculate]
llm_with_tools = llm.bind_tools(tools)
available_tools = {
    "search_documents": search_documents,
    "calculate": calculate,
}

def run_agent(question: str) -> str:
    """Takes a question, routes through tools if needed, returns the final answer as a string."""

    # Beat 1 — send question + tools to the model
    messages = [("human", question)]
    response = llm_with_tools.invoke(messages)

    # Beat 2 — no tool needed? return the direct answer
    if not response.tool_calls:
        return response.content

    # Keep the model's request in the conversation
    messages.append(response)

    # Beat 3 — run each tool the model asked for
    for call in response.tool_calls:
        chosen_tool = available_tools[call["name"]]
        result = chosen_tool.invoke(call["args"])

        # Beat 4 — hand the result back to the model
        messages.append({
            "role": "tool",
            "content": str(result),
            "tool_call_id": call["id"],
        })

    # Beat 5 — model reads the result, writes the final answer
    final = llm_with_tools.invoke(messages)
    return final.content


if __name__ == "__main__":
    # Still runnable from the terminal for testing
    print(run_agent("What does my document say about RAG?"))
    print(run_agent("What is 4200 * 0.15?"))
    print(run_agent("Who wrote Hamlet?"))