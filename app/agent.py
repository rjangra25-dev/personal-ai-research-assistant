import os
from dotenv import load_dotenv
from retriever import search_vector_store
from langchain_groq import ChatGroq
from langchain_core.tools import tool

load_dotenv()

# --- Tool 1: wraps the retriever you already built and tested ---
@tool
def search_documents(query: str) -> str:
    """Search the user's personal documents for information on a topic."""
    results = search_vector_store(query, k=3)
    return "\n\n---\n\n".join([doc.page_content for doc in results])

# --- Tool 2: a calculator, so the model has a real choice to make ---
@tool
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '4200 * 0.15'."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: could not evaluate '{expression}' ({e})"


if __name__ == "__main__":
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    tools = [search_documents, calculate]
    llm_with_tools = llm.bind_tools(tools)

    # A lookup so we can find the real function by the name the model gives us
    available_tools = {
        "search_documents": search_documents,
        "calculate": calculate,
    }

    def run_agent(question):
        print(f"\nQUESTION: {question}")

        # Beat 1 — send question + tools to the model
        messages = [("human", question)]
        response = llm_with_tools.invoke(messages)

        # Beat 2 — did it ask for a tool, or answer directly?
        if not response.tool_calls:
            print("ANSWER (no tool needed):", response.content)
            return

        # Keep the model's request in the conversation
        messages.append(response)

        # Beat 3 — run each tool the model asked for
        for call in response.tool_calls:
            tool_name = call["name"]
            tool_args = call["args"]
            print(f"  -> model wants '{tool_name}' with {tool_args}")

            chosen_tool = available_tools[tool_name]
            result = chosen_tool.invoke(tool_args)

            # Beat 4 — hand the result back to the model
            messages.append({
                "role": "tool",
                "content": str(result),
                "tool_call_id": call["id"],
            })

        # Beat 5 — model reads the result and writes the final answer
        final = llm_with_tools.invoke(messages)
        print("ANSWER:", final.content)
        
    run_agent("What does my document say about RAG?")
    run_agent("What is 4200 * 0.15?")
    run_agent("Who wrote Hamlet?")

    