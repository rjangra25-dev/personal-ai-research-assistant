import sys
import os
from mcp.server.fastmcp import FastMCP

# Let this file import from the app/ folder (retriever lives there)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))
from retriever import search_vector_store

# Create the MCP server — the name is how clients identify it
mcp = FastMCP("research-assistant-tools")

@mcp.tool()
def search_documents(query: str) -> str:
    """Search the user's personal documents for information on a topic."""
    results = search_vector_store(query, k=3)
    return "\n\n---\n\n".join([doc.page_content for doc in results])

@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '4200 * 0.15'."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: could not evaluate '{expression}' ({e})"

if __name__ == "__main__":
    # Run the server over stdio — it listens on standard input/output
    mcp.run(transport="stdio")