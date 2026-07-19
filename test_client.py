import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Tell the client HOW to start the server: run this command
server_params = StdioServerParameters(
    command=sys.executable,
    args=["mcp_server/server.py"],
)

async def main():
    # Launch the server as a subprocess and open a stdio connection to it
    async with stdio_client(server_params) as (read, write):
        # A session is one conversation with the server
        async with ClientSession(read, write) as session:

            # Step 1 — the handshake
            await session.initialize()
            print("Connected to server.\n")

            # Step 2 — ask the server what tools it has
            tools = await session.list_tools()
            print("Tools the server exposes:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")
            print()

            # Step 3 — actually call one tool
            print("Calling calculate with '4200 * 0.15'...")
            result = await session.call_tool("calculate", {"expression": "4200 * 0.15"})
            print("Result:", result.content[0].text)

if __name__ == "__main__":
    asyncio.run(main())