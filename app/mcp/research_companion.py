"""Read-only research-agent tool on the official MCP SDK, stdio only.

    python -m app.mcp.research_companion

No filesystem, shell, mutation or approval tools are registered. Agent credentials
are the local server environment, never tool arguments or tool output.
Uses the SDK's MCPServer 2.x entry point, matching the existing MetaNavT server.
"""
from __future__ import annotations

import httpx

from app.agent.companion import AskRequest
from app.agent.companion_api import AgentConfig, create_app


async def ask_research(question: str, mode: str = "demo") -> dict:
    """Investigate research files through a bounded read-only tool loop.

    mode: demo for synthetic fixtures/scripted decisions; live for an operator-configured Ollama model and retrieval service.
    """
    request = AskRequest(question=question, mode=mode)
    config = AgentConfig.from_env()
    app = create_app(config)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://internal-agent") as client:
            response = await client.post("/agent/ask", json=request.model_dump(),
                                         headers={"X-API-Key": config.api_key} if request.mode == "live" else {})
            if response.status_code != 200:
                return {"error": "research_agent_unavailable", "status_code": response.status_code,
                        "detail": response.json().get("detail", "The request failed.")}
            return response.json()


def build_server():
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("metanavt-research-companion",
                       instructions="Read-only research questions with cited excerpts and executed tool traces. Demo uses scripted decisions and synthetic files. No mutation, execution or approval tools.")
    server.add_tool(ask_research, name="ask_research", description=ask_research.__doc__)
    return server


def main():
    server = build_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
