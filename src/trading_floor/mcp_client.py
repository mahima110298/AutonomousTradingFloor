"""MCP client that connects to all six servers.

Servers run in-process, connected over MCP's in-memory transport
(`mcp.shared.memory.create_connected_server_and_client_session`) rather than
as separate subprocesses. This keeps the 6-server / 44-tool architecture
intact (each server is still a standalone `FastMCP` instance, and the
standalone entry points in `trading_floor.servers.*` still work for spawning
them independently) while avoiding 6 extra Python interpreter processes at
runtime, which matters on memory-constrained hosts. Tools are loaded lazily
and cached for the lifetime of the process.
"""

from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from typing import AsyncIterator

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared.memory import create_connected_server_and_client_session

from trading_floor.servers import (
    audit_server,
    execution_server,
    market_data_server,
    news_server,
    portfolio_server,
    risk_server,
)

_SERVERS = {
    "market_data": market_data_server.mcp,
    "news": news_server.mcp,
    "portfolio": portfolio_server.mcp,
    "execution": execution_server.mcp,
    "risk": risk_server.mcp,
    "audit": audit_server.mcp,
}


@asynccontextmanager
async def open_floor() -> AsyncIterator[dict[str, BaseTool]]:
    """Yield a name->tool dict spanning every server. Tool names are namespaced
    by the originating server (e.g., 'market_data_get_quote'). Tools are async
    LangChain tools, ready to be passed to LangGraph or CrewAI agents."""
    named: dict[str, BaseTool] = {}
    async with AsyncExitStack() as stack:
        for server_name, server in _SERVERS.items():
            session = await stack.enter_async_context(
                create_connected_server_and_client_session(server)
            )
            tools = await load_mcp_tools(
                session, server_name=server_name, tool_name_prefix=True
            )
            for tool in tools:
                named[tool.name] = tool
        yield named


def select_tools(
    all_tools: dict[str, BaseTool], wanted: list[str]
) -> list[BaseTool]:
    """Return tools whose names match (substring or exact). Order preserved."""
    out: list[BaseTool] = []
    for w in wanted:
        for name, tool in all_tools.items():
            if name == w or name.endswith(f"_{w}") or w in name:
                if tool not in out:
                    out.append(tool)
    return out
