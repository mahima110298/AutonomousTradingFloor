"""MCP client that connects to all six servers.

Servers run as subprocesses over stdio. Tools are loaded lazily and cached
for the lifetime of the process.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


_SERVER_MODULES = {
    "market_data": "trading_floor.servers.market_data_server",
    "news": "trading_floor.servers.news_server",
    "portfolio": "trading_floor.servers.portfolio_server",
    "execution": "trading_floor.servers.execution_server",
    "risk": "trading_floor.servers.risk_server",
    "audit": "trading_floor.servers.audit_server",
}


def _server_specs() -> dict:
    python = sys.executable
    return {
        name: {"command": python, "args": ["-m", module], "transport": "stdio"}
        for name, module in _SERVER_MODULES.items()
    }


@asynccontextmanager
async def open_floor() -> AsyncIterator[dict[str, BaseTool]]:
    """Yield a name->tool dict spanning every server. Tool names are namespaced
    by the originating server (e.g., 'market_data.get_quote'). Tools are async
    LangChain tools, ready to be passed to LangGraph or CrewAI agents."""
    client = MultiServerMCPClient(_server_specs())
    try:
        tools = await client.get_tools()
        named: dict[str, BaseTool] = {}
        for tool in tools:
            named[tool.name] = tool
        yield named
    finally:
        try:
            await client.aclose()
        except Exception:
            pass


def select_tools(
    all_tools: dict[str, BaseTool], wanted: list[str]
) -> list[BaseTool]:
    """Return tools whose names match (substring or exact). Order preserved."""
    out: list[BaseTool] = []
    for w in wanted:
        for name, tool in all_tools.items():
            if name == w or name.endswith(f".{w}") or w in name:
                if tool not in out:
                    out.append(tool)
    return out
