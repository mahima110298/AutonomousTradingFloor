"""Verify the 6 MCP servers expose exactly 44 tools as advertised."""

from __future__ import annotations

import os

os.environ.setdefault("OPENAI_API_KEY", "test")

from trading_floor.servers import (  # noqa: E402
    audit_server,
    execution_server,
    market_data_server,
    news_server,
    portfolio_server,
    risk_server,
)


_EXPECTED = {
    "market_data": 8,
    "news": 7,
    "portfolio": 8,
    "execution": 6,
    "risk": 8,
    "audit": 7,
}

_SERVERS = {
    "market_data": market_data_server.mcp,
    "news": news_server.mcp,
    "portfolio": portfolio_server.mcp,
    "execution": execution_server.mcp,
    "risk": risk_server.mcp,
    "audit": audit_server.mcp,
}


async def _tool_count(server) -> int:
    tools = await server.list_tools()
    return len(tools)


async def test_each_server_tool_count():
    for name, server in _SERVERS.items():
        count = await _tool_count(server)
        assert count == _EXPECTED[name], (
            f"{name} server has {count} tools, expected {_EXPECTED[name]}"
        )


async def test_total_tool_count_is_44():
    total = 0
    for server in _SERVERS.values():
        total += await _tool_count(server)
    assert total == 44, f"expected 44 tools across all servers, found {total}"
