"""Six MCP servers exposing the 44 tools the trading floor uses.

Each server is a standalone process that can be launched independently:

    python -m trading_floor.servers.market_data_server
    python -m trading_floor.servers.news_server
    ...

Servers communicate via the standard MCP stdio transport. The orchestrator
spawns them as subprocesses; clients in other processes (other agents, other
teams) can connect to the same servers without modification.
"""
