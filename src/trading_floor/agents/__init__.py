"""Per-role agent factories.

Each module exposes a `build_*_agent(tools)` callable that returns either a
LangGraph prebuilt agent or a CrewAI Agent, configured with the subset of
MCP tools that role needs.
"""
