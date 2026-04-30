"""Audit trail MCP server (7 tools)."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from trading_floor.config import get_settings
from trading_floor.storage import audit_repo

mcp = FastMCP("trading-floor-audit")


@mcp.tool()
def log_decision(actor: str, decision: str, context: dict | None = None) -> dict:
    """Record an agent's decision (e.g., trade proposal, risk verdict)."""
    return audit_repo.write_entry(actor, "decision", {"decision": decision, "context": context or {}})


@mcp.tool()
def log_trade(actor: str, order: dict) -> dict:
    """Record an order or fill in the audit trail."""
    return audit_repo.write_entry(actor, "trade", {"order": order})


@mcp.tool()
def log_event(actor: str, event: str, payload: dict | None = None) -> dict:
    """Record any other notable event (status change, error, etc.)."""
    return audit_repo.write_entry(actor, "event", {"event": event, "payload": payload or {}})


@mcp.tool()
def query_audit_trail(actor: str | None = None, kind: str | None = None, limit: int = 50) -> list[dict]:
    """Query the audit trail by actor and/or kind."""
    return audit_repo.query(actor=actor, kind=kind, limit=limit)


@mcp.tool()
def get_decision_history(actor: str | None = None, limit: int = 20) -> list[dict]:
    """Recent decision-class entries, optionally filtered to one actor."""
    return audit_repo.query(actor=actor, kind="decision", limit=limit)


@mcp.tool()
def get_trade_history_audit(limit: int = 20) -> list[dict]:
    """Recent trade-class audit entries (a complement to portfolio.get_trade_history)."""
    return audit_repo.query(kind="trade", limit=limit)


@mcp.tool()
def export_audit_csv(filename: str = "audit_trail.csv") -> dict:
    """Dump the entire audit trail to CSV under the configured data directory."""
    path = get_settings().data_dir / filename
    return audit_repo.export_csv(str(path))


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
