"""Portfolio MCP server (8 tools)."""

from __future__ import annotations

from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

from trading_floor.storage import portfolio_repo

mcp = FastMCP("trading-floor-portfolio")


@mcp.tool()
def get_cash_balance() -> dict:
    """Available cash for new orders."""
    return {"cash": portfolio_repo.cash_balance()}


@mcp.tool()
def get_positions() -> list[dict]:
    """All open positions with quantity, avg cost, and unrealized P&L."""
    return portfolio_repo.positions()


@mcp.tool()
def get_position(ticker: str) -> dict | None:
    """Open position for a single ticker, or null if flat."""
    return portfolio_repo.position_for(ticker)


@mcp.tool()
def get_pnl() -> dict:
    """Realized and unrealized P&L summary."""
    return {
        "realized": portfolio_repo.realized_pnl(),
        "unrealized": portfolio_repo.unrealized_pnl_total(),
    }


@mcp.tool()
def get_total_equity() -> dict:
    """Cash plus market value of all positions."""
    return {"total_equity": portfolio_repo.total_equity()}


@mcp.tool()
def get_exposure_by_sector() -> dict:
    """Percent of total equity allocated to each sector."""
    return portfolio_repo.exposure_by_sector()


@mcp.tool()
def get_trade_history(limit: int = 50) -> list[dict]:
    """Most recent filled trades."""
    return portfolio_repo.trade_history(limit=limit)


@mcp.tool()
def get_account_snapshot() -> dict:
    """Single call returning cash, positions, equity, and P&L."""
    return {
        "asof": datetime.now(timezone.utc).isoformat(),
        "cash": portfolio_repo.cash_balance(),
        "realized_pnl": portfolio_repo.realized_pnl(),
        "unrealized_pnl": portfolio_repo.unrealized_pnl_total(),
        "total_equity": portfolio_repo.total_equity(),
        "positions": portfolio_repo.positions(),
        "sector_exposure_pct": portfolio_repo.exposure_by_sector(),
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
