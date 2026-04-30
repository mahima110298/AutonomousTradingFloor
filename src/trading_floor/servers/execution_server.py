"""Order execution MCP server (6 tools)."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from trading_floor.storage import portfolio_repo

mcp = FastMCP("trading-floor-execution")


@mcp.tool()
def submit_market_order(ticker: str, side: str, quantity: int) -> dict:
    """Submit a market order. Fills immediately at the simulated bid/ask."""
    return portfolio_repo.submit_order(ticker, side, quantity, order_type="market")


@mcp.tool()
def submit_limit_order(ticker: str, side: str, quantity: int, limit_price: float) -> dict:
    """Submit a limit order. Fills only if the market crosses the limit price."""
    return portfolio_repo.submit_order(
        ticker, side, quantity, order_type="limit", limit_price=limit_price
    )


@mcp.tool()
def cancel_order(order_id: str) -> dict:
    """Cancel a pending order. No effect if it has already been filled."""
    return portfolio_repo.cancel_order(order_id)


@mcp.tool()
def modify_order(
    order_id: str, new_quantity: int | None = None, new_limit_price: float | None = None
) -> dict:
    """Adjust quantity or limit price of a pending order."""
    return portfolio_repo.modify_order(order_id, new_quantity, new_limit_price)


@mcp.tool()
def get_order_status(order_id: str) -> dict | None:
    """Look up the current status of an order."""
    return portfolio_repo.order_by_id(order_id)


@mcp.tool()
def list_open_orders() -> list[dict]:
    """All currently pending orders."""
    return portfolio_repo.open_orders()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
