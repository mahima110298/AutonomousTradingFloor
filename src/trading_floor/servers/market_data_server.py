"""Market data MCP server (8 tools).

Exposes synthetic price/quote/candle data for the simulated universe.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from trading_floor.data import market_store
from trading_floor.data.universe import UNIVERSE

mcp = FastMCP("trading-floor-market-data")


@mcp.tool()
def list_tickers() -> list[dict]:
    """Return every ticker the simulator knows about, with name and sector."""
    return [
        {"ticker": t.ticker, "name": t.name, "sector": t.sector} for t in UNIVERSE
    ]


@mcp.tool()
def get_quote(ticker: str) -> dict:
    """Return the latest bid/ask/last for one ticker."""
    return market_store.latest_quote(ticker.upper())


@mcp.tool()
def get_quote_batch(tickers: list[str]) -> list[dict]:
    """Return latest quotes for several tickers in one call."""
    return [market_store.latest_quote(t.upper()) for t in tickers]


@mcp.tool()
def get_candles(ticker: str, days: int = 30) -> list[dict]:
    """Return OHLCV candles for the last `days` sessions."""
    return market_store.candles(ticker.upper(), days=days)


@mcp.tool()
def get_technical_indicator(ticker: str, indicator: str, days: int = 14) -> dict:
    """Compute a technical indicator. Supported: 'sma', 'rsi', 'volatility'."""
    ticker = ticker.upper()
    indicator = indicator.lower()
    if indicator == "sma":
        return {"ticker": ticker, "indicator": "sma", "window": days, "value": market_store.simple_moving_average(ticker, days)}
    if indicator == "rsi":
        return {"ticker": ticker, "indicator": "rsi", "window": days, "value": market_store.rsi(ticker, days)}
    if indicator == "volatility":
        return {"ticker": ticker, "indicator": "volatility", "window": days, "value": market_store.volatility(ticker, days)}
    raise ValueError(f"unsupported indicator: {indicator}")


@mcp.tool()
def get_fundamentals(ticker: str) -> dict:
    """Return basic fundamental metrics for a ticker."""
    return market_store.fundamentals(ticker.upper())


@mcp.tool()
def get_market_summary() -> dict:
    """Snapshot of all tickers ranked by daily move."""
    return market_store.market_summary()


@mcp.tool()
def get_volatility(ticker: str, days: int = 30) -> dict:
    """Annualized realized volatility over the lookback window."""
    return {
        "ticker": ticker.upper(),
        "window_days": days,
        "annualized_volatility": market_store.volatility(ticker.upper(), days),
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
