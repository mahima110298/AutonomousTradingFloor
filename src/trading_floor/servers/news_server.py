"""News MCP server (7 tools)."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from trading_floor.data import news_store

mcp = FastMCP("trading-floor-news")


@mcp.tool()
def list_recent_headlines(limit: int = 20) -> list[dict]:
    """Return the most recent headlines across the universe."""
    return news_store.recent_headlines(limit=limit)


@mcp.tool()
def search_news(query: str, limit: int = 10) -> list[dict]:
    """Substring search across headline and body."""
    return news_store.search_news(query, limit=limit)


@mcp.tool()
def get_news_for_ticker(ticker: str, limit: int = 5) -> list[dict]:
    """Recent news items mentioning the given ticker."""
    return news_store.headlines_for(ticker, limit=limit)


@mcp.tool()
def get_sentiment_score(ticker: str, lookback_hours: int = 72) -> dict:
    """Aggregate sentiment for a ticker over the lookback window (-1 to +1)."""
    return {
        "ticker": ticker.upper(),
        "lookback_hours": lookback_hours,
        "sentiment": news_store.aggregate_sentiment(ticker, lookback_hours=lookback_hours),
    }


@mcp.tool()
def summarize_news_window(ticker: str, lookback_hours: int = 72) -> dict:
    """Summary stats over a recent news window for a ticker."""
    return news_store.summarize_window(ticker, lookback_hours=lookback_hours)


@mcp.tool()
def list_filings(ticker: str, limit: int = 5) -> list[dict]:
    """Recent regulatory filings for a ticker."""
    return news_store.list_filings(ticker, limit=limit)


@mcp.tool()
def get_market_themes() -> list[dict]:
    """Return active cross-cutting market themes and their related tickers."""
    return news_store.market_themes()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
