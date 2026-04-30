"""Universe + market data smoke tests (no LLM, no MCP)."""

from __future__ import annotations

import os

os.environ.setdefault("OPENAI_API_KEY", "test")

from trading_floor.data import market_store
from trading_floor.data.universe import UNIVERSE, all_tickers, by_ticker


def test_universe_nonempty():
    assert len(UNIVERSE) >= 5
    assert "AAPL" in all_tickers()


def test_quote_shape():
    q = market_store.latest_quote("AAPL")
    assert {"ticker", "price", "bid", "ask", "timestamp"}.issubset(q.keys())
    assert q["ask"] > q["bid"] > 0


def test_candles_count():
    c = market_store.candles("MSFT", days=10)
    assert len(c) == 10
    assert c[0]["high"] >= c[0]["low"]


def test_unknown_ticker():
    assert by_ticker("ZZZZ") is None
