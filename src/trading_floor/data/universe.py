"""The simulated ticker universe.

A small, hand-picked set of well-known names so that the agents have a
finite, knowable space to reason over. Adjust freely — every other module
discovers tickers via list_tickers().
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TickerInfo:
    ticker: str
    name: str
    sector: str
    base_price: float
    annual_drift: float
    annual_vol: float


UNIVERSE: tuple[TickerInfo, ...] = (
    TickerInfo("AAPL", "Apple Inc.", "Technology", 195.0, 0.10, 0.28),
    TickerInfo("MSFT", "Microsoft Corp.", "Technology", 410.0, 0.12, 0.26),
    TickerInfo("GOOGL", "Alphabet Inc.", "Technology", 165.0, 0.09, 0.30),
    TickerInfo("NVDA", "NVIDIA Corp.", "Technology", 880.0, 0.25, 0.55),
    TickerInfo("AMZN", "Amazon.com Inc.", "Consumer Discretionary", 185.0, 0.11, 0.32),
    TickerInfo("TSLA", "Tesla Inc.", "Consumer Discretionary", 220.0, 0.05, 0.60),
    TickerInfo("META", "Meta Platforms Inc.", "Communication Services", 480.0, 0.10, 0.40),
    TickerInfo("JPM", "JPMorgan Chase & Co.", "Financials", 195.0, 0.07, 0.22),
    TickerInfo("XOM", "Exxon Mobil Corp.", "Energy", 115.0, 0.04, 0.30),
    TickerInfo("JNJ", "Johnson & Johnson", "Healthcare", 155.0, 0.05, 0.18),
)


def by_ticker(symbol: str) -> TickerInfo | None:
    symbol = symbol.upper()
    for t in UNIVERSE:
        if t.ticker == symbol:
            return t
    return None


def all_tickers() -> list[str]:
    return [t.ticker for t in UNIVERSE]
