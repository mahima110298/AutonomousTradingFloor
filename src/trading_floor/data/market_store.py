"""Synthetic market data generator.

Prices follow a deterministic geometric Brownian motion seeded per ticker so
that runs are reproducible and the agents can reason over consistent history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np

from trading_floor.config import get_settings
from trading_floor.data.universe import UNIVERSE, TickerInfo, by_ticker

_TRADING_DAYS = 252
_LOOKBACK_DAYS = 200


@dataclass(frozen=True)
class _PriceSeries:
    timestamps: list[datetime]
    closes: list[float]
    highs: list[float]
    lows: list[float]
    opens: list[float]
    volumes: list[int]


def _seed_for(ticker: str) -> int:
    base = get_settings().price_seed
    return base + (sum(ord(c) for c in ticker) * 1009) % 100000


@lru_cache(maxsize=None)
def _series(ticker: str) -> _PriceSeries:
    info: TickerInfo | None = by_ticker(ticker)
    if info is None:
        raise ValueError(f"Unknown ticker: {ticker}")

    rng = np.random.default_rng(_seed_for(ticker))
    dt = 1.0 / _TRADING_DAYS
    drift = info.annual_drift
    vol = info.annual_vol

    n = _LOOKBACK_DAYS
    shocks = rng.standard_normal(n)
    log_returns = (drift - 0.5 * vol * vol) * dt + vol * math.sqrt(dt) * shocks
    closes_arr = info.base_price * np.exp(np.cumsum(log_returns))
    closes = [float(x) for x in closes_arr]

    intraday_vol = vol * math.sqrt(dt) * 0.6
    highs: list[float] = []
    lows: list[float] = []
    opens: list[float] = []
    volumes: list[int] = []
    prev_close = info.base_price
    for c in closes:
        o = prev_close * (1.0 + float(rng.normal(0, intraday_vol * 0.3)))
        h = max(o, c) * (1.0 + abs(float(rng.normal(0, intraday_vol))))
        lo = min(o, c) * (1.0 - abs(float(rng.normal(0, intraday_vol))))
        v = int(abs(rng.normal(5_000_000, 1_500_000)))
        opens.append(o)
        highs.append(h)
        lows.append(lo)
        volumes.append(v)
        prev_close = c

    today = datetime.now(timezone.utc).replace(hour=20, minute=0, second=0, microsecond=0)
    timestamps = [today - timedelta(days=(n - 1 - i)) for i in range(n)]
    return _PriceSeries(timestamps, closes, highs, lows, opens, volumes)


def latest_close(ticker: str) -> float:
    return _series(ticker).closes[-1]


def latest_quote(ticker: str) -> dict:
    s = _series(ticker)
    last = s.closes[-1]
    spread = max(0.01, last * 0.0005)
    return {
        "ticker": ticker,
        "price": round(last, 2),
        "bid": round(last - spread, 2),
        "ask": round(last + spread, 2),
        "timestamp": s.timestamps[-1].isoformat(),
    }


def candles(ticker: str, days: int = 30) -> list[dict]:
    s = _series(ticker)
    days = max(1, min(days, len(s.closes)))
    out = []
    for i in range(len(s.closes) - days, len(s.closes)):
        out.append(
            {
                "ticker": ticker,
                "timestamp": s.timestamps[i].isoformat(),
                "open": round(s.opens[i], 2),
                "high": round(s.highs[i], 2),
                "low": round(s.lows[i], 2),
                "close": round(s.closes[i], 2),
                "volume": s.volumes[i],
            }
        )
    return out


def returns_window(ticker: str, days: int) -> list[float]:
    s = _series(ticker)
    days = min(days, len(s.closes) - 1)
    closes = s.closes[-(days + 1):]
    return [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]


def volatility(ticker: str, days: int = 30) -> float:
    rets = returns_window(ticker, days)
    if not rets:
        return 0.0
    arr = np.asarray(rets, dtype=float)
    return float(arr.std(ddof=1) * math.sqrt(_TRADING_DAYS))


def simple_moving_average(ticker: str, days: int) -> float:
    s = _series(ticker)
    window = s.closes[-days:]
    return float(sum(window) / len(window)) if window else 0.0


def rsi(ticker: str, days: int = 14) -> float:
    rets = returns_window(ticker, days)
    if not rets:
        return 50.0
    gains = [r for r in rets if r > 0]
    losses = [-r for r in rets if r < 0]
    avg_gain = sum(gains) / len(rets)
    avg_loss = sum(losses) / len(rets) if losses else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def fundamentals(ticker: str) -> dict:
    info = by_ticker(ticker)
    if info is None:
        raise ValueError(f"Unknown ticker: {ticker}")
    rng = np.random.default_rng(_seed_for(ticker) + 7)
    return {
        "ticker": info.ticker,
        "name": info.name,
        "sector": info.sector,
        "market_cap": float(round(latest_close(ticker) * float(rng.integers(2_000_000_000, 18_000_000_000)), 0)),
        "pe_ratio": float(round(15 + rng.normal(0, 7), 2)),
        "dividend_yield": float(round(max(0.0, rng.normal(0.015, 0.012)), 4)),
    }


def market_summary() -> dict:
    quotes = []
    for t in UNIVERSE:
        rets = returns_window(t.ticker, 1)
        change = rets[-1] if rets else 0.0
        quotes.append(
            {
                "ticker": t.ticker,
                "price": round(latest_close(t.ticker), 2),
                "day_change_pct": round(change * 100, 2),
            }
        )
    quotes.sort(key=lambda q: q["day_change_pct"], reverse=True)
    return {
        "asof": datetime.now(timezone.utc).isoformat(),
        "movers": quotes,
    }
