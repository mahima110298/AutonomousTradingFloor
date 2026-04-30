"""Synthetic news generator.

Generates plausible-sounding headlines per ticker so the agents have material
to reason over. Sentiment is computed from a small lexicon. Real deployments
would swap this for a feed adapter (Benzinga, Polygon, etc.).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np

from trading_floor.config import get_settings
from trading_floor.data.universe import UNIVERSE, by_ticker

_POSITIVE = {
    "beats", "raises", "expands", "wins", "upgraded", "record", "surges",
    "approves", "launches", "partners", "exceeds", "rallies",
}
_NEGATIVE = {
    "misses", "cuts", "downgraded", "lawsuit", "probe", "recall", "delays",
    "warns", "plunges", "investigation", "fraud", "guidance lowered",
}

_TEMPLATES_POS = [
    "{name} beats earnings expectations as revenue grows",
    "Analysts raise price target on {name} after strong quarter",
    "{name} expands operations into new market segment",
    "{name} wins major contract worth billions",
    "{name} launches breakthrough product line",
    "{name} surges on better-than-expected guidance",
]
_TEMPLATES_NEG = [
    "{name} misses estimates, shares slide",
    "Regulators open probe into {name} business practices",
    "{name} warns of weaker outlook for next quarter",
    "{name} cuts guidance amid slowing demand",
    "{name} faces lawsuit over disclosure issues",
    "{name} delays product launch, citing supply issues",
]
_TEMPLATES_NEU = [
    "{name} announces leadership changes",
    "{name} hosts annual investor day",
    "{name} files routine SEC disclosure",
    "{name} schedules earnings call for next month",
]

_FILING_KINDS = ("10-K", "10-Q", "8-K", "DEF 14A")


def _hash_id(*parts: str) -> str:
    h = hashlib.sha1("|".join(parts).encode()).hexdigest()
    return h[:12]


def _score(text: str) -> float:
    lo = text.lower()
    pos = sum(1 for w in _POSITIVE if w in lo)
    neg = sum(1 for w in _NEGATIVE if w in lo)
    if pos == 0 and neg == 0:
        return 0.0
    return float(np.tanh((pos - neg) / 2.0))


@lru_cache(maxsize=None)
def _generate_for_ticker(ticker: str, n: int = 8) -> list[dict]:
    info = by_ticker(ticker)
    if info is None:
        return []
    rng = np.random.default_rng(get_settings().price_seed + sum(ord(c) for c in ticker))
    items: list[dict] = []
    now = datetime.now(timezone.utc)
    for i in range(n):
        bucket = rng.integers(0, 3)
        if bucket == 0:
            template = _TEMPLATES_POS[int(rng.integers(0, len(_TEMPLATES_POS)))]
        elif bucket == 1:
            template = _TEMPLATES_NEG[int(rng.integers(0, len(_TEMPLATES_NEG)))]
        else:
            template = _TEMPLATES_NEU[int(rng.integers(0, len(_TEMPLATES_NEU)))]
        headline = template.format(name=info.name)
        body = (
            f"{headline}. The development comes amid broader moves in the "
            f"{info.sector} sector. Market participants are watching closely "
            f"for follow-through over the next several sessions."
        )
        published = now - timedelta(hours=int(rng.integers(1, 96)))
        items.append(
            {
                "id": _hash_id(ticker, str(i), headline),
                "headline": headline,
                "body": body,
                "tickers": [ticker],
                "sentiment": _score(headline),
                "published_at": published.isoformat(),
            }
        )
    items.sort(key=lambda x: x["published_at"], reverse=True)
    return items


def headlines_for(ticker: str, limit: int = 5) -> list[dict]:
    items = _generate_for_ticker(ticker.upper())
    return items[:limit]


def recent_headlines(limit: int = 20) -> list[dict]:
    pool: list[dict] = []
    for t in UNIVERSE:
        pool.extend(_generate_for_ticker(t.ticker))
    pool.sort(key=lambda x: x["published_at"], reverse=True)
    return pool[:limit]


def search_news(query: str, limit: int = 10) -> list[dict]:
    q = query.lower().strip()
    if not q:
        return recent_headlines(limit)
    pool: list[dict] = []
    for t in UNIVERSE:
        for item in _generate_for_ticker(t.ticker):
            haystack = (item["headline"] + " " + item["body"]).lower()
            if q in haystack or any(part in haystack for part in q.split()):
                pool.append(item)
    pool.sort(key=lambda x: x["published_at"], reverse=True)
    return pool[:limit]


def aggregate_sentiment(ticker: str, lookback_hours: int = 72) -> float:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    items = _generate_for_ticker(ticker.upper())
    relevant = [
        i for i in items if datetime.fromisoformat(i["published_at"]) >= cutoff
    ]
    if not relevant:
        return 0.0
    return float(sum(i["sentiment"] for i in relevant) / len(relevant))


def summarize_window(ticker: str, lookback_hours: int = 72) -> dict:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    items = [
        i for i in _generate_for_ticker(ticker.upper())
        if datetime.fromisoformat(i["published_at"]) >= cutoff
    ]
    pos = [i for i in items if i["sentiment"] > 0.1]
    neg = [i for i in items if i["sentiment"] < -0.1]
    return {
        "ticker": ticker.upper(),
        "lookback_hours": lookback_hours,
        "total_items": len(items),
        "positive_count": len(pos),
        "negative_count": len(neg),
        "avg_sentiment": (
            float(sum(i["sentiment"] for i in items) / len(items)) if items else 0.0
        ),
        "top_headlines": [i["headline"] for i in items[:3]],
    }


def list_filings(ticker: str, limit: int = 5) -> list[dict]:
    info = by_ticker(ticker)
    if info is None:
        return []
    rng = np.random.default_rng(get_settings().price_seed + sum(ord(c) for c in ticker) + 99)
    out: list[dict] = []
    now = datetime.now(timezone.utc)
    for i in range(limit):
        kind = _FILING_KINDS[int(rng.integers(0, len(_FILING_KINDS)))]
        out.append(
            {
                "ticker": ticker.upper(),
                "kind": kind,
                "filed_at": (now - timedelta(days=int(rng.integers(7, 365)))).isoformat(),
                "title": f"{info.name} {kind} filing",
            }
        )
    out.sort(key=lambda x: x["filed_at"], reverse=True)
    return out


def market_themes() -> list[dict]:
    return [
        {
            "label": "AI infrastructure spend",
            "description": "Hyperscaler capex driving demand for accelerators and data-center buildouts.",
            "related_tickers": ["NVDA", "MSFT", "GOOGL", "AMZN"],
            "momentum": 0.6,
        },
        {
            "label": "Consumer demand softness",
            "description": "Discretionary spending showing signs of cooling.",
            "related_tickers": ["AMZN", "TSLA"],
            "momentum": -0.3,
        },
        {
            "label": "Rate-sensitive financials",
            "description": "Bank earnings sensitive to net interest margin and credit costs.",
            "related_tickers": ["JPM"],
            "momentum": 0.1,
        },
        {
            "label": "Energy macro",
            "description": "Crude pricing and OPEC+ supply discipline driving energy names.",
            "related_tickers": ["XOM"],
            "momentum": 0.0,
        },
    ]
