"""Risk management MCP server (8 tools)."""

from __future__ import annotations

import math

import numpy as np
from mcp.server.fastmcp import FastMCP

from trading_floor.config import get_settings
from trading_floor.data import market_store
from trading_floor.data.universe import by_ticker
from trading_floor.storage import portfolio_repo

mcp = FastMCP("trading-floor-risk")


def _equity() -> float:
    eq = portfolio_repo.total_equity()
    return eq if eq > 0 else 1.0


@mcp.tool()
def compute_position_size(ticker: str, conviction: float, risk_budget_pct: float = 2.0) -> dict:
    """Suggest a share count given conviction (0..1) and a risk budget % of equity.

    Sizes the position so that a 1-sigma daily move costs about
    risk_budget_pct% of equity, scaled by conviction.
    """
    ticker = ticker.upper()
    price = market_store.latest_close(ticker)
    daily_vol = market_store.volatility(ticker, days=30) / math.sqrt(252)
    if daily_vol <= 0 or price <= 0:
        return {"ticker": ticker, "suggested_quantity": 0, "reason": "insufficient data"}
    equity = _equity()
    risk_dollars = equity * (risk_budget_pct / 100.0) * max(0.05, min(conviction, 1.0))
    one_sigma_dollars = price * daily_vol
    qty = int(risk_dollars / one_sigma_dollars) if one_sigma_dollars > 0 else 0
    return {
        "ticker": ticker,
        "price": price,
        "daily_vol_pct": round(daily_vol * 100, 3),
        "risk_dollars": round(risk_dollars, 2),
        "suggested_quantity": max(qty, 0),
    }


@mcp.tool()
def check_position_limit(ticker: str, additional_quantity: int) -> dict:
    """Check if adding `additional_quantity` shares would exceed the per-name cap."""
    ticker = ticker.upper()
    settings = get_settings()
    price = market_store.latest_close(ticker)
    held = portfolio_repo.position_for(ticker)
    held_qty = held["quantity"] if held else 0
    new_value = (held_qty + additional_quantity) * price
    pct = new_value / _equity() * 100
    return {
        "ticker": ticker,
        "new_position_pct": round(pct, 2),
        "limit_pct": settings.max_position_pct,
        "within_limit": pct <= settings.max_position_pct,
    }


@mcp.tool()
def check_sector_exposure_limit(ticker: str, additional_quantity: int) -> dict:
    """Check if adding shares would push the ticker's sector over the cap."""
    ticker = ticker.upper()
    info = by_ticker(ticker)
    if info is None:
        return {"ticker": ticker, "error": "unknown ticker"}
    settings = get_settings()
    price = market_store.latest_close(ticker)
    add_value = additional_quantity * price
    sector_pct = portfolio_repo.exposure_by_sector().get(info.sector, 0.0)
    new_pct = sector_pct + (add_value / _equity() * 100)
    return {
        "ticker": ticker,
        "sector": info.sector,
        "new_sector_pct": round(new_pct, 2),
        "limit_pct": settings.max_sector_pct,
        "within_limit": new_pct <= settings.max_sector_pct,
    }


@mcp.tool()
def estimate_value_at_risk(confidence: float = 0.95, lookback_days: int = 60) -> dict:
    """Parametric historical VaR over the existing portfolio."""
    positions = portfolio_repo.positions()
    if not positions:
        return {"value_at_risk": 0.0, "confidence": confidence, "note": "no positions"}
    series = []
    weights = []
    equity = _equity()
    for p in positions:
        rets = market_store.returns_window(p["ticker"], lookback_days)
        if not rets:
            continue
        series.append(np.asarray(rets, dtype=float))
        weights.append(p["market_value"] / equity)
    if not series:
        return {"value_at_risk": 0.0, "confidence": confidence}
    n = min(len(s) for s in series)
    matrix = np.vstack([s[-n:] for s in series])
    portfolio_returns = np.array(weights) @ matrix
    z = float(np.quantile(portfolio_returns, 1.0 - confidence))
    var_dollars = -z * equity
    return {
        "value_at_risk": round(var_dollars, 2),
        "confidence": confidence,
        "lookback_days": lookback_days,
    }


@mcp.tool()
def compute_portfolio_volatility(lookback_days: int = 60) -> dict:
    """Annualized portfolio return volatility."""
    positions = portfolio_repo.positions()
    if not positions:
        return {"annualized_volatility": 0.0}
    series, weights = [], []
    equity = _equity()
    for p in positions:
        rets = market_store.returns_window(p["ticker"], lookback_days)
        if not rets:
            continue
        series.append(np.asarray(rets, dtype=float))
        weights.append(p["market_value"] / equity)
    if not series:
        return {"annualized_volatility": 0.0}
    n = min(len(s) for s in series)
    matrix = np.vstack([s[-n:] for s in series])
    portfolio_returns = np.array(weights) @ matrix
    return {
        "annualized_volatility": float(portfolio_returns.std(ddof=1) * math.sqrt(252)),
        "lookback_days": lookback_days,
    }


@mcp.tool()
def check_drawdown(threshold_pct: float = 10.0) -> dict:
    """Flag when current equity has dropped below the configured threshold from the start."""
    equity = portfolio_repo.total_equity()
    starting = get_settings().starting_cash
    drawdown_pct = max(0.0, (starting - equity) / starting * 100)
    return {
        "current_equity": equity,
        "starting_equity": starting,
        "drawdown_pct": round(drawdown_pct, 2),
        "threshold_pct": threshold_pct,
        "breached": drawdown_pct >= threshold_pct,
    }


@mcp.tool()
def compute_correlation(ticker_a: str, ticker_b: str, lookback_days: int = 60) -> dict:
    """Pearson correlation of daily returns between two tickers."""
    a = np.asarray(market_store.returns_window(ticker_a.upper(), lookback_days), dtype=float)
    b = np.asarray(market_store.returns_window(ticker_b.upper(), lookback_days), dtype=float)
    if len(a) < 2 or len(b) < 2:
        return {"correlation": 0.0, "note": "insufficient data"}
    n = min(len(a), len(b))
    corr = float(np.corrcoef(a[-n:], b[-n:])[0, 1])
    return {"ticker_a": ticker_a.upper(), "ticker_b": ticker_b.upper(), "correlation": corr}


@mcp.tool()
def assess_trade_risk(ticker: str, side: str, quantity: int, conviction: float = 0.5) -> dict:
    """Composite risk assessment combining the position, sector, and volatility checks."""
    ticker = ticker.upper()
    side = side.lower()
    signed_qty = quantity if side == "buy" else -quantity
    position_check = check_position_limit(ticker, signed_qty)
    sector_check = check_sector_exposure_limit(ticker, signed_qty)
    daily_vol = market_store.volatility(ticker, days=30) / math.sqrt(252)
    composite = 0.0
    composite += 0.0 if position_check.get("within_limit", True) else 0.5
    composite += 0.0 if sector_check.get("within_limit", True) else 0.3
    composite += min(daily_vol * 4.0, 0.4)
    composite *= max(0.5, 1.5 - conviction)
    return {
        "ticker": ticker,
        "side": side,
        "quantity": quantity,
        "position_check": position_check,
        "sector_check": sector_check,
        "daily_vol_pct": round(daily_vol * 100, 3),
        "risk_score": round(min(composite, 1.0), 3),
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
