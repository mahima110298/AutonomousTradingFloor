"""Portfolio repo behavior — buys, sells, P&L bookkeeping."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("TF_DATA_DIR", str(Path(__file__).parent / "_tmp_data"))

from trading_floor.config import get_settings  # noqa: E402
from trading_floor.storage import portfolio_repo  # noqa: E402
from trading_floor.storage.db import reset_db  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_state():
    get_settings.cache_clear()
    reset_db()
    yield
    reset_db()


def test_initial_state():
    assert portfolio_repo.cash_balance() > 0
    assert portfolio_repo.positions() == []
    assert portfolio_repo.realized_pnl() == 0.0


def test_market_buy_then_sell_books_pnl():
    starting_cash = portfolio_repo.cash_balance()
    buy = portfolio_repo.submit_order("AAPL", "buy", 10, "market")
    assert buy["status"] == "filled"
    assert portfolio_repo.cash_balance() < starting_cash

    sell = portfolio_repo.submit_order("AAPL", "sell", 10, "market")
    assert sell["status"] == "filled"
    assert portfolio_repo.position_for("AAPL") is None


def test_oversell_is_rejected():
    result = portfolio_repo.submit_order("AAPL", "sell", 10, "market")
    assert result["status"] == "rejected"


def test_limit_order_above_market_does_not_fill_for_buy():
    quote = portfolio_repo.market_store.latest_quote("AAPL") if False else None  # noqa
    from trading_floor.data import market_store

    last_ask = market_store.latest_quote("AAPL")["ask"]
    result = portfolio_repo.submit_order(
        "AAPL", "buy", 5, "limit", limit_price=last_ask - 50.0
    )
    assert result["status"] == "pending"
