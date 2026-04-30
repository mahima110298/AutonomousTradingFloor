"""Repository for portfolio state: cash, positions, orders, trades."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from trading_floor.data import market_store
from trading_floor.storage.db import connect, init_db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cash_balance() -> float:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT cash FROM account WHERE id = 1").fetchone()
        return float(row["cash"]) if row else 0.0


def realized_pnl() -> float:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT realized_pnl FROM account WHERE id = 1").fetchone()
        return float(row["realized_pnl"]) if row else 0.0


def positions() -> list[dict]:
    init_db()
    with connect() as conn:
        rows = conn.execute("SELECT ticker, quantity, average_price FROM positions").fetchall()
    out: list[dict] = []
    for r in rows:
        if r["quantity"] == 0:
            continue
        market_price = market_store.latest_close(r["ticker"])
        out.append(
            {
                "ticker": r["ticker"],
                "quantity": int(r["quantity"]),
                "average_price": float(r["average_price"]),
                "market_price": round(float(market_price), 2),
                "market_value": round(float(market_price) * int(r["quantity"]), 2),
                "unrealized_pnl": round(
                    (float(market_price) - float(r["average_price"])) * int(r["quantity"]), 2
                ),
            }
        )
    return out


def position_for(ticker: str) -> dict | None:
    for p in positions():
        if p["ticker"] == ticker.upper():
            return p
    return None


def total_equity() -> float:
    return cash_balance() + sum(p["market_value"] for p in positions())


def unrealized_pnl_total() -> float:
    return sum(p["unrealized_pnl"] for p in positions())


def exposure_by_sector() -> dict[str, float]:
    from trading_floor.data.universe import by_ticker

    equity = total_equity() or 1.0
    sectors: dict[str, float] = {}
    for p in positions():
        info = by_ticker(p["ticker"])
        if info is None:
            continue
        sectors[info.sector] = sectors.get(info.sector, 0.0) + p["market_value"]
    return {sector: round(value / equity * 100, 2) for sector, value in sectors.items()}


def trade_history(limit: int = 50) -> list[dict]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT trade_id, order_id, ticker, side, quantity, price, timestamp "
            "FROM trades ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def open_orders() -> list[dict]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM orders WHERE status = 'pending' ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def order_by_id(order_id: str) -> dict | None:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    return dict(row) if row else None


def _apply_fill(conn, ticker: str, side: str, quantity: int, price: float) -> dict:
    cur = conn.execute(
        "SELECT quantity, average_price FROM positions WHERE ticker = ?", (ticker,)
    ).fetchone()
    cash_row = conn.execute("SELECT cash, realized_pnl FROM account WHERE id = 1").fetchone()
    cash = float(cash_row["cash"])
    realized = float(cash_row["realized_pnl"])

    if side == "buy":
        cost = quantity * price
        if cost > cash:
            raise ValueError(f"Insufficient cash: need {cost:.2f}, have {cash:.2f}")
        if cur is None or cur["quantity"] == 0:
            new_qty = quantity
            new_avg = price
        else:
            old_qty = int(cur["quantity"])
            old_avg = float(cur["average_price"])
            new_qty = old_qty + quantity
            new_avg = (old_qty * old_avg + quantity * price) / new_qty
        conn.execute(
            "INSERT INTO positions (ticker, quantity, average_price) VALUES (?, ?, ?) "
            "ON CONFLICT(ticker) DO UPDATE SET quantity = excluded.quantity, "
            "average_price = excluded.average_price",
            (ticker, new_qty, new_avg),
        )
        conn.execute("UPDATE account SET cash = ? WHERE id = 1", (cash - cost,))
    elif side == "sell":
        if cur is None or int(cur["quantity"]) < quantity:
            held = int(cur["quantity"]) if cur else 0
            raise ValueError(f"Insufficient shares of {ticker}: holding {held}, sell {quantity}")
        old_qty = int(cur["quantity"])
        old_avg = float(cur["average_price"])
        proceeds = quantity * price
        realized_delta = (price - old_avg) * quantity
        new_qty = old_qty - quantity
        conn.execute(
            "UPDATE positions SET quantity = ?, average_price = ? WHERE ticker = ?",
            (new_qty, old_avg, ticker),
        )
        if new_qty == 0:
            conn.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))
        conn.execute(
            "UPDATE account SET cash = ?, realized_pnl = ? WHERE id = 1",
            (cash + proceeds, realized + realized_delta),
        )
    else:
        raise ValueError(f"Unknown side: {side}")
    return {"cash_after": cash + (-quantity * price if side == "buy" else quantity * price)}


def submit_order(
    ticker: str,
    side: str,
    quantity: int,
    order_type: str = "market",
    limit_price: float | None = None,
) -> dict:
    """Place an order. Market orders fill immediately; limit orders may not."""
    init_db()
    ticker = ticker.upper()
    side = side.lower()
    order_type = order_type.lower()
    if quantity <= 0:
        raise ValueError("quantity must be positive")

    order_id = uuid.uuid4().hex[:12]
    now = _now()

    quote = market_store.latest_quote(ticker)
    market_price = float(quote["ask"] if side == "buy" else quote["bid"])

    with connect() as conn:
        conn.execute(
            "INSERT INTO orders (order_id, ticker, side, quantity, order_type, limit_price, "
            "status, filled_quantity, filled_price, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, NULL, ?, ?)",
            (order_id, ticker, side, quantity, order_type, limit_price, now, now),
        )

        fill_price: float | None = None
        if order_type == "market":
            fill_price = market_price
        elif order_type == "limit":
            if limit_price is None:
                conn.execute(
                    "UPDATE orders SET status = 'rejected', updated_at = ? WHERE order_id = ?",
                    (now, order_id),
                )
                return {"order_id": order_id, "status": "rejected", "reason": "missing limit_price"}
            if (side == "buy" and market_price <= limit_price) or (
                side == "sell" and market_price >= limit_price
            ):
                fill_price = market_price
        else:
            conn.execute(
                "UPDATE orders SET status = 'rejected', updated_at = ? WHERE order_id = ?",
                (now, order_id),
            )
            return {"order_id": order_id, "status": "rejected", "reason": f"unknown order_type {order_type}"}

        if fill_price is None:
            return order_by_id(order_id) or {"order_id": order_id, "status": "pending"}

        try:
            _apply_fill(conn, ticker, side, quantity, fill_price)
        except ValueError as exc:
            conn.execute(
                "UPDATE orders SET status = 'rejected', updated_at = ? WHERE order_id = ?",
                (now, order_id),
            )
            return {"order_id": order_id, "status": "rejected", "reason": str(exc)}

        trade_id = uuid.uuid4().hex[:12]
        conn.execute(
            "INSERT INTO trades (trade_id, order_id, ticker, side, quantity, price, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (trade_id, order_id, ticker, side, quantity, fill_price, now),
        )
        conn.execute(
            "UPDATE orders SET status = 'filled', filled_quantity = ?, filled_price = ?, "
            "updated_at = ? WHERE order_id = ?",
            (quantity, fill_price, now, order_id),
        )

    return order_by_id(order_id) or {"order_id": order_id, "status": "filled"}


def cancel_order(order_id: str) -> dict:
    init_db()
    now = _now()
    with connect() as conn:
        row = conn.execute("SELECT status FROM orders WHERE order_id = ?", (order_id,)).fetchone()
        if row is None:
            return {"order_id": order_id, "status": "not_found"}
        if row["status"] != "pending":
            return {"order_id": order_id, "status": row["status"], "reason": "not cancellable"}
        conn.execute(
            "UPDATE orders SET status = 'cancelled', updated_at = ? WHERE order_id = ?",
            (now, order_id),
        )
    return order_by_id(order_id) or {"order_id": order_id, "status": "cancelled"}


def modify_order(order_id: str, new_quantity: int | None = None, new_limit_price: float | None = None) -> dict:
    init_db()
    now = _now()
    with connect() as conn:
        row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
        if row is None:
            return {"order_id": order_id, "status": "not_found"}
        if row["status"] != "pending":
            return {"order_id": order_id, "status": row["status"], "reason": "not modifiable"}
        qty = int(new_quantity) if new_quantity is not None else int(row["quantity"])
        lp = float(new_limit_price) if new_limit_price is not None else row["limit_price"]
        conn.execute(
            "UPDATE orders SET quantity = ?, limit_price = ?, updated_at = ? WHERE order_id = ?",
            (qty, lp, now, order_id),
        )
    return order_by_id(order_id) or {"order_id": order_id, "status": "modified"}
