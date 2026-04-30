"""SQLite connection manager for portfolio + audit persistence."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from trading_floor.config import get_settings

_LOCK = threading.Lock()
_INITIALIZED = False


def db_path() -> Path:
    return get_settings().data_dir / "trading_floor.db"


@contextmanager
def connect():
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
    finally:
        conn.close()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    cash REAL NOT NULL,
    realized_pnl REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS positions (
    ticker TEXT PRIMARY KEY,
    quantity INTEGER NOT NULL,
    average_price REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    order_type TEXT NOT NULL,
    limit_price REAL,
    status TEXT NOT NULL,
    filled_quantity INTEGER NOT NULL DEFAULT 0,
    filled_price REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price REAL NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

CREATE TABLE IF NOT EXISTS audit (
    entry_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    actor TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit(actor);
CREATE INDEX IF NOT EXISTS idx_audit_kind ON audit(kind);
CREATE INDEX IF NOT EXISTS idx_orders_ticker ON orders(ticker);
"""


def init_db() -> None:
    """Create schema if missing and seed account row."""
    global _INITIALIZED
    with _LOCK:
        if _INITIALIZED:
            return
        starting = get_settings().starting_cash
        with connect() as conn:
            conn.executescript(_SCHEMA)
            cur = conn.execute("SELECT cash FROM account WHERE id = 1")
            if cur.fetchone() is None:
                conn.execute(
                    "INSERT INTO account (id, cash, realized_pnl) VALUES (1, ?, 0.0)",
                    (starting,),
                )
        _INITIALIZED = True


def reset_db() -> None:
    """Wipe all state. Useful between simulation runs."""
    global _INITIALIZED
    with _LOCK:
        with connect() as conn:
            conn.executescript(
                "DROP TABLE IF EXISTS audit;"
                "DROP TABLE IF EXISTS trades;"
                "DROP TABLE IF EXISTS orders;"
                "DROP TABLE IF EXISTS positions;"
                "DROP TABLE IF EXISTS account;"
            )
        _INITIALIZED = False
    init_db()
