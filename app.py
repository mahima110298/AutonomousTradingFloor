"""Gradio web UI for the trading floor."""

from __future__ import annotations

import os

import asyncio
import queue
import threading

import gradio as gr
import pandas as pd

from trading_floor.data.universe import UNIVERSE, all_tickers
from trading_floor.orchestrator import run_session
from trading_floor.storage import audit_repo, portfolio_repo
from trading_floor.storage.db import reset_db

_LOCK = threading.Lock()


def _snapshot_text() -> str:
    return (
        f"**Cash:** ${portfolio_repo.cash_balance():,.2f}  \n"
        f"**Total equity:** ${portfolio_repo.total_equity():,.2f}  \n"
        f"**Realized P&L:** ${portfolio_repo.realized_pnl():,.2f}  \n"
        f"**Unrealized P&L:** ${portfolio_repo.unrealized_pnl_total():,.2f}"
    )


def _positions_df() -> pd.DataFrame:
    rows = portfolio_repo.positions()
    if not rows:
        return pd.DataFrame(columns=["Ticker", "Qty", "Avg", "Mkt", "Mkt Value", "Unrealized"])
    return pd.DataFrame(
        [
            {
                "Ticker": r["ticker"],
                "Qty": r["quantity"],
                "Avg": r["average_price"],
                "Mkt": r["market_price"],
                "Mkt Value": r["market_value"],
                "Unrealized": r["unrealized_pnl"],
            }
            for r in rows
        ]
    )


def _audit_df(limit: int = 30) -> pd.DataFrame:
    rows = audit_repo.query(limit=limit)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "actor", "kind", "payload"])
    return pd.DataFrame(
        [
            {
                "timestamp": r["timestamp"],
                "actor": r["actor"],
                "kind": r["kind"],
                "payload": str(r["payload"])[:200],
            }
            for r in rows
        ]
    )


async def _run_async(watchlist: list[str], cycles: int):
    seen = 0
    log_lines: list[str] = []
    yield (
        "\n".join(log_lines) or "_Starting…_",
        _snapshot_text(),
        _positions_df(),
        _audit_df(),
    )
    async for state in run_session(watchlist=watchlist, max_cycles=cycles):
        log = state.get("status_log") or []
        if len(log) > seen:
            log_lines.extend(log[seen:])
            seen = len(log)
            yield (
                "\n".join(f"- {line}" for line in log_lines),
                _snapshot_text(),
                _positions_df(),
                _audit_df(),
            )

    yield (
        "\n".join(f"- {line}" for line in log_lines) + "\n\n**Session complete.**",
        _snapshot_text(),
        _positions_df(),
        _audit_df(),
    )


def run_session_handler(tickers_text: str, cycles: int):
    """Drive the async session generator to completion on a dedicated
    background thread with its own event loop, handing each yielded UI
    update back to this sync generator over a thread-safe queue.

    This keeps the whole async generator — including the MCP client's
    task group in `open_floor()` — running under a single top-level
    asyncio Task for its entire lifetime. Driving it step-by-step with
    repeated `loop.run_until_complete(agen.__anext__())` calls (the
    previous approach) wraps each step in a *new* Task on the same loop,
    which anyio rejects once a cancel scope opened under an earlier Task
    is touched from a later one (`RuntimeError: Attempted to exit cancel
    scope in a different task than it was entered in`).
    """
    watchlist = [t.strip().upper() for t in tickers_text.split(",") if t.strip()] or all_tickers()
    if not _LOCK.acquire(blocking=False):
        yield "_A session is already running. Wait for it to finish._", _snapshot_text(), _positions_df(), _audit_df()
        return
    try:
        q: queue.Queue = queue.Queue()
        done = object()

        def _worker() -> None:
            async def _consume() -> None:
                async for item in _run_async(watchlist, cycles):
                    q.put(item)

            try:
                asyncio.run(_consume())
            except Exception as exc:  # surfaced to the UI thread below
                q.put(exc)
            finally:
                q.put(done)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        while True:
            item = q.get()
            if item is done:
                break
            if isinstance(item, Exception):
                raise item
            yield item
        thread.join()
    finally:
        _LOCK.release()


def reset_handler():
    reset_db()
    return "State reset.", _snapshot_text(), _positions_df(), _audit_df()


with gr.Blocks(title="Autonomous Trading Floor") as ui:
    gr.Markdown(
        "# Autonomous Trading Floor\n"
        "Four agents — **Researcher**, **Strategist**, **Risk Officer**, **Trader** — collaborate "
        "via 6 MCP servers (44 tools) to research a watchlist, propose trades, vet them against "
        "risk limits, and execute against a simulated portfolio with a full audit trail."
    )

    with gr.Row():
        tickers_in = gr.Textbox(
            label="Watchlist",
            placeholder=f"e.g. {', '.join(t.ticker for t in UNIVERSE[:4])}",
            value=", ".join(t.ticker for t in UNIVERSE[:4]),
            scale=4,
        )
        cycles_in = gr.Number(label="Cycles", value=1, precision=0, minimum=1, maximum=5, scale=1)
        run_btn = gr.Button("Run session", variant="primary", scale=1)
        reset_btn = gr.Button("Reset", variant="secondary", scale=1)

    with gr.Row():
        with gr.Column(scale=2):
            status = gr.Markdown("_Idle._", label="Progress")
        with gr.Column(scale=1):
            snapshot = gr.Markdown(_snapshot_text(), label="Account")

    positions_table = gr.Dataframe(value=_positions_df(), label="Open positions", interactive=False)
    audit_table = gr.Dataframe(value=_audit_df(), label="Audit trail (most recent)", interactive=False)

    run_btn.click(
        run_session_handler,
        inputs=[tickers_in, cycles_in],
        outputs=[status, snapshot, positions_table, audit_table],
    )
    reset_btn.click(
        reset_handler, outputs=[status, snapshot, positions_table, audit_table]
    )


if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
