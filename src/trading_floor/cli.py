"""Command-line entry point for the trading floor."""

from __future__ import annotations

import asyncio
import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trading_floor.data.universe import all_tickers
from trading_floor.orchestrator import run_session
from trading_floor.storage import audit_repo, portfolio_repo
from trading_floor.storage.db import reset_db

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def _print_snapshot(title: str) -> None:
    snapshot = {
        "cash": portfolio_repo.cash_balance(),
        "total_equity": portfolio_repo.total_equity(),
        "realized_pnl": portfolio_repo.realized_pnl(),
        "unrealized_pnl": portfolio_repo.unrealized_pnl_total(),
    }
    console.print(Panel.fit(title, style="bold cyan"))
    console.print(f"Cash:           ${snapshot['cash']:,.2f}")
    console.print(f"Total equity:   ${snapshot['total_equity']:,.2f}")
    console.print(f"Realized P&L:   ${snapshot['realized_pnl']:,.2f}")
    console.print(f"Unrealized P&L: ${snapshot['unrealized_pnl']:,.2f}")
    positions = portfolio_repo.positions()
    if positions:
        table = Table(title="Open positions")
        for col in ("Ticker", "Qty", "Avg", "Mkt", "Mkt Value", "Unrealized"):
            table.add_column(col, justify="right")
        for p in positions:
            table.add_row(
                p["ticker"],
                str(p["quantity"]),
                f"${p['average_price']:.2f}",
                f"${p['market_price']:.2f}",
                f"${p['market_value']:.2f}",
                f"${p['unrealized_pnl']:.2f}",
            )
        console.print(table)


async def _stream_session(watchlist: list[str], cycles: int) -> None:
    last_log = 0
    async for state in run_session(watchlist=watchlist, max_cycles=cycles):
        log = state.get("status_log") or []
        for line in log[last_log:]:
            console.log(line)
        last_log = len(log)


@app.command()
def trade(
    tickers: str = typer.Option(
        "",
        "--tickers",
        "-t",
        help="Comma-separated watchlist (default: full simulated universe).",
    ),
    cycles: int = typer.Option(1, "--cycles", "-c", help="How many trade cycles to run."),
    reset: bool = typer.Option(False, "--reset", help="Wipe portfolio + audit before running."),
) -> None:
    """Run one or more autonomous trading cycles."""
    if reset:
        reset_db()
        console.log("State reset to starting cash.")

    watchlist = [t.strip().upper() for t in tickers.split(",") if t.strip()] or all_tickers()
    console.log(f"Watchlist: {', '.join(watchlist)}")
    _print_snapshot("Before")
    asyncio.run(_stream_session(watchlist, cycles))
    _print_snapshot("After")


@app.command()
def status() -> None:
    """Print the current portfolio snapshot."""
    _print_snapshot("Portfolio")


@app.command()
def audit(
    actor: str = typer.Option("", "--actor", help="Filter by actor name."),
    kind: str = typer.Option("", "--kind", help="decision | trade | event | risk"),
    limit: int = typer.Option(20, "--limit", "-n"),
) -> None:
    """Show recent audit-trail entries."""
    rows = audit_repo.query(actor=actor or None, kind=kind or None, limit=limit)
    for r in rows:
        payload_summary = json.dumps(r["payload"], default=str)[:140]
        console.print(
            f"[dim]{r['timestamp']}[/dim] [bold]{r['actor']}[/bold] "
            f"[cyan]{r['kind']}[/cyan] {payload_summary}"
        )


@app.command()
def reset() -> None:
    """Wipe portfolio, orders, trades, and audit trail."""
    reset_db()
    console.log("State reset.")


if __name__ == "__main__":
    app()
