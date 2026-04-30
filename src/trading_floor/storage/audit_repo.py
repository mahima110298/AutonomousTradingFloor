"""Persistent audit trail repository."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from trading_floor.storage.db import connect, init_db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_entry(actor: str, kind: str, payload: dict) -> dict:
    init_db()
    if kind not in {"decision", "trade", "event", "risk"}:
        raise ValueError(f"unknown audit kind: {kind}")
    entry_id = uuid.uuid4().hex[:12]
    timestamp = _now()
    with connect() as conn:
        conn.execute(
            "INSERT INTO audit (entry_id, timestamp, actor, kind, payload_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (entry_id, timestamp, actor, kind, json.dumps(payload, default=str)),
        )
    return {
        "entry_id": entry_id,
        "timestamp": timestamp,
        "actor": actor,
        "kind": kind,
        "payload": payload,
    }


def query(
    actor: str | None = None,
    kind: str | None = None,
    limit: int = 100,
) -> list[dict]:
    init_db()
    sql = "SELECT * FROM audit WHERE 1=1"
    args: list = []
    if actor:
        sql += " AND actor = ?"
        args.append(actor)
    if kind:
        sql += " AND kind = ?"
        args.append(kind)
    sql += " ORDER BY timestamp DESC LIMIT ?"
    args.append(int(limit))
    with connect() as conn:
        rows = conn.execute(sql, tuple(args)).fetchall()
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        d["payload"] = json.loads(d.pop("payload_json"))
        out.append(d)
    return out


def export_csv(path: str) -> dict:
    import csv

    rows = query(limit=100_000)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["entry_id", "timestamp", "actor", "kind", "payload"])
        for r in rows:
            writer.writerow([r["entry_id"], r["timestamp"], r["actor"], r["kind"], json.dumps(r["payload"])])
    return {"path": path, "rows_written": len(rows)}
