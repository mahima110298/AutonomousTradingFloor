"""LangGraph shared state for the trading-floor orchestrator."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from trading_floor.models import RiskAssessment, TradeProposal


def _replace(_old, new):
    return new


class FloorState(TypedDict, total=False):
    watchlist: Annotated[list[str], _replace]
    cycle_number: int
    max_cycles: int

    portfolio_snapshot: dict
    research_brief: str
    proposals: Annotated[list[TradeProposal], _replace]
    assessments: Annotated[list[RiskAssessment], _replace]
    executed_orders: Annotated[list[dict], operator.add]

    status_log: Annotated[list[str], operator.add]
    done: bool
