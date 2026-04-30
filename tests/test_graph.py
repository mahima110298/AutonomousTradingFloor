"""Smoke test: orchestrator graph compiles and registers all nodes."""

from __future__ import annotations

import os

os.environ.setdefault("OPENAI_API_KEY", "test")


def test_graph_has_expected_nodes():
    from trading_floor.orchestrator import build_graph

    graph = build_graph()
    nodes = set(graph.nodes.keys()) if hasattr(graph, "nodes") else set()
    for n in {"gather_context", "research_crew", "review_risk", "execute_trades", "close_cycle"}:
        assert n in nodes, f"missing node: {n}"


def test_models_roundtrip():
    from trading_floor.models import OrderType, RiskAssessment, Side, TradeProposal

    p = TradeProposal(
        ticker="AAPL",
        side=Side.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
        rationale="example",
        conviction=0.7,
    )
    a = RiskAssessment(proposal=p, approved=True, reasons=["fits limits"], risk_score=0.2)
    assert a.proposal.ticker == "AAPL"
    assert a.approved is True
