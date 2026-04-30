"""LangGraph orchestrator that drives the full trade cycle.

Topology:
    START
      -> gather_context
      -> research_crew (CrewAI Researcher + Strategist)
      -> review_risk (Risk Officer agent, one pass per proposal)
      -> execute_trades (Trader agent, only on approved proposals)
      -> close_cycle (write audit summary; decide whether to loop)
      -> {next_cycle | END}
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from trading_floor.agents.risk_officer import build_risk_officer
from trading_floor.agents.trader import build_trader
from trading_floor.crews.research_crew import run_research_crew
from trading_floor.data.universe import all_tickers
from trading_floor.mcp_client import open_floor
from trading_floor.models import RiskAssessment, TradeProposal
from trading_floor.state import FloorState
from trading_floor.storage import audit_repo, portfolio_repo


def _filter(tools: dict[str, BaseTool], substrings: list[str]) -> list[BaseTool]:
    return [t for t in tools.values() if any(s in t.name for s in substrings)]


def _portfolio_snapshot() -> dict:
    return {
        "cash": portfolio_repo.cash_balance(),
        "total_equity": portfolio_repo.total_equity(),
        "realized_pnl": portfolio_repo.realized_pnl(),
        "unrealized_pnl": portfolio_repo.unrealized_pnl_total(),
        "positions": portfolio_repo.positions(),
        "sector_exposure_pct": portfolio_repo.exposure_by_sector(),
    }


def _gather_context(state: FloorState) -> dict:
    watchlist = state.get("watchlist") or all_tickers()
    snapshot = _portfolio_snapshot()
    return {
        "watchlist": watchlist,
        "portfolio_snapshot": snapshot,
        "cycle_number": state.get("cycle_number", 0) + 1,
        "status_log": [
            f"Cycle {state.get('cycle_number', 0) + 1}: gathered context "
            f"(equity ${snapshot['total_equity']:.0f}, "
            f"{len(snapshot['positions'])} open positions)"
        ],
    }


def _build_node_with_tools(coro):
    """Wrap an async node so LangGraph can invoke it. Tools are loaded once
    per graph invocation via `open_floor` context manager held at the graph
    layer (set as a module-global by `run_session`)."""

    async def node(state: FloorState) -> dict:
        tools = _CURRENT_TOOLS
        if tools is None:
            raise RuntimeError("MCP tools not loaded; call run_session().")
        return await coro(state, tools)

    return node


_CURRENT_TOOLS: dict[str, BaseTool] | None = None


async def _research_node(state: FloorState, tools: dict[str, BaseTool]) -> dict:
    brief, proposals = await asyncio.to_thread(
        run_research_crew,
        state["watchlist"],
        state["portfolio_snapshot"],
        tools,
    )
    return {
        "research_brief": brief,
        "proposals": proposals,
        "status_log": [f"Research crew produced {len(proposals)} proposal(s)"],
    }


async def _risk_node(state: FloorState, tools: dict[str, BaseTool]) -> dict:
    proposals: list[TradeProposal] = state.get("proposals") or []
    if not proposals:
        return {"assessments": [], "status_log": ["No proposals to review"]}

    risk_tools = _filter(tools, ["portfolio", "risk", "audit"])
    agent = build_risk_officer(risk_tools)

    assessments: list[RiskAssessment] = []
    for proposal in proposals:
        instruction = (
            "Evaluate this TradeProposal. Use the available risk and portfolio tools "
            "to check limits, then call log_decision on the audit server with your verdict.\n\n"
            f"Proposal: {proposal.model_dump_json()}\n\n"
            "After your tool calls, output a single JSON object on the final line with keys: "
            "approved (bool), reasons (list of strings), suggested_quantity (int or null), "
            "risk_score (0..1)."
        )
        result = await agent.ainvoke({"messages": [HumanMessage(content=instruction)]})
        last = result["messages"][-1].content if result.get("messages") else ""
        verdict = _parse_verdict(last)
        assessments.append(
            RiskAssessment(
                proposal=proposal,
                approved=bool(verdict.get("approved", False)),
                reasons=list(verdict.get("reasons", [])),
                suggested_quantity=verdict.get("suggested_quantity"),
                risk_score=float(verdict.get("risk_score", 0.5)),
            )
        )

    approved = sum(1 for a in assessments if a.approved)
    return {
        "assessments": assessments,
        "status_log": [f"Risk Officer approved {approved}/{len(assessments)} proposal(s)"],
    }


def _parse_verdict(text: str) -> dict:
    import re

    matches = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    for chunk in reversed(matches):
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            continue
    return {}


async def _execute_node(state: FloorState, tools: dict[str, BaseTool]) -> dict:
    assessments = state.get("assessments") or []
    approved = [a for a in assessments if a.approved]
    if not approved:
        return {"executed_orders": [], "status_log": ["Nothing approved to execute"]}

    exec_tools = _filter(tools, ["execution", "portfolio", "audit"])
    agent = build_trader(exec_tools)

    executed: list[dict] = []
    for assessment in approved:
        proposal = assessment.proposal
        qty = assessment.suggested_quantity or proposal.quantity
        instruction = (
            "Place this approved trade using the execution server. Then call log_trade "
            "on the audit server with the resulting order payload.\n\n"
            f"Approved trade: ticker={proposal.ticker}, side={proposal.side.value}, "
            f"quantity={qty}, order_type={proposal.order_type.value}, "
            f"limit_price={proposal.limit_price}\n\n"
            "On the final line, return a JSON object with the order details you submitted."
        )
        result = await agent.ainvoke({"messages": [HumanMessage(content=instruction)]})
        last = result["messages"][-1].content if result.get("messages") else ""
        executed.append(
            {
                "ticker": proposal.ticker,
                "side": proposal.side.value,
                "quantity": qty,
                "result_summary": last,
            }
        )

    return {
        "executed_orders": executed,
        "status_log": [f"Trader placed {len(executed)} order(s)"],
    }


async def _close_cycle(state: FloorState, tools: dict[str, BaseTool]) -> dict:
    cycle = state.get("cycle_number", 1)
    audit_repo.write_entry(
        actor="orchestrator",
        kind="event",
        payload={
            "event": "cycle_closed",
            "cycle": cycle,
            "executed_orders": state.get("executed_orders") or [],
            "snapshot_after": _portfolio_snapshot(),
        },
    )
    max_cycles = state.get("max_cycles", 1)
    done = cycle >= max_cycles
    return {
        "done": done,
        "status_log": [f"Cycle {cycle} closed; done={done}"],
    }


def _route_next(state: FloorState) -> str:
    return "end" if state.get("done") else "continue"


def build_graph():
    builder = StateGraph(FloorState)
    builder.add_node("gather_context", _gather_context)
    builder.add_node("research_crew", _build_node_with_tools(_research_node))
    builder.add_node("review_risk", _build_node_with_tools(_risk_node))
    builder.add_node("execute_trades", _build_node_with_tools(_execute_node))
    builder.add_node("close_cycle", _build_node_with_tools(_close_cycle))

    builder.add_edge(START, "gather_context")
    builder.add_edge("gather_context", "research_crew")
    builder.add_edge("research_crew", "review_risk")
    builder.add_edge("review_risk", "execute_trades")
    builder.add_edge("execute_trades", "close_cycle")
    builder.add_conditional_edges(
        "close_cycle",
        _route_next,
        {"continue": "gather_context", "end": END},
    )
    return builder.compile()


async def run_session(
    watchlist: list[str] | None = None, max_cycles: int = 1
) -> AsyncIterator[FloorState]:
    """Run a complete trading session, yielding state snapshots as it progresses."""
    global _CURRENT_TOOLS
    graph = build_graph()
    async with open_floor() as tools:
        _CURRENT_TOOLS = tools
        try:
            async for event in graph.astream(
                {
                    "watchlist": watchlist or all_tickers(),
                    "max_cycles": max_cycles,
                    "cycle_number": 0,
                    "executed_orders": [],
                    "status_log": [],
                },
                stream_mode="values",
                config={"recursion_limit": 50},
            ):
                yield event
        finally:
            _CURRENT_TOOLS = None
