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

import json
from typing import AsyncIterator

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from trading_floor.agents.risk_officer import build_risk_officer
from trading_floor.agents.trader import build_trader
from trading_floor.config import get_settings
from trading_floor.crews.research_crew import run_research_crew
from trading_floor.data.universe import all_tickers
from trading_floor.mcp_client import open_floor
from trading_floor.models import RiskAssessment, TradeProposal
from trading_floor.state import FloorState
from trading_floor.storage import audit_repo, portfolio_repo


class _VerdictPayload(BaseModel):
    approved: bool = Field(..., description="True only if the trade should be executed.")
    reasons: list[str] = Field(default_factory=list)
    suggested_quantity: int | None = None
    risk_score: float = Field(0.5, ge=0.0, le=1.0)


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
    # Awaits CrewAI's async kickoff (crew.kickoff_async()) directly, so this
    # stays on the same task/event loop that's servicing the MCP session in
    # mcp_client.open_floor — no thread handoff, no separate task/loop for
    # the tool-call bridge to desync against.
    brief, proposals = await run_research_crew(
        state["watchlist"],
        state["portfolio_snapshot"],
        tools,
    )
    sized = _enforce_sizing(proposals, state["portfolio_snapshot"])
    return {
        "research_brief": brief,
        "proposals": sized,
        "status_log": [
            f"Research crew produced {len(proposals)} proposal(s); "
            f"sizing-clipped to fit per-name limit"
        ],
    }


def _enforce_sizing(
    proposals: list[TradeProposal], snapshot: dict
) -> list[TradeProposal]:
    """Apply hard policy guardrails before risk review.

    - Drop SELL proposals on tickers the portfolio doesn't hold.
    - Clip BUY quantities so no single trade can exceed the per-name cap.

    This decouples policy enforcement from LLM compliance.
    """
    from trading_floor.data import market_store

    settings = get_settings()
    equity = float(snapshot.get("total_equity") or settings.starting_cash)
    cap_per_name_dollars = equity * (settings.max_position_pct / 100.0)
    held = {p["ticker"]: int(p["quantity"]) for p in snapshot.get("positions") or []}

    sized: list[TradeProposal] = []
    for p in proposals:
        if p.side.value == "sell":
            held_qty = held.get(p.ticker, 0)
            if held_qty <= 0:
                continue
            new_qty = min(p.quantity, held_qty)
            if new_qty != p.quantity:
                sized.append(p.model_copy(update={"quantity": new_qty}))
            else:
                sized.append(p)
            continue

        try:
            price = market_store.latest_close(p.ticker)
        except ValueError:
            continue
        max_qty = max(1, int(cap_per_name_dollars / price))
        new_qty = min(p.quantity, max_qty)
        if new_qty != p.quantity:
            sized.append(p.model_copy(update={"quantity": new_qty}))
        else:
            sized.append(p)
    return sized


async def _risk_node(state: FloorState, tools: dict[str, BaseTool]) -> dict:
    proposals: list[TradeProposal] = state.get("proposals") or []
    if not proposals:
        return {"assessments": [], "status_log": ["No proposals to review"]}

    risk_tool_names = {
        "assess_trade_risk", "check_drawdown", "check_position_limit",
        "check_sector_exposure_limit", "compute_position_size",
    }
    risk_tools = [t for t in tools.values() if any(n in t.name for n in risk_tool_names)]
    agent = build_risk_officer(risk_tools)

    extractor_llm = (
        ChatOpenAI(model=get_settings().risk_model, temperature=0.0)
        .with_structured_output(_VerdictPayload)
    )

    assessments: list[RiskAssessment] = []
    for proposal in proposals:
        instruction = (
            "Evaluate this TradeProposal. Use the risk and portfolio tools to check "
            "position limits, sector exposure, drawdown, and cash availability. Then "
            "call log_decision on the audit server with your verdict and reasoning.\n\n"
            f"Proposal: {proposal.model_dump_json()}\n\n"
            "Finally, write a brief one-paragraph summary of your decision. The system "
            "will extract the structured verdict from your summary."
        )
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=instruction)]},
            config={"recursion_limit": 60},
        )
        last = result["messages"][-1].content if result.get("messages") else ""
        try:
            verdict: _VerdictPayload = await extractor_llm.ainvoke(
                f"Extract the structured verdict from this Risk Officer summary:\n\n{last}"
            )
            assessment = RiskAssessment(
                proposal=proposal,
                approved=verdict.approved,
                reasons=verdict.reasons,
                suggested_quantity=verdict.suggested_quantity,
                risk_score=verdict.risk_score,
            )
        except Exception as exc:
            assessment = RiskAssessment(
                proposal=proposal,
                approved=False,
                reasons=[f"verdict extraction failed: {exc}"],
                risk_score=1.0,
            )

        audit_repo.write_entry(
            actor="risk_officer",
            kind="decision",
            payload={
                "proposal": proposal.model_dump(mode="json"),
                "approved": assessment.approved,
                "reasons": assessment.reasons,
                "suggested_quantity": assessment.suggested_quantity,
                "risk_score": assessment.risk_score,
                "officer_summary": last,
            },
        )
        assessments.append(assessment)

    approved = sum(1 for a in assessments if a.approved)
    return {
        "assessments": assessments,
        "status_log": [f"Risk Officer approved {approved}/{len(assessments)} proposal(s)"],
    }


async def _execute_node(state: FloorState, tools: dict[str, BaseTool]) -> dict:
    assessments = state.get("assessments") or []
    approved = [a for a in assessments if a.approved]
    if not approved:
        return {"executed_orders": [], "status_log": ["Nothing approved to execute"]}

    trader_tool_names = {
        "submit_market_order", "submit_limit_order", "get_order_status",
        "list_open_orders", "get_position", "get_cash_balance",
    }
    exec_tools = [t for t in tools.values() if any(n in t.name for n in trader_tool_names)]
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
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=instruction)]},
            config={"recursion_limit": 60},
        )
        last = result["messages"][-1].content if result.get("messages") else ""
        record = {
            "ticker": proposal.ticker,
            "side": proposal.side.value,
            "quantity": qty,
            "result_summary": last,
        }
        audit_repo.write_entry(actor="trader", kind="trade", payload=record)
        executed.append(record)

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
