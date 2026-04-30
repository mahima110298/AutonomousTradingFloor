"""The research crew: Market Researcher + Strategy Analyst, run via CrewAI.

The crew takes a watchlist and produces structured trade proposals. Tools
come from the MCP servers; we hand each agent only the tools it needs.
"""

from __future__ import annotations

import json
import re

from crewai import Crew, Process, Task
from langchain_core.tools import BaseTool

from trading_floor.agents.researcher import build_researcher
from trading_floor.agents.strategist import build_strategist
from trading_floor.models import OrderType, Side, TradeProposal


def _filter(tools: dict[str, BaseTool], substrings: list[str]) -> list[BaseTool]:
    out: list[BaseTool] = []
    for tool in tools.values():
        if any(s in tool.name for s in substrings):
            out.append(tool)
    return out


def _coerce_proposals(raw: str) -> list[TradeProposal]:
    """Pull the first JSON array out of the strategist's free-form output."""
    matches = re.findall(r"\[\s*\{.*?\}\s*\]", raw, flags=re.DOTALL)
    for chunk in matches:
        try:
            payload = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        proposals: list[TradeProposal] = []
        for item in payload:
            try:
                proposals.append(
                    TradeProposal(
                        ticker=str(item["ticker"]).upper(),
                        side=Side(str(item["side"]).lower()),
                        quantity=int(item["quantity"]),
                        order_type=OrderType(str(item.get("order_type", "market")).lower()),
                        limit_price=item.get("limit_price"),
                        rationale=str(item.get("rationale", "")),
                        conviction=float(item.get("conviction", 0.5)),
                        target_horizon_days=int(item.get("target_horizon_days", 5)),
                    )
                )
            except (KeyError, ValueError, TypeError):
                continue
        if proposals:
            return proposals
    return []


def run_research_crew(
    watchlist: list[str],
    portfolio_snapshot: dict,
    all_tools: dict[str, BaseTool],
) -> tuple[str, list[TradeProposal]]:
    """Synchronously run the CrewAI crew. Returns (research_brief, proposals)."""
    research_tools = _filter(all_tools, substrings=["market_data", "news"])
    strategy_tools = _filter(
        all_tools, substrings=["market_data", "news", "portfolio", "compute_position_size"]
    )

    researcher = build_researcher(research_tools)
    strategist = build_strategist(strategy_tools)

    research_task = Task(
        description=(
            "Build a research brief covering the watchlist below. "
            "For each ticker, include latest price, a 14-day RSI, a 30-day SMA, "
            "30-day annualized volatility, recent news sentiment, and 1-2 recent "
            "headlines verbatim. End the brief with a 'Signals to watch' list.\n\n"
            f"Watchlist: {', '.join(watchlist)}"
        ),
        expected_output="A markdown brief with one section per ticker, plus signals.",
        agent=researcher,
    )

    strategy_task = Task(
        description=(
            "Using the research brief and the portfolio snapshot below, propose "
            "1-3 TradeProposal objects as a JSON array. Each object must have keys: "
            "ticker, side ('buy'|'sell'), quantity (int), order_type "
            "('market'|'limit'), limit_price (number or null), rationale (string), "
            "conviction (0..1), target_horizon_days (int).\n\n"
            f"Portfolio snapshot: {json.dumps(portfolio_snapshot, default=str)}\n\n"
            "Return ONLY the JSON array as the final answer, with no surrounding prose."
        ),
        expected_output="A JSON array of trade proposals.",
        agent=strategist,
        context=[research_task],
    )

    crew = Crew(
        agents=[researcher, strategist],
        tasks=[research_task, strategy_task],
        process=Process.sequential,
        verbose=False,
    )
    result = crew.kickoff()
    final_text = str(result)
    return final_text, _coerce_proposals(final_text)
