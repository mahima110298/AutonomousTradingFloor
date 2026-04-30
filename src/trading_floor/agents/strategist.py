"""Strategy Analyst: a CrewAI agent that turns a research brief into trade proposals."""

from __future__ import annotations

from crewai import Agent, LLM
from langchain_core.tools import BaseTool

from trading_floor.agents.tool_adapters import adapt_many
from trading_floor.config import get_settings
from trading_floor.prompts import STRATEGIST_SYSTEM


def build_strategist(tools: list[BaseTool]) -> Agent:
    settings = get_settings()
    return Agent(
        role="Strategy Analyst",
        goal="Convert the research brief into 1-3 high-conviction TradeProposal objects.",
        backstory=STRATEGIST_SYSTEM,
        llm=LLM(model=f"openai/{settings.strategist_model}", temperature=0.2),
        tools=adapt_many(tools),
        allow_delegation=False,
        verbose=False,
    )
