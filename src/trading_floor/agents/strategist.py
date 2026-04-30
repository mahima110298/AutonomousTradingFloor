"""Strategy Analyst: a CrewAI agent that turns a research brief into trade proposals."""

from __future__ import annotations

from crewai import Agent
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from trading_floor.config import get_settings
from trading_floor.prompts import STRATEGIST_SYSTEM


def build_strategist(tools: list[BaseTool]) -> Agent:
    settings = get_settings()
    return Agent(
        role="Strategy Analyst",
        goal="Convert the research brief into 1-3 high-conviction TradeProposal objects.",
        backstory=STRATEGIST_SYSTEM,
        llm=ChatOpenAI(model=settings.strategist_model, temperature=0.2),
        tools=tools,
        allow_delegation=False,
        verbose=False,
    )
