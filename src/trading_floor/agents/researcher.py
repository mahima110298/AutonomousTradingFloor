"""Market Researcher: a CrewAI agent that produces a factual brief on a watchlist."""

from __future__ import annotations

from crewai import Agent, LLM
from langchain_core.tools import BaseTool

from trading_floor.agents.tool_adapters import adapt_many
from trading_floor.config import get_settings
from trading_floor.prompts import RESEARCHER_SYSTEM


def build_researcher(tools: list[BaseTool]) -> Agent:
    settings = get_settings()
    return Agent(
        role="Market Research Analyst",
        goal="Build a concise, factual brief on each ticker in the watchlist.",
        backstory=RESEARCHER_SYSTEM,
        llm=LLM(model=f"openai/{settings.researcher_model}", temperature=0.1),
        tools=adapt_many(tools),
        allow_delegation=False,
        verbose=False,
    )
