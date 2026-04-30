"""Risk Officer: a LangGraph tool-calling agent that approves or rejects proposals."""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from trading_floor.config import get_settings
from trading_floor.prompts import RISK_OFFICER_SYSTEM


def build_risk_officer(tools: list[BaseTool]):
    settings = get_settings()
    llm = ChatOpenAI(model=settings.risk_model, temperature=0.0)
    return create_react_agent(llm, tools=tools, prompt=RISK_OFFICER_SYSTEM)
