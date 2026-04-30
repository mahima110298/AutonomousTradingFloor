"""Execution Trader: a LangGraph tool-calling agent that places approved orders."""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from trading_floor.config import get_settings
from trading_floor.prompts import TRADER_SYSTEM


def build_trader(tools: list[BaseTool]):
    settings = get_settings()
    llm = ChatOpenAI(model=settings.trader_model, temperature=0.0)
    return create_react_agent(llm, tools=tools, prompt=TRADER_SYSTEM)
