"""Pydantic models shared across the system."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class Quote(BaseModel):
    ticker: str
    price: float
    bid: float
    ask: float
    timestamp: datetime


class Candle(BaseModel):
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime


class Fundamentals(BaseModel):
    ticker: str
    name: str
    sector: str
    market_cap: float
    pe_ratio: float
    dividend_yield: float


class NewsItem(BaseModel):
    id: str
    headline: str
    body: str
    tickers: list[str]
    sentiment: float = Field(..., ge=-1.0, le=1.0)
    published_at: datetime


class Position(BaseModel):
    ticker: str
    quantity: int
    average_price: float
    market_price: float

    @property
    def market_value(self) -> float:
        return self.quantity * self.market_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.market_price - self.average_price) * self.quantity


class AccountSnapshot(BaseModel):
    cash: float
    positions: list[Position]
    total_equity: float
    realized_pnl: float
    unrealized_pnl: float
    timestamp: datetime


class TradeProposal(BaseModel):
    ticker: str
    side: Side
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    rationale: str
    conviction: float = Field(..., ge=0.0, le=1.0)
    target_horizon_days: int = 5


class RiskAssessment(BaseModel):
    proposal: TradeProposal
    approved: bool
    reasons: list[str]
    suggested_quantity: Optional[int] = None
    risk_score: float = Field(..., ge=0.0, le=1.0)


class Order(BaseModel):
    order_id: str
    ticker: str
    side: Side
    quantity: int
    order_type: OrderType
    limit_price: Optional[float]
    status: OrderStatus
    filled_quantity: int = 0
    filled_price: Optional[float] = None
    created_at: datetime
    updated_at: datetime


class AuditEntry(BaseModel):
    entry_id: str
    timestamp: datetime
    actor: str
    kind: Literal["decision", "trade", "event", "risk"]
    payload: dict


class MarketTheme(BaseModel):
    label: str
    description: str
    related_tickers: list[str]
    momentum: float = Field(..., ge=-1.0, le=1.0)
