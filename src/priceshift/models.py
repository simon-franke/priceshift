"""Shared Pydantic models for all priceshift modules."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Platform(str, Enum):
    POLYMARKET = "polymarket"
    KALSHI = "kalshi"


class MarketStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    RESOLVED = "resolved"
    UNKNOWN = "unknown"


class TradeDirection(str, Enum):
    BUY_YES_PM_SELL_YES_KALSHI = "buy_yes_pm_sell_yes_kalshi"
    SELL_YES_PM_BUY_YES_KALSHI = "sell_yes_pm_buy_yes_kalshi"


class TradeStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


class Market(BaseModel):
    """Normalized market from either platform."""

    id: str
    platform: Platform
    title: str
    description: str = ""
    category: str = ""
    status: MarketStatus = MarketStatus.UNKNOWN
    resolution_date: Optional[datetime] = None
    yes_price: Optional[float] = None  # 0.0–1.0
    no_price: Optional[float] = None  # 0.0–1.0
    volume: Optional[float] = None
    liquidity: Optional[float] = None
    created_at: Optional[datetime] = None
    fetched_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def mid_price(self) -> Optional[float]:
        if self.yes_price is not None:
            return self.yes_price
        return None


class PriceSnapshot(BaseModel):
    """Timestamped yes/no price for one market."""

    market_id: str
    platform: Platform
    yes_price: float
    no_price: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    volume: Optional[float] = None
    liquidity: Optional[float] = None


class MatchedPair(BaseModel):
    """Linked polymarket_id ↔ kalshi_ticker + similarity score."""

    id: Optional[int] = None
    polymarket_id: str
    kalshi_ticker: str
    polymarket_title: str = ""
    kalshi_title: str = ""
    similarity_score: float
    match_source: str = "semantic"  # "ground_truth", "rule", "semantic"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True


class ArbitrageGap(BaseModel):
    """Gap observation per pair per timestamp."""

    pair_id: int
    polymarket_id: str
    kalshi_ticker: str
    pm_yes_price: float
    kalshi_yes_price: float
    gap_pp: float  # polymarket - kalshi, in percentage points (×100)
    abs_gap_pp: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def from_prices(
        cls,
        pair_id: int,
        polymarket_id: str,
        kalshi_ticker: str,
        pm_yes: float,
        kalshi_yes: float,
        timestamp: Optional[datetime] = None,
    ) -> "ArbitrageGap":
        gap = (pm_yes - kalshi_yes) * 100
        return cls(
            pair_id=pair_id,
            polymarket_id=polymarket_id,
            kalshi_ticker=kalshi_ticker,
            pm_yes_price=pm_yes,
            kalshi_yes_price=kalshi_yes,
            gap_pp=gap,
            abs_gap_pp=abs(gap),
            timestamp=timestamp or datetime.utcnow(),
        )


class PaperTrade(BaseModel):
    """Open or closed simulated position."""

    id: Optional[int] = None
    pair_id: int
    polymarket_id: str
    kalshi_ticker: str
    direction: TradeDirection
    position_size: float  # notional dollars
    open_gap_pp: float
    open_pm_price: float
    open_kalshi_price: float
    close_gap_pp: Optional[float] = None
    close_pm_price: Optional[float] = None
    close_kalshi_price: Optional[float] = None
    realized_pnl: Optional[float] = None
    status: TradeStatus = TradeStatus.OPEN
    opened_at: datetime = Field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None

    @property
    def hold_duration_seconds(self) -> Optional[float]:
        if self.closed_at is None:
            return None
        return (self.closed_at - self.opened_at).total_seconds()


class BacktestResult(BaseModel):
    """Aggregated stats from a historical replay."""

    start_time: datetime
    end_time: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    avg_pnl_per_trade: float
    win_rate: float
    avg_hold_duration_seconds: float
    max_drawdown: float
    sharpe_ratio: Optional[float] = None

    @property
    def lose_rate(self) -> float:
        return 1.0 - self.win_rate
