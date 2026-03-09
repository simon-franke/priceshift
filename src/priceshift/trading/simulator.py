"""Mean-reversion paper trading simulator."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from priceshift.db.store import DataStore
from priceshift.models import (
    ArbitrageGap,
    PaperTrade,
    TradeDirection,
    TradeStatus,
)

logger = logging.getLogger(__name__)


def _estimate_pnl(trade: dict, close_gap_pp: float) -> float:
    """
    Simplified P&L estimate.

    When we open: we buy the cheaper side and sell the more expensive side.
    Gap at open = open_gap_pp (positive: PM > Kalshi).
    Gap at close = close_gap_pp.

    P&L ≈ (abs_gap_open - abs_gap_close) / 100 * position_size
    """
    open_gap = abs(trade["open_gap_pp"])
    close_gap = abs(close_gap_pp)
    position_size = trade["position_size"]
    return (open_gap - close_gap) / 100.0 * position_size


class PaperTrader:
    """Mean-reversion paper trading strategy."""

    def __init__(
        self,
        store: DataStore,
        min_gap_open_pp: float = 3.0,
        min_gap_close_pp: float = 1.0,
        position_size: float = 100.0,
        max_open_trades: int = 20,
    ) -> None:
        self._store = store
        self._min_gap_open = min_gap_open_pp
        self._min_gap_close = min_gap_close_pp
        self._position_size = position_size
        self._max_open_trades = max_open_trades

    def process_gap(self, gap: ArbitrageGap) -> Optional[PaperTrade]:
        """
        Evaluate a gap observation. Opens or closes trades as needed.
        Returns the trade if an action was taken, else None.
        """
        existing = self._store.get_open_trade_for_pair(gap.pair_id)

        if existing:
            return self._maybe_close(existing, gap)
        else:
            return self._maybe_open(gap)

    def _maybe_open(self, gap: ArbitrageGap) -> Optional[PaperTrade]:
        if gap.abs_gap_pp < self._min_gap_open:
            return None

        open_count = len(self._store.get_open_trades())
        if open_count >= self._max_open_trades:
            logger.warning("Max open trades (%d) reached, skipping", self._max_open_trades)
            return None

        direction = (
            TradeDirection.BUY_YES_PM_SELL_YES_KALSHI
            if gap.gap_pp < 0  # PM cheaper → buy YES on PM
            else TradeDirection.SELL_YES_PM_BUY_YES_KALSHI  # PM more expensive → sell YES on PM
        )

        trade = PaperTrade(
            pair_id=gap.pair_id,
            polymarket_id=gap.polymarket_id,
            kalshi_ticker=gap.kalshi_ticker,
            direction=direction,
            position_size=self._position_size,
            open_gap_pp=gap.gap_pp,
            open_pm_price=gap.pm_yes_price,
            open_kalshi_price=gap.kalshi_yes_price,
            opened_at=gap.timestamp,
        )

        trade_id = self._store.create_paper_trade(trade)
        trade.id = trade_id
        logger.info(
            "OPENED trade #%d: %s / %s gap=%.2fpp direction=%s",
            trade_id,
            gap.polymarket_id,
            gap.kalshi_ticker,
            gap.abs_gap_pp,
            direction.value,
        )
        return trade

    def _maybe_close(self, existing: dict, gap: ArbitrageGap) -> Optional[PaperTrade]:
        if gap.abs_gap_pp >= self._min_gap_close:
            return None

        realized_pnl = _estimate_pnl(existing, gap.gap_pp)
        self._store.close_paper_trade(
            trade_id=existing["id"],
            close_gap_pp=gap.gap_pp,
            close_pm_price=gap.pm_yes_price,
            close_kalshi_price=gap.kalshi_yes_price,
            realized_pnl=realized_pnl,
            closed_at=gap.timestamp,
        )

        closed_trade = PaperTrade(
            id=existing["id"],
            pair_id=existing["pair_id"],
            polymarket_id=existing["polymarket_id"],
            kalshi_ticker=existing["kalshi_ticker"],
            direction=existing["direction"],  # type: ignore
            position_size=existing["position_size"],
            open_gap_pp=existing["open_gap_pp"],
            open_pm_price=existing["open_pm_price"],
            open_kalshi_price=existing["open_kalshi_price"],
            close_gap_pp=gap.gap_pp,
            close_pm_price=gap.pm_yes_price,
            close_kalshi_price=gap.kalshi_yes_price,
            realized_pnl=realized_pnl,
            status=TradeStatus.CLOSED,
            opened_at=datetime.fromisoformat(existing["opened_at"]),
            closed_at=gap.timestamp,
        )

        logger.info(
            "CLOSED trade #%d: pnl=%.2f gap_closed=%.2fpp",
            existing["id"],
            realized_pnl,
            gap.abs_gap_pp,
        )

        try:
            self._store.append_completed_trade(closed_trade)
        except Exception as exc:
            logger.warning("Failed to write completed trade to DuckDB: %s", exc)

        return closed_trade

    def force_close_all(self, current_gaps: dict[int, ArbitrageGap]) -> list[PaperTrade]:
        """Close all open trades at current prices (e.g. on market resolution)."""
        closed = []
        for trade in self._store.get_open_trades():
            pair_id = trade["pair_id"]
            if pair_id in current_gaps:
                result = self._maybe_close(trade, current_gaps[pair_id])
                if result:
                    closed.append(result)
        return closed
