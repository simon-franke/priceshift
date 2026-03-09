"""Replay historical gap data from DuckDB and compute P&L stats."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from priceshift.db.store import DataStore
from priceshift.models import ArbitrageGap, BacktestResult, Platform
from priceshift.trading.simulator import PaperTrader

logger = logging.getLogger(__name__)


class Backtester:
    """Replay gap history and compute aggregate stats."""

    def __init__(
        self,
        store: DataStore,
        min_gap_open_pp: float = 3.0,
        min_gap_close_pp: float = 1.0,
        position_size: float = 100.0,
    ) -> None:
        self._store = store
        self._min_gap_open = min_gap_open_pp
        self._min_gap_close = min_gap_close_pp
        self._position_size = position_size

    def run(
        self,
        polymarket_id: Optional[str] = None,
        kalshi_ticker: Optional[str] = None,
    ) -> BacktestResult:
        """
        Replay gaps from DuckDB in time order.
        If polymarket_id + kalshi_ticker given, filter to that pair.
        """
        if polymarket_id and kalshi_ticker:
            gaps_raw = self._store.get_gaps_for_pair(polymarket_id, kalshi_ticker)
        else:
            gaps_raw = self._store.get_latest_gaps(limit=10_000)
            # sort ascending
            gaps_raw = sorted(gaps_raw, key=lambda g: g["timestamp"])

        if not gaps_raw:
            logger.warning("No gap data found for backtest")
            return self._empty_result()

        # Simulate using in-memory state (don't write to real DB)
        open_trades: dict[int, dict] = {}  # pair_id → trade
        closed_trades: list[dict] = []
        next_id = 1

        for row in gaps_raw:
            pair_id = int(row["pair_id"])
            gap_pp = float(row["gap_pp"])
            abs_gap = abs(gap_pp)
            ts = row["timestamp"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)

            if pair_id in open_trades:
                if abs_gap < self._min_gap_close:
                    t = open_trades.pop(pair_id)
                    pnl = (abs(t["open_gap_pp"]) - abs_gap) / 100.0 * self._position_size
                    closed_trades.append({**t, "close_gap_pp": gap_pp, "pnl": pnl, "closed_at": ts})
            else:
                if abs_gap >= self._min_gap_open:
                    open_trades[pair_id] = {
                        "id": next_id,
                        "pair_id": pair_id,
                        "open_gap_pp": gap_pp,
                        "opened_at": ts,
                    }
                    next_id += 1

        if not closed_trades:
            return self._empty_result()

        pnls = [t["pnl"] for t in closed_trades]
        holds = [
            (t["closed_at"] - t["opened_at"]).total_seconds()
            for t in closed_trades
            if isinstance(t["opened_at"], datetime) and isinstance(t["closed_at"], datetime)
        ]

        total = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / total
        win_rate = wins / total if total else 0.0
        avg_hold = sum(holds) / len(holds) if holds else 0.0

        # Max drawdown (running cumulative)
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cumulative += p
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        start = closed_trades[0]["opened_at"] if closed_trades else datetime.utcnow()
        end = closed_trades[-1]["closed_at"] if closed_trades else datetime.utcnow()

        return BacktestResult(
            start_time=start if isinstance(start, datetime) else datetime.utcnow(),
            end_time=end if isinstance(end, datetime) else datetime.utcnow(),
            total_trades=total,
            winning_trades=wins,
            losing_trades=total - wins,
            total_pnl=round(total_pnl, 4),
            avg_pnl_per_trade=round(avg_pnl, 4),
            win_rate=round(win_rate, 4),
            avg_hold_duration_seconds=round(avg_hold, 1),
            max_drawdown=round(max_dd, 4),
        )

    def _empty_result(self) -> BacktestResult:
        now = datetime.utcnow()
        return BacktestResult(
            start_time=now,
            end_time=now,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_pnl=0.0,
            avg_pnl_per_trade=0.0,
            win_rate=0.0,
            avg_hold_duration_seconds=0.0,
            max_drawdown=0.0,
        )
