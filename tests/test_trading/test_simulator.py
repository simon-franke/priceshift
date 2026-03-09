"""Tests for the paper trading simulator."""
from __future__ import annotations

from datetime import datetime

import pytest

from priceshift.models import ArbitrageGap, MatchedPair
from priceshift.trading.simulator import PaperTrader, _estimate_pnl


def make_gap(pair_id: int, pm_yes: float, kalshi_yes: float) -> ArbitrageGap:
    return ArbitrageGap.from_prices(
        pair_id=pair_id,
        polymarket_id="pm-1",
        kalshi_ticker="KAL-A",
        pm_yes=pm_yes,
        kalshi_yes=kalshi_yes,
    )


def setup_pair(tmp_store) -> int:
    pair = MatchedPair(polymarket_id="pm-1", kalshi_ticker="KAL-A", similarity_score=0.9)
    return tmp_store.upsert_matched_pair(pair)


def test_no_trade_below_threshold(tmp_store):
    pair_id = setup_pair(tmp_store)
    trader = PaperTrader(tmp_store, min_gap_open_pp=3.0)
    gap = make_gap(pair_id, pm_yes=0.52, kalshi_yes=0.50)  # 2pp gap
    result = trader.process_gap(gap)
    assert result is None
    assert len(tmp_store.get_open_trades()) == 0


def test_opens_trade_above_threshold(tmp_store):
    pair_id = setup_pair(tmp_store)
    trader = PaperTrader(tmp_store, min_gap_open_pp=3.0)
    gap = make_gap(pair_id, pm_yes=0.60, kalshi_yes=0.55)  # 5pp
    result = trader.process_gap(gap)
    assert result is not None
    assert result.id is not None
    assert len(tmp_store.get_open_trades()) == 1


def test_closes_trade_when_gap_narrows(tmp_store):
    pair_id = setup_pair(tmp_store)
    trader = PaperTrader(tmp_store, min_gap_open_pp=3.0, min_gap_close_pp=1.0)

    # Open
    open_gap = make_gap(pair_id, pm_yes=0.60, kalshi_yes=0.55)
    trader.process_gap(open_gap)
    assert len(tmp_store.get_open_trades()) == 1

    # Close
    close_gap = make_gap(pair_id, pm_yes=0.57, kalshi_yes=0.565)  # ~0.5pp
    result = trader.process_gap(close_gap)
    assert result is not None
    assert len(tmp_store.get_open_trades()) == 0
    all_trades = tmp_store.get_all_trades()
    assert all_trades[0]["status"] == "closed"
    assert all_trades[0]["realized_pnl"] is not None


def test_does_not_open_second_trade_for_same_pair(tmp_store):
    pair_id = setup_pair(tmp_store)
    trader = PaperTrader(tmp_store, min_gap_open_pp=3.0)
    gap = make_gap(pair_id, pm_yes=0.60, kalshi_yes=0.55)
    trader.process_gap(gap)
    trader.process_gap(gap)  # second call with same pair_id
    assert len(tmp_store.get_open_trades()) == 1


def test_estimate_pnl_positive_when_gap_closes():
    trade = {
        "open_gap_pp": 5.0,
        "position_size": 100.0,
    }
    pnl = _estimate_pnl(trade, close_gap_pp=0.5)
    assert pnl > 0


def test_estimate_pnl_negative_when_gap_widens():
    trade = {
        "open_gap_pp": 5.0,
        "position_size": 100.0,
    }
    pnl = _estimate_pnl(trade, close_gap_pp=8.0)
    assert pnl < 0
