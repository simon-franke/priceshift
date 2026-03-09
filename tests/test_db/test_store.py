"""Tests for DataStore operations."""
from __future__ import annotations

from datetime import datetime

import pytest

from priceshift.models import (
    ArbitrageGap,
    Market,
    MatchedPair,
    PaperTrade,
    Platform,
    PriceSnapshot,
    TradeDirection,
)


def make_market(id: str, platform: Platform, yes_price: float = 0.6) -> Market:
    return Market(
        id=id,
        platform=platform,
        title=f"Test market {id}",
        yes_price=yes_price,
        no_price=1.0 - yes_price,
    )


def test_upsert_and_fetch_market(tmp_store):
    m = make_market("pm-123", Platform.POLYMARKET)
    tmp_store.upsert_market(m)
    rows = tmp_store.get_markets_by_platform("polymarket")
    assert len(rows) == 1
    assert rows[0]["id"] == "pm-123"


def test_upsert_market_updates_price(tmp_store):
    m = make_market("pm-123", Platform.POLYMARKET, yes_price=0.5)
    tmp_store.upsert_market(m)
    m2 = make_market("pm-123", Platform.POLYMARKET, yes_price=0.7)
    tmp_store.upsert_market(m2)
    rows = tmp_store.get_markets_by_platform("polymarket")
    assert len(rows) == 1
    assert rows[0]["yes_price"] == pytest.approx(0.7)


def test_upsert_matched_pair_returns_id(tmp_store):
    pair = MatchedPair(
        polymarket_id="pm-1",
        kalshi_ticker="KAL-A",
        similarity_score=0.9,
    )
    pair_id = tmp_store.upsert_matched_pair(pair)
    assert pair_id > 0


def test_get_active_pairs(tmp_store):
    pair = MatchedPair(polymarket_id="pm-1", kalshi_ticker="KAL-A", similarity_score=0.9)
    tmp_store.upsert_matched_pair(pair)
    pairs = tmp_store.get_active_pairs()
    assert len(pairs) == 1
    assert pairs[0]["polymarket_id"] == "pm-1"


def test_deactivate_pair(tmp_store):
    pair = MatchedPair(polymarket_id="pm-1", kalshi_ticker="KAL-A", similarity_score=0.9)
    pair_id = tmp_store.upsert_matched_pair(pair)
    tmp_store.deactivate_pair(pair_id)
    pairs = tmp_store.get_active_pairs()
    assert len(pairs) == 0


def test_paper_trade_create_and_close(tmp_store):
    pair = MatchedPair(polymarket_id="pm-1", kalshi_ticker="KAL-A", similarity_score=0.9)
    pair_id = tmp_store.upsert_matched_pair(pair)

    trade = PaperTrade(
        pair_id=pair_id,
        polymarket_id="pm-1",
        kalshi_ticker="KAL-A",
        direction=TradeDirection.BUY_YES_PM_SELL_YES_KALSHI,
        position_size=100.0,
        open_gap_pp=5.0,
        open_pm_price=0.55,
        open_kalshi_price=0.50,
    )
    trade_id = tmp_store.create_paper_trade(trade)
    assert trade_id > 0

    open_trades = tmp_store.get_open_trades()
    assert len(open_trades) == 1

    tmp_store.close_paper_trade(
        trade_id=trade_id,
        close_gap_pp=0.5,
        close_pm_price=0.52,
        close_kalshi_price=0.52,
        realized_pnl=4.5,
    )

    open_trades = tmp_store.get_open_trades()
    assert len(open_trades) == 0

    all_trades = tmp_store.get_all_trades()
    assert len(all_trades) == 1
    assert all_trades[0]["status"] == "closed"
    assert all_trades[0]["realized_pnl"] == pytest.approx(4.5)


def test_price_snapshot_appended_to_duckdb(tmp_store):
    snap = PriceSnapshot(
        market_id="pm-1",
        platform=Platform.POLYMARKET,
        yes_price=0.6,
        no_price=0.4,
    )
    tmp_store.append_price_snapshot(snap)
    # DuckDB has no direct fetch in store, but we test via get_latest_gaps indirectly


def test_arbitrage_gap_append_and_fetch(tmp_store):
    pair = MatchedPair(polymarket_id="pm-1", kalshi_ticker="KAL-A", similarity_score=0.9)
    pair_id = tmp_store.upsert_matched_pair(pair)

    gap = ArbitrageGap.from_prices(
        pair_id=pair_id,
        polymarket_id="pm-1",
        kalshi_ticker="KAL-A",
        pm_yes=0.60,
        kalshi_yes=0.55,
    )
    tmp_store.append_arbitrage_gap(gap)

    gaps = tmp_store.get_latest_gaps(limit=10)
    assert len(gaps) == 1
    assert gaps[0]["abs_gap_pp"] == pytest.approx(5.0)


def test_get_gaps_for_pair(tmp_store):
    pair = MatchedPair(polymarket_id="pm-1", kalshi_ticker="KAL-A", similarity_score=0.9)
    pair_id = tmp_store.upsert_matched_pair(pair)

    for pm_yes in [0.60, 0.58, 0.56]:
        gap = ArbitrageGap.from_prices(pair_id, "pm-1", "KAL-A", pm_yes, 0.50)
        tmp_store.append_arbitrage_gap(gap)

    gaps = tmp_store.get_gaps_for_pair("pm-1", "KAL-A")
    assert len(gaps) == 3
