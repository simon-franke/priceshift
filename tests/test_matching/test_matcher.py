"""Tests for the event matching pipeline."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from priceshift.matching.matcher import EventMatcher, _tokenize
from priceshift.models import Market, Platform


def make_market(
    id: str,
    title: str,
    platform: Platform = Platform.POLYMARKET,
    resolution_date: datetime | None = None,
) -> Market:
    return Market(
        id=id,
        platform=platform,
        title=title,
        resolution_date=resolution_date,
    )


def test_tokenize_removes_stopwords():
    tokens = _tokenize("Will the Fed cut rates in March?")
    assert "will" not in tokens
    assert "the" not in tokens
    assert "fed" in tokens
    assert "cut" in tokens
    assert "rates" in tokens
    assert "march" in tokens


def test_tokenize_empty():
    assert _tokenize("") == set()


def test_no_match_below_threshold(tmp_path):
    matcher = EventMatcher(
        semantic_threshold=0.99,
        ground_truth_path=str(tmp_path / "gt.json"),
        cache_dir=str(tmp_path / "cache"),
    )
    pm = make_market("pm-1", "Will Apple release a new iPhone in 2025?")
    kalshi = make_market("KAL-1", "Will GDP growth exceed 3% in Q4?", platform=Platform.KALSHI)
    result = matcher.match_one(pm, [kalshi])
    assert result is None


def test_rule_filter_rejects_date_mismatch(tmp_path):
    matcher = EventMatcher(
        semantic_threshold=0.5,
        date_window_days=14,
        ground_truth_path=str(tmp_path / "gt.json"),
        cache_dir=str(tmp_path / "cache"),
    )
    base = datetime(2025, 6, 1)
    pm = make_market("pm-1", "Fed rate cut June", resolution_date=base)
    kalshi = make_market(
        "KAL-1",
        "Fed rate cut September",
        platform=Platform.KALSHI,
        resolution_date=base + timedelta(days=100),
    )
    result = matcher.match_one(pm, [kalshi])
    assert result is None


def test_ground_truth_returns_score_1(tmp_path):
    import json

    gt_file = tmp_path / "gt.json"
    gt_file.write_text(json.dumps([{
        "polymarket_id": "pm-abc",
        "kalshi_ticker": "KAL-XYZ",
        "similarity_score": 1.0,
    }]))
    matcher = EventMatcher(
        ground_truth_path=str(gt_file),
        cache_dir=str(tmp_path / "cache"),
    )
    pm = make_market("pm-abc", "Some event")
    kalshi = make_market("KAL-XYZ", "Some event", platform=Platform.KALSHI)
    result = matcher.match_one(pm, [kalshi])
    assert result is not None
    assert result.similarity_score == 1.0
    assert result.match_source == "ground_truth"


def test_match_all_returns_list(tmp_path):
    matcher = EventMatcher(
        semantic_threshold=0.99,
        ground_truth_path=str(tmp_path / "gt.json"),
        cache_dir=str(tmp_path / "cache"),
    )
    pm_markets = [make_market(f"pm-{i}", f"Market {i}") for i in range(3)]
    kalshi_markets = [
        make_market(f"KAL-{i}", f"Kalshi {i}", platform=Platform.KALSHI) for i in range(3)
    ]
    results = matcher.match_all(pm_markets, kalshi_markets)
    assert isinstance(results, list)
