"""Tests for the event matching pipeline."""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from priceshift.db.store import DataStore
from priceshift.matching.matcher import EventMatcher, _tokenize
from priceshift.matching.verifier import MatchVerifier
from priceshift.models import Market, Platform


def make_market(
    id: str,
    title: str,
    platform: Platform = Platform.POLYMARKET,
    resolution_date: datetime | None = None,
    description: str = "",
) -> Market:
    return Market(
        id=id,
        platform=platform,
        title=title,
        resolution_date=resolution_date,
        description=description,
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


# ---------------------------------------------------------------------------
# Verifier integration tests
# ---------------------------------------------------------------------------


@patch("priceshift.matching.verifier._get_nli_model")
def test_verifier_rejects_win_vs_qualify(mock_get_model, tmp_path):
    """Regression: 'win' vs 'qualify' must not match through full pipeline."""
    mock_model = MagicMock()
    # Return high contradiction scores
    mock_model.predict.return_value = [np.array([0.85, 0.05, 0.10])]
    mock_get_model.return_value = mock_model

    store = DataStore(sqlite_path=str(tmp_path / "test.sqlite"))
    verifier = MatchVerifier(store=store, use_ollama_fallback=False)
    matcher = EventMatcher(
        semantic_threshold=0.5,  # low threshold so semantic passes
        ground_truth_path=str(tmp_path / "gt.json"),
        cache_dir=str(tmp_path / "cache"),
        verifier=verifier,
    )
    pm = make_market("pm-spain-win", "Spain win World Cup 2026")
    kalshi = make_market(
        "kal-spain-qual", "Spain qualify for World Cup 2026",
        platform=Platform.KALSHI,
    )
    result = matcher.match_one(pm, [kalshi])
    assert result is None


@patch("priceshift.matching.verifier._get_nli_model")
def test_verified_match_has_correct_source(mock_get_model, tmp_path):
    """Verified matches should have source='nli_verified'."""
    mock_model = MagicMock()
    mock_model.predict.return_value = [np.array([0.05, 0.90, 0.05])]
    mock_get_model.return_value = mock_model

    store = DataStore(sqlite_path=str(tmp_path / "test.sqlite"))
    verifier = MatchVerifier(store=store, use_ollama_fallback=False)
    matcher = EventMatcher(
        semantic_threshold=0.5,
        ground_truth_path=str(tmp_path / "gt.json"),
        cache_dir=str(tmp_path / "cache"),
        verifier=verifier,
    )
    pm = make_market("pm-btc", "Bitcoin over 100k by December 2025")
    kalshi = make_market(
        "kal-btc", "Bitcoin above 100k December 2025",
        platform=Platform.KALSHI,
    )
    result = matcher.match_one(pm, [kalshi])
    assert result is not None
    assert result.match_source == "nli_verified"


def test_matcher_works_without_verifier(tmp_path):
    """When no verifier is configured, matcher still works (backward compat)."""
    matcher = EventMatcher(
        semantic_threshold=0.99,
        ground_truth_path=str(tmp_path / "gt.json"),
        cache_dir=str(tmp_path / "cache"),
        verifier=None,
    )
    pm = make_market("pm-1", "Some unique event")
    kalshi = make_market("KAL-1", "Totally different", platform=Platform.KALSHI)
    result = matcher.match_one(pm, [kalshi])
    assert result is None


@patch("priceshift.matching.verifier._get_nli_model")
def test_enriched_description_distinguishes_winner_from_qualifier(mock_get_model, tmp_path):
    """Regression: enriched Kalshi description must prevent winner/qualifier false positives.

    With enrichment:
    - 'winner' market → description='FIFA World Cup 2026 Winner: Spain' → NLI match
    - 'qualifier' market → description='FIFA World Cup 2026 Qualifier: Spain' → NLI reject
    """
    call_count = 0

    def side_effect(pairs):
        nonlocal call_count
        call_count += 1
        text_a, text_b = pairs[0]
        # Contradict when "qualifier" appears in either text
        if "qualifier" in text_a.lower() or "qualifier" in text_b.lower():
            return [np.array([0.85, 0.05, 0.10])]
        return [np.array([0.05, 0.90, 0.05])]

    mock_model = MagicMock()
    mock_model.predict.side_effect = side_effect
    mock_get_model.return_value = mock_model

    store = DataStore(sqlite_path=str(tmp_path / "test.sqlite"))
    verifier = MatchVerifier(store=store, use_ollama_fallback=False)
    matcher = EventMatcher(
        semantic_threshold=0.5,
        ground_truth_path=str(tmp_path / "gt.json"),
        cache_dir=str(tmp_path / "cache"),
        verifier=verifier,
    )

    pm = make_market("pm-spain-win", "Will Spain win the 2026 FIFA World Cup?")

    kalshi_winner = make_market(
        "kal-spain-winner", "Spain",
        platform=Platform.KALSHI,
        description="FIFA World Cup 2026 Winner: Spain",
    )
    kalshi_qualifier = make_market(
        "kal-spain-qual", "Spain",
        platform=Platform.KALSHI,
        description="FIFA World Cup 2026 Qualifier: Spain",
    )

    result_winner = matcher.match_one(pm, [kalshi_winner])
    assert result_winner is not None, "Winner market should match"

    result_qualifier = matcher.match_one(pm, [kalshi_qualifier])
    assert result_qualifier is None, "Qualifier market must NOT match"
