"""Tests for API client market normalization."""
from __future__ import annotations

import pytest

from priceshift.apis.kalshi import KalshiClient
from priceshift.apis.polymarket import PolymarketGammaClient
from priceshift.config import KalshiConfig, PolymarketConfig
from priceshift.models import MarketStatus, Platform


def pm_client() -> PolymarketGammaClient:
    return PolymarketGammaClient(PolymarketConfig())


def kal_client() -> KalshiClient:
    return KalshiClient(KalshiConfig())


def test_polymarket_normalizes_yes_price():
    client = pm_client()
    raw = {
        "conditionId": "0xabc",
        "question": "Will X happen?",
        "active": True,
        "outcomes": ["YES", "NO"],
        "outcomePrices": ["0.65", "0.35"],
    }
    m = client.normalize_market(raw)
    assert m is not None
    assert m.id == "0xabc"
    assert m.platform == Platform.POLYMARKET
    assert m.yes_price == pytest.approx(0.65)
    assert m.no_price == pytest.approx(0.35)
    assert m.status == MarketStatus.OPEN


def test_polymarket_returns_none_without_id():
    client = pm_client()
    m = client.normalize_market({})
    assert m is None


def test_kalshi_normalizes_cents_to_fraction():
    client = kal_client()
    raw = {
        "ticker": "FED-25JAN",
        "title": "Fed cuts in January 2025?",
        "status": "open",
        "yes_bid": 45,
        "no_bid": 55,
    }
    m = client.normalize_market(raw)
    assert m is not None
    assert m.id == "FED-25JAN"
    assert m.platform == Platform.KALSHI
    assert m.yes_price == pytest.approx(0.45)
    assert m.status == MarketStatus.OPEN


def test_kalshi_derives_no_price():
    client = kal_client()
    raw = {
        "ticker": "FED-25JAN",
        "title": "Test",
        "status": "open",
        "yes_bid": 60,
    }
    m = client.normalize_market(raw)
    assert m is not None
    assert m.no_price == pytest.approx(0.40)


def test_kalshi_settled_maps_to_resolved():
    client = kal_client()
    raw = {
        "ticker": "OLD-EVENT",
        "title": "Past event",
        "status": "settled",
        "yes_bid": 100,
    }
    m = client.normalize_market(raw)
    assert m is not None
    assert m.status == MarketStatus.RESOLVED


def test_kalshi_returns_none_without_ticker():
    client = kal_client()
    m = client.normalize_market({"title": "no ticker"})
    assert m is None
