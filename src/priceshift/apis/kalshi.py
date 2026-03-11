"""Kalshi API client — public read-only REST (no auth required)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from priceshift.apis.base import BaseAPIClient
from priceshift.config import KalshiConfig
from priceshift.models import Market, MarketStatus, Platform, PriceSnapshot

logger = logging.getLogger(__name__)


def _parse_status(raw: str) -> MarketStatus:
    mapping = {
        "open": MarketStatus.OPEN,
        "active": MarketStatus.OPEN,
        "closed": MarketStatus.CLOSED,
        "settled": MarketStatus.RESOLVED,
        "finalized": MarketStatus.RESOLVED,
        "determined": MarketStatus.RESOLVED,
    }
    return mapping.get(raw.lower(), MarketStatus.UNKNOWN)


def _parse_dt(val: Optional[str]) -> Optional[datetime]:
    if not val:
        return None
    try:
        return datetime.fromisoformat(val.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _safe_float(val: Any) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


class KalshiClient(BaseAPIClient):
    """Public read-only client for Kalshi market data."""

    def __init__(self, config: KalshiConfig) -> None:
        super().__init__(config.base_url, config.request_timeout_seconds)
        self._config = config

    # Categories that have meaningful overlap with Polymarket
    MATCHABLE_CATEGORIES = {
        "Politics", "Economics", "Financials", "Elections",
        "Science and Technology", "Companies", "World", "Health",
        "Entertainment", "Sports", "Crypto",
    }

    def fetch_markets(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        status: str = "open",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "limit": min(limit, self._config.max_markets_per_fetch),
            "status": status,
        }
        if cursor:
            params["cursor"] = cursor
        return self.get("/markets", params=params)

    def fetch_events_with_markets(self, limit: int = 200) -> list[dict[str, Any]]:
        """Fetch all events with nested markets in one pass using with_nested_markets=true."""
        events = []
        cursor: Optional[str] = None
        while len(events) < limit:
            params: dict[str, Any] = {
                "limit": min(100, limit - len(events)),
                "with_nested_markets": "true",
                "status": "open",
            }
            if cursor:
                params["cursor"] = cursor
            data = self.get("/events", params=params)
            batch = data.get("events", [])
            if not batch:
                break
            events.extend(batch)
            cursor = data.get("cursor")
            if not cursor:
                break
        return events

    def fetch_market(self, ticker: str) -> dict[str, Any]:
        return self.get(f"/markets/{ticker}")

    def normalize_market(self, raw: dict[str, Any]) -> Optional[Market]:
        ticker = raw.get("ticker")
        if not ticker:
            return None

        # Skip MVE multi-leg parlay markets — not matchable against Polymarket
        if raw.get("mve_collection_ticker"):
            return None

        # Kalshi prices are in cents (0–100); use mid of bid/ask
        yes_price: Optional[float] = None
        no_price: Optional[float] = None

        yes_bid = _safe_float(raw.get("yes_bid"))
        yes_ask = _safe_float(raw.get("yes_ask"))

        if yes_bid is not None and yes_ask is not None and (yes_bid > 0 or yes_ask > 0):
            yes_price = (yes_bid + yes_ask) / 2.0 / 100.0
        elif yes_bid is not None and yes_bid > 0:
            yes_price = yes_bid / 100.0
        elif yes_ask is not None and yes_ask > 0:
            yes_price = yes_ask / 100.0
        else:
            # Fall back to last_price
            last = _safe_float(raw.get("last_price"))
            if last is not None and last > 0:
                yes_price = last / 100.0

        if yes_price is not None:
            no_price = 1.0 - yes_price

        event_title = raw.get("_event_title", "")
        subtitle = raw.get("subtitle", "")
        if event_title and subtitle and event_title.lower() != subtitle.lower():
            description = f"{event_title}: {subtitle}"
        elif event_title:
            description = event_title
        else:
            description = subtitle

        return Market(
            id=str(ticker),
            platform=Platform.KALSHI,
            title=raw.get("title", raw.get("question", "")),
            description=description,
            category=raw.get("category", raw.get("event_ticker", "")),
            status=_parse_status(raw.get("status", "unknown")),
            resolution_date=_parse_dt(raw.get("close_time") or raw.get("expected_expiration_time")),
            yes_price=yes_price,
            no_price=no_price,
            volume=_safe_float(raw.get("volume")),
            liquidity=_safe_float(raw.get("liquidity")),
            created_at=_parse_dt(raw.get("created_time")),
        )

    def fetch_and_normalize(self, limit: int = 100) -> list[Market]:
        """Fetch markets via the events API (matchable categories only).

        Collects ALL matchable markets from up to 2000 events, ignoring the
        limit for collection (limit is only used for the final return slice).
        This ensures we don't miss important markets that appear later in the
        event list due to Kalshi's ordering.
        """
        now = datetime.now(timezone.utc)

        events = self.fetch_events_with_markets(limit=500)
        markets: list[Market] = []

        for event in events:
            category = event.get("category", "")
            if category not in self.MATCHABLE_CATEGORIES:
                continue
            for raw in event.get("markets", []):
                raw = {**raw, "category": category, "_event_title": event.get("title", "")}
                m = self.normalize_market(raw)
                if m and m.yes_price is not None:
                    # Skip already-closed markets
                    if m.resolution_date is not None and m.resolution_date <= now:
                        continue
                    markets.append(m)

        logger.info("Fetched %d Kalshi markets from events API", len(markets))
        return markets

    def get_price_snapshot(self, ticker: str) -> Optional[PriceSnapshot]:
        try:
            data = self.fetch_market(ticker)
            market_data = data.get("market", data)
            m = self.normalize_market(market_data)
            if m is None or m.yes_price is None:
                return None
            return PriceSnapshot(
                market_id=ticker,
                platform=Platform.KALSHI,
                yes_price=m.yes_price,
                no_price=m.no_price or (1.0 - m.yes_price),
            )
        except Exception as exc:
            logger.warning("Kalshi price fetch failed for %s: %s", ticker, exc)
            return None
