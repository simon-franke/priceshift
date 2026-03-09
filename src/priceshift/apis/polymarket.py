"""Polymarket API client — Gamma REST + CLOB REST."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from priceshift.apis.base import BaseAPIClient
from priceshift.config import PolymarketConfig
from priceshift.models import Market, MarketStatus, Platform, PriceSnapshot

logger = logging.getLogger(__name__)


def _parse_status(raw: str) -> MarketStatus:
    mapping = {
        "open": MarketStatus.OPEN,
        "closed": MarketStatus.CLOSED,
        "resolved": MarketStatus.RESOLVED,
    }
    return mapping.get(raw.lower(), MarketStatus.UNKNOWN)


def _parse_dt(val: Optional[str]) -> Optional[datetime]:
    if not val:
        return None
    try:
        return datetime.fromisoformat(val.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


class PolymarketGammaClient(BaseAPIClient):
    """Client for the Gamma metadata API (market listings)."""

    def __init__(self, config: PolymarketConfig) -> None:
        super().__init__(config.gamma_base_url, config.request_timeout_seconds)
        self._config = config

    def fetch_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        active: bool = True,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "limit": min(limit, self._config.max_markets_per_fetch),
            "offset": offset,
            "active": str(active).lower(),
            "closed": "false",
        }
        data = self.get("/markets", params=params)
        if isinstance(data, list):
            return data
        return data.get("data", []) if isinstance(data, dict) else []

    def normalize_market(self, raw: dict[str, Any]) -> Optional[Market]:
        cid = raw.get("conditionId") or raw.get("id")
        if not cid:
            return None

        # Extract yes price — prefer bestBid/bestAsk mid, fall back to outcomePrices
        yes_price: Optional[float] = None
        no_price: Optional[float] = None

        best_bid = _safe_float(raw.get("bestBid"))
        best_ask = _safe_float(raw.get("bestAsk"))
        if best_bid is not None and best_ask is not None:
            yes_price = (best_bid + best_ask) / 2.0
            no_price = 1.0 - yes_price
        elif best_bid is not None:
            yes_price = best_bid
            no_price = 1.0 - yes_price
        else:
            # outcomePrices is a JSON string in the Gamma API
            outcomes_raw = raw.get("outcomes", "[]")
            prices_raw = raw.get("outcomePrices", "[]")
            if isinstance(outcomes_raw, str):
                try:
                    outcomes_raw = json.loads(outcomes_raw)
                except (ValueError, TypeError):
                    outcomes_raw = []
            if isinstance(prices_raw, str):
                try:
                    prices_raw = json.loads(prices_raw)
                except (ValueError, TypeError):
                    prices_raw = []
            for i, outcome in enumerate(outcomes_raw):
                if str(outcome).upper() == "YES" and i < len(prices_raw):
                    yes_price = _safe_float(prices_raw[i])
                elif str(outcome).upper() == "NO" and i < len(prices_raw):
                    no_price = _safe_float(prices_raw[i])

        # Derive missing side
        if yes_price is not None and no_price is None:
            no_price = 1.0 - yes_price

        active = raw.get("active", True)
        closed = raw.get("closed", False)
        if closed:
            status_str = "closed"
        elif active:
            status_str = "open"
        else:
            status_str = "closed"

        return Market(
            id=str(cid),
            platform=Platform.POLYMARKET,
            title=raw.get("question", raw.get("title", "")),
            description=raw.get("description", ""),
            category=raw.get("category", ""),
            status=_parse_status(status_str),
            resolution_date=_parse_dt(raw.get("endDate") or raw.get("resolutionDate")),
            yes_price=yes_price,
            no_price=no_price,
            volume=_safe_float(raw.get("volume")),
            liquidity=_safe_float(raw.get("liquidity")),
            created_at=_parse_dt(raw.get("createdAt")),
        )

    def fetch_and_normalize(self, limit: int = 100) -> list[Market]:
        now = datetime.now(timezone.utc)
        markets = []
        offset = 0
        page_size = min(limit, self._config.max_markets_per_fetch)

        while len(markets) < limit:
            raw_markets = self.fetch_markets(limit=page_size, offset=offset)
            if not raw_markets:
                break
            for raw in raw_markets:
                m = self.normalize_market(raw)
                if m and (m.resolution_date is None or m.resolution_date > now):
                    markets.append(m)
            offset += len(raw_markets)
            if len(raw_markets) < page_size:
                break  # last page

        logger.info("Fetched %d Polymarket markets", len(markets))
        return markets


class PolymarketCLOBClient(BaseAPIClient):
    """Client for the CLOB order book API (live prices)."""

    def __init__(self, config: PolymarketConfig) -> None:
        super().__init__(config.clob_base_url, config.request_timeout_seconds)
        self._config = config

    def get_market(self, condition_id: str) -> dict[str, Any]:
        return self.get(f"/markets/{condition_id}")

    def get_price_snapshot(self, condition_id: str, token_id: str) -> Optional[PriceSnapshot]:
        """Fetch mid-price from CLOB for a specific token."""
        try:
            data = self.get("/midpoint", params={"token_id": token_id})
            mid = _safe_float(data.get("mid"))
            if mid is None:
                return None
            return PriceSnapshot(
                market_id=condition_id,
                platform=Platform.POLYMARKET,
                yes_price=mid,
                no_price=1.0 - mid,
            )
        except Exception as exc:
            logger.warning("CLOB price fetch failed for %s: %s", condition_id, exc)
            return None


def _safe_float(val: Any) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None
