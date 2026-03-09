"""DataStore: owns both DB connections and all SQL operations."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb

from priceshift.db.schema import DUCKDB_ALL, SQLITE_ALL
from priceshift.models import (
    ArbitrageGap,
    Market,
    MatchedPair,
    PaperTrade,
    PriceSnapshot,
    TradeStatus,
)


class DataStore:
    def __init__(self, sqlite_path: str, duckdb_path: str) -> None:
        Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        Path(duckdb_path).parent.mkdir(parents=True, exist_ok=True)

        self._sqlite = sqlite3.connect(sqlite_path, check_same_thread=False)
        self._sqlite.row_factory = sqlite3.Row
        self._duckdb = duckdb.connect(duckdb_path)

        self._init_schemas()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _init_schemas(self) -> None:
        cur = self._sqlite.cursor()
        for ddl in SQLITE_ALL:
            cur.execute(ddl)
        self._sqlite.commit()

        for ddl in DUCKDB_ALL:
            self._duckdb.execute(ddl)

    def close(self) -> None:
        self._sqlite.close()
        self._duckdb.close()

    def __enter__(self) -> "DataStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Markets (SQLite)
    # ------------------------------------------------------------------

    def upsert_market(self, market: Market) -> None:
        self._sqlite.execute(
            """
            INSERT INTO markets
                (id, platform, title, description, category, status,
                 resolution_date, yes_price, no_price, volume, liquidity,
                 created_at, fetched_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                status=excluded.status,
                yes_price=excluded.yes_price,
                no_price=excluded.no_price,
                volume=excluded.volume,
                liquidity=excluded.liquidity,
                fetched_at=excluded.fetched_at
            """,
            (
                market.id,
                market.platform.value,
                market.title,
                market.description,
                market.category,
                market.status.value,
                market.resolution_date.isoformat() if market.resolution_date else None,
                market.yes_price,
                market.no_price,
                market.volume,
                market.liquidity,
                market.created_at.isoformat() if market.created_at else None,
                market.fetched_at.isoformat(),
            ),
        )
        self._sqlite.commit()

    def upsert_markets(self, markets: list[Market]) -> None:
        for market in markets:
            self.upsert_market(market)

    def get_markets_by_platform(self, platform: str) -> list[dict]:
        rows = self._sqlite.execute(
            "SELECT * FROM markets WHERE platform = ?", (platform,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Matched Pairs (SQLite)
    # ------------------------------------------------------------------

    def upsert_matched_pair(self, pair: MatchedPair) -> int:
        cur = self._sqlite.execute(
            """
            INSERT INTO matched_pairs
                (polymarket_id, kalshi_ticker, polymarket_title, kalshi_title,
                 similarity_score, match_source, created_at, is_active)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(polymarket_id, kalshi_ticker) DO UPDATE SET
                similarity_score=excluded.similarity_score,
                is_active=excluded.is_active
            RETURNING id
            """,
            (
                pair.polymarket_id,
                pair.kalshi_ticker,
                pair.polymarket_title,
                pair.kalshi_title,
                pair.similarity_score,
                pair.match_source,
                pair.created_at.isoformat(),
                1 if pair.is_active else 0,
            ),
        )
        # Consume cursor BEFORE commit (SQLite requires this with RETURNING)
        row = cur.fetchone()
        self._sqlite.commit()
        if row:
            return int(row[0])
        # fetch existing id
        existing = self._sqlite.execute(
            "SELECT id FROM matched_pairs WHERE polymarket_id=? AND kalshi_ticker=?",
            (pair.polymarket_id, pair.kalshi_ticker),
        ).fetchone()
        return int(existing[0]) if existing else -1

    def get_active_pairs(self) -> list[dict]:
        rows = self._sqlite.execute(
            "SELECT * FROM matched_pairs WHERE is_active = 1"
        ).fetchall()
        return [dict(r) for r in rows]

    def deactivate_pair(self, pair_id: int) -> None:
        self._sqlite.execute(
            "UPDATE matched_pairs SET is_active = 0 WHERE id = ?", (pair_id,)
        )
        self._sqlite.commit()

    # ------------------------------------------------------------------
    # Paper Trades (SQLite)
    # ------------------------------------------------------------------

    def create_paper_trade(self, trade: PaperTrade) -> int:
        cur = self._sqlite.execute(
            """
            INSERT INTO paper_trades
                (pair_id, polymarket_id, kalshi_ticker, direction, position_size,
                 open_gap_pp, open_pm_price, open_kalshi_price, status, opened_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                trade.pair_id,
                trade.polymarket_id,
                trade.kalshi_ticker,
                trade.direction.value,
                trade.position_size,
                trade.open_gap_pp,
                trade.open_pm_price,
                trade.open_kalshi_price,
                trade.status.value,
                trade.opened_at.isoformat(),
            ),
        )
        self._sqlite.commit()
        return cur.lastrowid or -1

    def close_paper_trade(
        self,
        trade_id: int,
        close_gap_pp: float,
        close_pm_price: float,
        close_kalshi_price: float,
        realized_pnl: float,
        closed_at: Optional[datetime] = None,
    ) -> None:
        self._sqlite.execute(
            """
            UPDATE paper_trades SET
                close_gap_pp = ?,
                close_pm_price = ?,
                close_kalshi_price = ?,
                realized_pnl = ?,
                status = 'closed',
                closed_at = ?
            WHERE id = ?
            """,
            (
                close_gap_pp,
                close_pm_price,
                close_kalshi_price,
                realized_pnl,
                (closed_at or datetime.utcnow()).isoformat(),
                trade_id,
            ),
        )
        self._sqlite.commit()

    def get_open_trades(self) -> list[dict]:
        rows = self._sqlite.execute(
            "SELECT * FROM paper_trades WHERE status = 'open'"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_open_trade_for_pair(self, pair_id: int) -> Optional[dict]:
        row = self._sqlite.execute(
            "SELECT * FROM paper_trades WHERE pair_id = ? AND status = 'open'",
            (pair_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_all_trades(self) -> list[dict]:
        rows = self._sqlite.execute("SELECT * FROM paper_trades").fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Price Snapshots (DuckDB)
    # ------------------------------------------------------------------

    def append_price_snapshot(self, snap: PriceSnapshot) -> None:
        self._duckdb.execute(
            """
            INSERT INTO price_snapshots
                (market_id, platform, yes_price, no_price, timestamp, volume, liquidity)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                snap.market_id,
                snap.platform.value,
                snap.yes_price,
                snap.no_price,
                snap.timestamp,
                snap.volume,
                snap.liquidity,
            ),
        )

    def append_price_snapshots(self, snaps: list[PriceSnapshot]) -> None:
        for snap in snaps:
            self.append_price_snapshot(snap)

    # ------------------------------------------------------------------
    # Arbitrage Gaps (DuckDB)
    # ------------------------------------------------------------------

    def append_arbitrage_gap(self, gap: ArbitrageGap) -> None:
        self._duckdb.execute(
            """
            INSERT INTO arbitrage_gaps
                (pair_id, polymarket_id, kalshi_ticker, pm_yes_price,
                 kalshi_yes_price, gap_pp, abs_gap_pp, timestamp)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                gap.pair_id,
                gap.polymarket_id,
                gap.kalshi_ticker,
                gap.pm_yes_price,
                gap.kalshi_yes_price,
                gap.gap_pp,
                gap.abs_gap_pp,
                gap.timestamp,
            ),
        )

    def get_latest_gaps(self, limit: int = 50) -> list[dict]:
        rows = self._duckdb.execute(
            """
            SELECT * FROM arbitrage_gaps
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        cols = [d[0] for d in self._duckdb.description]
        return [dict(zip(cols, row)) for row in rows]

    def get_gaps_for_pair(self, polymarket_id: str, kalshi_ticker: str) -> list[dict]:
        rows = self._duckdb.execute(
            """
            SELECT * FROM arbitrage_gaps
            WHERE polymarket_id = ? AND kalshi_ticker = ?
            ORDER BY timestamp ASC
            """,
            [polymarket_id, kalshi_ticker],
        ).fetchall()
        cols = [d[0] for d in self._duckdb.description]
        return [dict(zip(cols, row)) for row in rows]

    # ------------------------------------------------------------------
    # Completed Trades (DuckDB)
    # ------------------------------------------------------------------

    def append_completed_trade(self, trade: PaperTrade) -> None:
        if trade.id is None or trade.closed_at is None or trade.realized_pnl is None:
            raise ValueError("Trade must be closed before appending to DuckDB")
        hold = trade.hold_duration_seconds or 0.0
        self._duckdb.execute(
            """
            INSERT INTO completed_trades
                (trade_id, pair_id, polymarket_id, kalshi_ticker, direction,
                 position_size, open_gap_pp, close_gap_pp, realized_pnl,
                 hold_duration_seconds, opened_at, closed_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                trade.id,
                trade.pair_id,
                trade.polymarket_id,
                trade.kalshi_ticker,
                trade.direction.value,
                trade.position_size,
                trade.open_gap_pp,
                trade.close_gap_pp,
                trade.realized_pnl,
                hold,
                trade.opened_at,
                trade.closed_at,
            ),
        )

    def get_trade_summary(self) -> dict:
        row = self._duckdb.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(realized_pnl) as total_pnl,
                AVG(realized_pnl) as avg_pnl,
                AVG(hold_duration_seconds) as avg_hold
            FROM completed_trades
            """
        ).fetchone()
        if row is None:
            return {}
        cols = [d[0] for d in self._duckdb.description]
        return dict(zip(cols, row))
