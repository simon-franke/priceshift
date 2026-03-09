"""DDL constants for SQLite (operational) and DuckDB (analytics)."""

# ---------------------------------------------------------------------------
# SQLite — operational state
# ---------------------------------------------------------------------------

SQLITE_MARKETS = """
CREATE TABLE IF NOT EXISTS markets (
    id           TEXT PRIMARY KEY,
    platform     TEXT NOT NULL,
    title        TEXT NOT NULL,
    description  TEXT DEFAULT '',
    category     TEXT DEFAULT '',
    status       TEXT DEFAULT 'unknown',
    resolution_date TEXT,
    yes_price    REAL,
    no_price     REAL,
    volume       REAL,
    liquidity    REAL,
    created_at   TEXT,
    fetched_at   TEXT NOT NULL
)
"""

SQLITE_MATCHED_PAIRS = """
CREATE TABLE IF NOT EXISTS matched_pairs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    polymarket_id    TEXT NOT NULL,
    kalshi_ticker    TEXT NOT NULL,
    polymarket_title TEXT DEFAULT '',
    kalshi_title     TEXT DEFAULT '',
    similarity_score REAL NOT NULL,
    match_source     TEXT DEFAULT 'semantic',
    created_at       TEXT NOT NULL,
    is_active        INTEGER DEFAULT 1,
    UNIQUE (polymarket_id, kalshi_ticker)
)
"""

SQLITE_PAPER_TRADES = """
CREATE TABLE IF NOT EXISTS paper_trades (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    pair_id             INTEGER NOT NULL,
    polymarket_id       TEXT NOT NULL,
    kalshi_ticker       TEXT NOT NULL,
    direction           TEXT NOT NULL,
    position_size       REAL NOT NULL,
    open_gap_pp         REAL NOT NULL,
    open_pm_price       REAL NOT NULL,
    open_kalshi_price   REAL NOT NULL,
    close_gap_pp        REAL,
    close_pm_price      REAL,
    close_kalshi_price  REAL,
    realized_pnl        REAL,
    status              TEXT DEFAULT 'open',
    opened_at           TEXT NOT NULL,
    closed_at           TEXT
)
"""

SQLITE_ALL = [SQLITE_MARKETS, SQLITE_MATCHED_PAIRS, SQLITE_PAPER_TRADES]

# ---------------------------------------------------------------------------
# DuckDB — analytics (append-only)
# ---------------------------------------------------------------------------

DUCKDB_PRICE_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS price_snapshots (
    market_id  VARCHAR NOT NULL,
    platform   VARCHAR NOT NULL,
    yes_price  DOUBLE NOT NULL,
    no_price   DOUBLE NOT NULL,
    timestamp  TIMESTAMP NOT NULL,
    volume     DOUBLE,
    liquidity  DOUBLE
)
"""

DUCKDB_ARBITRAGE_GAPS = """
CREATE TABLE IF NOT EXISTS arbitrage_gaps (
    pair_id        INTEGER NOT NULL,
    polymarket_id  VARCHAR NOT NULL,
    kalshi_ticker  VARCHAR NOT NULL,
    pm_yes_price   DOUBLE NOT NULL,
    kalshi_yes_price DOUBLE NOT NULL,
    gap_pp         DOUBLE NOT NULL,
    abs_gap_pp     DOUBLE NOT NULL,
    timestamp      TIMESTAMP NOT NULL
)
"""

DUCKDB_COMPLETED_TRADES = """
CREATE TABLE IF NOT EXISTS completed_trades (
    trade_id               INTEGER NOT NULL,
    pair_id                INTEGER NOT NULL,
    polymarket_id          VARCHAR NOT NULL,
    kalshi_ticker          VARCHAR NOT NULL,
    direction              VARCHAR NOT NULL,
    position_size          DOUBLE NOT NULL,
    open_gap_pp            DOUBLE NOT NULL,
    close_gap_pp           DOUBLE NOT NULL,
    realized_pnl           DOUBLE NOT NULL,
    hold_duration_seconds  DOUBLE NOT NULL,
    opened_at              TIMESTAMP NOT NULL,
    closed_at              TIMESTAMP NOT NULL
)
"""

DUCKDB_ALL = [DUCKDB_PRICE_SNAPSHOTS, DUCKDB_ARBITRAGE_GAPS, DUCKDB_COMPLETED_TRADES]
