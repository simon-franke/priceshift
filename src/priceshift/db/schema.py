"""DDL constants for SQLite."""

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

SQLITE_ARBITRAGE_GAPS = """
CREATE TABLE IF NOT EXISTS arbitrage_gaps (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    pair_id        INTEGER NOT NULL,
    polymarket_id  TEXT NOT NULL,
    kalshi_ticker  TEXT NOT NULL,
    pm_yes_price   REAL NOT NULL,
    kalshi_yes_price REAL NOT NULL,
    gap_pp         REAL NOT NULL,
    abs_gap_pp     REAL NOT NULL,
    timestamp      TEXT NOT NULL
)
"""

SQLITE_PRICE_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS price_snapshots (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id  TEXT NOT NULL,
    platform   TEXT NOT NULL,
    yes_price  REAL NOT NULL,
    no_price   REAL NOT NULL,
    timestamp  TEXT NOT NULL,
    volume     REAL,
    liquidity  REAL
)
"""

SQLITE_COMPLETED_TRADES = """
CREATE TABLE IF NOT EXISTS completed_trades (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id               INTEGER NOT NULL,
    pair_id                INTEGER NOT NULL,
    polymarket_id          TEXT NOT NULL,
    kalshi_ticker          TEXT NOT NULL,
    direction              TEXT NOT NULL,
    position_size          REAL NOT NULL,
    open_gap_pp            REAL NOT NULL,
    close_gap_pp           REAL NOT NULL,
    realized_pnl           REAL NOT NULL,
    hold_duration_seconds  REAL NOT NULL,
    opened_at              TEXT NOT NULL,
    closed_at              TEXT NOT NULL
)
"""

SQLITE_MATCH_VERDICTS = """
CREATE TABLE IF NOT EXISTS match_verdicts (
    polymarket_id TEXT NOT NULL,
    kalshi_ticker TEXT NOT NULL,
    is_match      INTEGER NOT NULL,
    confidence    REAL,
    source        TEXT NOT NULL,
    explanation   TEXT,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (polymarket_id, kalshi_ticker)
)
"""

SQLITE_ALL = [
    SQLITE_MARKETS,
    SQLITE_MATCHED_PAIRS,
    SQLITE_PAPER_TRADES,
    SQLITE_ARBITRAGE_GAPS,
    SQLITE_PRICE_SNAPSHOTS,
    SQLITE_COMPLETED_TRADES,
    SQLITE_MATCH_VERDICTS,
]
