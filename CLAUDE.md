# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate virtual environment (required before any python commands)
source .venv/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_trading/test_simulator.py -v

# Run a single test by name
python -m pytest tests/ -k "test_name" -v

# Lint
ruff check src/ tests/

# Type check
mypy src/

# Run the app (CLI entry point)
priceshift smoke         # test API connectivity
priceshift gaps          # show arbitrage gaps
priceshift portfolio     # show portfolio
priceshift dashboard     # live Rich dashboard
priceshift backtest      # replay historical gaps
priceshift               # start polling loop
```

**Note:** `uv` is not installed. Use `python3.12` / `.venv/bin/activate` directly.

## Architecture

Priceshift monitors price discrepancies between Polymarket and Kalshi prediction markets, matches equivalent events across platforms, detects arbitrage gaps, and simulates paper trades.

### Data flow
```
APIs (Polymarket + Kalshi) → Matcher → DataStore → PaperTrader → Dashboard/Backtest
```

### Key design decisions

- **Single database**: SQLite (`data/operational.sqlite`) for all writes and queries. WAL mode enabled so the dashboard and polling loop can run concurrently without lock conflicts. `DataStore` in `db/store.py` takes a single `sqlite_path` argument.
- **Config loading**: `get_config()` in `config.py` is a cached singleton. It reads `config.toml` via tomllib, then overlays env vars with prefixes (`POLYMARKET_`, `KALSHI_`, etc.). Always call `get_config()` — never instantiate config classes directly.
- **All shared models in `models.py`**: `Market`, `PriceSnapshot`, `MatchedPair`, `ArbitrageGap`, `PaperTrade`, `BacktestResult`. `ArbitrageGap.from_prices()` is the factory for gap creation.
- **3-stage matching pipeline** (`matching/matcher.py`): ground-truth hardcoded pairs → rule-based keyword filter → semantic embeddings (all-MiniLM-L6-v2, threshold configurable in `config.toml`).
- **SQLite RETURNING id gotcha**: cursor rows must be fetched *before* `commit()` — already fixed in `store.upsert_matched_pair`.
- **API clients** (`apis/polymarket.py`, `apis/kalshi.py`) are public read-only; no auth needed for market data. Auth keys are optional (only for live trading).
- **Trading** (`trading/simulator.py`): mean-reversion paper trader; gaps above `min_gap_open_pp` open a position, gaps below `min_gap_close_pp` close it.
