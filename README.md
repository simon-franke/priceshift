# priceshift

Prediction market arbitrage intelligence system. Monitors price discrepancies between [Polymarket](https://polymarket.com) and [Kalshi](https://kalshi.com), matches equivalent events across platforms, detects arbitrage gaps, and simulates paper trades.

## Architecture

```
APIs (Polymarket + Kalshi) → Matcher → DataStore → PaperTrader → Dashboard/Backtest
```

**Dual database:**
- `data/operational.sqlite` — markets, matched pairs, paper trades (SQLite)
- `data/priceshift.duckdb` — gap history, backtest replay (DuckDB)

**3-stage event matching pipeline** (`matching/matcher.py`):
1. Ground-truth hardcoded pairs
2. Rule-based keyword filter
3. Semantic embeddings (`all-MiniLM-L6-v2`, threshold configurable in `config.toml`)

**Paper trading** (`trading/simulator.py`): mean-reversion strategy — opens a position when a gap exceeds `min_gap_open_pp`, closes it when the gap narrows below `min_gap_close_pp`.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
priceshift smoke        # test API connectivity (10 markets from each platform)
priceshift gaps         # show current arbitrage gaps
priceshift portfolio    # show open paper trades
priceshift dashboard    # live Rich dashboard
priceshift backtest     # replay historical gaps
priceshift              # start continuous polling loop
```

## Development

```bash
# Run tests
python -m pytest tests/ -v

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Configuration

Edit `config.toml` to tune polling intervals, matching thresholds, and trading parameters. Environment variables with prefixes `POLYMARKET_`, `KALSHI_`, etc. override config file values.

## Project Structure

```
src/priceshift/
├── main.py              # CLI entry point & polling loop
├── models.py            # Shared Pydantic models
├── config.py            # Config singleton (get_config())
├── apis/
│   ├── polymarket.py    # Gamma REST client
│   └── kalshi.py        # Public read-only REST client
├── matching/
│   ├── matcher.py       # 3-stage event matching pipeline
│   └── embeddings.py    # Sentence-transformer wrapper
├── db/
│   └── store.py         # DataStore (SQLite + DuckDB)
├── trading/
│   ├── simulator.py     # PaperTrader
│   └── backtest.py      # Gap history replay
└── dashboard/
    └── cli.py           # Rich live dashboard
```
