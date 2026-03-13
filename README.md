# priceshift

Prediction market arbitrage intelligence system. Monitors price discrepancies between [Polymarket](https://polymarket.com) and [Kalshi](https://kalshi.com), matches equivalent events across platforms, detects arbitrage gaps, and simulates paper trades.

## Architecture

```
APIs (Polymarket + Kalshi) → Matcher → DataStore → PaperTrader → Dashboard/Backtest
```

**Single database:** `data/operational.sqlite` (SQLite, WAL mode) — markets, matched pairs, paper trades, gap history, price snapshots, completed trades. WAL mode allows the dashboard and polling loop to run concurrently without lock conflicts.

**5-stage event matching pipeline** (`matching/matcher.py` + `matching/verifier.py`):
1. Ground-truth hardcoded pairs
2. Rule-based filter (date proximity + keyword overlap)
3. Semantic bi-encoder ranking (`all-MiniLM-L6-v2`, threshold configurable in `config.toml`) — uses enriched text (title + description)
4. NLI cross-encoder verification (`nli-deberta-v3-small`) — bidirectional entailment check on enriched text
5. Ollama LLM fallback (`phi3:mini`) — for uncertain NLI results only

**Kalshi description enrichment:** Kalshi markets are hierarchical (Event → Market). The parent event title (e.g. `"FIFA World Cup 2026 Winner"`) is threaded into the market description during normalization, producing `"FIFA World Cup 2026 Winner: Spain"` instead of the bare `"Spain"`. This prevents false positives between related-but-non-equivalent markets (e.g. winner vs. qualifier).

Verification results are cached in SQLite (`match_verdicts` table) to avoid repeated model calls.

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
priceshift once         # run a single fetch → match → store cycle and exit
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
│   ├── matcher.py       # 5-stage event matching pipeline
│   ├── verifier.py      # NLI cross-encoder + Ollama LLM verification
│   └── embeddings.py    # Sentence-transformer wrapper
├── db/
│   └── store.py         # DataStore (SQLite, WAL mode)
├── trading/
│   ├── simulator.py     # PaperTrader
│   └── backtest.py      # Gap history replay
└── dashboard/
    └── cli.py           # Rich live dashboard
```
