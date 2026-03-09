# Prediction Market Intelligence System
### Claude Code Project Brief

---

## Project Goal

Build a system that monitors price dynamics on **Polymarket** and **Kalshi**, identifies cross-platform arbitrage opportunities, and analyzes how news events move prediction market prices. Fully implemented in Python using Claude Code.

---

## Phase 1: Arbitrage Monitor & Paper Trading Simulator

### What to Build

A pipeline that fetches live market data from both platforms, matches equivalent events, tracks price gaps, and simulates trading strategies — no real capital, full analytical depth.

### Key Questions to Answer

- How large are typical price gaps between Kalshi and Polymarket for the same event?
- How long do gaps persist before closing?
- How often would arbitrage trades be profitable after transaction costs?
- Which event categories (politics, economics, sports) show the largest inefficiencies?

### Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  Polymarket API │     │   Kalshi API    │
│  (REST + WS)    │     │   (REST)        │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│            Event Matcher                │
│  Sentence embeddings to link identical  │
│  events across platforms                │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│           Price Gap Tracker             │
│  - Implied probability per platform     │
│  - Delta calculation + timestamps       │
│  - Volume weighting                     │
└────────────────────┬────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐  ┌──────────────────────┐
│  DuckDB / SQLite │  │  Paper Trading       │
│  (logging)       │  │  Simulator + Backtest│
└──────────────────┘  └──────────────────────┘
```

### Event Matching Strategy

The core challenge: Polymarket calls an event *"Will the Fed raise rates in June 2025?"*, Kalshi calls it *"Fed rate hike – June FOMC meeting"*. Matching approach:

1. Rule-based pre-filter by date and known keywords
2. Semantic similarity via `sentence-transformers` embeddings
3. Manual validation list as ground truth for ~10–20 frequent events

### Paper Trading Logic

Simple **mean-reversion strategy**:
- Open position when gap > threshold X (e.g. 3 percentage points)
- Close when gap < Y or event resolves
- Track P&L, win rate, average hold duration

### Deliverables

- [ ] Polymarket API wrapper (REST + WebSocket)
- [ ] Kalshi API wrapper
- [ ] Event matching pipeline
- [ ] Gap tracking database (historical logs)
- [ ] Paper trading simulator with backtest
- [ ] CLI or lightweight web dashboard

### Timeline

| Week | Milestone |
|------|-----------|
| 1 | API wrappers, data pipeline |
| 2 | Event matching + validation |
| 3 | Gap tracker + database logging |
| 4 | Paper trading simulator + first analysis |

---

## Phase 2: News Reaction Analysis

### What to Build

Extend the Phase 1 infrastructure with a news ingestion layer. Measure how quickly and strongly prediction markets react to relevant news — and whether reactions are efficient, over- or under-shooting.

### Key Questions to Answer

- What is the typical market reaction time after a relevant article is published?
- Do Polymarket and Kalshi differ in reaction speed?
- Are there news categories markets consistently over- or under-react to?
- Can early price moves anticipate news (reverse causality)?

### Data Sources

| Source | Content | Cost |
|--------|---------|------|
| **GDELT Project** | Global news articles, real-time, structured | Free |
| **NewsAPI** | Aggregated news, simple REST | Free (limited) |
| **Polymarket + Kalshi** | Price time series from Phase 1 | Free |

### Extended Architecture

```
Phase 1 Infrastructure
         +
┌─────────────────────────────────────────┐
│         News Ingestion Layer            │
│  GDELT API → article fetching           │
│  Relevance filtering per event          │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│        News-Event Linker (NLP)          │
│  - Article → market mapping             │
│  - Sentiment score per article          │
│  - Named Entity Recognition (spaCy)     │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│          Event Study Framework          │
│  - Price window: T-60min to T+180min    │
│  - Abnormal return calculation          │
│  - Aggregation across many events       │
└─────────────────────────────────────────┘
```

### Event Study Methodology

Borrowed from financial econometrics:

1. **Define event:** Timestamp T when a relevant article is published
2. **Observe price window:** T−60min to T+180min
3. **Subtract baseline:** Remove "normal" price drift without news
4. **Measure abnormal reaction:** What moved beyond baseline?
5. **Aggregate across events:** Draw robust conclusions about reaction patterns

### ML Component

- **Classification:** Which articles are price-moving? (binary)
- **Regression:** How much does price change given sentiment score, source, time of day?
- **Features:** Sentiment, named entities, publication outlet, hour, market volume, volatility

### Deliverables

- [ ] GDELT ingestion pipeline
- [ ] News-event linker (NLP matching)
- [ ] Event study framework
- [ ] ML model: predict price reaction magnitude
- [ ] Visualizations: average reaction curves by category
- [ ] Final report with findings

### Timeline

| Week | Milestone |
|------|-----------|
| 5 | GDELT integration, news preprocessing |
| 6 | News-event linker |
| 7 | Event study framework + first analyses |
| 8 | ML model + evaluation + documentation |

---

## Tech Stack

```
Language:       Python 3.11+
Storage:        DuckDB (analytics) + SQLite (operational)
ML/NLP:         scikit-learn, XGBoost, sentence-transformers, spaCy
APIs:           Polymarket REST/WS, Kalshi REST, GDELT, NewsAPI
Visualization:  Plotly, matplotlib
Dev tooling:    Claude Code, Git, uv
```

---

## First Steps for Claude Code

1. **Explore Polymarket API** — read docs, write first wrapper
2. **Explore Kalshi API** — create account (required for API access), test endpoints
3. **Design database schema** — Events, Prices, Gaps, News as a relational model
4. **Event matching prototype** — manually match 10–20 events as ground truth

> *Estimated effort: 6–8 weeks at ~10h/week. Note: Kalshi is restricted to US persons — this system is designed as an analytical tool only, no real trading.*
