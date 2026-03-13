"""Main polling loop entry point."""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime

from priceshift.apis.kalshi import KalshiClient
from priceshift.apis.polymarket import PolymarketGammaClient
from priceshift.config import get_config
from priceshift.db.store import DataStore
from priceshift.matching.matcher import EventMatcher
from priceshift.matching.verifier import MatchVerifier
from priceshift.models import ArbitrageGap, Platform, PriceSnapshot
from priceshift.trading.simulator import PaperTrader

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def smoke_test() -> None:
    """Print 10 live markets from each platform."""
    cfg = get_config()

    print("=== Polymarket (10 markets) ===")
    pm_client = PolymarketGammaClient(cfg.polymarket)
    try:
        pm_markets = pm_client.fetch_and_normalize(limit=10)
        for m in pm_markets[:10]:
            print(f"  [{m.status.value}] {m.title[:70]}  yes={m.yes_price}")
    except Exception as exc:
        print(f"  ERROR: {exc}")
    finally:
        pm_client.close()

    print()
    print("=== Kalshi (10 markets) ===")
    kalshi_client = KalshiClient(cfg.kalshi)
    try:
        kalshi_markets = kalshi_client.fetch_and_normalize(limit=10)
        for m in kalshi_markets[:10]:
            print(f"  [{m.status.value}] {m.title[:70]}  yes={m.yes_price}")
    except Exception as exc:
        print(f"  ERROR: {exc}")
    finally:
        kalshi_client.close()


def run_once(store: DataStore, cfg: object) -> None:
    """Single poll cycle: fetch → match → gap → store → trade."""
    from priceshift.config import AppConfig

    assert isinstance(cfg, AppConfig)

    pm_client = PolymarketGammaClient(cfg.polymarket)
    kalshi_client = KalshiClient(cfg.kalshi)
    verifier = MatchVerifier(
        store=store,
        nli_model=cfg.matching.nli_model,
        nli_threshold=cfg.matching.nli_threshold,
        nli_contradiction_threshold=cfg.matching.nli_contradiction_threshold,
        ollama_model=cfg.matching.ollama_model,
        ollama_url=cfg.matching.ollama_url,
        use_ollama_fallback=cfg.matching.use_ollama_fallback,
    )
    matcher = EventMatcher(
        semantic_threshold=cfg.matching.semantic_threshold,
        min_keyword_overlap=cfg.matching.min_keyword_overlap,
        cache_dir=cfg.matching.cache_dir,
        model_name=cfg.matching.embedding_model,
        verifier=verifier,
    )
    trader = PaperTrader(
        store=store,
        min_gap_open_pp=cfg.trading.min_gap_open_pp,
        min_gap_close_pp=cfg.trading.min_gap_close_pp,
        position_size=cfg.trading.default_position_size,
        max_open_trades=cfg.trading.max_open_trades,
    )

    try:
        pm_markets = pm_client.fetch_and_normalize(limit=cfg.polymarket.max_markets_per_fetch)
        kalshi_markets = kalshi_client.fetch_and_normalize(limit=cfg.kalshi.max_markets_per_fetch)
    except Exception as exc:
        logger.error("API fetch failed: %s", exc)
        return
    finally:
        pm_client.close()
        kalshi_client.close()

    # Store markets
    store.upsert_markets(pm_markets)
    store.upsert_markets(kalshi_markets)

    # Build price snapshots
    now = datetime.utcnow()
    snapshots = []
    pm_prices = {m.id: m for m in pm_markets if m.yes_price is not None}
    kalshi_prices = {m.id: m for m in kalshi_markets if m.yes_price is not None}

    for m in pm_markets:
        if m.yes_price is not None:
            snapshots.append(PriceSnapshot(
                market_id=m.id,
                platform=Platform.POLYMARKET,
                yes_price=m.yes_price,
                no_price=m.no_price or (1.0 - m.yes_price),
                timestamp=now,
                volume=m.volume,
                liquidity=m.liquidity,
            ))
    for m in kalshi_markets:
        if m.yes_price is not None:
            snapshots.append(PriceSnapshot(
                market_id=m.id,
                platform=Platform.KALSHI,
                yes_price=m.yes_price,
                no_price=m.no_price or (1.0 - m.yes_price),
                timestamp=now,
                volume=m.volume,
                liquidity=m.liquidity,
            ))

    store.append_price_snapshots(snapshots)

    # Match events
    pairs = matcher.match_all(pm_markets, kalshi_markets)
    pair_id_map: dict[tuple[str, str], int] = {}
    for pair in pairs:
        pair_id = store.upsert_matched_pair(pair)
        pair_id_map[(pair.polymarket_id, pair.kalshi_ticker)] = pair_id

    # Compute and store gaps; run paper trading
    for pair in pairs:
        pid = pair_id_map.get((pair.polymarket_id, pair.kalshi_ticker))
        if pid is None or pid < 0:
            continue
        pm_m = pm_prices.get(pair.polymarket_id)
        kal_m = kalshi_prices.get(pair.kalshi_ticker)
        if pm_m is None or kal_m is None:
            continue

        gap = ArbitrageGap.from_prices(
            pair_id=pid,
            polymarket_id=pair.polymarket_id,
            kalshi_ticker=pair.kalshi_ticker,
            pm_yes=pm_m.yes_price,  # type: ignore
            kalshi_yes=kal_m.yes_price,  # type: ignore
            timestamp=now,
        )
        store.append_arbitrage_gap(gap)
        trader.process_gap(gap)

    logger.info(
        "Cycle complete: %d PM markets, %d Kalshi markets, %d pairs, %d snapshots",
        len(pm_markets),
        len(kalshi_markets),
        len(pairs),
        len(snapshots),
    )


def main() -> None:
    cfg = get_config()
    setup_logging(cfg.log_level)

    # CLI dispatch
    args = sys.argv[1:]

    if "smoke" in args:
        smoke_test()
        return

    if "gaps" in args or "portfolio" in args or "dashboard" in args:
        from priceshift.dashboard.cli import run_live, show_gaps, show_portfolio

        with DataStore(cfg.db.sqlite_path) as store:
            if "gaps" in args:
                show_gaps(store)
            elif "portfolio" in args:
                show_portfolio(store)
            else:
                run_live(store)
        return

    if "once" in args:
        with DataStore(cfg.db.sqlite_path) as store:
            run_once(store, cfg)
        return

    if "backtest" in args:
        from priceshift.trading.backtest import Backtester

        with DataStore(cfg.db.sqlite_path) as store:
            bt = Backtester(
                store,
                min_gap_open_pp=cfg.trading.min_gap_open_pp,
                min_gap_close_pp=cfg.trading.min_gap_close_pp,
                position_size=cfg.trading.default_position_size,
            )
            result = bt.run()
            print(result.model_dump_json(indent=2))
        return

    # Default: run polling loop
    logger.info("Starting priceshift polling loop (interval=%ds)", cfg.polling.main_loop_interval_seconds)
    with DataStore(cfg.db.sqlite_path) as store:
        while True:
            try:
                run_once(store, cfg)
            except Exception as exc:
                logger.error("Poll cycle error: %s", exc, exc_info=True)
            time.sleep(cfg.polling.main_loop_interval_seconds)


if __name__ == "__main__":
    main()
