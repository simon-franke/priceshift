"""Configuration loader: pydantic-settings + tomllib + .env."""
from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_toml() -> dict[str, Any]:
    config_path = Path(__file__).parent.parent.parent / "config.toml"
    if config_path.exists():
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    return {}


_TOML = _load_toml()


def _get(section: str, key: str, default: Any = None) -> Any:
    return _TOML.get(section, {}).get(key, default)


class PolymarketConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POLYMARKET_")

    gamma_base_url: str = Field(default=_get("polymarket", "gamma_base_url", "https://gamma-api.polymarket.com"))
    clob_base_url: str = Field(default=_get("polymarket", "clob_base_url", "https://clob.polymarket.com"))
    ws_url: str = Field(default=_get("polymarket", "ws_url", "wss://ws-subscriptions-clob.polymarket.com/ws/market"))
    poll_interval_seconds: int = Field(default=_get("polymarket", "poll_interval_seconds", 60))
    request_timeout_seconds: int = Field(default=_get("polymarket", "request_timeout_seconds", 30))
    max_markets_per_fetch: int = Field(default=_get("polymarket", "max_markets_per_fetch", 100))


class KalshiConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KALSHI_")

    base_url: str = Field(default=_get("kalshi", "base_url", "https://api.elections.kalshi.com/trade-api/v2"))
    poll_interval_seconds: int = Field(default=_get("kalshi", "poll_interval_seconds", 60))
    request_timeout_seconds: int = Field(default=_get("kalshi", "request_timeout_seconds", 30))
    max_markets_per_fetch: int = Field(default=_get("kalshi", "max_markets_per_fetch", 100))


class MatchingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MATCHING_")

    semantic_threshold: float = Field(default=_get("matching", "semantic_threshold", 0.75))
    min_keyword_overlap: int = Field(default=_get("matching", "min_keyword_overlap", 1))
    embedding_model: str = Field(default=_get("matching", "embedding_model", "all-MiniLM-L6-v2"))
    cache_dir: str = Field(default=_get("matching", "cache_dir", ".cache/embeddings"))
    nli_model: str = Field(default=_get("matching", "nli_model", "cross-encoder/nli-deberta-v3-small"))
    nli_threshold: float = Field(default=_get("matching", "nli_threshold", 0.65))
    ollama_model: str = Field(default=_get("matching", "ollama_model", "phi3:mini"))
    ollama_url: str = Field(default=_get("matching", "ollama_url", "http://localhost:11434"))
    use_ollama_fallback: bool = Field(default=_get("matching", "use_ollama_fallback", True))


class TradingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRADING_")

    min_gap_open_pp: float = Field(default=_get("trading", "min_gap_open_pp", 3.0))
    min_gap_close_pp: float = Field(default=_get("trading", "min_gap_close_pp", 1.0))
    default_position_size: float = Field(default=_get("trading", "default_position_size", 100.0))
    max_open_trades: int = Field(default=_get("trading", "max_open_trades", 20))


class DbConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_")

    sqlite_path: str = Field(default=_get("db", "sqlite_path", "data/operational.sqlite"))


class PollingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POLLING_")

    main_loop_interval_seconds: int = Field(
        default=_get("polling", "main_loop_interval_seconds", 300)
    )


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    log_level: str = Field(default="INFO")

    polymarket: PolymarketConfig = Field(default_factory=PolymarketConfig)
    kalshi: KalshiConfig = Field(default_factory=KalshiConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    db: DbConfig = Field(default_factory=DbConfig)
    polling: PollingConfig = Field(default_factory=PollingConfig)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return AppConfig()
