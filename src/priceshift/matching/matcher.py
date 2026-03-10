"""Event matcher: ground truth → rule filter → semantic similarity → NLI/LLM verification."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from priceshift.matching.embeddings import EmbeddingCache, cosine_similarity
from priceshift.matching.verifier import MatchVerifier
from priceshift.models import Market, MatchedPair

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "will", "the", "a", "an", "be", "is", "are", "was", "were",
    "in", "on", "at", "to", "of", "and", "or", "for", "by", "with",
    "this", "that", "it", "its", "from", "as", "up", "if", "do",
    # Common words that appear in many unrelated markets
    "before", "after", "any", "first", "next", "get", "have", "has",
    "2024", "2025", "2026", "2027", "2028", "2029", "2030",
}


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


class EventMatcher:
    """Five-stage matching pipeline:
    1. Ground truth lookup
    2. Rule filter (date proximity + keyword overlap)
    3. Semantic bi-encoder ranking
    4. NLI cross-encoder verification
    5. Ollama LLM fallback (for uncertain NLI results)
    """

    def __init__(
        self,
        semantic_threshold: float = 0.75,
        date_window_days: int = 14,
        min_keyword_overlap: int = 1,
        ground_truth_path: str = "data/ground_truth_pairs.json",
        cache_dir: str = ".cache/embeddings",
        model_name: str = "all-MiniLM-L6-v2",
        verifier: Optional[MatchVerifier] = None,
    ) -> None:
        self._threshold = semantic_threshold
        self._date_window = timedelta(days=date_window_days)
        self._min_overlap = min_keyword_overlap
        self._ground_truth = self._load_ground_truth(ground_truth_path)
        self._embed = EmbeddingCache(cache_dir=cache_dir, model_name=model_name)
        self._verifier = verifier

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------

    def _load_ground_truth(self, path: str) -> dict[tuple[str, str], float]:
        """Load manually validated pairs. Returns {(pm_id, kalshi_ticker): score}."""
        p = Path(path)
        if not p.exists():
            logger.warning("Ground truth file not found: %s", path)
            return {}
        with open(p) as f:
            pairs = json.load(f)
        result = {}
        for entry in pairs:
            key = (entry["polymarket_id"], entry["kalshi_ticker"])
            result[key] = entry.get("similarity_score", 1.0)
        logger.info("Loaded %d ground truth pairs", len(result))
        return result

    # ------------------------------------------------------------------
    # Rule pre-filter
    # ------------------------------------------------------------------

    def _passes_rule_filter(self, pm: Market, kalshi: Market) -> bool:
        # Date proximity
        if pm.resolution_date and kalshi.resolution_date:
            diff = abs((pm.resolution_date - kalshi.resolution_date).total_seconds())
            if diff > self._date_window.total_seconds():
                return False

        # Keyword overlap
        pm_tokens = _tokenize(pm.title + " " + pm.description)
        kalshi_tokens = _tokenize(kalshi.title + " " + kalshi.description)
        overlap = pm_tokens & kalshi_tokens
        if len(overlap) < self._min_overlap:
            return False

        return True

    # ------------------------------------------------------------------
    # Public matching API
    # ------------------------------------------------------------------

    def match_one(self, pm: Market, kalshi_markets: list[Market]) -> Optional[MatchedPair]:
        """Find the best Kalshi match for a single Polymarket market."""
        # Stage 1: ground truth
        for kalshi in kalshi_markets:
            key = (pm.id, kalshi.id)
            if key in self._ground_truth:
                return MatchedPair(
                    polymarket_id=pm.id,
                    kalshi_ticker=kalshi.id,
                    polymarket_title=pm.title,
                    kalshi_title=kalshi.title,
                    similarity_score=self._ground_truth[key],
                    match_source="ground_truth",
                )

        # Stage 2 + 3: rule filter then semantic ranking
        candidates = [k for k in kalshi_markets if self._passes_rule_filter(pm, k)]
        if not candidates:
            return None

        pm_emb = self._embed.encode(pm.title)
        best_score = -1.0
        best_match: Optional[Market] = None

        for kalshi in candidates:
            kalshi_emb = self._embed.encode(kalshi.title)
            score = cosine_similarity(pm_emb, kalshi_emb)
            if score > best_score:
                best_score = score
                best_match = kalshi

        if best_match is None or best_score < self._threshold:
            return None

        # Stage 4 + 5: NLI verification (+ Ollama fallback)
        if self._verifier is not None:
            is_match, confidence, source = self._verifier.verify_pair(pm, best_match)
            if not is_match:
                logger.debug(
                    "Verifier rejected %s vs %s (source=%s, conf=%.2f)",
                    pm.id, best_match.id, source, confidence,
                )
                return None
            return MatchedPair(
                polymarket_id=pm.id,
                kalshi_ticker=best_match.id,
                polymarket_title=pm.title,
                kalshi_title=best_match.title,
                similarity_score=best_score,
                match_source=source,
            )

        # No verifier configured — return unverified semantic match
        return MatchedPair(
            polymarket_id=pm.id,
            kalshi_ticker=best_match.id,
            polymarket_title=pm.title,
            kalshi_title=best_match.title,
            similarity_score=best_score,
            match_source="semantic",
        )

    def match_all(
        self,
        pm_markets: list[Market],
        kalshi_markets: list[Market],
    ) -> list[MatchedPair]:
        """Match all Polymarket markets against all Kalshi markets."""
        pairs = []
        for pm in pm_markets:
            pair = self.match_one(pm, kalshi_markets)
            if pair:
                pairs.append(pair)
        logger.info(
            "Matched %d / %d Polymarket markets to Kalshi events",
            len(pairs),
            len(pm_markets),
        )
        return pairs
