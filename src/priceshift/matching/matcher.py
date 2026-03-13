"""Event matcher: rule filter → semantic similarity → NLI/LLM verification."""
from __future__ import annotations

import logging
import re
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


def _market_text(market: Market) -> str:
    """Build search text from market title + description."""
    if market.description:
        return f"{market.title} {market.description}".strip()
    return market.title


class EventMatcher:
    """Four-stage matching pipeline:
    1. Rule filter (keyword overlap)
    2. Semantic bi-encoder ranking
    3. NLI cross-encoder verification
    4. Ollama LLM fallback (for uncertain NLI results)
    """

    def __init__(
        self,
        semantic_threshold: float = 0.75,
        min_keyword_overlap: int = 1,
        cache_dir: str = ".cache/embeddings",
        model_name: str = "all-MiniLM-L6-v2",
        verifier: Optional[MatchVerifier] = None,
    ) -> None:
        self._threshold = semantic_threshold
        self._min_overlap = min_keyword_overlap
        self._embed = EmbeddingCache(cache_dir=cache_dir, model_name=model_name)
        self._verifier = verifier

    def _passes_rule_filter(self, pm: Market, kalshi: Market) -> bool:
        pm_tokens = _tokenize(_market_text(pm))
        kalshi_tokens = _tokenize(_market_text(kalshi))
        return len(pm_tokens & kalshi_tokens) >= self._min_overlap

    def _find_best_semantic(
        self, pm: Market, candidates: list[Market],
    ) -> tuple[Optional[Market], float]:
        """Return the best semantic match and its score from candidates."""
        pm_emb = self._embed.encode(_market_text(pm))
        best_score = -1.0
        best_match: Optional[Market] = None
        for kalshi in candidates:
            score = cosine_similarity(pm_emb, self._embed.encode(_market_text(kalshi)))
            if score > best_score:
                best_score = score
                best_match = kalshi
        return best_match, best_score

    def _make_pair(
        self, pm: Market, kalshi: Market, score: float, source: str,
    ) -> MatchedPair:
        return MatchedPair(
            polymarket_id=pm.id,
            kalshi_ticker=kalshi.id,
            polymarket_title=pm.title,
            kalshi_title=kalshi.title,
            similarity_score=score,
            match_source=source,
        )

    def match_one(self, pm: Market, kalshi_markets: list[Market]) -> Optional[MatchedPair]:
        """Find the best Kalshi match for a single Polymarket market."""
        # Stage 1: rule filter
        candidates = [k for k in kalshi_markets if self._passes_rule_filter(pm, k)]
        logger.debug(
            "Rule filter: %d / %d Kalshi candidates for PM '%s'",
            len(candidates), len(kalshi_markets), pm.title[:60],
        )
        if not candidates:
            return None

        # Stage 2: semantic ranking
        best_match, best_score = self._find_best_semantic(pm, candidates)
        if best_match is None or best_score < self._threshold:
            logger.debug(
                "Semantic threshold not met (%.3f < %.3f) for PM '%s'",
                best_score, self._threshold, pm.title[:60],
            )
            return None

        # Stage 3 + 4: NLI verification (+ Ollama fallback)
        if self._verifier is not None:
            is_match, confidence, source = self._verifier.verify_pair(pm, best_match)
            if not is_match:
                logger.debug(
                    "Verifier rejected %s vs %s (source=%s, conf=%.2f)",
                    pm.id, best_match.id, source, confidence,
                )
                return None
            return self._make_pair(pm, best_match, best_score, source)

        return self._make_pair(pm, best_match, best_score, "semantic")

    def match_all(
        self,
        pm_markets: list[Market],
        kalshi_markets: list[Market],
    ) -> list[MatchedPair]:
        """Match all Polymarket markets against all Kalshi markets."""
        logger.info(
            "Starting match_all: %d PM markets × %d Kalshi markets",
            len(pm_markets), len(kalshi_markets),
        )
        pairs = []
        for pm in pm_markets:
            pair = self.match_one(pm, kalshi_markets)
            if pair is not None:
                pairs.append(pair)

        logger.info("match_all done: %d matched out of %d PM markets", len(pairs), len(pm_markets))
        return pairs
