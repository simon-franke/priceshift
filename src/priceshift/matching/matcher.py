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

    # ------------------------------------------------------------------
    # Rule pre-filter
    # ------------------------------------------------------------------

    def _passes_rule_filter(self, pm: Market, kalshi: Market) -> bool:
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
        # Stage 1 + 2: rule filter then semantic ranking
        candidates = [k for k in kalshi_markets if self._passes_rule_filter(pm, k)]
        if not candidates:
            return None

        pm_text = f"{pm.title} {pm.description}".strip() if pm.description else pm.title
        pm_emb = self._embed.encode(pm_text)
        best_score = -1.0
        best_match: Optional[Market] = None

        for kalshi in candidates:
            kalshi_text = f"{kalshi.title} {kalshi.description}".strip() if kalshi.description else kalshi.title
            kalshi_emb = self._embed.encode(kalshi_text)
            score = cosine_similarity(pm_emb, kalshi_emb)
            if score > best_score:
                best_score = score
                best_match = kalshi

        if best_match is None or best_score < self._threshold:
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
