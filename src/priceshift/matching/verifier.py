"""NLI cross-encoder + Ollama LLM verification for match candidates."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

import urllib.request
import urllib.error

from priceshift.db.store import DataStore
from priceshift.models import Market

logger = logging.getLogger(__name__)

# Lazy-loaded cross-encoder model
_nli_model: Optional[object] = None


def _get_nli_model(model_name: str = "cross-encoder/nli-deberta-v3-small") -> object:
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder  # type: ignore

        logger.info("Loading NLI model %s ...", model_name)
        _nli_model = CrossEncoder(model_name)
        logger.info("NLI model loaded.")
    return _nli_model


@dataclass
class VerifiedPair:
    pm: Market
    kalshi: Market
    is_match: bool
    confidence: float
    source: str  # "nli", "ollama", "cached"
    explanation: str = ""


class NLIVerifier:
    """Cross-encoder NLI verification using nli-deberta-v3-small."""

    # Label order for nli-deberta-v3-small: contradiction, entailment, neutral
    _LABELS = ["contradiction", "entailment", "neutral"]

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        threshold: float = 0.65,
        contradiction_threshold: float = 0.7,
    ) -> None:
        self._model_name = model_name
        self._threshold = threshold
        self._contradiction_threshold = contradiction_threshold

    def _predict(self, text_a: str, text_b: str) -> dict[str, float]:
        """Run cross-encoder and return label→score dict."""
        import numpy as np

        model = _get_nli_model(self._model_name)
        scores = model.predict([(text_a, text_b)])  # type: ignore
        # scores shape: (1, 3) — softmax over [contradiction, entailment, neutral]
        probs = scores[0]
        if not hasattr(probs, "__len__") or len(probs) != 3:
            # Some CrossEncoder versions return logits; apply softmax
            probs = np.exp(probs) / np.exp(probs).sum()
        return dict(zip(self._LABELS, [float(p) for p in probs]))

    def verify(self, title_a: str, title_b: str) -> tuple[bool | None, float, str]:
        """Bidirectional NLI check.

        Returns:
            (is_match, confidence, verdict_label)
            is_match=True  → match
            is_match=False → reject
            is_match=None  → uncertain (send to Ollama)
        """
        scores_ab = self._predict(title_a, title_b)
        scores_ba = self._predict(title_b, title_a)

        entailment_score = min(scores_ab["entailment"], scores_ba["entailment"])
        contradiction_score = max(scores_ab["contradiction"], scores_ba["contradiction"])

        if entailment_score >= self._threshold:
            return True, entailment_score, "nli_match"

        if contradiction_score >= self._contradiction_threshold:
            return False, contradiction_score, "nli_contradiction"

        # Uncertain
        return None, entailment_score, "nli_uncertain"


class OllamaVerifier:
    """Fallback verification using local Ollama LLM."""

    def __init__(
        self,
        model: str = "phi3:mini",
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            req = urllib.request.Request(f"{self._base_url}/api/tags", method="GET")
            urllib.request.urlopen(req, timeout=5)
            return True
        except (urllib.error.URLError, OSError):
            return False

    def verify(
        self,
        pm_title: str,
        pm_desc: str,
        kalshi_title: str,
        kalshi_desc: str,
    ) -> tuple[bool, float, str]:
        """Ask Ollama whether two markets are equivalent.

        Returns (is_match, confidence_0_to_1, explanation).
        """
        prompt = (
            "Do these two prediction markets ask the same question and resolve "
            "to the same outcome?\n\n"
            f"Market A: {pm_title}\n"
            f"Description A: {pm_desc}\n\n"
            f"Market B: {kalshi_title}\n"
            f"Description B: {kalshi_desc}\n\n"
            "Respond with exactly one line in this format:\n"
            "YES|NO <confidence 0-100> <one-sentence explanation>\n"
            "Example: YES 85 Both markets ask whether Bitcoin exceeds $100k by end of 2025."
        )

        payload = json.dumps({
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            resp = urllib.request.urlopen(req, timeout=self._timeout)
            body = json.loads(resp.read().decode())
            response_text = body.get("response", "").strip()
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            logger.warning("Ollama request failed: %s", e)
            return False, 0.0, f"ollama_error: {e}"

        return self._parse_response(response_text)

    @staticmethod
    def _parse_response(text: str) -> tuple[bool, float, str]:
        """Parse 'YES|NO <confidence> <explanation>' from LLM output."""
        # Try to find YES/NO at start of any line
        for line in text.splitlines():
            line = line.strip()
            match = re.match(r"^(YES|NO)\s+(\d+)\s+(.*)", line, re.IGNORECASE)
            if match:
                is_match = match.group(1).upper() == "YES"
                confidence = min(100, max(0, int(match.group(2)))) / 100.0
                explanation = match.group(3).strip()
                return is_match, confidence, explanation

        # Fallback: look for yes/no anywhere
        lower = text.lower()
        if "yes" in lower:
            return True, 0.5, text[:200]
        return False, 0.5, text[:200]


class MatchVerifier:
    """Orchestrates NLI + Ollama verification with SQLite caching."""

    def __init__(
        self,
        store: DataStore,
        nli_model: str = "cross-encoder/nli-deberta-v3-small",
        nli_threshold: float = 0.65,
        ollama_model: str = "phi3:mini",
        ollama_url: str = "http://localhost:11434",
        use_ollama_fallback: bool = True,
    ) -> None:
        self._store = store
        self._nli = NLIVerifier(model_name=nli_model, threshold=nli_threshold)
        self._use_ollama = use_ollama_fallback
        self._ollama: Optional[OllamaVerifier] = None
        if use_ollama_fallback:
            self._ollama = OllamaVerifier(model=ollama_model, base_url=ollama_url)

    @staticmethod
    def _build_nli_text(market: Market) -> str:
        """Compose title + description into a single NLI input string."""
        title = market.title.strip()
        desc = market.description.strip()
        if desc and desc.lower() != title.lower():
            return f"{title}. {desc}"
        return title

    def verify_pair(self, pm: Market, kalshi: Market) -> tuple[bool, float, str]:
        """Verify a candidate pair. Returns (is_match, confidence, source)."""
        # 1. Check cache
        cached = self._store.get_match_verdict(pm.id, kalshi.id)
        if cached is not None:
            return bool(cached["is_match"]), cached["confidence"], "cached"

        # 2. Run NLI with enriched text (title + description)
        pm_text = self._build_nli_text(pm)
        kalshi_text = self._build_nli_text(kalshi)
        nli_result, nli_conf, nli_label = self._nli.verify(pm_text, kalshi_text)

        if nli_result is True:
            self._store.save_match_verdict(
                pm.id, kalshi.id, True, nli_conf, "nli", nli_label,
            )
            return True, nli_conf, "nli_verified"

        if nli_result is False:
            self._store.save_match_verdict(
                pm.id, kalshi.id, False, nli_conf, "nli", nli_label,
            )
            return False, nli_conf, "nli_verified"

        # 3. Uncertain — try Ollama fallback
        if self._ollama and self._ollama.is_available():
            is_match, conf, explanation = self._ollama.verify(
                pm.title, pm.description, kalshi.title, kalshi.description,
            )
            self._store.save_match_verdict(
                pm.id, kalshi.id, is_match, conf, "ollama", explanation,
            )
            source = "llm_verified"
            return is_match, conf, source

        # 4. No Ollama — reject uncertain pairs conservatively
        logger.debug(
            "NLI uncertain and Ollama unavailable for %s vs %s (conf=%.2f)",
            pm.id, kalshi.id, nli_conf,
        )
        self._store.save_match_verdict(
            pm.id, kalshi.id, False, nli_conf, "nli", "uncertain_no_fallback",
        )
        return False, nli_conf, "nli_verified"

    def verify_batch(
        self, pairs: list[tuple[Market, Market]]
    ) -> list[VerifiedPair]:
        """Verify a batch of candidate pairs."""
        results = []
        for pm, kalshi in pairs:
            is_match, confidence, source = self.verify_pair(pm, kalshi)
            results.append(VerifiedPair(
                pm=pm,
                kalshi=kalshi,
                is_match=is_match,
                confidence=confidence,
                source=source,
            ))
        return results
