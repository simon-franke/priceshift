"""Tests for NLI + Ollama match verification."""
from __future__ import annotations

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from priceshift.db.store import DataStore
from priceshift.matching.verifier import (
    MatchVerifier,
    NLIVerifier,
    OllamaVerifier,
)
from priceshift.models import Market, Platform


def _market(id: str, title: str, platform: Platform = Platform.POLYMARKET, desc: str = "") -> Market:
    return Market(id=id, platform=platform, title=title, description=desc)


# ---------------------------------------------------------------------------
# NLIVerifier
# ---------------------------------------------------------------------------


class TestNLIVerifier:
    def _make_verifier(self, threshold: float = 0.65) -> NLIVerifier:
        return NLIVerifier(threshold=threshold)

    @patch("priceshift.matching.verifier._get_nli_model")
    def test_nli_accepts_equivalent_titles(self, mock_get_model):
        """Bidirectional high entailment → match."""
        mock_model = MagicMock()
        # Both directions: high entailment
        mock_model.predict.return_value = [
            np.array([0.05, 0.90, 0.05])  # [contradiction, entailment, neutral]
        ]
        mock_get_model.return_value = mock_model

        verifier = self._make_verifier()
        is_match, confidence, label = verifier.verify(
            "Spain win World Cup 2026",
            "Spain World Cup 2026 champion",
        )
        assert is_match is True
        assert confidence >= 0.65
        assert label == "nli_match"

    @patch("priceshift.matching.verifier._get_nli_model")
    def test_nli_rejects_contradictory_titles(self, mock_get_model):
        """High contradiction in either direction → reject."""
        mock_model = MagicMock()
        # High contradiction
        mock_model.predict.return_value = [
            np.array([0.85, 0.05, 0.10])
        ]
        mock_get_model.return_value = mock_model

        verifier = self._make_verifier()
        is_match, confidence, label = verifier.verify(
            "Spain win World Cup",
            "Spain qualify for World Cup",
        )
        assert is_match is False
        assert label == "nli_contradiction"

    @patch("priceshift.matching.verifier._get_nli_model")
    def test_nli_uncertain_when_scores_ambiguous(self, mock_get_model):
        """Low entailment + low contradiction → uncertain."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [
            np.array([0.30, 0.35, 0.35])
        ]
        mock_get_model.return_value = mock_model

        verifier = self._make_verifier()
        is_match, confidence, label = verifier.verify(
            "Fed cuts rates in June",
            "Federal Reserve rate decision June",
        )
        assert is_match is None
        assert label == "nli_uncertain"


# ---------------------------------------------------------------------------
# OllamaVerifier
# ---------------------------------------------------------------------------


class TestOllamaVerifier:
    def test_parse_response_yes(self):
        text = "YES 85 Both markets ask whether Bitcoin exceeds $100k by end of 2025."
        is_match, conf, explanation = OllamaVerifier._parse_response(text)
        assert is_match is True
        assert conf == 0.85
        assert "Bitcoin" in explanation

    def test_parse_response_no(self):
        text = "NO 20 Market A asks about winning, Market B about qualifying."
        is_match, conf, explanation = OllamaVerifier._parse_response(text)
        assert is_match is False
        assert conf == 0.20

    def test_parse_response_fallback(self):
        text = "I think the answer is yes, they seem similar."
        is_match, conf, explanation = OllamaVerifier._parse_response(text)
        assert is_match is True
        assert conf == 0.5

    def test_parse_response_no_fallback(self):
        text = "These markets are completely different."
        is_match, conf, explanation = OllamaVerifier._parse_response(text)
        assert is_match is False
        assert conf == 0.5

    def test_is_available_returns_false_when_down(self):
        verifier = OllamaVerifier(base_url="http://localhost:19999")
        assert verifier.is_available() is False

    @patch("urllib.request.urlopen")
    def test_verify_handles_http_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        verifier = OllamaVerifier()
        is_match, conf, explanation = verifier.verify("A", "", "B", "")
        assert is_match is False
        assert conf == 0.0
        assert "ollama_error" in explanation


# ---------------------------------------------------------------------------
# MatchVerifier (orchestrator)
# ---------------------------------------------------------------------------


class TestMatchVerifier:
    @pytest.fixture
    def store(self, tmp_path) -> DataStore:
        return DataStore(sqlite_path=str(tmp_path / "test.sqlite"))

    @patch("priceshift.matching.verifier._get_nli_model")
    def test_caches_verdict(self, mock_get_model, store):
        """Second call should hit cache, not the model."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [np.array([0.05, 0.90, 0.05])]
        mock_get_model.return_value = mock_model

        verifier = MatchVerifier(store=store, use_ollama_fallback=False)
        pm = _market("pm-1", "Bitcoin over 100k")
        kalshi = _market("kal-1", "Bitcoin above 100k", Platform.KALSHI)

        # First call — hits model
        r1 = verifier.verify_pair(pm, kalshi)
        assert r1[0] is True
        assert r1[2] == "nli_verified"
        assert mock_model.predict.call_count == 2  # bidirectional

        # Second call — should hit cache
        mock_model.predict.reset_mock()
        r2 = verifier.verify_pair(pm, kalshi)
        assert r2[0] is True
        assert r2[2] == "cached"
        assert mock_model.predict.call_count == 0

    @patch("priceshift.matching.verifier._get_nli_model")
    def test_uncertain_without_ollama_rejects(self, mock_get_model, store):
        """When NLI is uncertain and no Ollama, reject conservatively."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [np.array([0.30, 0.35, 0.35])]
        mock_get_model.return_value = mock_model

        verifier = MatchVerifier(store=store, use_ollama_fallback=False)
        pm = _market("pm-2", "Fed rate cut")
        kalshi = _market("kal-2", "Federal Reserve decision", Platform.KALSHI)

        is_match, conf, source = verifier.verify_pair(pm, kalshi)
        assert is_match is False

    @patch("priceshift.matching.verifier._get_nli_model")
    def test_uncertain_with_ollama_calls_ollama(self, mock_get_model, store):
        """When NLI uncertain and Ollama available, use Ollama."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [np.array([0.30, 0.35, 0.35])]
        mock_get_model.return_value = mock_model

        verifier = MatchVerifier(store=store, use_ollama_fallback=True)
        pm = _market("pm-3", "Fed rate cut", desc="Will the Fed cut?")
        kalshi = _market("kal-3", "Federal Reserve decision", Platform.KALSHI, desc="Rate decision")

        # Mock Ollama to be available and return YES
        with patch.object(verifier._ollama, "is_available", return_value=True), \
             patch.object(verifier._ollama, "verify", return_value=(True, 0.9, "Same question")):
            is_match, conf, source = verifier.verify_pair(pm, kalshi)
            assert is_match is True
            assert source == "llm_verified"

    @patch("priceshift.matching.verifier._get_nli_model")
    def test_verify_batch(self, mock_get_model, store):
        """verify_batch returns VerifiedPair objects."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [np.array([0.05, 0.90, 0.05])]
        mock_get_model.return_value = mock_model

        verifier = MatchVerifier(store=store, use_ollama_fallback=False)
        pairs = [
            (_market("pm-a", "BTC 100k"), _market("kal-a", "Bitcoin 100k", Platform.KALSHI)),
        ]
        results = verifier.verify_batch(pairs)
        assert len(results) == 1
        assert results[0].is_match is True
        assert results[0].source == "nli_verified"

    @patch("priceshift.matching.verifier._get_nli_model")
    def test_verify_pair_passes_enriched_text_to_nli(self, mock_get_model, store):
        """verify_pair should feed 'title. description' to NLI, not just title."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [np.array([0.05, 0.90, 0.05])]
        mock_get_model.return_value = mock_model

        verifier = MatchVerifier(store=store, use_ollama_fallback=False)
        pm = _market("pm-5", "Will Spain win the 2026 FIFA World Cup?")
        kalshi = _market(
            "kal-5", "Spain",
            platform=Platform.KALSHI,
            desc="FIFA World Cup 2026 Winner: Spain",
        )

        verifier.verify_pair(pm, kalshi)

        # NLI should have been called with enriched text, not bare title "Spain"
        calls = mock_model.predict.call_args_list
        assert len(calls) == 2  # bidirectional
        all_texts = [str(c) for c in calls]
        assert any("FIFA World Cup 2026 Winner: Spain" in t for t in all_texts), (
            "Expected enriched Kalshi text in NLI call"
        )

    def test_build_nli_text_combines_title_and_desc(self):
        """_build_nli_text returns 'title. desc' when they differ."""
        m = _market("x", "My Title", desc="My description")
        assert MatchVerifier._build_nli_text(m) == "My Title. My description"

    def test_build_nli_text_deduplicates_when_same(self):
        """_build_nli_text returns just title when desc equals title."""
        m = _market("x", "Same text", desc="Same text")
        assert MatchVerifier._build_nli_text(m) == "Same text"

    def test_build_nli_text_empty_desc(self):
        """_build_nli_text returns just title when desc is empty."""
        m = _market("x", "Just title")
        assert MatchVerifier._build_nli_text(m) == "Just title"
