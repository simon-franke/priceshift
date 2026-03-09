"""Shared pytest fixtures."""
from __future__ import annotations

import pytest

from priceshift.db.store import DataStore


@pytest.fixture
def tmp_store(tmp_path):
    """In-memory-ish store using temp paths."""
    sqlite = str(tmp_path / "test.sqlite")
    duckdb = str(tmp_path / "test.duckdb")
    store = DataStore(sqlite, duckdb)
    yield store
    store.close()
