"""Export current DB state as JSON for the GitHub Pages static dashboard."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from priceshift.db.store import DataStore


def export_json(store: DataStore, output_dir: str = "docs/data") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gaps = store.get_latest_gaps(limit=50)
    (out / "gaps.json").write_text(json.dumps(gaps, default=str))

    open_trades = store.get_open_trades()
    (out / "open_trades.json").write_text(json.dumps(open_trades, default=str))

    summary = store.get_trade_summary()
    (out / "summary.json").write_text(json.dumps(summary or {}, default=str))

    meta = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "gap_count": len(gaps),
        "open_trade_count": len(open_trades),
    }
    (out / "meta.json").write_text(json.dumps(meta))
