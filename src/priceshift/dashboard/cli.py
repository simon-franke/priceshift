"""Rich live dashboard: gaps, portfolio, live polling."""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from priceshift.config import get_config
from priceshift.db.store import DataStore

logger = logging.getLogger(__name__)
console = Console()


def _gap_color(abs_gap: float) -> str:
    if abs_gap >= 5.0:
        return "red"
    elif abs_gap >= 3.0:
        return "yellow"
    return "green"


def show_gaps(store: DataStore, limit: int = 20) -> None:
    """Print a snapshot of the most recent arbitrage gaps."""
    gaps = store.get_latest_gaps(limit=limit)

    table = Table(title="Recent Arbitrage Gaps", show_lines=True)
    table.add_column("Timestamp", style="dim")
    table.add_column("Polymarket", max_width=40)
    table.add_column("Kalshi", max_width=40)
    table.add_column("PM Yes", justify="right")
    table.add_column("Kalshi Yes", justify="right")
    table.add_column("Gap (pp)", justify="right")

    for g in gaps:
        abs_gap = abs(float(g["gap_pp"]))
        color = _gap_color(abs_gap)
        ts = g["timestamp"]
        if isinstance(ts, datetime):
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts_str = str(ts)[:19]
        pm_name = str(g.get("polymarket_title") or g["polymarket_id"])[:40]
        kal_name = str(g.get("kalshi_title") or g["kalshi_ticker"])[:40]
        table.add_row(
            ts_str,
            pm_name,
            kal_name,
            f"{float(g['pm_yes_price']):.3f}",
            f"{float(g['kalshi_yes_price']):.3f}",
            Text(f"{float(g['gap_pp']):+.2f}", style=color),
        )

    console.print(table)


def show_portfolio(store: DataStore) -> None:
    """Print open trades and P&L summary."""
    open_trades = store.get_open_trades()
    summary = store.get_trade_summary()

    # Open trades
    open_table = Table(title=f"Open Trades ({len(open_trades)})", show_lines=True)
    open_table.add_column("ID", justify="right")
    open_table.add_column("Pair")
    open_table.add_column("Direction", max_width=30)
    open_table.add_column("Open Gap (pp)", justify="right")
    open_table.add_column("Size $", justify="right")
    open_table.add_column("Opened At", style="dim")

    for t in open_trades:
        open_table.add_row(
            str(t["id"]),
            f"{t['polymarket_id'][:15]}…/{t['kalshi_ticker']}",
            t["direction"],
            f"{float(t['open_gap_pp']):+.2f}",
            f"${float(t['position_size']):.0f}",
            str(t["opened_at"])[:19],
        )

    console.print(open_table)

    # Summary
    if summary:
        total = summary.get("total") or 0
        wins = summary.get("wins") or 0
        total_pnl = summary.get("total_pnl") or 0.0
        avg_pnl = summary.get("avg_pnl") or 0.0
        win_rate = (wins / total * 100) if total else 0.0
        pnl_color = "green" if total_pnl >= 0 else "red"

        console.print(
            f"[bold]Closed trades:[/bold] {total}  "
            f"[bold]Win rate:[/bold] {win_rate:.1f}%  "
            f"[bold]Total P&L:[/bold] [{pnl_color}]${total_pnl:+.2f}[/{pnl_color}]  "
            f"[bold]Avg P&L:[/bold] ${avg_pnl:+.2f}"
        )


def run_live(store: DataStore, refresh_seconds: int = 10) -> None:
    """Live-updating dashboard that refreshes every N seconds."""

    def build_table() -> Table:
        grid = Table.grid(expand=True)
        grid.add_column()

        # Reserve rows for the trades table so both fit in the terminal.
        # Each table row with show_lines uses ~2 terminal lines; headers/borders ~4.
        term_height = console.height or 50
        trade_rows = 10
        # trades table overhead: title + header + border lines ≈ 5, each row ≈ 2 lines
        trades_height = 5 + trade_rows * 2
        # gaps table overhead: title + header + border ≈ 5, each row ≈ 2 lines
        gap_budget = max(3, (term_height - trades_height - 5) // 2)

        gaps = store.get_latest_gaps(limit=gap_budget)
        gap_table = Table(
            title=f"[bold]Arbitrage Gaps[/bold] — {datetime.utcnow().strftime('%H:%M:%S UTC')}",
            show_lines=True,
            expand=True,
        )
        gap_table.add_column("Time", style="dim")
        gap_table.add_column("Polymarket", max_width=35)
        gap_table.add_column("Kalshi", max_width=35)
        gap_table.add_column("PM Yes", justify="right")
        gap_table.add_column("KAL Yes", justify="right")
        gap_table.add_column("Gap pp", justify="right")

        for g in gaps:
            abs_gap = abs(float(g["gap_pp"]))
            color = _gap_color(abs_gap)
            ts = str(g["timestamp"])[:19]
            pm_name = str(g.get("polymarket_title") or g["polymarket_id"])[:35]
            kal_name = str(g.get("kalshi_title") or g["kalshi_ticker"])[:35]
            gap_table.add_row(
                ts,
                pm_name,
                kal_name,
                f"{float(g['pm_yes_price']):.3f}",
                f"{float(g['kalshi_yes_price']):.3f}",
                Text(f"{float(g['gap_pp']):+.2f}", style=color),
            )

        open_trades = store.get_open_trades()
        trade_table = Table(
            title=f"[bold]Open Trades ({len(open_trades)})[/bold]",
            show_lines=True,
            expand=True,
        )
        trade_table.add_column("ID", justify="right")
        trade_table.add_column("Ticker")
        trade_table.add_column("Gap Open", justify="right")
        trade_table.add_column("Size $", justify="right")

        for t in open_trades[:trade_rows]:
            trade_table.add_row(
                str(t["id"]),
                str(t["kalshi_ticker"])[:18],
                f"{float(t['open_gap_pp']):+.2f}",
                f"${float(t['position_size']):.0f}",
            )

        grid.add_row(gap_table)
        grid.add_row(trade_table)
        return grid

    try:
        with Live(build_table(), refresh_per_second=0.2, console=console) as live:
            while True:
                time.sleep(refresh_seconds)
                live.update(build_table())
    except KeyboardInterrupt:
        console.print("[yellow]Dashboard stopped.[/yellow]")
