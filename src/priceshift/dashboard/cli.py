"""Dashboard: Rich one-shot tables + Textual live TUI."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from rich.console import Console
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


def _pm_link(condition_id: str) -> str:
    return f"https://polymarket.com/event/{condition_id}"


def _kalshi_link(ticker: str) -> str:
    return f"https://kalshi.com/markets/{ticker.lower()}"


# ------------------------------------------------------------------
# One-shot Rich commands (priceshift gaps / portfolio)
# ------------------------------------------------------------------


def show_gaps(store: DataStore, limit: int = 20) -> None:
    """Print a snapshot of the most recent arbitrage gaps."""
    gaps = store.get_latest_gaps(limit=limit)

    table = Table(title="Recent Arbitrage Gaps", show_lines=True)
    table.add_column("Time", style="dim", max_width=19)
    table.add_column("Polymarket", max_width=50)
    table.add_column("Kalshi", max_width=50)
    table.add_column("PM\nYes/No", justify="right")
    table.add_column("KAL\nYes/No", justify="right")
    table.add_column("Gap\n(pp)", justify="right")
    table.add_column("Arb\nCost", justify="right")

    for g in gaps:
        abs_gap = abs(float(g["gap_pp"]))
        color = _gap_color(abs_gap)
        ts = g["timestamp"]
        if isinstance(ts, datetime):
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts_str = str(ts)[:19]

        pm_id = str(g["polymarket_id"])
        kal_id = str(g["kalshi_ticker"])
        pm_title = str(g.get("polymarket_title") or pm_id)
        kal_title = str(g.get("kalshi_title") or kal_id)

        pm_yes = float(g["pm_yes_price"])
        pm_no = 1.0 - pm_yes
        kal_yes = float(g["kalshi_yes_price"])
        kal_no = 1.0 - kal_yes

        arb_cost = min(pm_yes + kal_no, pm_no + kal_yes)

        pm_label = f"{pm_title}\n[dim]{pm_id[:30]}[/dim]"
        kal_label = f"{kal_title}\n[dim]{kal_id}[/dim]"

        table.add_row(
            ts_str,
            pm_label,
            kal_label,
            f"{pm_yes:.2%}\n{pm_no:.2%}",
            f"{kal_yes:.2%}\n{kal_no:.2%}",
            Text(f"{float(g['gap_pp']):+.2f}", style=color),
            Text(f"{arb_cost:.4f}", style="green" if arb_cost < 1.0 else "dim"),
        )

    console.print(table)


def show_portfolio(store: DataStore) -> None:
    """Print open trades and P&L summary."""
    open_trades = store.get_open_trades()
    summary = store.get_trade_summary()

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


# ------------------------------------------------------------------
# Textual live dashboard (priceshift dashboard)
# ------------------------------------------------------------------


def run_live(store: DataStore, refresh_seconds: int = 10) -> None:
    """Launch the Textual TUI dashboard."""
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Vertical
    from textual.widgets import DataTable, Footer, Header, Static

    class PnlBar(Static):
        """Single-line P&L summary bar."""

        def update_summary(self, summary: dict) -> None:
            total = summary.get("total") or 0
            wins = summary.get("wins") or 0
            total_pnl = summary.get("total_pnl") or 0.0
            avg_pnl = summary.get("avg_pnl") or 0.0
            win_rate = (wins / total * 100) if total else 0.0
            pnl_style = "green" if total_pnl >= 0 else "red"
            self.update(
                f" Closed: {total}  |  Win rate: {win_rate:.1f}%  |  "
                f"P&L: [{pnl_style}]${total_pnl:+.2f}[/{pnl_style}]  |  "
                f"Avg: ${avg_pnl:+.2f}"
            )

    class DashboardApp(App):
        CSS = """
        Screen {
            layout: vertical;
        }
        #gaps-label, #trades-label {
            height: 1;
            padding: 0 1;
            background: $primary-background;
            color: $text;
            text-style: bold;
        }
        #gaps-table {
            height: 2fr;
        }
        #trades-table {
            height: 1fr;
        }
        #pnl-bar {
            height: 1;
            padding: 0 1;
            background: $surface;
        }
        DataTable {
            overflow-y: auto;
        }
        """

        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("r", "refresh", "Refresh"),
            Binding("tab", "focus_next", "Next table"),
            Binding("shift+tab", "focus_previous", "Prev table"),
        ]

        TITLE = "Priceshift Dashboard"

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("Arbitrage Gaps", id="gaps-label")
            yield DataTable(id="gaps-table")
            yield Static("Open Trades", id="trades-label")
            yield DataTable(id="trades-table")
            yield PnlBar(id="pnl-bar")
            yield Footer()

        def on_mount(self) -> None:
            gaps_table = self.query_one("#gaps-table", DataTable)
            gaps_table.cursor_type = "row"
            gaps_table.add_columns(
                "Time", "Polymarket", "PM ID",
                "Kalshi", "KAL Ticker",
                "PM Yes", "PM No", "KAL Yes", "KAL No",
                "Gap pp", "Arb Cost",
            )

            trades_table = self.query_one("#trades-table", DataTable)
            trades_table.cursor_type = "row"
            trades_table.add_columns(
                "ID", "PM ID", "Kalshi Ticker",
                "Direction", "Gap Open (pp)", "Size $", "Opened At",
            )

            self._load_data()
            self.set_interval(refresh_seconds, self._load_data)

        def _load_data(self) -> None:
            self._load_gaps()
            self._load_trades()

        def _load_gaps(self) -> None:
            table = self.query_one("#gaps-table", DataTable)
            table.clear()

            gaps = store.get_latest_gaps(limit=200)
            label = self.query_one("#gaps-label", Static)
            label.update(
                f"Arbitrage Gaps ({len(gaps)}) — "
                f"{datetime.utcnow().strftime('%H:%M:%S UTC')}"
            )

            for g in gaps:
                abs_gap = abs(float(g["gap_pp"]))
                pm_yes = float(g["pm_yes_price"])
                pm_no = 1.0 - pm_yes
                kal_yes = float(g["kalshi_yes_price"])
                kal_no = 1.0 - kal_yes
                arb_cost = min(pm_yes + kal_no, pm_no + kal_yes)

                pm_id = str(g["polymarket_id"])
                kal_id = str(g["kalshi_ticker"])
                pm_title = str(g.get("polymarket_title") or pm_id)
                kal_title = str(g.get("kalshi_title") or kal_id)

                ts = str(g["timestamp"])[:19]
                gap_val = float(g["gap_pp"])

                gap_text = Text(f"{gap_val:+.2f}")
                gap_text.stylize(_gap_color(abs_gap))

                arb_text = Text(f"{arb_cost:.4f}")
                arb_text.stylize("green" if arb_cost < 1.0 else "dim")

                table.add_row(
                    ts,
                    pm_title,
                    pm_id[:30],
                    kal_title,
                    kal_id,
                    f"{pm_yes:.2%}",
                    f"{pm_no:.2%}",
                    f"{kal_yes:.2%}",
                    f"{kal_no:.2%}",
                    gap_text,
                    arb_text,
                )

        def _load_trades(self) -> None:
            table = self.query_one("#trades-table", DataTable)
            table.clear()

            open_trades = store.get_open_trades()
            label = self.query_one("#trades-label", Static)
            label.update(f"Open Trades ({len(open_trades)})")

            for t in open_trades:
                table.add_row(
                    str(t["id"]),
                    str(t["polymarket_id"])[:30],
                    str(t["kalshi_ticker"]),
                    str(t.get("direction", "")),
                    f"{float(t['open_gap_pp']):+.2f}",
                    f"${float(t['position_size']):.0f}",
                    str(t["opened_at"])[:19],
                )

            summary = store.get_trade_summary()
            self.query_one("#pnl-bar", PnlBar).update_summary(summary)

        def action_refresh(self) -> None:
            self._load_data()

    app = DashboardApp()
    app.run()
