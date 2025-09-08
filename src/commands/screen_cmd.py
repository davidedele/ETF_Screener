# src/commands/screen_cmd.py
from __future__ import annotations

import pandas as pd
import typer

from src.utils.cli_utils import resolve_tickers_or_die
from src.utils.io_utils import ensure_dir, save_csv
from src.utils.screen_utils import snapshot_row

def run_screen(
    *,
    tickers: list[str],
    show: bool,
    QUIET: bool,
    VERBOSE: bool,  # kept for consistency with other commands; not used here
) -> None:
    # We still reuse the resolver (it tries suffixes and validates symbol by downloading quietly)
    # Pass a very wide date range just to exercise the resolution logic.
    resolved = resolve_tickers_or_die(
        tickers, start="1900-01-01", end="2100-01-01", quiet=QUIET
    )

    # Ensure per-ticker folder and write a single fund-facts CSV: screen.csv
    for t in resolved:
        ensure_dir(f"data/output/tickers/{t}")
        row = snapshot_row(t)
        df = pd.DataFrame([row])

        out_path = f"data/output/tickers/{t}/screen.csv"
        save_csv(df, out_path)

        if not QUIET:
            typer.secho(f"Saved screen snapshot → {out_path}", fg=typer.colors.GREEN)

        if show and not QUIET:
            with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 120):
                typer.echo(f"\n=== {t} fund facts ===")
                typer.echo(df.to_string(index=False))