# src/fetch_cmd.py
from __future__ import annotations

import typer

from src.data import download_prices
from src.utils.io_utils import save_period_all_tickers
from src.utils.cli_utils import (
    validate_dates_or_die,
    resolve_tickers_or_die,
    filter_downloaded_or_die,
    print_head_block,
    maybe_print_debug_bounds,
)

def run_fetch(
    *,
    tickers: list[str],
    start: str,
    end: str,
    QUIET: bool,
    VERBOSE: bool,
) -> None:
    # Check if the start and end dates given make sense
    validate_dates_or_die(start, end)

    # Resolve tickers (try common suffixes if needed)
    resolved = resolve_tickers_or_die(tickers, start, end, quiet=QUIET)

    # Download data
    prices, volumes = download_prices(resolved, start, end)

    # Check if there is data downloaded + filter so only valid tickers with data are shown
    prices, volumes, resolved = filter_downloaded_or_die(
        prices, volumes, resolved, ctx="fetch", quiet=QUIET
    )

    # Optional console preview (per ticker, compact)
    for t in resolved:
        print_head_block(prices, volumes, t, quiet=QUIET)         # show preview
        maybe_print_debug_bounds(prices, t, verbose=VERBOSE)       # add debug info

    # Save two CSV files for each ticker into: data/output/<TICKER>/<START>_to_<END>/
    saved_paths = save_period_all_tickers(prices, volumes, start, end, out_base="data/output")
    if not QUIET:
        typer.secho("\nSaved per-ticker period bundles:", fg=typer.colors.GREEN)
        for p in saved_paths:
            typer.echo(f"- {p}")