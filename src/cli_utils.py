# src/cli_utils.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import typer
import pandas as pd

from src.data import download_prices, find_ticker
from src.io_utils import ensure_dir, save_csv


# Keep pack names here so cli.py can import a single constant.
PACK_NAMES: tuple[str, ...] = ("performance", "risk", "risk_adjusted", "tracking", "liquidity")


# ---------------------------
# Paths & tokens
# ---------------------------
#Make a folder-friendly token: no slashes, compact, no dots/spaces.
def safe_name(x: str) -> str:
    return x.replace("/", "-").replace("\\", "-").replace(" ", "").replace(".", "")

#creates consistent folder/file name for a time period
def period_token(start: str, end: str) -> str:
    return f"{start}_to_{end}"

#data/output/<TICKER>/<START>_to_<END>
def ticker_period_dir(ticker: str, start: str, end: str) -> str:
    return f"data/output/{ticker}/{period_token(start, end)}"

#data/output/<TICKER>/<START>_to_<END>/metrics
def ticker_metrics_dir(ticker: str, start: str, end: str) -> str:
    return f"{ticker_period_dir(ticker, start, end)}/metrics"

#Return comparison output dir and the combo token (Multi-Ticket).
def comparison_dir(resolved_tickers: list[str], start: str, end: str) -> tuple[str, str]:
    combo = "-".join(safe_name(t) for t in sorted(resolved_tickers))
    path = f"data/output/comparisons/{combo}/metrics_{period_token(start, end)}"
    return path, combo

#data/output/all_tickers/<START>_to_<END>
def all_tickers_dir(start: str, end: str) -> str:
    return f"data/output/all_tickers/{period_token(start, end)}"


# ---------------------------
# Dates
# ---------------------------

#Strict ISO (YYYY-MM-DD); raises ValueError if invalid.
def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

#Validate input dates and exit(1) with a friendly message if invalid.
def validate_dates_or_die(start: str, end: str) -> None:
    try:
        ds = parse_date(start)
        de = parse_date(end)
    except ValueError:
        typer.secho("Dates must be in YYYY-MM-DD format.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if ds > de:
        typer.secho(f"Start date {start} is after end date {end}.", fg=typer.colors.RED)
        raise typer.Exit(code=1)


# ---------------------------
# I/O helpers
# ---------------------------

#Save 'all.csv' , 'BM_all.csv' if bm=True,  (if detailed=True) the split packs.
def save_packs(metrics_dir: str, packs: dict[str, pd.DataFrame], *, detailed: bool, bm: bool) -> None:
    ensure_dir(metrics_dir)
    save_csv(packs["all"], f"{metrics_dir}/all.csv")
    if bm:
        save_csv(packs["all"], f"{metrics_dir}/BM_all.csv")
    if detailed:
        for pack_name in PACK_NAMES:
            if pack_name in packs:
                save_csv(packs[pack_name], f"{metrics_dir}/{pack_name}.csv")


# ---------------------------
# Download & selection
# ---------------------------


#Try to resolve each provided ticker (adding suffixes, etc.). Die if none resolve.
def resolve_tickers_or_die(raw_tickers: list[str], start: str, end: str, *, quiet: bool) -> list[str]:
    resolved: list[str] = []
    for t in raw_tickers:
        real = find_ticker(t, start, end)
        if real:
            resolved.append(real)
        else:
            if not quiet:
                typer.secho(f"Could not resolve {t}", fg=typer.colors.RED)

    if not resolved:
        typer.secho("No valid tickers found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    unique = remove_duplicates(resolved)
    if len(unique) != len(resolved) and not quiet:
        removed = [t for i, t in enumerate(resolved) if t not in unique[:unique.index(t)+1]]
        typer.secho(f"Removed duplicate tickers. Using: {unique}", fg=typer.colors.YELLOW)

    if not quiet:
        typer.secho(f"Resolved tickers: {resolved}", fg=typer.colors.GREEN)
    return resolved

#Download benchmark prices and return its Series if OK, otherwise warn (once) and return None.
def maybe_download_benchmark(benchmark: str | None, start: str, end: str, *, quiet: bool) -> pd.Series | None:
    if not benchmark:
        return None
    b_prices, _ = download_prices([benchmark], start, end)
    if not b_prices.empty and benchmark in b_prices.columns:
        return b_prices[benchmark]
    if not quiet:
        typer.secho(f"Warning: benchmark '{benchmark}' not downloaded. Skipping.", fg=typer.colors.YELLOW)
    return None

#Keep only columns that were actually downloaded; warn for missing; exit if none.
def filter_downloaded_or_die(
    prices: pd.DataFrame,
    volumes: pd.DataFrame | None,
    resolved: list[str],
    *,
    ctx: str,         # "fetch" | "metrics"
    quiet: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    if prices is None or prices.empty or prices.shape[1] == 0:
        typer.secho(
            f"No price data downloaded for any ticker in {ctx}. "
            "Check symbols/suffixes and date range.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    downloaded = [t for t in resolved if t in prices.columns]
    missing = [t for t in resolved if t not in prices.columns]

    if missing and not quiet:
        typer.secho(f"Skipped tickers with no data: {missing}", fg=typer.colors.YELLOW)

    prices = prices[downloaded]
    if volumes is not None:
        volumes = volumes.reindex(columns=downloaded)

    if not downloaded:
        typer.secho("After filtering, no valid tickers remain.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Ensure unique columns
    prices = prices.loc[:, ~prices.columns.duplicated(keep="first")]
    if volumes is not None:
        volumes = volumes.loc[:, ~volumes.columns.duplicated(keep="first")]

    return prices, volumes, downloaded

#Return a new list with duplicates removed
def remove_duplicates(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ---------------------------
# Console helpers
# ---------------------------

#Show a small head() preview for a single ticker (no in quiet mode).
def print_head_block(prices: pd.DataFrame, volumes: pd.DataFrame, ticker: str, *, quiet: bool) -> None:
    if quiet:
        return
    if ticker not in prices.columns:
        return
    typer.echo(f"\n=== {ticker} ===")
    df = pd.DataFrame({"Price": prices[ticker], "Volume": volumes[ticker]})
    typer.echo(df.head().to_string())

#Print start/end rows info for a ticker when verbose is on.
def maybe_print_debug_bounds(prices: pd.DataFrame, t: str, *, verbose: bool) -> None:
    if verbose and t in prices.columns:
        typer.echo(
            f"[DEBUG] {t} → rows={len(prices[t])}, "
            f"start={prices.index.min()}, end={prices.index.max()}"
        )