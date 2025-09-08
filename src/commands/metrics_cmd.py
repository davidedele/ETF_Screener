# src/metrics_cmd.py
from __future__ import annotations

import typer
import pandas as pd

from src.factors import metrics_for_prices, split_metric_packs
from src.utils.io_utils import save_csv, ensure_dir
from src.utils.cli_utils import (
    ticker_metrics_dir,
    comparison_dir,
    all_tickers_dir,
    validate_dates_or_die,
    resolve_tickers_or_die,
    maybe_download_benchmark,
    filter_downloaded_or_die,
    maybe_print_debug_bounds,
    save_packs,
)

# Keep these here to avoid depending on cli.py
NUMERIC_COLS = [
    "total_return", "cagr", "vol_ann", "max_dd", "sharpe", "sortino",
    "avg_daily_volume", "best_month", "worst_month", "dd_duration_days",
    "recovery_days", "downside_vol", "skew", "kurtosis",
    "bench_cagr", "bench_vol_ann", "bench_max_dd",
    "excess_cagr", "rel_vol", "dd_gap", "beta", "corr", "te", "info_ratio",
]

PREFERRED_ORDER = [
    "ticker", "total_return", "cagr", "vol_ann", "max_dd",
    "sharpe", "sortino", "avg_daily_volume",
    "best_month", "worst_month", "dd_duration_days", "recovery_days",
    "downside_vol", "skew", "kurtosis",
    "bench_cagr", "bench_vol_ann", "bench_max_dd",
    "excess_cagr", "rel_vol", "dd_gap", "beta", "corr", "te", "info_ratio",
]

def run_metrics(
    *,
    tickers: list[str],
    start: str,
    end: str,
    benchmark: str | None,
    detailed: bool,
    show: bool,
    QUIET: bool,
    VERBOSE: bool,
) -> None:
    # Check if the start and end dates given make sense
    validate_dates_or_die(start, end)

    # Resolve tickers
    resolved = resolve_tickers_or_die(tickers, start, end, quiet=QUIET)

    # Download data for the resolved list
    from src.data import download_prices
    prices, volumes = download_prices(resolved, start, end)

    # Check if there is data downloaded + filter so only valid tickers with data are shown
    prices, volumes, _ = filter_downloaded_or_die(
        prices, volumes, resolved, ctx="metrics", quiet=QUIET
    )

    # Optional benchmark download
    bench_series = maybe_download_benchmark(benchmark, start, end, quiet=QUIET)

    combined_rows: list[pd.DataFrame] = []

    # ---- Metrics for each ticker and save ----
    for t in prices.columns:
        maybe_print_debug_bounds(prices, t, verbose=VERBOSE)

        # Compute all metrics for that single asset, optionally vs the benchmark
        m = metrics_for_prices(prices[[t]], volumes.get(t), benchmark=bench_series, rf_daily=0.0)
        packs = split_metric_packs(m)  # Split all.csv metrics into mini-tables

        # Accumulate the "all" row for the cross-ticker summary
        row_all = packs["all"].copy()
        row_all["ticker"] = t
        combined_rows.append(row_all)

        metrics_dir = ticker_metrics_dir(t, start, end)  # data/output/<TICKER>/<START>_to_<END>/metrics

        # always writes all.csv; writes BM_all.csv if BM; if --detailed writes the split packs
        save_packs(metrics_dir, packs, detailed=detailed, bm=(bench_series is not None))

        if show and not QUIET:
            typer.echo(f"\n=== {t} | {start} → {end} ===")
            typer.echo(packs["all"].to_string(index=False))

        if not QUIET:
            typer.secho(f"Saved metrics for {t} → {metrics_dir}/", fg=typer.colors.GREEN)

    # ---- Comparisons (only if >1 ticker) ----
    if len(prices.columns) > 1:
        m_comp = metrics_for_prices(prices, volumes, benchmark=bench_series, rf_daily=0.0)  # across tickers
        packs_comp = split_metric_packs(m_comp)  # logical groups

        comp_dir, combo = comparison_dir(list(prices.columns), start, end)  # folder path + combo name
        save_packs(comp_dir, packs_comp, detailed=detailed, bm=(bench_series is not None))  # write comparison metrics

        if show and not QUIET:
            typer.secho(f"\n=== COMPARISON: {combo} ===", fg=typer.colors.CYAN)
            typer.echo(packs_comp["all"].to_string())

        if not QUIET:
            typer.secho(f"Saved comparison metrics → {comp_dir}/", fg=typer.colors.GREEN)

    # ---- Global period summary across all processed tickers ----
    if combined_rows:
        # Filter out None or empty DataFrames
        rows = [df for df in combined_rows if isinstance(df, pd.DataFrame) and not df.empty]
        if not rows:
            if not QUIET:
                typer.secho("No per-ticker rows to summarize.", fg=typer.colors.YELLOW)
        else:
            summary = pd.concat(rows, ignore_index=True, sort=False)

            # Remove duplicate tickers
            if "ticker" in summary.columns:
                summary = summary.drop_duplicates(subset=["ticker"], keep="first")

            # Enforce numeric dtype for key columns (converts invalid values to NaN)
            for num_col in NUMERIC_COLS:
                if num_col in summary.columns:
                    summary[num_col] = pd.to_numeric(summary[num_col], errors="coerce")

            # Column ordering: preferred first, extras at the end
            cols = [c for c in PREFERRED_ORDER if c in summary.columns] + \
                   [c for c in summary.columns if c not in PREFERRED_ORDER]
            summary = summary[cols]

            # Sort rows by CAGR if present (highest first, NaN pushed to bottom)
            if "cagr" in summary.columns:
                summary = summary.sort_values("cagr", ascending=False, na_position="last")

            # Save final multi-ticker summary
            out_dir = all_tickers_dir(start, end)
            ensure_dir(out_dir)
            save_csv(summary, f"{out_dir}/metrics_all.csv")

            # Print to console if requested
            if show and not QUIET:
                typer.echo("\n=== Comparative summary (all tickers) ===")
                with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 120):
                    typer.echo(summary.to_string(index=False))

            if not QUIET:
                typer.secho(f"Saved summary → {out_dir}/metrics_all.csv", fg=typer.colors.GREEN)