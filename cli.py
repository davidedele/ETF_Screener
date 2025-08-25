# cli.py — Minimal multi-command Typer app
import typer
import pandas as pd
import typing

from src.data import download_prices
from src.io_utils import save_csv, save_period_all_tickers, ensure_dir
from src.factors import metrics_for_prices, split_metric_packs
from src.cli_utils import (
    ticker_metrics_dir,
    comparison_dir,
    all_tickers_dir,
    validate_dates_or_die,
    resolve_tickers_or_die,
    maybe_download_benchmark,
    filter_downloaded_or_die,
    print_head_block,
    maybe_print_debug_bounds,
    save_packs,
)

VERBOSE = False
QUIET = False


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

app = typer.Typer(
    help=(
        "A small CLI to download ETF data, compute metrics, and save tidy outputs."
    )
    ,
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100},
)

def _version_callback(value: bool):
    if value:
        typer.echo("ETF Screener CLI v0.1.0")
        raise typer.Exit()


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="More logs."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Less logs."),
    version: bool = typer.Option(None, "--version", help="Show version and exit.", callback=_version_callback, is_eager=True
    ),
):
    global VERBOSE, QUIET
    VERBOSE = verbose
    QUIET = quiet



# ------------------------- FETCH -------------------------

@app.command( #help and epilog so for the user is simpler to use the tool
    help="Download ETF price/volume data and save per-ticker/per-period bundles.",
    epilog=(
            "[bold]Examples[/]\n\n"
            "  [dim]Single ticker (auto-suffix)[/]\n"
            "  [bold]python cli.py fetch[/] [cyan]--tickers[/] [yellow]VWCE[/] "
            "[cyan]--start[/] [yellow]2022-01-01[/] [cyan]--end[/] [yellow]2023-01-01[/]\n\n"
            "  [dim]Multiple tickers (mix with/without suffix)[/]\n"
            "  [bold]python cli.py fetch[/] [cyan]--tickers[/] [yellow]VWCE[/] "
            "[cyan]--tickers[/] [yellow]CSPX.L[/] "
            "[cyan]--start[/] [yellow]2022-01-01[/] [cyan]--end[/] [yellow]2023-01-01[/]\n\n"
            "  [dim]Longer time period (5 years)[/]\n"
            "  [bold]python cli.py fetch[/] [cyan]--tickers[/] [yellow]IWDA.L[/] "
            "[cyan]--start[/] [yellow]2018-01-01[/] [cyan]--end[/] [yellow]2023-01-01[/]\n\n"
            "\n\n"
            "  [dim]Quiet mode (minimal logs)[/]\n"
            "  [bold]python cli.py fetch[/] [cyan]--tickers[/] [yellow]VWCE[/] "
            "[cyan]--start[/] [yellow]2022-01-01[/] [cyan]--end[/] [yellow]2023-01-01[/] -q\n\n"
            "  [dim]Verbose mode (debug logs)[/]\n"
            "  [bold]python cli.py fetch[/] [cyan]--tickers[/] [yellow]VWCE[/] "
            "[cyan]--start[/] [yellow]2022-01-01[/] [cyan]--end[/] [yellow]2023-01-01[/] -v"
    )
)
#---------------------------------- Retrieve and save price/volume data for one or more tickers--------------------------
def fetch(
    tickers: list[str] = typer.Option(
        ...,
        "--tickers",
        help=(
            "Repeat for multiple symbols (e.g. VWCE, CSPX, IWDA). "
            "You can omit exchange suffixes; the CLI will try to resolve them."
        ),

        metavar="TICKER",
    ),
    start: str = typer.Option(
        ..., "--start", help="Start date (YYYY-MM-DD).", metavar="YYYY-MM-DD"
    ),
    end: str = typer.Option(
        ..., "--end", help="End date (YYYY-MM-DD).", metavar="YYYY-MM-DD"
    ),
    quiet_cmd: typing.Optional[bool] = typer.Option(None, "--quiet", "-q", help="Less logs."),
    verbose_cmd: typing.Optional[bool] = typer.Option(None, "--verbose", "-v", help="More logs."),
):
    global QUIET, VERBOSE
    if quiet_cmd is not None:
        QUIET = quiet_cmd
    if verbose_cmd is not None:
        VERBOSE = verbose_cmd
    if QUIET and VERBOSE:
        typer.secho("Choose either --quiet or --verbose, not both.", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    # Check if the start and end dates given make sense
    validate_dates_or_die(start, end)

    # Resolve tickers (try common suffixes if needed)
    resolved = resolve_tickers_or_die(tickers, start, end, quiet=QUIET)

    # Download data
    prices, volumes = download_prices(resolved, start, end)

    # Check if there is data downloaded + filter so only valid tickers with data are shown
    prices, volumes, resolved = filter_downloaded_or_die(prices, volumes, resolved, ctx="fetch", quiet = QUIET)

    # Optional console preview (per ticker, compact)
    for t in resolved:
        print_head_block(prices, volumes, t, quiet = QUIET) #show preview
        maybe_print_debug_bounds(prices, t, verbose = VERBOSE) #add debug info

    # save two CSV files for each ticker into a folder like: data/output/<TICKER>/<START>_to_<END>/
    saved_paths = save_period_all_tickers(prices, volumes, start, end, out_base="data/output")
    if not QUIET:
        typer.secho("\nSaved per-ticker period bundles:", fg=typer.colors.GREEN)
        for p in saved_paths:
            typer.echo(f"- {p}")


# ------------------------- METRICS -------------------------
# Compute metrics for each ticker and save CSVs under:
# data/output/<TICKER>/<START>_to_<END>/metrics/{all.csv[, BM_all.csv][, packs if --detailed]}
# and comparisons under:
# data/output/comparisons/<T1-T2-...>/metrics_<START>_to_<END>/{all.csv[, BM_all.csv][, packs if --detailed]}

@app.command(
    help="Compute performance/risk metrics and save them per ticker (and comparisons).",
    epilog=(
            "[bold]Examples[/]\n\n"
            "  [dim]# Two tickers + show summary[/]\n"
            "  [bold]python cli.py metrics[/] [cyan]--tickers[/] [yellow]VWCE[/] "
            "[cyan]--tickers[/] [yellow]CSPX.L[/] "
            "[cyan]--start[/] [yellow]2022-01-01[/] [cyan]--end[/] [yellow]2023-01-01[/] [cyan]--show[/]\n\n"
            "  [dim]# With benchmark (adds BM_all.csv)[/]\n"
            "  [bold]python cli.py metrics[/] [cyan]--tickers[/] [yellow]VWCE[/] "
            "[cyan]--start[/] [yellow]2022-01-01[/] [cyan]--end[/] [yellow]2023-01-01[/] "
            "[cyan]--benchmark[/] [yellow]^GSPC[/]\n\n"
            "  [dim]# Detailed packs[/]\n"
            "  [bold]python cli.py metrics[/] [cyan]--tickers[/] [yellow]VWCE[/] "
            "[cyan]--tickers[/] [yellow]CSPX.L[/] "
            "[cyan]--start[/] [yellow]2022-01-01[/] [cyan]--end[/] [yellow]2023-01-01[/] [cyan]--detailed[/]\n\n"
            "  [dim]# Two tickers + benchmark + detailed[/]\n"
            "  [bold]python cli.py metrics[/] [cyan]--tickers[/] [yellow]VWCE[/] "
            "[cyan]--tickers[/] [yellow]IWDA.L[/] "
            "[cyan]--start[/] [yellow]2021-01-01[/] [cyan]--end[/] [yellow]2023-01-01[/] "
            "[cyan]--benchmark[/] [yellow]^GSPC[/] [cyan]--detailed --show[/]\n\n"
            "\n\n"
            "  [dim]# Quiet mode (minimal logs)[/]\n"
            "  [bold]python cli.py metrics[/] [cyan]--tickers[/] [yellow]VWCE[/] "
            "[cyan]--start[/] [yellow]2022-01-01[/] [cyan]--end[/] [yellow]2023-01-01[/] -q\n\n"
            "  [dim]# Verbose mode (debug logs)[/]\n"
            "  [bold]python cli.py metrics[/] [cyan]--tickers[/] [yellow]VWCE[/] "
            "[cyan]--start[/] [yellow]2022-01-01[/] [cyan]--end[/] [yellow]2023-01-01[/] -v"
    ),
)

def metrics(
    tickers: list[str] = typer.Option(
        ...,
        "--tickers",
        help="Repeat for multiple symbols (e.g. VWCE, CSPX, IWDA). You may omit suffixes.",
        metavar="TICKER",
    ),
    start: str = typer.Option(
        ..., "--start", help="Start date (YYYY-MM-DD).", metavar="YYYY-MM-DD"
    ),
    end: str = typer.Option(
        ..., "--end", help="End date (YYYY-MM-DD).", metavar="YYYY-MM-DD"
    ),
    benchmark: str = typer.Option(
        None,
        "--benchmark",
        help="Optional benchmark (e.g. ^GSPC). Adds BM_all.csv alongside all.csv.",
        metavar="TICKER",
    ),
    detailed: bool = typer.Option(
        False, "--detailed",
        help="Also save split packs (performance, risk, risk_adjusted, tracking, liquidity)."
    ),
    show: bool = typer.Option(False, "--show", help="Print a compact overview in console."),
    quiet_cmd: typing.Optional[bool] = typer.Option(None, "--quiet", "-q", help="Less logs."),
    verbose_cmd: typing.Optional[bool] = typer.Option(None, "--verbose", "-v", help="More logs."),
):
    global QUIET, VERBOSE
    if quiet_cmd is not None:
        QUIET = quiet_cmd
    if verbose_cmd is not None:
        VERBOSE = verbose_cmd
    if QUIET and VERBOSE:
        typer.secho("Choose either --quiet or --verbose, not both.", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    # Check if the start and end dates given make sense
    validate_dates_or_die(start, end)

    # Resolve tickers
    resolved = resolve_tickers_or_die(tickers, start, end, quiet=QUIET)

    # Download data for the resolved list
    prices, volumes = download_prices(resolved, start, end)

    # Check if there is data downloaded + filter so only valid tickers with data are shown
    prices, volumes, _ = filter_downloaded_or_die(prices, volumes, resolved, ctx="metrics", quiet=QUIET)

    # Optional benchmark download
    bench_series = maybe_download_benchmark(benchmark, start, end, quiet=QUIET)

    combined_rows: list[pd.DataFrame] = []

    # ---- Metrics for each ticker and save ----
    for t in prices.columns:
        maybe_print_debug_bounds(prices, t, verbose=VERBOSE)
        #Compute all metrics for that single asset, optionally vs the benchmark (bench_series).
        m = metrics_for_prices(prices[[t]], volumes.get(t), benchmark=bench_series, rf_daily=0.0)
        packs = split_metric_packs(m) #Split all.csv metrics into mini‑tables

        # Accumulate the "all" row for the cross‑ticker summary
        row_all = packs["all"].copy()
        row_all["ticker"] = t
        combined_rows.append(row_all)

        metrics_dir = ticker_metrics_dir(t, start, end) #data/output/<TICKER>/<START>_to_<END>/metrics
        #always writes all.csv, writes BM_all.csv if BM,  if --detailed  writes the split packs
        save_packs(metrics_dir, packs, detailed=detailed, bm=(bench_series is not None))

        if show and not QUIET:
            typer.echo(f"\n=== {t} | {start} → {end} ===")
            typer.echo(packs["all"].to_string(index=False))

        if not QUIET:
            typer.secho(f"Saved metrics for {t} → {metrics_dir}/", fg=typer.colors.GREEN)

    # ---- Comparisons (only if >1 ticker) ----
    if len(prices.columns) > 1:
        m_comp = metrics_for_prices(prices, volumes, benchmark=bench_series, rf_daily=0.0) #computes metrics across tickers
        packs_comp = split_metric_packs(m_comp) #breaks comparison metrics table in logical groups of metrics

        comp_dir, combo = comparison_dir(list(prices.columns), start, end) #builds folder path and name (combo)
        save_packs(comp_dir, packs_comp, detailed=detailed, bm=(bench_series is not None)) #writes comparison metrics

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
            # nothing to summarize so exit
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

            # Column ordering: preferred  first, extras at the end
            preferred = PREFERRED_ORDER

            cols = [c for c in preferred if c in summary.columns] + \
                   [c for c in summary.columns if c not in preferred]
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


if __name__ == "__main__":
    app()