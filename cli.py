# Minimal multi-command Typer app
import typer
import pandas as pd

from src.data import download_prices, find_ticker
from src.io_utils import save_csv, save_period_all_tickers, ensure_dir
from src.factors import metrics_for_prices, split_metric_packs


app = typer.Typer(help="ETF Screener CLI")

def _safe_name(x: str) -> str:
    # folder-friendly: no slash, compact spaces, dots
    return x.replace("/", "-").replace("\\", "-").replace(" ", "").replace(".", "")

# ------------------------- FETCH -------------------------

@app.command()
def fetch( #Retrieve necessary infos for downloading data
    tickers: list[str] = typer.Option(
        ..., "--tickers", help="Repeat Option from multiple symbols (e.g. VWCE.DE, CSPX, NUKL.DE, etc.)"),
    start: str = typer.Option(..., "--start", help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="End date YYYY-MM-DD"),
):

    resolved = []
    for ticker in tickers:
        real = find_ticker(ticker, start, end)
        if real:
            resolved.append(real)
        else:
            typer.secho(f"Could not resolve {ticker}", fg=typer.colors.RED)

    if not resolved:
        typer.secho(f"No valid Tickers found", fg=typer.colors.RED)
        raise typer.Exit(code = 1)

    typer.secho(f"Resolved tickers: {resolved}", fg=typer.colors.GREEN)

    #Download and print first rows of prices.
    prices, volumes = download_prices(resolved, start, end)
    if prices.empty:
        typer.echo("No price data downloaded")
        raise typer.Exit(code=1)

     # ------------------------------ OPTION 1: each ticker has its own column ------------------
    # combined = pd.concat(
    #     {"Price": prices, "Volume": volumes}, axis=1
    # )
    # typer.echo(combined.head())

    # ------------------------------ OPTION 2: each ticker printed as its own block ------------------

    for i in resolved:
        if i in prices.columns:
            typer.echo(f"\n=== {i} ===")
            df = pd.DataFrame({
                "Price": prices[i],
                "Volume": volumes[i]
            })
            typer.echo(df.head().to_string())

    saved_paths = save_period_all_tickers(prices, volumes, start, end, out_base="data/output")
    typer.secho("\nSaved per-ticker period bundles:", fg=typer.colors.GREEN)
    for p in saved_paths:
        typer.echo(f"- {p}")

# ------------------------- METRICS -------------------------
# Compute metrics for each ticker and save CSVs under:
# data/output/<TICKER>/<START>_to_<END>/metrics/{overview|risk|stability|efficiency}.csv

@app.command()
def metrics(
    tickers: list[str] = typer.Option(
        ..., "--tickers", help="Repeat option for multiple symbols (e.g. VWCE.DE, CSPX.L)"
    ),
    start: str = typer.Option(..., "--start", help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="End date YYYY-MM-DD"),
    benchmark: str = typer.Option(
        None, "--benchmark", help="Optional benchmark ticker (e.g. CSPX.L)"
    ),
    show: bool = typer.Option(False, "--show", help="Print summary to console"),
):

    prices, volumes = download_prices(tickers, start, end)
    if prices.empty:
        typer.secho("No data. Check tickers/suffixes.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Optional benchmark download
    bench_series = None
    if benchmark:
        b_prices, _ = download_prices([benchmark], start, end)
        if not b_prices.empty and benchmark in b_prices.columns:
            bench_series = b_prices[benchmark]
        else:
            typer.secho(f"Warning: benchmark '{benchmark}' not downloaded. Skipping.", fg=typer.colors.YELLOW)

    combined_rows = []

    # Compute + save per ticker
    for t in prices.columns:
        from src.factors import metrics_for_prices, split_metric_packs
        from src.io_utils import save_csv, ensure_dir

        m = metrics_for_prices(prices[[t]], volumes.get(t), benchmark=bench_series, rf_daily=0.0)
        packs = split_metric_packs(m)

        # Collect the "all" row for cross-ticker comparison
        row_all = packs["all"].copy()
        row_all["ticker"] = t  # make sure ticker is correct
        combined_rows.append(row_all)

        base_dir = f"data/output/{t}/{start}_to_{end}/metrics"
        ensure_dir(base_dir)

        for pack_name, df in packs.items():
            save_csv(df, f"{base_dir}/{pack_name}.csv")

        if show:
            typer.echo(f"\n=== {t} | {start} → {end} ===")
            typer.echo(packs["all"].to_string(index=False))

        typer.secho(f"Saved metrics for {t} → {base_dir}/", fg=typer.colors.GREEN)

    # COMPARISON BLOCK (only for multiple tickers)
    if len(tickers) > 1:
        # compute metrics for all the DataFrame (more columns, more tickers)
        m_comp = metrics_for_prices(prices, volumes)

        # split in packs
        packs_comp = split_metric_packs(m_comp)

        # folder: data/output/comparisons/<T1-T2-...>/<PERIODO>/metrics
        combo = "-".join(_safe_name(t) for t in sorted(tickers))
        comp_dir = f"data/output/comparisons/{combo}/{start}_to_{end}/metrics"
        ensure_dir(comp_dir)

        # saves all CSVs (all.csv, performance.csv, risk.csv, …)
        for pack_name, df in packs_comp.items():
            save_csv(df, f"{comp_dir}/{pack_name}.csv")

        # optional print
        if show:
            typer.secho(f"\n=== COMPARISON: {combo} | {start} → {end} ===", fg=typer.colors.CYAN)
            # tabella 'all' è la più completa per un colpo d’occhio
            typer.echo(packs_comp["all"].to_string())
            typer.secho(f"Saved comparisons → {comp_dir}/", fg=typer.colors.GREEN)

    # ---- comparative table cross-ticker ----
    if combined_rows:
        summary = pd.concat(combined_rows, ignore_index=True)

        # order columns (puts key ones first )
        preferred = [
            "ticker", "total_return", "cagr", "vol_ann", "max_dd",
            "sharpe", "sortino", "avg_daily_volume",
            "best_month", "worst_month", "dd_duration_days", "recovery_days",
            "downside_vol", "skew", "kurtosis"
        ]
        cols = [c for c in preferred if c in summary.columns] + \
               [c for c in summary.columns if c not in preferred]
        summary = summary[cols]

        # Saves a general copy for the period
        out_dir = f"data/output/all_tickers/{start}_to_{end}"
        ensure_dir(out_dir)
        save_csv(summary, f"{out_dir}/metrics_all.csv")

        if show:
            typer.echo("\n=== Comparative summary (all tickers) ===")
            to_show = summary.sort_values("cagr", ascending=False, na_position="last")
            typer.echo(to_show.to_string(index=False))
            typer.secho(f"\nSaved comparative summary → {out_dir}/metrics_all.csv",
                        fg=typer.colors.GREEN)

if __name__ == "__main__":
    app()