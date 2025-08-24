# Minimal multi-command Typer app
import typer
from src.data import download_prices, find_ticker
import pandas as pd
from src.io_utils import save_csv, save_period_all_tickers

app = typer.Typer(help="ETF Screener CLI")

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

@app.command()
def hello():
    # Test command to see if subcommands work.
    typer.echo(f"Hope this works")

if __name__ == "__main__":
    app()