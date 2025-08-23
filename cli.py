# Minimal multi-command Typer app
import typer
from src.data import download_prices
import pandas as pd

app = typer.Typer(help="ETF Screener CLI")

@app.command()
def fetch( #Retrieve necessary infos for downloading data
    tickers: list[str] = typer.Option(
        ..., "--tickers", help="Repeat Option fro multiple symbols (e.g. VWCE.DE, etc.)"),
    start: str = typer.Option(..., "--start", help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="End date YYYY-MM-DD"),
):
    #Download and print first rows of prices.
    prices, volumes = download_prices(tickers, start, end)
    if prices.empty:
        typer.echo("No data. Try an exchange suffix like .DE, .L, .AS, .MI.")
        raise typer.Exit(code=1)

     # ------------------------------ OPTION 1: each ticker has its own column ------------------
    # combined = pd.concat(
    #     {"Price": prices, "Volume": volumes}, axis=1
    # )
    # typer.echo(combined.head())

    # ------------------------------ OPTION 2: each ticker printed as its own block ------------------

    for i in tickers:
        if i in prices.columns:
            typer.echo(f"\n=== {i} ===")
            df = pd.DataFrame({
                "Price": prices[i],
                "Volume": volumes[i]
            })
            typer.echo(df.head().to_string())

@app.command()
def hello():
    # Test command to see if subcommands work.
    typer.echo(f"Hope this works")

if __name__ == "__main__":
    app()