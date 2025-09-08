# cli.py
import typer
import typing

from src.commands.fetch_cmd import run_fetch
from src.commands.metrics_cmd import run_metrics
from src.commands.screen_cmd import run_screen


app = typer.Typer(
    help=(
        "A small CLI to download ETF data, compute metrics, and save tidy outputs.\n\n"
    ),
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100},
)

VERBOSE = False
QUIET = False


def _version_callback(value: bool):
    if value:
        typer.echo("ETF Screener CLI v0.1.0")
        raise typer.Exit()


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="More logs."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Less logs."),
    version: bool = typer.Option(
        None,
        "--version",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
):
    global VERBOSE, QUIET
    VERBOSE = verbose
    QUIET = quiet


# ------------------------- FETCH -------------------------

@app.command(
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
    ),
)
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
    start: str = typer.Option(..., "--start", help="Start date (YYYY-MM-DD).", metavar="YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="End date (YYYY-MM-DD).", metavar="YYYY-MM-DD"),
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

    return run_fetch(
        tickers=tickers,
        start=start,
        end=end,
        QUIET=QUIET,
        VERBOSE=VERBOSE,
    )


# ------------------------- METRICS -------------------------

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
    start: str = typer.Option(..., "--start", help="Start date (YYYY-MM-DD).", metavar="YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="End date (YYYY-MM-DD).", metavar="YYYY-MM-DD"),
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

    return run_metrics(
        tickers=tickers,
        start=start,
        end=end,
        benchmark=benchmark,
        detailed=detailed,
        show=show,
        QUIET=QUIET,
        VERBOSE=VERBOSE,
    )

# ------------------------- SCREEN -------------------------


@app.command(
    help="Quick lookup of ETF metadata (ISIN, expense ratio, family, domicile, etc.). ",

    epilog=(
        "[bold]Examples[/]\n\n"
        "  [dim]# Snapshot only[/]\n"
        "  [bold]python cli.py screen[/] [cyan]--tickers[/] [yellow]VWCE[/]\n\n"
        "  [dim]# Multiple tickers + show output[/]\n"
        "  [bold]python cli.py screen[/] [cyan]--tickers[/] [yellow]VWCE[/] "
        "[cyan]--tickers[/] [yellow]CSPX.L[/] [cyan]--show[/]\n"
    ),
)
def screen(
    tickers: list[str] = typer.Option(
        ..., "--tickers", metavar="TICKER", help="Repeat for multiple symbols."
    ),
    show: bool = typer.Option(False, "--show", help="Print a compact snapshot in console."),
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

    return run_screen(
        tickers=tickers,
        show=show,
        QUIET=QUIET,
        VERBOSE=VERBOSE,
    )

if __name__ == "__main__":
    app()