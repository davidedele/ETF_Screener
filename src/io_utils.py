from pathlib import Path
import pandas as pd

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_csv(df: pd.DataFrame, out_dir: str | Path, filename: str) -> Path:
    out_path = ensure_dir(out_dir) / filename
    df.to_csv(out_path, index=True)
    return out_path

def load_csv(path: str) -> pd.DataFrame: #loads CSV file into a DataFrame
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"❌ File not found: {p}")
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    print(f"CSV loaded from {p}")
    return df


def _period_folder(start: str, end: str) -> str:
    # create folder for each period in this format "YYYY-MM-DD_to_YYYY-MM-DD"
    return f"{start}_to_{end}".replace(":", "-").strip()

def save_period_for_ticker(
    #saves 2 CSV for each ticker in a period folder data/output/<TICKER>/<start>_to_<end>/{prices.csv, volumes.csv}
    price_series: pd.Series,
    volume_series: pd.Series,
    out_base: str | Path,
    ticker: str,
    start: str,
    end: str,
) -> tuple[Path, Path]:

    period = _period_folder(start, end)
    out_dir = Path(out_base) / ticker / period
    ensure_dir(out_dir)

    prices_path = out_dir / "prices.csv"
    volumes_path = out_dir / "volumes.csv"

    # creates a one column DataFrame <ticker>
    p_df = pd.DataFrame({ticker: price_series})
    v_df = pd.DataFrame({ticker: volume_series})
    p_df.index.name = "Date"
    v_df.index.name = "Date"

    p_df.to_csv(prices_path)
    v_df.to_csv(volumes_path)

    return prices_path, volumes_path

def save_period_all_tickers(
    #for each ticker in prices_df/volumes_df saves 2 CSV in data/output/<TICKER>/<period>
    prices_df: pd.DataFrame,
    volumes_df: pd.DataFrame,
    start: str,
    end: str,
    out_base: str | Path = "data/output",
) -> list[Path]:
    saved: list[Path] = []
    for t in prices_df.columns:
        p = prices_df[t].dropna()
        v = volumes_df[t].dropna()
        pth_prices, pth_vols = save_period_for_ticker(p, v, out_base, t, start, end)
        saved.extend([pth_prices, pth_vols])
    return saved