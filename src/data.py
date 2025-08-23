#Price of ETF Downloader
import yfinance as yf
import pandas as pd
import numpy as np

def _ensure_series(x):
    #return a 1 Dimension Panda Series even if x is a Dataframe with a single column.
    if isinstance(x, pd.DataFrame): # if is a DataFrame, take the first column as a Series
        return x.iloc[:, 0]
    if isinstance(x, pd.Series):
        return x

    # fallback: array/list → make it one dimension and make it a Series
    arr = np.asarray(x).reshape(-1)
    return pd.Series(arr)

def download_prices(tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # download Historical data (from start, to end) of a given list of tickers.
    # always return two DataFrames (prices, volumes) with Date index and one column per ticker.
    all_closes: dict[str, pd.Series] = {}
    all_vols: dict[str, pd.Series] = {}

    for i in tickers:
        data = yf.download(i, start=start, end=end, auto_adjust=True, progress=False)
        if data is None or data.empty:
            continue

        close = _ensure_series(data.get("Close"))
        vol = _ensure_series(data.get("Volume"))

        # Skip if Close/Volume not present
        if close is None or vol is None:
            continue

        # Make sure they are Series indexed by date
        close = pd.to_numeric(close, errors="coerce").astype("float64")
        vol = pd.to_numeric(vol, errors="coerce").astype("float64")

        all_closes[i] = close
        all_vols[i] = vol

    # Build wide DataFrames (one column per ticker)
    prices = pd.DataFrame(all_closes)
    volumes = pd.DataFrame(all_vols)

    # Align indexes and sort
    if not prices.empty:
        prices = prices.sort_index()
    if not volumes.empty:
        volumes = volumes.sort_index()

    return prices, volumes