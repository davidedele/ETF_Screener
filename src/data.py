#Price of ETF Downloader
import yfinance as yf
import pandas as pd
import numpy as np
import io
import contextlib
import warnings

#try these exchange suffixes so the user doesn't have to type it always
SUFFIX_CANDIDATES = [
    # --- Core markets ---
    "",       # USA (default: Nasdaq / NYSE / AMEX)
    ".L",     # UK - London Stock Exchange
    ".DE",    # Germany - XETRA / Frankfurt
    ".AS",    # Netherlands - Euronext Amsterdam
    ".PA",    # France - Euronext Paris
    ".MI",    # Italy - Borsa Italiana (Milan)
    ".IR",    # Ireland - Euronext Dublin
    ".BR",    # Belgium - Euronext Brussels
    ".LS",    # Portugal - Euronext Lisbon
    ".SW",    # Switzerland - SIX Swiss Exchange
    ".ST",    # Sweden - Stockholm
    ".CO",    # Denmark - Copenhagen
    ".HE",    # Finland - Helsinki
    ".OL",    # Norway - Oslo
    ".VI",    # Austria - Vienna
    ".MA",    # Spain - Madrid
    ".WA",    # Poland - Warsaw

    # --- Americas ---
    ".TO",    # Canada - Toronto
    ".V",     # Canada - TSX Venture
    ".MX",    # Mexico - Bolsa Mexicana
    ".SA",    # Brazil - B3 (São Paulo)

    # --- Asia-Pacific ---
    ".T",     # Japan - Tokyo
    ".HK",    # Hong Kong
    ".TW",    # Taiwan
    ".AX",    # Australia - ASX
    ".NZ",    # New Zealand
    ".KS",    # South Korea - KOSPI
    ".KQ",    # South Korea - KOSDAQ
    ".SI",    # Singapore
    ".JK",    # Indonesia
    ".BK",    # Thailand - Bangkok
    ".IN",    # India - NSE/BSE (varies, sometimes not consistent)

    # --- Middle East & Africa ---
    ".IL",    # Israel - Tel Aviv
    ".IS",    # Turkey - Istanbul
    ".SR",    # Saudi Arabia - Tadawul
    ".QA",    # Qatar
    ".AE",    # UAE - Dubai/Abu Dhabi

    # --- China (less common for ETFs on Yahoo) ---
    ".SS",    # Shanghai A-shares
    ".SZ",    # Shenzhen A-shares
]

#try different Exchange Suffixes until a ticker works. Returns the first ticker that works or None if none is found
def find_ticker(symbol: str, start:str, end:str) -> str | None:
    has_suffix = any(symbol.endswith(suffix) for suffix in SUFFIX_CANDIDATES)
    candidate = [symbol] if has_suffix else [symbol + suffix for suffix in SUFFIX_CANDIDATES]

    for suffix in SUFFIX_CANDIDATES:
        candidate = f"{symbol}{suffix}"
        data = _silent_download(candidate, start, end)
        if not data.empty and "Close" in data:
            return candidate
    return None

 #downloads data from YahooFinance without printing messages like  Failed download and without FutureWarning/stacktrace
def _silent_download(ticker: str, start: str, end: str) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore FutureWarning etc.
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            try:
                # progress=False avoids bar, threads=False reduce noise
                return yf.download(
                    ticker, start=start, end=end,
                    auto_adjust=True, progress=False, threads=False
                )
            except (KeyError, ValueError) as e:
                # common data-shape/column issues -> return empty silently
                return pd.DataFrame()
            except Exception as e:
                # let unexpected errors arise
                raise

#return a 1 Dimension Panda Series even if x is a Dataframe with a single column.
def _ensure_series(x):
    if isinstance(x, pd.DataFrame): # if is a DataFrame, take the first column as a Series
        return x.iloc[:, 0]
    if isinstance(x, pd.Series):
        return x

    # fallback: array/list → make it one dimension and make it a Series
    arr = np.asarray(x).reshape(-1)
    return pd.Series(arr)
# download Historical data (from start, to end) of a given list of tickers.
# Return two DataFrames (prices, volumes) with Date index and one column per ticker.
def download_prices(tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
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