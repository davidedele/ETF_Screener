# src/screen_utils.py
from __future__ import annotations

import typing
import warnings
import yfinance as yf
from datetime import datetime


# Try yfinance new API first (get_info), fallback to legacy (info).
def get_info_safely(tkr: yf.Ticker) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            info = tkr.get_info()
            if isinstance(info, dict) and info:
                return info
        except Exception:
            pass
        try:
            info = tkr.info
            if isinstance(info, dict) and info:
                return info
        except Exception:
            pass
    return {}


def get_fast_safely(tkr: yf.Ticker) -> typing.Mapping[str, typing.Any]:
    try:
        return tkr.fast_info or {}
    except Exception:
        return {}


def safe_pick(d: dict | typing.Mapping[str, typing.Any], *keys: str, default: typing.Any = None):
    for k in keys:
        try:
            v = d[k]  # type: ignore[index]
        except Exception:
            continue
        if v is not None:
            return v
    return default

#Convert epoch into YYYY-MM-DD string if possible.
def _safe_date(value: typing.Any) -> str | None:
    try:
        if value is None:
            return None
        v = int(value)
        # Heuristic: if it's in milliseconds, scale down
        if v > 10**12:  # ~ year 33658
            v //= 1000
        from datetime import datetime, timezone
        return datetime.fromtimestamp(v, tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return None


# Return a single-row dict of stable 'fund facts' only (no time-dependent fields).
def snapshot_row(ticker: str) -> dict[str, typing.Any]:
    yt = yf.Ticker(ticker)
    info = get_info_safely(yt)
    fast = get_fast_safely(yt)  # kept in case you want currency fallback, etc.

    row: dict[str, typing.Any] = {
        "ticker": ticker,
        # Names
        "shortName": safe_pick(info, "shortName", "longName", "name"),
        "longName": safe_pick(info, "longName", "shortName", "name"),

        # Class/category
        "quoteType": safe_pick(info, "quoteType"),
        "category": safe_pick(info, "category", "fundCategory"),
        "fundFamily": safe_pick(info, "fundFamily"),
        "exchange": safe_pick(info, "fullExchangeName", "exchange"),
        "currency": safe_pick(info, "currency", default=safe_pick(fast, "currency")),

        # Costs
        "expenseRatio": safe_pick(info, "expenseRatio", "annualReportExpenseRatio"),

        # Size / basics (generally stable, but may refresh periodically on Yahoo)
        "totalAssets": safe_pick(info, "totalAssets", "totalAssetsRaw"),
        "sharesOutstanding": safe_pick(info, "sharesOutstanding"),

        # Identifiers / inception / domicile
        "isin": safe_pick(info, "isin", "ISIN"),
        "symbol": safe_pick(info, "symbol", default=ticker),
        "firstTradeDate": _safe_date(
            safe_pick(info, "firstTradeDateEpochUtc", "firstTradeDateMilliseconds")
        ),
        "inceptionDate": _safe_date(safe_pick(info, "fundInceptionDate")),
        "domicile": safe_pick(info, "fundDomicile"),
    }

    return row