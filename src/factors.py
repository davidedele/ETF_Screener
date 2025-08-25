# src/factors.py
from __future__ import annotations

from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
from scipy.stats import skew as _skew, kurtosis as _kurtosis

TRADING_DAYS = 252

# ---------------------------
# Helpers
# ---------------------------

#Return a 1-D Series (first column if DataFrame), or None.
def _to_series(x: pd.Series | pd.DataFrame | None) -> Optional[pd.Series]:
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            return None
        return x.iloc[:, 0]
    return x

#Align all inputs on the common date index and sort by date.
def _align_on_common_index(
    prices: pd.DataFrame,
    volumes: Optional[pd.DataFrame] = None,
    benchmark: Optional[pd.Series | pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.Series]]:

    idx = prices.index
    if volumes is not None:
        idx = idx.intersection(volumes.index)
    if benchmark is not None:
        b = _to_series(benchmark)
        if b is not None:
            idx = idx.intersection(b.index)

    prices = prices.loc[idx].sort_index()
    if volumes is not None:
        volumes = volumes.loc[idx].sort_index()
    if benchmark is not None:
        b = _to_series(benchmark)
        benchmark = b.loc[idx].sort_index() if b is not None else None

    return prices, volumes, benchmark  # type: ignore[return-value]

#Daily log-returns for each column.
def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")

#CAGR from first/last price and actual elapsed years.
def cagr_from_prices(price_series: pd.Series) -> float:
    s = price_series.dropna()
    if s.shape[0] < 2:
        return np.nan
    start_price = float(s.iloc[0])
    end_price = float(s.iloc[-1])
    days = (s.index[-1] - s.index[0]).days
    years = days / 365.25 if days > 0 else np.nan
    if not years or years <= 0:
        return np.nan
    return (end_price / start_price) ** (1.0 / years) - 1.0

#Annualized volatility from daily returns.
def annual_vol_from_returns(rets: pd.Series) -> float:
    r = rets.dropna()
    if r.empty:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))

#Annualized downside deviation (daily returns below MAR).
def downside_vol_from_returns(rets: pd.Series, mar: float = 0.0) -> float:
    r = rets.dropna()
    if r.empty:
        return np.nan
    downside = r[r < mar]
    if downside.empty:
        return 0.0
    return float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS))

#Maximum drawdown as a negative number (e.g., -0.35).
def max_drawdown(price_series: pd.Series) -> float:
    s = price_series.dropna()
    if s.empty:
        return np.nan
    running_max = s.cummax()
    dd = s / running_max - 1.0
    return float(dd.min())

#Longest stretch (in days) spent under the previous peak."""
def max_drawdown_duration_days(price_series: pd.Series) -> int:
    s = price_series.dropna()
    if s.empty:
        return 0
    cummax = s.cummax()
    draw = s < cummax
    longest = cur = 0
    prev = False
    for v in draw:
        if v:
            cur = cur + 1 if prev else 1
            longest = max(longest, cur)
        else:
            cur = 0
        prev = v
    return int(longest)

#Days from the most recent peak to recovery of that same peak. If not recovered by series end, returns NaN.
def recovery_days_after_last_drawdown(price_series: pd.Series) -> float:
    s = price_series.dropna()
    if s.empty:
        return np.nan
    cummax = s.cummax()
    # dates where a new peak was set
    peaks = cummax[cummax.diff().fillna(0) > 0]
    if peaks.empty:
        return np.nan
    last_peak_date = peaks.index[-1]
    peak_val = float(cummax.loc[last_peak_date])

    after = s.loc[last_peak_date:]
    recovered_idx = after[after >= peak_val].index
    if len(recovered_idx) == 0:
        return np.nan
    first_recovery_date = recovered_idx[0]
    return float((first_recovery_date - last_peak_date).days)

#Annualized Sharpe ratio using daily returns.
def sharpe_from_returns(rets: pd.Series, rf_daily: float = 0.0) -> float:
    r = rets.dropna()
    if r.empty:
        return np.nan
    ex = r - rf_daily
    denom = ex.std(ddof=1)
    if denom == 0 or np.isnan(denom):
        return np.nan
    return float(ex.mean() / denom * np.sqrt(TRADING_DAYS))

#Annualized Sortino ratio using downside deviation.
def sortino_from_returns(rets: pd.Series, rf_daily: float = 0.0) -> float:
    r = rets.dropna()
    if r.empty:
        return np.nan
    ex = r - rf_daily
    downside = ex[ex < 0]
    denom = downside.std(ddof=1)
    if denom == 0 or np.isnan(denom):
        return np.nan
    return float(ex.mean() / denom * np.sqrt(TRADING_DAYS))

#Basic benchmark absolutes (CAGR, vol, max DD) for the benchmark itself."""
def _single_price_metrics(series: pd.Series) -> Dict[str, float]:
    s = series.dropna()
    if s.shape[0] < 2:
        return {"bench_cagr": np.nan, "bench_vol_ann": np.nan, "bench_max_dd": np.nan}
    bench_cagr = cagr_from_prices(s)
    bench_rets = compute_log_returns(s.to_frame()).iloc[:, 0]
    bench_vol_ann = annual_vol_from_returns(bench_rets)
    bench_max_dd = max_drawdown(s)
    return {
        "bench_cagr": bench_cagr,
        "bench_vol_ann": bench_vol_ann,
        "bench_max_dd": bench_max_dd,
    }

#Beta, correlation, tracking error (annualized), information ratio. IR = annualized mean(excess) / TE, where excess = asset - benchmark (daily).
def _beta_corr_te_ir(asset_rets: pd.Series, bench_rets: pd.Series) -> Dict[str, float]:
    a = asset_rets.dropna()
    b = bench_rets.dropna()
    common = a.index.intersection(b.index)
    if len(common) < 2:
        return {"beta": np.nan, "corr": np.nan, "te": np.nan, "info_ratio": np.nan}

    a = a.loc[common]
    b = b.loc[common]

    var_b = float(np.var(b, ddof=1))
    if var_b == 0 or np.isnan(var_b):
        beta = np.nan
    else:
        cov_ab = float(np.cov(a, b, ddof=1)[0, 1])
        beta = cov_ab / var_b

    corr = float(np.corrcoef(a, b)[0, 1]) if a.std(ddof=1) > 0 and b.std(ddof=1) > 0 else np.nan

    diff = a - b
    te_daily = float(diff.std(ddof=1))
    te = te_daily * np.sqrt(TRADING_DAYS) if not np.isnan(te_daily) else np.nan

    mean_excess_daily = float(diff.mean())
    ann_excess = mean_excess_daily * TRADING_DAYS if not np.isnan(mean_excess_daily) else np.nan
    info_ratio = (ann_excess / te) if te and not np.isnan(ann_excess) and te != 0 else np.nan

    return {"beta": beta, "corr": corr, "te": te, "info_ratio": info_ratio}

# ---------------------------
# ---------------------------

# Compute set of metrics for each column (ticker) in `prices`.
def metrics_for_prices(
    prices: pd.DataFrame,
    volumes: Optional[pd.DataFrame] = None,
    benchmark: Optional[pd.Series | pd.DataFrame] = None,
    rf_daily: float = 0.0,
) -> pd.DataFrame:
    """
    If benchmark  provided,  output also includes:
      Absolute benchmark fields: bench_cagr, bench_vol_ann, bench_max_dd
      Relative fields: excess_cagr, rel_vol, dd_gap
      Tracking fields: beta, corr, te (tracking error), info_ratio
    """
    # Normalize inputs
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    if isinstance(volumes, pd.Series):
        volumes = volumes.to_frame()
    if prices is None or prices.empty:
        return pd.DataFrame()

    prices, volumes, benchmark = _align_on_common_index(prices, volumes, benchmark)

    # Daily returns
    rets = compute_log_returns(prices)
    bench_rets = None
    if benchmark is not None:
        bench_rets = compute_log_returns(benchmark.to_frame()).iloc[:, 0]

    # Absolute benchmark metrics
    bench_abs: Dict[str, float] = {}
    if benchmark is not None:
        bench_abs = _single_price_metrics(_to_series(benchmark))  # type: ignore[arg-type]

    out_rows: List[Dict[str, float | str | int]] = []

    for col in prices.columns:
        p = prices[col].dropna()
        if p.shape[0] < 2:
            # Not enough data
            out_rows.append({"ticker": col})
            continue

        r = rets[col].dropna()

        total_return = float(p.iloc[-1] / p.iloc[0] - 1.0)
        cagr = cagr_from_prices(p)
        vol_ann = annual_vol_from_returns(r)
        dvol = downside_vol_from_returns(r)
        mdd = max_drawdown(p)
        dd_dur = max_drawdown_duration_days(p)
        rec_days = recovery_days_after_last_drawdown(p)

        # Monthly best/worst
        monthly_last = p.resample("ME").last()
        monthly_rets = monthly_last.pct_change().dropna()
        best_month = float(monthly_rets.max()) if not monthly_rets.empty else np.nan
        worst_month = float(monthly_rets.min()) if not monthly_rets.empty else np.nan

        # Moments
        skew_val = float(_skew(r, bias=False)) if r.shape[0] > 2 else np.nan
        kurt_val = float(_kurtosis(r, fisher=True, bias=False)) if r.shape[0] > 3 else np.nan

        # Risk-adjusted
        sharpe = sharpe_from_returns(r, rf_daily=rf_daily)
        sortino = sortino_from_returns(r, rf_daily=rf_daily)

        # Liquidity proxy
        if volumes is not None:
            if col in volumes.columns:
                avg_daily_volume = float(volumes[col].mean())
            elif getattr(volumes, "name", None) == col:
                avg_daily_volume = float(volumes.mean())
            else:
                avg_daily_volume = np.nan
        else:
            avg_daily_volume = np.nan

        row: Dict[str, float | str | int] = {
            "ticker": col,
            # Performance
            "total_return": total_return,
            "cagr": cagr,
            "best_month": best_month,
            "worst_month": worst_month,
            # Risk
            "vol_ann": vol_ann,
            "max_dd": mdd,
            "dd_duration_days": dd_dur,
            "recovery_days": rec_days,
            "downside_vol": dvol,
            "skew": skew_val,
            "kurtosis": kurt_val,
            # Risk-adjusted
            "sharpe": sharpe,
            "sortino": sortino,
            # Liquidity
            "avg_daily_volume": avg_daily_volume,
        }

        # Benchmark-dependent fields
        if bench_rets is not None:
            bc = bench_abs.get("bench_cagr", np.nan)
            bv = bench_abs.get("bench_vol_ann", np.nan)
            bd = bench_abs.get("bench_max_dd", np.nan)

            row.update({
                # Absolute benchmark (for side-by-side view)
                "bench_cagr": bc,
                "bench_vol_ann": bv,
                "bench_max_dd": bd,
                # Relative vs benchmark
                "excess_cagr": (cagr - bc) if pd.notna(cagr) and pd.notna(bc) else np.nan,
                "rel_vol": (vol_ann / bv) if pd.notna(vol_ann) and pd.notna(bv) and bv != 0 else np.nan,
                "dd_gap": (mdd - bd) if pd.notna(mdd) and pd.notna(bd) else np.nan,
            })

            # Tracking metrics (beta, corr, te, info_ratio) — align asset & benchmark on common dates
            a_aligned, b_aligned = r.align(bench_rets, join="inner")
            tr = _beta_corr_te_ir(a_aligned, b_aligned)
            row.update(tr)

        out_rows.append(row)

    df = pd.DataFrame(out_rows)
    # Reasonable default ordering
    if "cagr" in df.columns:
        df = df.sort_values("cagr", ascending=False, na_position="last").reset_index(drop=True)
    return df

#Split the all metrics table into themed packs for easier saving/reading. Returns a dict {pack_name: dataframe}.
def split_metric_packs(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    packs: Dict[str, List[str]] = {
        "performance": [
            "ticker", "total_return", "cagr", "best_month", "worst_month",
            "bench_cagr", "excess_cagr"
        ],
        "risk": [
            "ticker", "vol_ann", "bench_vol_ann", "rel_vol",
            "max_dd", "bench_max_dd", "dd_gap",
            "dd_duration_days", "recovery_days", "downside_vol", "skew", "kurtosis"
        ],
        "risk_adjusted": ["ticker", "sharpe", "sortino"],
        "tracking": ["ticker", "beta", "corr", "te", "info_ratio"],
        "liquidity": ["ticker", "avg_daily_volume"],
        "all": list(df.columns),  # full dump
    }

    out: Dict[str, pd.DataFrame] = {}
    for name, cols in packs.items():
        present = [c for c in cols if c in df.columns]
        if not present:
            continue
        sub = df[present].copy()
        out[name] = sub
    return out