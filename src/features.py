"""
features.py – Protocol-aware feature engineering for the leakage benchmark.

All features are computed per-asset using only data that would be available at
the close of day *t* under the CLEAN protocol.  The TEMP_CENTER protocol
introduces look-ahead by shifting the rolling window forward by *future_days*.

Feature list (10 features)
--------------------------
ret_1          log(C_t / C_{t-1})
ret_5          log(C_t / C_{t-5})
ret_10         log(C_t / C_{t-10})
ret_20         log(C_t / C_{t-20})
vol_5          rolling std of daily log-ret, 5-day window
vol_20         rolling std of daily log-ret, 20-day window
vol_ratio_20   vol_5 / vol_20
hl_range_5_mean  mean of (H-L)/C over last 5 days
turnover_5     mean of vol / rolling_max_vol over 5 days
mom_60         log(C_t / C_{t-60})
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import FUTURE_DAYS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_wide(ohlcv: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return *col* unstacked to wide format (dates × tickers)."""
    return ohlcv[col].unstack(level='ticker')


def _stack_feature(wide: pd.DataFrame, name: str) -> pd.Series:
    """Stack a wide feature DataFrame to a long Series with MultiIndex (date, ticker)."""
    s = wide.stack(future_stack=True)
    s.name = name
    s.index.names = ['date', 'ticker']
    return s


def _compute_features_from_wide(
    close_w: pd.DataFrame,
    high_w: pd.DataFrame,
    low_w: pd.DataFrame,
    vol_w: pd.DataFrame,
    protocol: str,
    future_days: int,
) -> pd.DataFrame:
    """Core feature computation on wide (dates × tickers) DataFrames.

    For TEMP_CENTER each source series is shifted by *-future_days* before any
    rolling computation, injecting look-ahead contamination.
    """
    shift = -future_days if protocol == 'TEMP_CENTER' else 0

    c = close_w.shift(shift)
    h = high_w.shift(shift)
    lo = low_w.shift(shift)
    v = vol_w.shift(shift)

    daily_ret = np.log(c / c.shift(1))

    vol5 = daily_ret.rolling(5, min_periods=3).std()
    vol20 = daily_ret.rolling(20, min_periods=10).std()

    max_vol5 = v.rolling(5, min_periods=1).max().replace(0.0, np.nan)

    feature_map: dict[str, pd.DataFrame] = {
        'ret_1':          daily_ret,
        'ret_5':          np.log(c / c.shift(5)),
        'ret_10':         np.log(c / c.shift(10)),
        'ret_20':         np.log(c / c.shift(20)),
        'vol_5':          vol5,
        'vol_20':         vol20,
        'vol_ratio_20':   vol5 / vol20.replace(0.0, np.nan),
        'hl_range_5_mean': ((h - lo) / c.replace(0.0, np.nan)).rolling(5, min_periods=3).mean(),
        'turnover_5':     (v / max_vol5).rolling(5, min_periods=3).mean(),
        'mom_60':         np.log(c / c.shift(60)),
    }

    series_list = [_stack_feature(wide, name) for name, wide in feature_map.items()]
    result = pd.concat(series_list, axis=1)
    result.index.names = ['date', 'ticker']
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_base_features(
    ohlcv: pd.DataFrame,
    protocol: str = 'CLEAN',
    future_days: int = FUTURE_DAYS,
) -> pd.DataFrame:
    """Compute all base features for every (date, ticker) pair.

    Parameters
    ----------
    ohlcv:
        MultiIndex (date, ticker) DataFrame with columns
        [open, high, low, close, volume].
    protocol:
        One of the six benchmark protocols.  Only ``'TEMP_CENTER'`` alters
        feature computation (rolling windows shifted forward by *future_days*).
    future_days:
        Look-ahead days for TEMP_CENTER contamination (default 3).

    Returns
    -------
    pd.DataFrame
        Same MultiIndex as *ohlcv*, columns:
        ret_1, ret_5, ret_10, ret_20, vol_5, vol_20, vol_ratio_20,
        hl_range_5_mean, turnover_5, mom_60.
    """
    close_w = _to_wide(ohlcv, 'close')
    high_w = _to_wide(ohlcv, 'high')
    low_w = _to_wide(ohlcv, 'low')
    vol_w = _to_wide(ohlcv, 'volume').astype(float)

    features = _compute_features_from_wide(
        close_w, high_w, low_w, vol_w, protocol, future_days
    )

    # Preserve original sort order
    return features.sort_index()


def compute_label(
    ohlcv: pd.DataFrame,
    horizon: int,
    protocol: str,
) -> pd.Series:
    """Compute the forward-return label for each (date, ticker).

    Label definitions by protocol
    ------------------------------
    CLEAN / TEMP_CENTER / NORM_GLOBAL / STRUCT_GRAPH:
        log(O_{t+h+1} / O_{t+1})   — enter at next open, exit *h* days later
    EXEC_CLOSE:
        log(C_{t+h} / C_t)          — enter at today's close, exit in *h* days
    EXEC_OPEN:
        log(O_{t+h} / O_t)          — enter at today's open (same-day leakage)

    Parameters
    ----------
    ohlcv:
        MultiIndex (date, ticker) DataFrame with [open, high, low, close, volume].
    horizon:
        Forecast horizon in trading days (5 or 20).
    protocol:
        Benchmark protocol string.

    Returns
    -------
    pd.Series
        MultiIndex (date, ticker) named ``'label'``.
        Future dates that exceed the data range will be NaN.
    """
    open_w = _to_wide(ohlcv, 'open')
    close_w = _to_wide(ohlcv, 'close')

    if protocol == 'EXEC_CLOSE':
        # Enter at today's close, exit at close in h days
        label_w = np.log(close_w.shift(-horizon) / close_w)
    elif protocol == 'EXEC_OPEN':
        # Enter at today's open, exit at open in h days
        label_w = np.log(open_w.shift(-horizon) / open_w)
    else:
        # CLEAN / TEMP_CENTER / NORM_GLOBAL / STRUCT_GRAPH
        # Enter at next open (t+1), exit at open h days later (t+h+1)
        label_w = np.log(open_w.shift(-(horizon + 1)) / open_w.shift(-1))

    label = _stack_feature(label_w, 'label')
    return label.sort_index()


def normalize_features(
    features: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    protocol: str,
    full_dates: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """Cross-sectionally normalise features using protocol-appropriate statistics.

    Normalisation strategy
    ----------------------
    NORM_GLOBAL:
        Mean and std estimated from the *full* panel (all dates in *full_dates*
        or, if not provided, all dates present in *features*).  This leaks
        future distributional information into the training and test periods.
    All other protocols:
        Mean and std estimated from the *training* dates only (*train_dates*).

    NaN values in *features* are ignored when computing statistics and are
    propagated unchanged in the output.

    Parameters
    ----------
    features:
        MultiIndex (date, ticker) feature DataFrame (output of
        :func:`compute_base_features`).
    train_dates:
        DatetimeIndex of dates belonging to the training set.
    protocol:
        Benchmark protocol string.
    full_dates:
        Optional DatetimeIndex covering the complete panel.  Used only for
        ``NORM_GLOBAL``; when ``None`` all unique dates in *features* are used.

    Returns
    -------
    pd.DataFrame
        Normalised features with the same shape and MultiIndex as *features*.
    """
    all_dates = features.index.get_level_values('date')

    if protocol == 'NORM_GLOBAL':
        if full_dates is not None:
            fit_mask = all_dates.isin(full_dates)
        else:
            fit_mask = np.ones(len(features), dtype=bool)
    else:
        fit_mask = all_dates.isin(train_dates)

    fit_data = features.loc[fit_mask]
    means = fit_data.mean(skipna=True)
    stds = fit_data.std(skipna=True)

    # Guard against division by near-zero stds (constant or near-constant columns).
    # 1e-8 is far below any legitimate financial-feature std yet safely above
    # floating-point noise on exactly constant synthetic series.
    stds = stds.where(stds > 1e-8, other=1.0)

    normalised = (features - means) / stds
    return normalised
