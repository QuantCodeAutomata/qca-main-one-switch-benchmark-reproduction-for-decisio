"""Tests for src/features.py – protocol-aware feature engineering."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import FUTURE_DAYS
from src.features import compute_base_features, compute_label, normalize_features


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_ohlcv():
    """300 business days × 10 synthetic stocks.

    Prices follow a random walk so all rolling features can be computed and
    the volatility / range columns are non-trivial.
    """
    rng = np.random.default_rng(42)
    n_days = 300
    n_stocks = 10
    dates = pd.date_range("2015-01-01", periods=n_days, freq="B")
    tickers = [f"S{i:02d}" for i in range(n_stocks)]

    # Geometric random walk
    log_ret = rng.normal(0, 0.01, size=(n_days, n_stocks))
    log_prices = np.cumsum(log_ret, axis=0)
    mid_prices = 100.0 * np.exp(log_prices)  # (n_days, n_stocks)

    noise_o = 1.0 + rng.normal(0, 0.002, (n_days, n_stocks))
    noise_c = 1.0 + rng.normal(0, 0.002, (n_days, n_stocks))
    noise_h = 1.0 + np.abs(rng.normal(0, 0.004, (n_days, n_stocks)))
    noise_l = 1.0 - np.abs(rng.normal(0, 0.004, (n_days, n_stocks)))

    opens  = mid_prices * noise_o
    closes = mid_prices * noise_c
    highs  = np.maximum.reduce([opens, closes, mid_prices * noise_h])
    lows   = np.minimum.reduce([opens, closes, mid_prices * noise_l])
    vols   = rng.integers(1_000, 100_000, (n_days, n_stocks)).astype(float)

    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    return pd.DataFrame(
        {
            "open":   opens.flatten(),
            "high":   highs.flatten(),
            "low":    lows.flatten(),
            "close":  closes.flatten(),
            "volume": vols.flatten(),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_clean_features_no_future_leakage(synthetic_ohlcv):
    """CLEAN features at date t must be identical with or without future rows."""
    dates = synthetic_ohlcv.index.get_level_values("date").unique().sort_values()
    cutoff = dates[len(dates) // 2]

    feat_full = compute_base_features(synthetic_ohlcv, "CLEAN", 0)
    ohlcv_trunc = synthetic_ohlcv[
        synthetic_ohlcv.index.get_level_values("date") <= cutoff
    ]
    feat_trunc = compute_base_features(ohlcv_trunc, "CLEAN", 0)

    # Features at the cutoff date should be the same in both cases
    at_full  = feat_full.xs(cutoff, level="date")
    at_trunc = feat_trunc.xs(cutoff, level="date")
    common   = at_full.index.intersection(at_trunc.index)
    assert len(common) > 0, "No common tickers at cutoff date"

    pd.testing.assert_frame_equal(
        at_full.loc[common].fillna(0.0),
        at_trunc.loc[common].fillna(0.0),
        atol=1e-10,
        check_names=False,
    )


def test_temp_center_features_use_future(synthetic_ohlcv):
    """TEMP_CENTER features at t differ from CLEAN features due to the future shift."""
    feat_clean = compute_base_features(synthetic_ohlcv, "CLEAN", 0)
    feat_tc    = compute_base_features(synthetic_ohlcv, "TEMP_CENTER", FUTURE_DAYS)

    dates    = synthetic_ohlcv.index.get_level_values("date").unique().sort_values()
    mid_date = dates[len(dates) // 2]

    at_clean = feat_clean.xs(mid_date, level="date").dropna()
    at_tc    = feat_tc.xs(mid_date, level="date").dropna()
    common   = at_clean.index.intersection(at_tc.index)
    assert len(common) > 0

    # Values must differ for at least one column
    differs = ~np.isclose(
        at_clean.loc[common].values,
        at_tc.loc[common].values,
        equal_nan=True,
    )
    assert differs.any(), "TEMP_CENTER features should differ from CLEAN at the same date"


def test_clean_label_formula(synthetic_ohlcv):
    """CLEAN label = log(O_{t+h+1} / O_{t+1})."""
    horizon = 5
    labels = compute_label(synthetic_ohlcv, horizon, "CLEAN")

    dates  = synthetic_ohlcv.index.get_level_values("date").unique().sort_values()
    ticker = synthetic_ohlcv.index.get_level_values("ticker").unique()[0]

    t_idx = 50
    if t_idx + horizon + 1 >= len(dates):
        pytest.skip("Not enough dates for formula check")
    t          = dates[t_idx]
    t_plus_1   = dates[t_idx + 1]
    t_plus_h1  = dates[t_idx + horizon + 1]

    if pd.isna(labels.loc[(t, ticker)]):
        pytest.skip("Label is NaN at test index")

    o_t1  = synthetic_ohlcv.loc[(t_plus_1, ticker), "open"]
    o_th1 = synthetic_ohlcv.loc[(t_plus_h1, ticker), "open"]
    expected = float(np.log(o_th1 / o_t1))

    np.testing.assert_almost_equal(float(labels.loc[(t, ticker)]), expected, decimal=10)


def test_exec_open_label_formula(synthetic_ohlcv):
    """EXEC_OPEN label = log(O_{t+h} / O_t)."""
    horizon = 5
    labels = compute_label(synthetic_ohlcv, horizon, "EXEC_OPEN")

    dates  = synthetic_ohlcv.index.get_level_values("date").unique().sort_values()
    ticker = synthetic_ohlcv.index.get_level_values("ticker").unique()[0]

    t_idx = 50
    if t_idx + horizon >= len(dates):
        pytest.skip("Not enough dates")
    t        = dates[t_idx]
    t_plus_h = dates[t_idx + horizon]

    if pd.isna(labels.loc[(t, ticker)]):
        pytest.skip("Label is NaN at test index")

    o_t  = synthetic_ohlcv.loc[(t, ticker), "open"]
    o_th = synthetic_ohlcv.loc[(t_plus_h, ticker), "open"]
    expected = float(np.log(o_th / o_t))

    np.testing.assert_almost_equal(float(labels.loc[(t, ticker)]), expected, decimal=10)


def test_exec_close_label_formula(synthetic_ohlcv):
    """EXEC_CLOSE label = log(C_{t+h} / C_t)."""
    horizon = 5
    labels = compute_label(synthetic_ohlcv, horizon, "EXEC_CLOSE")

    dates  = synthetic_ohlcv.index.get_level_values("date").unique().sort_values()
    ticker = synthetic_ohlcv.index.get_level_values("ticker").unique()[0]

    t_idx = 50
    if t_idx + horizon >= len(dates):
        pytest.skip("Not enough dates")
    t        = dates[t_idx]
    t_plus_h = dates[t_idx + horizon]

    if pd.isna(labels.loc[(t, ticker)]):
        pytest.skip("Label is NaN at test index")

    c_t  = synthetic_ohlcv.loc[(t, ticker), "close"]
    c_th = synthetic_ohlcv.loc[(t_plus_h, ticker), "close"]
    expected = float(np.log(c_th / c_t))

    np.testing.assert_almost_equal(float(labels.loc[(t, ticker)]), expected, decimal=10)


def test_normalize_clean_uses_train_only(synthetic_ohlcv):
    """CLEAN normalisation: per-column mean ≈ 0 and std ≈ 1 on the training set."""
    features   = compute_base_features(synthetic_ohlcv, "CLEAN", 0)
    dates      = features.index.get_level_values("date").unique().sort_values()
    n_train    = int(len(dates) * 0.7)
    train_dates = dates[:n_train]

    feat_norm = normalize_features(features, train_dates, "CLEAN")

    train_mask = feat_norm.index.get_level_values("date").isin(train_dates)
    feat_train = feat_norm[train_mask]

    for col in feat_train.columns:
        vals = feat_train[col].dropna()
        if len(vals) < 20:
            continue
        assert abs(vals.mean()) < 0.2, (
            f"Column '{col}' mean={vals.mean():.4f} not near 0 on train set"
        )
        assert 0.7 < vals.std() < 1.5, (
            f"Column '{col}' std={vals.std():.4f} not near 1 on train set"
        )


def test_normalize_global_uses_full_panel(synthetic_ohlcv):
    """NORM_GLOBAL and CLEAN normalisation differ outside the train period."""
    features    = compute_base_features(synthetic_ohlcv, "CLEAN", 0)
    dates       = features.index.get_level_values("date").unique().sort_values()
    n_half      = len(dates) // 2
    train_dates = dates[:n_half]

    norm_global = normalize_features(features, train_dates, "NORM_GLOBAL")
    norm_clean  = normalize_features(features, train_dates, "CLEAN")

    # On test dates, the two should differ (global uses future stats)
    test_dates = dates[n_half + 1 :]
    mask_g = norm_global.index.get_level_values("date").isin(test_dates)
    mask_c = norm_clean.index.get_level_values("date").isin(test_dates)
    g_vals = norm_global[mask_g].dropna().values
    c_vals = norm_clean[mask_c].dropna().values

    if g_vals.size == 0 or c_vals.size == 0:
        pytest.skip("No out-of-sample data to compare")

    assert not np.allclose(g_vals, c_vals, atol=1e-6), (
        "NORM_GLOBAL and CLEAN should produce different normalisation for out-of-sample dates"
    )


def test_feature_values_finite(synthetic_ohlcv):
    """After dropping NaN rows, all remaining feature values must be finite."""
    features = compute_base_features(synthetic_ohlcv, "CLEAN", 0)
    valid    = features.dropna()

    assert len(valid) > 0, "No valid feature rows after dropna"
    assert np.all(np.isfinite(valid.values)), (
        "Feature matrix contains non-finite (Inf/NaN) values in non-NaN rows"
    )
