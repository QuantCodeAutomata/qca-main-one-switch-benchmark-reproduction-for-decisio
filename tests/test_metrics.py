"""Tests for src/metrics.py – SR, RankIC, AUC, LG, CI, and Wilcoxon."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import (
    bootstrap_ci,
    compute_auc,
    compute_leakage_gain,
    compute_rankic,
    compute_sharpe,
    summarize_yearly_lg,
    wilcoxon_one_sided,
)


# ---------------------------------------------------------------------------
# compute_sharpe
# ---------------------------------------------------------------------------

def test_sharpe_formula():
    """SR = sqrt(252) * mean(r) / std(r, ddof=1)."""
    returns  = pd.Series([0.01, 0.02, -0.005, 0.015, 0.008, -0.003, 0.012])
    expected = float(np.sqrt(252.0) * returns.mean() / returns.std(ddof=1))
    actual   = compute_sharpe(returns)
    assert abs(actual - expected) < 1e-10, (
        f"SR mismatch: expected {expected:.6f}, got {actual:.6f}"
    )


def test_sharpe_zero_std_returns_nan():
    """Constant returns series (std=0) should yield NaN, not raise."""
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])
    result  = compute_sharpe(returns)
    assert np.isnan(result), f"Expected NaN for zero-std returns, got {result}"


def test_sharpe_empty_returns_nan():
    """Empty returns series should yield NaN."""
    assert np.isnan(compute_sharpe(pd.Series([], dtype=float)))


# ---------------------------------------------------------------------------
# compute_rankic
# ---------------------------------------------------------------------------

def test_rankic_perfect_score():
    """When scores equal labels, Spearman IC = 1.0 on every date."""
    rng     = np.random.default_rng(0)
    dates   = pd.date_range("2020-01-01", periods=5, freq="B")
    tickers = [f"T{i}" for i in range(10)]
    data    = rng.standard_normal((5, 10))

    scores = pd.DataFrame(data, index=dates, columns=tickers)
    labels = scores.copy()

    ic = compute_rankic(scores, labels)
    assert abs(ic - 1.0) < 1e-6, f"Expected RankIC = 1.0, got {ic:.6f}"


def test_rankic_skips_nans():
    """compute_rankic must not crash on rows with NaN values."""
    dates   = pd.date_range("2020-01-01", periods=3, freq="B")
    tickers = ["A", "B", "C", "D"]

    scores = pd.DataFrame(
        [[1.0, 2.0,  np.nan, 4.0],
         [np.nan, 1.0, 3.0, 2.0],
         [3.0, np.nan, 1.0, 2.0]],
        index=dates,
        columns=tickers,
    )
    labels = scores.copy()

    # Must not raise; result is finite or NaN (not +/-Inf)
    ic = compute_rankic(scores, labels)
    assert not np.isinf(ic), f"RankIC should not be infinite, got {ic}"


def test_rankic_anticorrelated_near_minus_one():
    """Perfectly anti-correlated scores should yield RankIC ≈ –1."""
    dates   = pd.date_range("2020-01-01", periods=4, freq="B")
    tickers = [f"T{i}" for i in range(8)]

    base    = np.tile(np.arange(8, dtype=float), (4, 1))
    scores  = pd.DataFrame(base,  index=dates, columns=tickers)
    labels  = pd.DataFrame(-base, index=dates, columns=tickers)

    ic = compute_rankic(scores, labels)
    assert abs(ic + 1.0) < 1e-6, f"Expected RankIC = -1.0, got {ic:.6f}"


# ---------------------------------------------------------------------------
# compute_auc
# ---------------------------------------------------------------------------

def test_auc_perfect_classifier():
    """Scores that perfectly separate positives from negatives → AUC = 1.0."""
    dates   = pd.date_range("2020-01-01", periods=3, freq="B")
    tickers = ["A", "B", "C", "D"]

    scores = pd.DataFrame(
        {"A": [10.0] * 3, "B": [8.0] * 3, "C": [-8.0] * 3, "D": [-10.0] * 3},
        index=dates,
    )
    labels = pd.DataFrame(
        {"A": [0.01] * 3, "B": [0.01] * 3, "C": [-0.01] * 3, "D": [-0.01] * 3},
        index=dates,
    )

    auc = compute_auc(scores, labels)
    assert abs(auc - 1.0) < 1e-6, f"Expected AUC = 1.0, got {auc:.6f}"


def test_auc_random_classifier_near_half():
    """A random classifier should yield AUC close to 0.5 on average."""
    rng     = np.random.default_rng(123)
    dates   = pd.date_range("2020-01-01", periods=100, freq="B")
    tickers = [f"T{i}" for i in range(20)]

    scores  = pd.DataFrame(rng.standard_normal((100, 20)), index=dates, columns=tickers)
    labels  = pd.DataFrame(rng.standard_normal((100, 20)), index=dates, columns=tickers)

    auc = compute_auc(scores, labels)
    if not np.isnan(auc):
        assert 0.3 < auc < 0.7, f"Random AUC should be near 0.5, got {auc:.4f}"


# ---------------------------------------------------------------------------
# compute_leakage_gain
# ---------------------------------------------------------------------------

def test_leakage_gain_formula():
    """LG = variant_metric - clean_metric for every key."""
    clean   = {"SR@0bps": 0.50, "SR@5bps": 0.40, "rankic": 0.10}
    variant = {"SR@0bps": 0.80, "SR@5bps": 0.70, "rankic": 0.15}

    lg = compute_leakage_gain(variant, clean)

    assert abs(lg["SR@0bps"] - 0.30) < 1e-10
    assert abs(lg["SR@5bps"] - 0.30) < 1e-10
    assert abs(lg["rankic"]  - 0.05) < 1e-10


def test_leakage_gain_zero_when_equal():
    """LG = 0 when the variant and CLEAN metrics are identical."""
    metrics = {"SR@5bps": 1.2, "rankic": 0.08}
    lg = compute_leakage_gain(metrics, metrics)
    for k, v in lg.items():
        assert v == 0.0, f"Expected LG=0 for key '{k}', got {v}"


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

def test_bootstrap_ci_bounds():
    """Bootstrap CI lower bound < mean < upper bound."""
    values         = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    lower, upper   = bootstrap_ci(values, n_bootstrap=5_000, ci=0.95, seed=0)
    mean_val       = float(np.mean(values))

    assert lower < upper,    f"Expected lower ({lower:.4f}) < upper ({upper:.4f})"
    assert lower < mean_val, f"Lower bound {lower:.4f} should be below the mean {mean_val:.4f}"
    assert upper > mean_val, f"Upper bound {upper:.4f} should be above the mean {mean_val:.4f}"


def test_bootstrap_ci_singleton():
    """A single value: CI bounds should both equal that value."""
    values       = np.array([0.42])
    lower, upper = bootstrap_ci(values, n_bootstrap=200, seed=0)
    assert abs(lower - 0.42) < 1e-10
    assert abs(upper - 0.42) < 1e-10


# ---------------------------------------------------------------------------
# wilcoxon_one_sided
# ---------------------------------------------------------------------------

def test_wilcoxon_positive_values():
    """All-positive LG values should yield a significant (small) p-value."""
    values        = np.array([0.5, 0.8, 0.3, 0.9, 0.1, 0.7, 0.4, 0.6, 0.2])
    stat, pval    = wilcoxon_one_sided(values)

    assert np.isfinite(stat) and np.isfinite(pval)
    assert pval < 0.05, (
        f"Expected p-value < 0.05 for all-positive values, got {pval:.4f}"
    )


def test_wilcoxon_all_zeros_returns_nan():
    """All-zero input: Wilcoxon test is undefined → (nan, nan)."""
    stat, pval = wilcoxon_one_sided(np.zeros(5))
    assert np.isnan(stat) and np.isnan(pval)


# ---------------------------------------------------------------------------
# summarize_yearly_lg
# ---------------------------------------------------------------------------

def test_summarize_yearly_lg_positive_count():
    """positive_years equals the count of values > 0 in the input array."""
    yearly_lg = np.array([0.1, -0.2, 0.3, 0.4, -0.1, 0.2, 0.5, 0.3, -0.1])
    summary   = summarize_yearly_lg(yearly_lg, n_bootstrap=500)

    expected_positive = int(np.sum(yearly_lg > 0))  # 6
    assert summary["positive_years"] == expected_positive, (
        f"Expected {expected_positive} positive years, got {summary['positive_years']}"
    )
    assert summary["ci_lower"] < summary["ci_upper"], "CI lower should be below upper"
    assert abs(summary["mean_lg"] - float(np.mean(yearly_lg))) < 1e-10
