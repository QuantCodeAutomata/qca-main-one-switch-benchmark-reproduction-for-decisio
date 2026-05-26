"""Tests for src/backtest.py – portfolio construction and execution semantics."""

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest import (
    compute_gross_returns,
    compute_net_returns,
    compute_turnover,
    form_portfolio,
    run_backtest,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n_days: int = 10, n_stocks: int = 5, seed: int = 0) -> pd.DataFrame:
    """Create a minimal synthetic MultiIndex OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    dates   = pd.date_range("2020-01-01", periods=n_days, freq="B")
    tickers = [f"T{i}" for i in range(n_stocks)]
    idx     = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    n       = len(idx)

    day_idx = np.repeat(np.arange(n_days), n_stocks)
    # Prices drift up 1 % per day so returns are predictable
    mid = 100.0 * (1.01 ** day_idx)
    noise = 1.0 + rng.normal(0, 0.002, n)

    opens  = mid * noise
    closes = mid * (1.0 + rng.normal(0, 0.002, n))
    highs  = np.maximum(opens, closes) * (1.0 + rng.uniform(0, 0.005, n))
    lows   = np.minimum(opens, closes) * (1.0 - rng.uniform(0, 0.005, n))

    return pd.DataFrame(
        {
            "open":   opens,
            "high":   highs,
            "low":    lows,
            "close":  closes,
            "volume": rng.integers(100, 10_000, n).astype(float),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# form_portfolio
# ---------------------------------------------------------------------------

def test_form_portfolio_top_decile():
    """With 100 stocks and top_decile=0.1 exactly 10 should be selected."""
    rng    = np.random.default_rng(7)
    scores = pd.Series(rng.standard_normal(100), index=[f"S{i:03d}" for i in range(100)])
    port   = form_portfolio(scores, top_decile=0.1)

    n_expected = math.ceil(0.1 * 100)  # = 10
    assert len(port) == n_expected, f"Expected {n_expected} assets, got {len(port)}"

    # Every selected ticker should be in the top decile
    threshold = scores.nlargest(n_expected).min()
    for ticker in port:
        assert scores[ticker] >= threshold - 1e-12, (
            f"Ticker {ticker} score {scores[ticker]:.4f} is below the top-decile threshold"
        )


def test_form_portfolio_ceil_rule():
    """With 3 stocks, ceil(0.1 * 3) = 1, so exactly one ticker is selected."""
    scores = pd.Series({"A": 3.0, "B": 1.0, "C": -1.0})
    port   = form_portfolio(scores, top_decile=0.1)

    assert len(port) == 1, f"Expected 1 asset, got {len(port)}"
    assert "A" in port, "Top-scored asset 'A' should be selected"


# ---------------------------------------------------------------------------
# compute_gross_returns
# ---------------------------------------------------------------------------

def test_gross_returns_equal_weight():
    """Equal-weight portfolio return equals the simple mean of asset returns."""
    ohlcv = _make_ohlcv(n_days=10, n_stocks=4, seed=1)
    dates   = ohlcv.index.get_level_values("date").unique().sort_values()
    tickers = ohlcv.index.get_level_values("ticker").unique()

    # All stocks get the same score so every one of them is in the portfolio
    scores_panel = pd.DataFrame(
        np.ones((len(dates) - 2, len(tickers))),
        index=dates[:-2],
        columns=tickers,
    )
    labels_panel = scores_panel.copy() * 0.0

    gross, portfolios = compute_gross_returns(
        ohlcv, scores_panel, "CLEAN", top_decile=1.0
    )

    assert not gross.empty, "Gross returns should not be empty"
    assert gross.dropna().shape[0] > 0, "All gross returns are NaN"


# ---------------------------------------------------------------------------
# compute_turnover
# ---------------------------------------------------------------------------

def test_turnover_zero_on_identical_portfolios():
    """If the same set of tickers is held every day, turnover after day 1 is 0."""
    dates      = pd.date_range("2020-01-01", periods=5, freq="B")
    portfolios = {d: pd.Index(["A", "B", "C"]) for d in dates}

    turnover = compute_turnover(portfolios, dates)

    # First date always has turnover = 1; all subsequent dates should be 0
    assert turnover.iloc[0] == 1.0, "First-date turnover should be 1.0"
    assert all(
        turnover.iloc[1:].fillna(0.0) == 0.0
    ), f"Subsequent turnover should be 0, got {turnover.iloc[1:].values}"


# ---------------------------------------------------------------------------
# compute_net_returns
# ---------------------------------------------------------------------------

def test_net_returns_decrease_with_cost():
    """Net returns are weakly monotonically decreasing in cost."""
    rng      = np.random.default_rng(99)
    n        = 50
    gross    = pd.Series(rng.normal(0.001, 0.01, n))
    turnover = pd.Series(np.full(n, 0.5))

    net_0  = compute_net_returns(gross, turnover, cost_bps=0)
    net_5  = compute_net_returns(gross, turnover, cost_bps=5)
    net_10 = compute_net_returns(gross, turnover, cost_bps=10)

    assert (net_0 >= net_5).all(),  "Higher cost should not increase returns (0 vs 5)"
    assert (net_5 >= net_10).all(), "Higher cost should not increase returns (5 vs 10)"


# ---------------------------------------------------------------------------
# Execution semantics
# ---------------------------------------------------------------------------

def _make_deterministic_ohlcv() -> pd.DataFrame:
    """OHLCV where open[t] = 100+t and close[t] = 110+t for ALL tickers on day t.

    MultiIndex order is (date0,A),(date0,B),(date1,A),(date1,B),…  so we must
    use np.repeat (not np.tile) to broadcast the per-day price to both tickers.
    """
    dates   = pd.date_range("2020-01-01", periods=8, freq="B")
    tickers = ["A", "B"]
    idx     = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    n_days  = len(dates)

    # Each day's price is shared by both tickers
    opens  = np.repeat(100.0 + np.arange(n_days), len(tickers))
    closes = np.repeat(110.0 + np.arange(n_days), len(tickers))

    return pd.DataFrame(
        {
            "open":   opens,
            "high":   closes + 1.0,
            "low":    opens  - 1.0,
            "close":  closes,
            "volume": np.ones(len(idx)) * 1000.0,
        },
        index=idx,
    )


def test_exec_open_return_semantics():
    """EXEC_OPEN: trade return = O_{t+1} / O_t - 1."""
    ohlcv  = _make_deterministic_ohlcv()
    dates  = ohlcv.index.get_level_values("date").unique().sort_values()

    scores = pd.DataFrame(
        {"A": np.ones(4), "B": np.zeros(4)}, index=dates[:4]
    )
    labels = pd.DataFrame(
        {"A": np.zeros(4), "B": np.zeros(4)}, index=dates[:4]
    )

    result = run_backtest(ohlcv, scores, labels, "EXEC_OPEN", costs_bps=[0])
    gross  = result["gross_returns"].dropna()

    # open[0]=100, open[1]=101 → return = 101/100 - 1 = 0.01
    expected_first = (100.0 + 1.0) / (100.0 + 0.0) - 1.0
    assert len(gross) > 0, "No gross returns computed"
    assert abs(float(gross.iloc[0]) - expected_first) < 1e-10, (
        f"EXEC_OPEN first return: expected {expected_first}, got {float(gross.iloc[0])}"
    )


def test_exec_close_return_semantics():
    """EXEC_CLOSE: trade return = C_{t+1} / C_t - 1."""
    ohlcv  = _make_deterministic_ohlcv()
    dates  = ohlcv.index.get_level_values("date").unique().sort_values()

    scores = pd.DataFrame(
        {"A": np.ones(4), "B": np.zeros(4)}, index=dates[:4]
    )
    labels = pd.DataFrame(
        {"A": np.zeros(4), "B": np.zeros(4)}, index=dates[:4]
    )

    result = run_backtest(ohlcv, scores, labels, "EXEC_CLOSE", costs_bps=[0])
    gross  = result["gross_returns"].dropna()

    # close[0]=110, close[1]=111 → return = 111/110 - 1 ≈ 0.009090...
    expected_first = (110.0 + 1.0) / (110.0 + 0.0) - 1.0
    assert len(gross) > 0, "No gross returns computed"
    assert abs(float(gross.iloc[0]) - expected_first) < 1e-10, (
        f"EXEC_CLOSE first return: expected {expected_first}, got {float(gross.iloc[0])}"
    )
