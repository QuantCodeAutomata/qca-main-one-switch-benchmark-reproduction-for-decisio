"""Performance and leakage metrics for the decision-time leakage benchmark."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score


def compute_sharpe(returns: pd.Series, annualize: float = 252.0) -> float:
    """Compute annualised Sharpe ratio.

    SR = sqrt(annualize) * mean(r) / std(r)

    Parameters
    ----------
    returns:
        Daily return series.
    annualize:
        Number of periods per year used for annualisation.

    Returns
    -------
    float
        Annualised Sharpe ratio, or NaN if the standard deviation is zero
        or there are no finite observations.
    """
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    s = float(r.std(ddof=1))
    if s == 0.0:
        return np.nan
    return float(np.sqrt(annualize) * r.mean() / s)


def compute_max_drawdown(returns: pd.Series) -> float:
    """Compute maximum peak-to-trough drawdown from a returns series.

    Parameters
    ----------
    returns:
        Daily return series.

    Returns
    -------
    float
        Maximum drawdown (a non-positive number), or NaN for an empty series.
    """
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    cum = (1.0 + r).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    return float(drawdown.min())


def compute_rankic(
    scores_panel: pd.DataFrame, labels_panel: pd.DataFrame
) -> float:
    """Daily cross-sectional Spearman rank-IC averaged across dates.

    For each date the Spearman correlation is computed between the model
    scores and the forward-return labels over the common set of tickers.
    Only dates with at least two finite paired observations are included.

    Parameters
    ----------
    scores_panel:
        DataFrame with dates as index and tickers as columns.
    labels_panel:
        DataFrame with dates as index and tickers as columns.

    Returns
    -------
    float
        Mean Spearman IC across eligible dates, or NaN if none qualify.
    """
    common_dates = scores_panel.index.intersection(labels_panel.index)
    ics: List[float] = []

    for t in common_dates:
        s = scores_panel.loc[t].dropna()
        y = labels_panel.loc[t].dropna()
        tickers = s.index.intersection(y.index)
        if len(tickers) < 2:
            continue

        sv = s[tickers].values.astype(float)
        yv = y[tickers].values.astype(float)
        finite = np.isfinite(sv) & np.isfinite(yv)
        sv, yv = sv[finite], yv[finite]
        if len(sv) < 2:
            continue

        corr, _ = spearmanr(sv, yv)
        if np.isfinite(corr):
            ics.append(float(corr))

    return float(np.mean(ics)) if ics else np.nan


def compute_auc(
    scores_panel: pd.DataFrame, labels_panel: pd.DataFrame
) -> float:
    """Daily ROC-AUC with binary label 1{y > 0}, averaged across dates.

    Only dates that contain both positive and negative binary labels are
    included in the average.

    Parameters
    ----------
    scores_panel:
        DataFrame with dates as index and tickers as columns.
    labels_panel:
        DataFrame with dates as index and tickers as columns.

    Returns
    -------
    float
        Mean ROC-AUC across eligible dates, or NaN if none qualify.
    """
    common_dates = scores_panel.index.intersection(labels_panel.index)
    aucs: List[float] = []

    for t in common_dates:
        s = scores_panel.loc[t].dropna()
        y = labels_panel.loc[t].dropna()
        tickers = s.index.intersection(y.index)
        if len(tickers) < 2:
            continue

        sv = s[tickers].values.astype(float)
        yv = y[tickers].values.astype(float)
        finite = np.isfinite(sv) & np.isfinite(yv)
        sv, yv = sv[finite], yv[finite]
        if len(sv) < 2:
            continue

        binary_y = (yv > 0).astype(int)
        if len(np.unique(binary_y)) < 2:
            continue  # need both classes for a meaningful AUC

        aucs.append(float(roc_auc_score(binary_y, sv)))

    return float(np.mean(aucs)) if aucs else np.nan


def compute_all_metrics(
    backtest_result: Dict,
    scores_panel: pd.DataFrame,
    labels_panel: pd.DataFrame,
    costs_bps: List[float] = None,
) -> Dict:
    """Compute the full metric suite for a single backtest result.

    Metrics computed
    ----------------
    SR@{c}bps   : Annualised Sharpe ratio of net returns at each cost level.
    max_drawdown: Maximum drawdown of net returns at 5 bps.
    mean_turnover: Mean one-way daily turnover.
    rankic      : Mean daily Spearman IC.
    auc         : Mean daily ROC-AUC.

    Parameters
    ----------
    backtest_result:
        Dict returned by :func:`run_backtest`.
    scores_panel:
        Model predictions panel.
    labels_panel:
        Forward returns panel.
    costs_bps:
        Cost levels for which SR is computed. Defaults to [0, 5, 10].

    Returns
    -------
    dict
        Flat mapping of metric name to scalar value.
    """
    if costs_bps is None:
        costs_bps = [0, 5, 10]

    metrics: Dict = {}
    for c in costs_bps:
        metrics[f"SR@{c}bps"] = compute_sharpe(backtest_result["net_returns"][c])

    metrics["max_drawdown"] = compute_max_drawdown(backtest_result["net_returns"][5])
    metrics["mean_turnover"] = float(backtest_result["turnover"].mean())
    metrics["rankic"] = compute_rankic(scores_panel, labels_panel)
    metrics["auc"] = compute_auc(scores_panel, labels_panel)
    return metrics


def compute_leakage_gain(variant_metrics: Dict, clean_metrics: Dict) -> Dict:
    """Compute leakage gain for each metric.

    LG_q = q(variant) - q(CLEAN)

    Parameters
    ----------
    variant_metrics:
        Metrics dict for the leaky protocol variant.
    clean_metrics:
        Metrics dict for the CLEAN baseline.

    Returns
    -------
    dict
        Same keys as the input dicts with difference values.
    """
    return {k: variant_metrics[k] - clean_metrics[k] for k in variant_metrics}


def compute_yearly_sharpe(
    net_returns_by_year: Dict[int, pd.Series], annualize: float = 252.0
) -> Dict[int, float]:
    """Compute Sharpe ratio separately for each year's net returns.

    Parameters
    ----------
    net_returns_by_year:
        Mapping from calendar year to daily net-return series.
    annualize:
        Annualisation factor.

    Returns
    -------
    dict
        Mapping from year to annualised Sharpe ratio.
    """
    return {
        year: compute_sharpe(returns, annualize)
        for year, returns in net_returns_by_year.items()
    }


def compute_yearly_leakage_gain(
    variant_yearly_sr: Dict[int, float],
    clean_yearly_sr: Dict[int, float],
) -> np.ndarray:
    """Compute per-year LG-SR values (variant SR minus CLEAN SR).

    Parameters
    ----------
    variant_yearly_sr:
        Per-year Sharpe ratios for the leaky variant.
    clean_yearly_sr:
        Per-year Sharpe ratios for the CLEAN baseline.

    Returns
    -------
    np.ndarray
        Array of LG-SR values, one per overlapping year, sorted ascending
        by year (9 values for 2016–2024).
    """
    years = sorted(set(variant_yearly_sr.keys()) & set(clean_yearly_sr.keys()))
    return np.array([variant_yearly_sr[y] - clean_yearly_sr[y] for y in years])


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for the mean of an array.

    Parameters
    ----------
    values:
        1-D array of observed values.
    n_bootstrap:
        Number of bootstrap resamples.
    ci:
        Nominal coverage probability (e.g. 0.95 for a 95 % CI).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    (lower, upper) : Tuple[float, float]
        Percentile bounds of the bootstrap distribution of the mean.
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = values[indices].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100.0))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    return lower, upper


def wilcoxon_one_sided(values: np.ndarray) -> Tuple[float, float]:
    """One-sided Wilcoxon signed-rank test.

    H0: median(values) <= 0
    H1: median(values) >  0

    Parameters
    ----------
    values:
        1-D array of paired differences (e.g. yearly LG-SR values).

    Returns
    -------
    (statistic, p_value) : Tuple[float, float]
        Test statistic and one-sided p-value.
        Returns (nan, nan) when the test is undefined (e.g. all zeros,
        insufficient non-zero observations).
    """
    try:
        stat, pval = wilcoxon(values, alternative="greater")
        return float(stat), float(pval)
    except ValueError:
        return np.nan, np.nan


def summarize_yearly_lg(yearly_lg: np.ndarray, n_bootstrap: int = 10000) -> Dict:
    """Produce a stability summary for an array of yearly leakage-gain values.

    Parameters
    ----------
    yearly_lg:
        Array of per-year LG-SR values (typically 9 values for 2016–2024).
    n_bootstrap:
        Number of bootstrap resamples for the CI.

    Returns
    -------
    dict with keys:
        mean_lg        : float  — mean of yearly LG values
        positive_years : int    — count of years with LG > 0
        ci_lower       : float  — lower bound of 95 % bootstrap CI on mean
        ci_upper       : float  — upper bound of 95 % bootstrap CI on mean
        wilcoxon_stat  : float  — Wilcoxon W statistic (or nan)
        wilcoxon_pval  : float  — one-sided p-value (or nan)
    """
    ci_lower, ci_upper = bootstrap_ci(yearly_lg, n_bootstrap=n_bootstrap)
    w_stat, w_pval = wilcoxon_one_sided(yearly_lg)
    return {
        "mean_lg": float(np.mean(yearly_lg)),
        "positive_years": int(np.sum(yearly_lg > 0)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "wilcoxon_stat": w_stat,
        "wilcoxon_pval": w_pval,
    }
