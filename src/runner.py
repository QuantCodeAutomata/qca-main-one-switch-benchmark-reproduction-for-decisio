"""
runner.py – Walk-forward experiment runner for the decision-time leakage benchmark.

Public API
----------
get_walk_forward_splits(dates, test_year)
run_protocol_configuration(ohlcv, market, model_name, horizon, protocol, test_years, costs_bps)
run_benchmark_grid(markets, models, horizons, protocols, costs_bps, save_dir)
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.config import (
    COSTS_BPS,
    FUTURE_DAYS,
    HORIZONS,
    LOOKBACK,
    MARKETS,
    MODELS,
    PROTOCOLS,
    RELATION_FEATURES,
    TEST_YEARS,
)
from src.data_loader import load_market_data
from src.features import compute_base_features, compute_label, normalize_features
from src.graph import (
    build_monthly_graph,
    compute_relation_features,
    get_monthly_anchors,
)
from src.metrics import compute_all_metrics
from src.models import GraphMLPModel, LSTMModel, MomentumModel, RidgeModel, TransformerModel, get_model

logger = logging.getLogger(__name__)

_GRAPH_MODELS = {"GraphMLP"}
_SEQ_MODELS = {"LSTM", "Transformer"}


# ---------------------------------------------------------------------------
# Walk-forward split helper
# ---------------------------------------------------------------------------

def get_walk_forward_splits(dates: pd.DatetimeIndex, test_year: int) -> Dict:
    """Return train / val / test DatetimeIndex slices for a given test year.

    Parameters
    ----------
    dates:
        Full sorted DatetimeIndex of trading days in the panel.
    test_year:
        Calendar year whose dates form the *test* fold.

    Returns
    -------
    dict with keys ``'train'``, ``'val'``, ``'test'``, each a DatetimeIndex.
    Train contains all dates strictly before *test_year − 1*.
    Val   contains dates in calendar year *test_year − 1*.
    Test  contains dates in calendar year *test_year*.
    """
    years = dates.year
    return {
        "train": dates[years < test_year - 1],
        "val":   dates[years == test_year - 1],
        "test":  dates[years == test_year],
    }


# ---------------------------------------------------------------------------
# Per-model scoring helpers
# ---------------------------------------------------------------------------

def _score_simple(
    model,
    features_normed: pd.DataFrame,
    test_dates: pd.DatetimeIndex,
) -> pd.Series:
    """Score every (date, ticker) in the test period using a simple model."""
    test_mask = features_normed.index.get_level_values("date").isin(test_dates)
    feat_test = features_normed[test_mask].dropna(how="all")
    if feat_test.empty:
        return pd.Series(dtype=float)
    # Impute residual NaNs (e.g. from TEMP_CENTER tail) with 0 to keep rows
    feat_vals = np.nan_to_num(feat_test.values, nan=0.0)
    preds = model.predict(feat_vals)
    return pd.Series(preds, index=feat_test.index)


def _fit_and_score_graph(
    features_normed: pd.DataFrame,
    labels: pd.Series,
    ohlcv: pd.DataFrame,
    train_val_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
    protocol: str,
) -> pd.Series:
    """Fit a GraphMLPModel and score the test period."""
    # Build wide log-return DataFrame for graph construction
    close_wide = ohlcv["close"].unstack(level="ticker").sort_index()
    close_returns = np.log(close_wide / close_wide.shift(1))
    tickers: List[str] = close_wide.columns.tolist()
    rel_names = [f for f in RELATION_FEATURES if f in features_normed.columns]

    def _collect_relation_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
        anchors = get_monthly_anchors(dates)
        parts: List[pd.DataFrame] = []
        for period, anchor_date in anchors.items():
            month_dates = dates[
                (dates.year == period.year) & (dates.month == period.month)
            ]
            W = build_monthly_graph(close_returns, anchor_date, protocol)
            rel = compute_relation_features(
                features_normed, W, rel_names, tickers, month_dates
            )
            parts.append(rel)
        return pd.concat(parts) if parts else pd.DataFrame(
            columns=[f"nbr_{f}" for f in rel_names],
            index=pd.MultiIndex.from_tuples([], names=["date", "ticker"]),
        )

    # ------------------------------------------------------------------
    # Training data: base + relation features for train+val period
    # ------------------------------------------------------------------
    tv_mask = features_normed.index.get_level_values("date").isin(train_val_dates)
    feat_tv = features_normed[tv_mask]
    rel_tv = _collect_relation_features(train_val_dates)

    common_tv = feat_tv.index.intersection(rel_tv.index)
    if len(common_tv) == 0:
        return pd.Series(dtype=float)

    lab_tv = labels.reindex(common_tv)
    valid = ~lab_tv.isna()
    X_base = feat_tv.loc[common_tv].values[valid]
    X_rel = rel_tv.loc[common_tv].values[valid]
    X_tv = np.concatenate([X_base, X_rel], axis=1)
    y_tv = lab_tv.values[valid]

    base_dim = feat_tv.shape[1]
    rel_dim = rel_tv.shape[1]
    model = GraphMLPModel(base_dim, rel_dim)
    model.fit(X_tv.astype(np.float32), y_tv.astype(np.float32))

    # ------------------------------------------------------------------
    # Scoring: test period
    # ------------------------------------------------------------------
    rel_test = _collect_relation_features(test_dates)
    test_mask = features_normed.index.get_level_values("date").isin(test_dates)
    feat_test = features_normed[test_mask]

    common_test = feat_test.index.intersection(rel_test.index)
    if len(common_test) == 0:
        return pd.Series(dtype=float)

    X_test = np.concatenate(
        [feat_test.loc[common_test].values, rel_test.loc[common_test].values], axis=1
    )
    preds = model.predict(X_test.astype(np.float32))
    return pd.Series(preds, index=common_test)


def _fit_and_score_sequence(
    model_name: str,
    features_normed: pd.DataFrame,
    labels: pd.Series,
    train_val_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
) -> pd.Series:
    """Fit an LSTM or Transformer model and score the test period.

    Training: concatenate each ticker's time-ordered feature vectors.
    Scoring:  per-ticker windowed inference including enough look-back before
              the first test date so that predictions are non-NaN.

    Performance notes
    -----------------
    * Tickers are pre-grouped once via groupby to avoid O(N×T) MultiIndex scans.
    * pandas .isin() replaces Python set(DatetimeIndex) which is ~100x slower.
    * X_window is trimmed to [window_start, last_test_pos] to skip unnecessary
      future dates that would otherwise be included for early test years.
    * Correct date alignment: preds[k] maps to window_dates[LOOKBACK-1+k],
      i.e. the sequence ending at that position is used for that day's score.
    """
    n_features = features_normed.shape[1]
    model = (
        LSTMModel(n_features)
        if model_name == "LSTM"
        else TransformerModel(n_features)
    )

    # Pre-group by ticker: O(N) once vs O(N×T) per ticker
    ticker_data: Dict[str, pd.DataFrame] = {
        ticker: grp.droplevel("ticker").sort_index()
        for ticker, grp in features_normed.groupby(level="ticker")
    }
    label_data: Dict[str, pd.Series] = {
        ticker: ser.droplevel("ticker").sort_index()
        for ticker, ser in labels.groupby(level="ticker")
    }
    all_tickers = sorted(ticker_data.keys())

    # ------------------------------------------------------------------
    # Build training arrays: one contiguous block per ticker
    # ------------------------------------------------------------------
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    for ticker in all_tickers:
        feat = ticker_data[ticker]
        # Vectorised membership — avoids Python-level Timestamp conversion
        tv_mask = feat.index.isin(train_val_dates)
        if not tv_mask.any():
            continue
        feat_tv = feat.loc[tv_mask].fillna(0.0)
        if len(feat_tv) < LOOKBACK:
            continue
        lab = label_data.get(ticker, pd.Series(dtype=float))
        lab_tv = lab.reindex(feat_tv.index).fillna(0.0)
        X_parts.append(feat_tv.values)
        y_parts.append(lab_tv.values)

    if not X_parts:
        return pd.Series(dtype=float)

    X_train = np.concatenate(X_parts, axis=0).astype(np.float32)
    y_train = np.concatenate(y_parts, axis=0).astype(np.float32)
    model.fit(X_train, y_train)

    # Build once — avoids repeated Timestamp object creation inside the loop
    test_date_set = set(test_dates.tolist())

    # ------------------------------------------------------------------
    # Score each ticker over the test window (with look-back prefix)
    # ------------------------------------------------------------------
    scores: Dict = {}
    for ticker in all_tickers:
        feat = ticker_data[ticker]
        all_idx = feat.index  # sorted DatetimeIndex (no ticker level)

        # Vectorised: find positions of test dates for this ticker
        test_mask = all_idx.isin(test_dates)
        if not test_mask.any():
            continue

        first_test_pos = int(np.argmax(test_mask))
        # last test position: argmax on reversed mask
        last_test_pos = len(all_idx) - 1 - int(np.argmax(test_mask[::-1]))

        window_start = max(0, first_test_pos - LOOKBACK + 1)
        window_stop = last_test_pos + 1  # trim: skip post-test-year data

        X_window = (
            feat.iloc[window_start:window_stop].fillna(0.0).values.astype(np.float32)
        )
        if len(X_window) < LOOKBACK:
            continue

        preds = model.predict(X_window)
        window_dates = all_idx[window_start:window_stop]

        # Correct alignment: preds[k] ↔ window_dates[LOOKBACK-1+k]
        pred_dates = window_dates[LOOKBACK - 1:]
        for date, pred in zip(pred_dates, preds):
            if date in test_date_set and np.isfinite(pred):
                scores[(date, ticker)] = float(pred)

    if not scores:
        return pd.Series(dtype=float)

    idx = pd.MultiIndex.from_tuples(list(scores.keys()), names=["date", "ticker"])
    return pd.Series(list(scores.values()), index=idx)


# ---------------------------------------------------------------------------
# Protocol configuration runner
# ---------------------------------------------------------------------------

def run_protocol_configuration(
    ohlcv: pd.DataFrame,
    market: str,
    model_name: str,
    horizon: int,
    protocol: str,
    test_years: List[int],
    costs_bps: List[float] = None,
) -> Dict:
    """Run one full (market, model, horizon, protocol) walk-forward experiment.

    For each year in *test_years* the function:
    1. Computes walk-forward train / val / test splits.
    2. Computes base features with the protocol's look-ahead setting.
    3. Computes forward-return labels.
    4. Normalises features (train-only or full-panel, depending on protocol).
    5. Fits the model on train + val data.
    6. Scores every test cross-section.
    7. Runs the backtest and computes performance metrics.

    Parameters
    ----------
    ohlcv:
        MultiIndex (date, ticker) DataFrame with OHLCV columns.
    market:
        ``'US'`` or ``'China'`` (used for logging only).
    model_name:
        One of ``'Momentum'``, ``'Ridge'``, ``'LightGBM'``, ``'GraphMLP'``,
        ``'LSTM'``, ``'Transformer'``.
    horizon:
        Forecast horizon in trading days (5 or 20).
    protocol:
        One of the six benchmark protocol identifiers.
    test_years:
        Calendar years to use as test folds (e.g. ``list(range(2016, 2025))``).
    costs_bps:
        Transaction-cost levels for which net returns are computed.
        5 bps must be included (required by ``compute_all_metrics``).

    Returns
    -------
    dict with keys:
        ``'metrics'``      : pd.DataFrame — one row per test year.
        ``'scores_panel'`` : pd.DataFrame — date × ticker score panels concatenated.
        ``'labels_panel'`` : pd.DataFrame — date × ticker label panels concatenated.
        ``'gross_returns'``: pd.Series.
        ``'net_returns'``  : dict mapping cost_bps → pd.Series.
        ``'turnover'``     : pd.Series.
        ``'portfolios'``   : dict mapping date → pd.Index.
    Returns an empty dict if no test years produced valid results.
    """
    if costs_bps is None:
        costs_bps = [0, 5, 10]
    # Ensure 5 bps present – required by compute_all_metrics
    effective_costs = sorted(set(costs_bps) | {5})

    # ------------------------------------------------------------------
    # Feature and label computation (done once over the full panel)
    # ------------------------------------------------------------------
    future_days = FUTURE_DAYS if protocol == "TEMP_CENTER" else 0
    features = compute_base_features(ohlcv, protocol, future_days)
    labels = compute_label(ohlcv, horizon, protocol)

    all_dates = features.index.get_level_values("date").unique().sort_values()

    yearly_metrics: List[Dict] = []
    all_scores: List[pd.DataFrame] = []
    all_labels: List[pd.DataFrame] = []
    all_gross: List[pd.Series] = []
    all_net: Dict[float, List[pd.Series]] = {c: [] for c in effective_costs}
    all_turnover: List[pd.Series] = []
    all_portfolios: Dict = {}

    for test_year in test_years:
        splits = get_walk_forward_splits(all_dates, test_year)
        if len(splits["test"]) == 0:
            logger.debug("No test data for year %d – skipping.", test_year)
            continue

        train_val_dates = splits["train"].append(splits["val"])
        if len(train_val_dates) < LOOKBACK:
            logger.debug(
                "Insufficient train+val data for year %d – skipping.", test_year
            )
            continue

        # ------------------------------------------------------------------
        # Normalisation
        # ------------------------------------------------------------------
        features_normed = normalize_features(
            features, train_dates=train_val_dates, protocol=protocol
        )

        try:
            # --------------------------------------------------------------
            # Model fit + score
            # --------------------------------------------------------------
            if model_name in _GRAPH_MODELS:
                scores_raw = _fit_and_score_graph(
                    features_normed, labels, ohlcv,
                    train_val_dates, splits["test"], protocol,
                )
            elif model_name in _SEQ_MODELS:
                scores_raw = _fit_and_score_sequence(
                    model_name, features_normed, labels,
                    train_val_dates, splits["test"],
                )
            else:
                # Momentum, Ridge, LightGBM
                tv_mask = features_normed.index.get_level_values("date").isin(
                    train_val_dates
                )
                feat_tv = features_normed[tv_mask]
                lab_tv = labels.reindex(feat_tv.index).dropna()
                feat_tv_valid = feat_tv.loc[lab_tv.index].dropna()
                lab_tv_valid = lab_tv.reindex(feat_tv_valid.index)

                if model_name == "Momentum":
                    model = MomentumModel()
                elif model_name == "Ridge":
                    model = RidgeModel()
                else:
                    model = get_model(model_name)

                # Impute any residual NaNs (e.g. TEMP_CENTER boundary effects)
                X_train = np.nan_to_num(feat_tv_valid.values, nan=0.0)
                model.fit(X_train, lab_tv_valid.values)
                scores_raw = _score_simple(
                    model, features_normed, splits["test"]
                )

        except Exception as exc:
            logger.warning(
                "Model %s failed for %s proto=%s year=%d: %s",
                model_name, market, protocol, test_year, exc,
            )
            continue

        if scores_raw.empty:
            logger.debug(
                "Empty scores for %s %s year=%d – skipping.", model_name, protocol, test_year
            )
            continue

        # Pivot scores and labels to date × ticker panels
        scores_panel = scores_raw.unstack(level="ticker")
        lab_test_mask = labels.index.get_level_values("date").isin(splits["test"])
        labels_panel = labels[lab_test_mask].unstack(level="ticker")

        # Align columns
        common_tickers = scores_panel.columns.intersection(labels_panel.columns)
        scores_panel = scores_panel[common_tickers]
        labels_panel = labels_panel[common_tickers]

        # ------------------------------------------------------------------
        # Backtest
        # ------------------------------------------------------------------
        bt = run_backtest(ohlcv, scores_panel, labels_panel, protocol, effective_costs)

        metrics = compute_all_metrics(bt, scores_panel, labels_panel, effective_costs)
        metrics["year"] = test_year
        yearly_metrics.append(metrics)

        all_scores.append(scores_panel)
        all_labels.append(labels_panel)
        all_gross.append(bt["gross_returns"])
        all_turnover.append(bt["turnover"])
        all_portfolios.update(bt["portfolios"])
        for c in effective_costs:
            all_net[c].append(bt["net_returns"][c])

    if not yearly_metrics:
        return {}

    return {
        "metrics": pd.DataFrame(yearly_metrics),
        "scores_panel": pd.concat(all_scores) if all_scores else pd.DataFrame(),
        "labels_panel": pd.concat(all_labels) if all_labels else pd.DataFrame(),
        "gross_returns": pd.concat(all_gross) if all_gross else pd.Series(dtype=float),
        "net_returns": {
            c: pd.concat(all_net[c]) if all_net[c] else pd.Series(dtype=float)
            for c in effective_costs
        },
        "turnover": pd.concat(all_turnover) if all_turnover else pd.Series(dtype=float),
        "portfolios": all_portfolios,
    }


# ---------------------------------------------------------------------------
# Benchmark grid
# ---------------------------------------------------------------------------

def run_benchmark_grid(
    markets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    horizons: Optional[List[int]] = None,
    protocols: Optional[List[str]] = None,
    costs_bps: Optional[List[float]] = None,
    save_dir: str = "results",
) -> pd.DataFrame:
    """Run the full benchmark grid and return a tidy results DataFrame.

    Each row corresponds to a single (market, model, horizon, protocol, year)
    combination with all metrics as columns.  Results are also written to
    ``{save_dir}/benchmark_results.parquet`` and
    ``{save_dir}/benchmark_results.csv``.

    Parameters
    ----------
    markets:   Markets to run (default: all in MARKETS).
    models:    Model names to run (default: all in MODELS).
    horizons:  Forecast horizons (default: all in HORIZONS).
    protocols: Protocol identifiers (default: all in PROTOCOLS).
    costs_bps: Transaction-cost levels (default: [0, 5, 10]).
    save_dir:  Directory for output files.

    Returns
    -------
    pd.DataFrame
        All results in a tidy table.  Empty if every configuration failed.
    """
    markets   = markets   or MARKETS
    models    = models    or MODELS
    horizons  = horizons  or HORIZONS
    protocols = protocols or PROTOCOLS
    costs_bps = costs_bps or [0, 5, 10]

    os.makedirs(save_dir, exist_ok=True)

    rows: List[Dict] = []
    ohlcv_cache: Dict[str, pd.DataFrame] = {}

    total = len(markets) * len(models) * len(horizons) * len(protocols)
    done = 0

    for market in markets:
        if market not in ohlcv_cache:
            logger.info("Loading %s market data …", market)
            ohlcv_cache[market] = load_market_data(market, force_download=False)
        ohlcv = ohlcv_cache[market]

        for model_name in models:
            for horizon in horizons:
                for protocol in protocols:
                    done += 1
                    key = f"{market}|{model_name}|h{horizon}|{protocol}"
                    logger.info("[%d/%d] Running %s", done, total, key)

                    result = run_protocol_configuration(
                        ohlcv=ohlcv,
                        market=market,
                        model_name=model_name,
                        horizon=horizon,
                        protocol=protocol,
                        test_years=TEST_YEARS,
                        costs_bps=costs_bps,
                    )
                    if not result:
                        logger.warning("No results for %s", key)
                        continue

                    for _, metric_row in result["metrics"].iterrows():
                        row = {
                            "market":   market,
                            "model":    model_name,
                            "horizon":  horizon,
                            "protocol": protocol,
                        }
                        row.update(metric_row.to_dict())
                        rows.append(row)

    results_df = pd.DataFrame(rows)

    if not results_df.empty:
        results_df.to_parquet(
            os.path.join(save_dir, "benchmark_results.parquet"), index=False
        )
        results_df.to_csv(
            os.path.join(save_dir, "benchmark_results.csv"), index=False
        )
        logger.info("Saved %d result rows to %s/", len(results_df), save_dir)

    return results_df
