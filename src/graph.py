"""
graph.py – Monthly correlation graph construction for the GraphMLP model.

Two graph variants (selected by *protocol*)
--------------------------------------------
CLEAN:
    W is estimated from the trailing *trailing_days* days **before** the month
    anchor τ_m.  No future information leaks into the graph weights.

STRUCT_GRAPH:
    W is estimated from a symmetric *trailing_days*-day window **centred at**
    τ_m, i.e. [τ_m − trailing_days//2, τ_m + trailing_days//2).  The forward
    half of the window introduces structural look-ahead leakage.

For all other protocols the CLEAN construction is used.

Graph topology
--------------
For each asset *i* the top-*k* peers by absolute Pearson correlation are kept;
all other edges are zeroed.  Each row is then L1-normalised so the weights
sum to 1 (softened graph convolution).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import GRAPH_TOP_PEERS, GRAPH_TRAILING_DAYS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_monthly_anchors(dates: pd.DatetimeIndex) -> pd.Series:
    """Return the first trading day of each calendar month present in *dates*.

    Parameters
    ----------
    dates:
        Sorted DatetimeIndex of all trading dates in the panel.

    Returns
    -------
    pd.Series
        Series whose index is ``pd.PeriodIndex`` (monthly) and whose values
        are the corresponding first-trading-day ``pd.Timestamp``.
    """
    s = pd.Series(dates, index=dates, dtype='datetime64[ns]')
    monthly = s.groupby(s.index.to_period('M')).first()
    return monthly


def build_monthly_graph(
    close_returns: pd.DataFrame,
    anchor_date: pd.Timestamp,
    protocol: str = 'CLEAN',
    trailing_days: int = GRAPH_TRAILING_DAYS,
    top_peers: int = GRAPH_TOP_PEERS,
) -> np.ndarray:
    """Build a row-normalised peer-correlation adjacency matrix.

    Parameters
    ----------
    close_returns:
        Wide DataFrame – index: trading dates, columns: ticker symbols,
        values: daily log-returns.
    anchor_date:
        Month anchor τ_m (first trading day of the month).
    protocol:
        ``'STRUCT_GRAPH'`` uses a symmetric window; all others use trailing.
    trailing_days:
        Window length in trading days (default 252).
    top_peers:
        Number of highest-correlated peers to keep per row (default 5).

    Returns
    -------
    np.ndarray
        Row-normalised adjacency matrix W of shape (n_assets, n_assets)
        where n_assets == ``len(close_returns.columns)``.
        Rows for assets with insufficient history are all-zero.
    """
    tickers = close_returns.columns.tolist()
    n = len(tickers)

    if n == 0:
        return np.zeros((0, 0), dtype=float)

    # ----------------------------------------------------------------
    # Select the estimation window
    # ----------------------------------------------------------------
    if protocol == 'STRUCT_GRAPH':
        half = trailing_days // 2
        idx = close_returns.index
        pos = idx.searchsorted(anchor_date, side='left')
        start_pos = max(0, pos - half)
        end_pos = min(len(idx), pos + half)
        window = close_returns.iloc[start_pos:end_pos]
    else:
        # CLEAN (and all other protocols): trailing window strictly before anchor
        mask = close_returns.index < anchor_date
        available = close_returns.loc[mask]
        window = available.iloc[-trailing_days:] if len(available) > 0 else available

    # ----------------------------------------------------------------
    # Require at least 20 observations per ticker (relaxed guard)
    # ----------------------------------------------------------------
    min_obs = max(20, int(len(window) * 0.25))
    enough = window.count() >= min_obs
    valid_cols = enough[enough].index.tolist()

    if len(valid_cols) < 2:
        logger.warning(
            'build_monthly_graph: fewer than 2 valid tickers at %s – returning zero matrix',
            anchor_date.date(),
        )
        return np.zeros((n, n), dtype=float)

    # ----------------------------------------------------------------
    # Compute Pearson correlation on valid columns
    # ----------------------------------------------------------------
    corr_valid = window[valid_cols].corr()  # (m, m)

    # Expand to full n×n matrix (NaN for tickers that had insufficient data)
    col_to_idx = {t: i for i, t in enumerate(tickers)}
    valid_pos = np.array([col_to_idx[t] for t in valid_cols], dtype=int)

    full_corr = np.full((n, n), np.nan, dtype=float)
    full_corr[np.ix_(valid_pos, valid_pos)] = corr_valid.values

    # ----------------------------------------------------------------
    # Sparse top-k graph construction
    # ----------------------------------------------------------------
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        row_abs = np.abs(full_corr[i]).copy()
        row_abs[i] = np.nan  # exclude self-loop

        valid_mask = ~np.isnan(row_abs)
        valid_j = np.where(valid_mask)[0]
        if len(valid_j) == 0:
            continue

        k = min(top_peers, len(valid_j))
        # argpartition gives indices of the k largest values (unsorted)
        top_rel = np.argpartition(row_abs[valid_j], -k)[-k:]
        top_j = valid_j[top_rel]
        W[i, top_j] = row_abs[top_j]

    # ----------------------------------------------------------------
    # Row-wise L1 normalisation
    # ----------------------------------------------------------------
    row_sums = W.sum(axis=1, keepdims=True)
    # Avoid division by zero for isolated nodes
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    W = W / row_sums
    return W


def compute_relation_features(
    features: pd.DataFrame,
    W: np.ndarray,
    rel_feature_names: list[str],
    tickers: list[str],
    month_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute neighbour-averaged relation features for a single month.

    For each asset *i* and trading day *t*:

        nbr_feat_{i,t} = Σ_j  W_{ij} · feat_{j,t}

    Parameters
    ----------
    features:
        MultiIndex (date, ticker) feature DataFrame (output of
        :func:`src.features.compute_base_features`).
    W:
        Row-normalised adjacency matrix of shape (n_assets, n_assets),
        aligned to *tickers*.
    rel_feature_names:
        Feature columns to aggregate across neighbours (e.g. ``RELATION_FEATURES``).
    tickers:
        Ordered list of asset symbols corresponding to rows/columns of *W*.
    month_dates:
        Trading dates belonging to the current month.

    Returns
    -------
    pd.DataFrame
        MultiIndex (date, ticker), columns ``[nbr_{f} for f in rel_feature_names]``.
    """
    n = len(tickers)
    W_arr = np.asarray(W, dtype=float)  # (n, n)

    available_dates = features.index.get_level_values('date')
    date_mask = available_dates.isin(month_dates)
    feat_subset = features.loc[date_mask, rel_feature_names]

    rel_col_names = [f'nbr_{f}' for f in rel_feature_names]
    frames: list[pd.DataFrame] = []

    for date in month_dates:
        # Extract (n_tickers, n_rel_features) matrix for this date
        if date not in feat_subset.index.get_level_values('date'):
            continue
        daily = feat_subset.xs(date, level='date')        # (tickers_present, n_feats)
        daily = daily.reindex(tickers)                    # align to graph ordering

        feat_matrix = daily.values.astype(float)          # (n, n_feats)
        # Replace NaN with 0 before matrix multiply (no neighbour signal)
        feat_filled = np.where(np.isnan(feat_matrix), 0.0, feat_matrix)

        rel_matrix = W_arr @ feat_filled                  # (n, n_feats)

        rel_df = pd.DataFrame(
            rel_matrix,
            index=pd.Index(tickers, name='ticker'),
            columns=rel_col_names,
        )
        rel_df.index = pd.MultiIndex.from_product(
            [[date], tickers], names=['date', 'ticker']
        )
        frames.append(rel_df)

    if not frames:
        logger.warning('compute_relation_features: no data found for month_dates provided')
        empty_idx = pd.MultiIndex.from_tuples([], names=['date', 'ticker'])
        return pd.DataFrame(columns=rel_col_names, index=empty_idx)

    result = pd.concat(frames)
    return result.sort_index()
