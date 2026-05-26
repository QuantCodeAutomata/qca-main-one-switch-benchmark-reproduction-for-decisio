"""Protocol-aware backtester for the decision-time leakage benchmark."""

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Protocols whose execution return uses opens two days forward
_OPEN_NEXT2_PROTOCOLS = {"CLEAN", "TEMP_CENTER", "NORM_GLOBAL", "STRUCT_GRAPH"}


def form_portfolio(scores: pd.Series, top_decile: float = 0.1) -> pd.Index:
    """Select top-decile assets by score (long-only, equal-weight).

    Parameters
    ----------
    scores:
        Series indexed by ticker for a single date.
    top_decile:
        Fraction of valid universe to select.

    Returns
    -------
    pd.Index
        Tickers selected for the portfolio.
    """
    valid = scores.dropna()
    n = math.ceil(top_decile * len(valid))
    return valid.nlargest(n).index


def compute_trade_returns(
    ohlcv: pd.DataFrame,
    portfolio_dates: pd.DatetimeIndex,
    portfolio_assets: Dict[pd.Timestamp, pd.Index],
    protocol: str,
) -> pd.Series:
    """Compute one-day realised trade returns for each decision date.

    Return formulae by protocol
    ---------------------------
    CLEAN / TEMP_CENTER / NORM_GLOBAL / STRUCT_GRAPH:
        r_t = O_{t+2} / O_{t+1} - 1
    EXEC_CLOSE:
        r_t = C_{t+1} / C_t - 1
    EXEC_OPEN:
        r_t = O_{t+1} / O_t - 1

    Parameters
    ----------
    ohlcv:
        DataFrame with MultiIndex (date, ticker) and columns including
        'open' and 'close'.
    portfolio_dates:
        Ordered DatetimeIndex of decision dates.
    portfolio_assets:
        Mapping from each decision date to its selected tickers.
    protocol:
        One of the six benchmark protocol identifiers.

    Returns
    -------
    pd.Series
        Equal-weight portfolio returns indexed by decision date.
        NaN where sufficient forward data is unavailable.
    """
    opens = ohlcv["open"].unstack()   # dates × tickers
    closes = ohlcv["close"].unstack()

    cal = opens.index.sort_values()
    date_pos: Dict[pd.Timestamp, int] = {d: i for i, d in enumerate(cal)}

    result: Dict[pd.Timestamp, float] = {}

    for t in portfolio_dates:
        assets = portfolio_assets.get(t)
        if assets is None or len(assets) == 0:
            result[t] = np.nan
            continue

        pos = date_pos.get(t)
        if pos is None:
            result[t] = np.nan
            continue

        if protocol == "EXEC_CLOSE":
            if pos + 1 >= len(cal):
                result[t] = np.nan
                continue
            t1 = cal[pos + 1]
            valid_cols = assets.intersection(closes.columns)
            p0 = closes.loc[t, valid_cols].dropna()
            p1 = closes.loc[t1, valid_cols].dropna()

        elif protocol == "EXEC_OPEN":
            if pos + 1 >= len(cal):
                result[t] = np.nan
                continue
            t1 = cal[pos + 1]
            valid_cols = assets.intersection(opens.columns)
            p0 = opens.loc[t, valid_cols].dropna()
            p1 = opens.loc[t1, valid_cols].dropna()

        else:  # CLEAN / TEMP_CENTER / NORM_GLOBAL / STRUCT_GRAPH
            if pos + 2 >= len(cal):
                result[t] = np.nan
                continue
            t1 = cal[pos + 1]
            t2 = cal[pos + 2]
            valid_cols = assets.intersection(opens.columns)
            p0 = opens.loc[t1, valid_cols].dropna()
            p1 = opens.loc[t2, valid_cols].dropna()

        common = p0.index.intersection(p1.index)
        if len(common) == 0:
            result[t] = np.nan
            continue

        result[t] = (p1[common] / p0[common] - 1.0).mean()

    return pd.Series(result)


def compute_gross_returns(
    ohlcv: pd.DataFrame,
    scores_panel: pd.DataFrame,
    protocol: str,
    top_decile: float = 0.1,
) -> Tuple[pd.Series, Dict]:
    """Form portfolios for every date in scores_panel and compute gross returns.

    Parameters
    ----------
    ohlcv:
        DataFrame with MultiIndex (date, ticker).
    scores_panel:
        DataFrame with dates as index and tickers as columns.
    protocol:
        Benchmark protocol identifier.
    top_decile:
        Fraction of valid universe to select per date.

    Returns
    -------
    gross_returns : pd.Series
        Equal-weight daily gross returns indexed by decision date.
    portfolios : dict
        Mapping from decision date to selected tickers (pd.Index).
    """
    portfolios: Dict[pd.Timestamp, pd.Index] = {}
    for t in scores_panel.index:
        portfolios[t] = form_portfolio(scores_panel.loc[t], top_decile)

    portfolio_dates = pd.DatetimeIndex(sorted(portfolios.keys()))
    gross_returns = compute_trade_returns(ohlcv, portfolio_dates, portfolios, protocol)
    return gross_returns, portfolios


def compute_turnover(
    portfolios: Dict[pd.Timestamp, pd.Index],
    all_dates: pd.DatetimeIndex,
) -> pd.Series:
    """Compute one-way daily portfolio turnover.

    turnover_t = |new_positions \\ old_positions| / portfolio_size

    On the first rebalance date all positions are treated as new
    (turnover = 1.0).

    Parameters
    ----------
    portfolios:
        Mapping from decision date to selected tickers.
    all_dates:
        Full date range used to reindex the result.

    Returns
    -------
    pd.Series
        One-way turnover indexed by date (NaN for dates not in portfolios).
    """
    result: Dict[pd.Timestamp, float] = {}
    prev_set: set = set()

    for t in sorted(portfolios.keys()):
        curr_set = set(portfolios[t])
        n = len(curr_set)
        if n == 0:
            result[t] = 0.0
        elif len(prev_set) == 0:
            result[t] = 1.0
        else:
            result[t] = len(curr_set - prev_set) / n
        prev_set = curr_set

    return pd.Series(result).reindex(all_dates)


def compute_net_returns(
    gross_returns: pd.Series,
    turnover: pd.Series,
    cost_bps: float,
) -> pd.Series:
    """Deduct transaction costs from gross returns.

    net_return_t = gross_return_t - turnover_t * cost_bps / 10_000

    Parameters
    ----------
    gross_returns:
        Portfolio gross returns.
    turnover:
        One-way turnover aligned to the same date index.
    cost_bps:
        Round-trip transaction cost in basis points applied per unit of
        one-way turnover.

    Returns
    -------
    pd.Series
        Net returns after costs.
    """
    return gross_returns - turnover.reindex(gross_returns.index) * cost_bps / 10_000


def run_backtest(
    ohlcv: pd.DataFrame,
    scores_panel: pd.DataFrame,
    labels_panel: pd.DataFrame,
    protocol: str,
    costs_bps: List[float] = None,
    top_decile: float = 0.1,
) -> Dict:
    """Execute the full backtest pipeline for a single protocol configuration.

    Parameters
    ----------
    ohlcv:
        DataFrame with MultiIndex (date, ticker) and OHLCV columns.
    scores_panel:
        Model predictions — DataFrame with dates as index, tickers as columns.
    labels_panel:
        Forward returns used for information-coefficient metrics.
    protocol:
        Benchmark protocol identifier.
    costs_bps:
        List of transaction-cost levels (in bps) for which net returns are
        computed. Defaults to [0, 5, 10].
    top_decile:
        Fraction of valid universe selected per rebalance date.

    Returns
    -------
    dict with keys:
        gross_returns : pd.Series
        net_returns   : dict mapping cost_bps → pd.Series
        turnover      : pd.Series
        portfolios    : dict mapping date → pd.Index
        labels        : pd.DataFrame (labels_panel, passed through)
    """
    if costs_bps is None:
        costs_bps = [0, 5, 10]

    gross_returns, portfolios = compute_gross_returns(
        ohlcv, scores_panel, protocol, top_decile
    )
    turnover = compute_turnover(portfolios, gross_returns.index)
    net_returns = {c: compute_net_returns(gross_returns, turnover, c) for c in costs_bps}

    return {
        "gross_returns": gross_returns,
        "net_returns": net_returns,
        "turnover": turnover,
        "portfolios": portfolios,
        "labels": labels_panel,
    }
