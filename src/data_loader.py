"""
data_loader.py – Download and cache daily OHLCV panels for US and China markets.

Primary source : massive REST API  (env var MASSIVE_TOKEN)
Fallback source: Yahoo Finance     (yfinance)

Saved format: parquet with MultiIndex (date, ticker), columns [open, high, low, close, volume].
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from src.config import (
    CHINA_DATA_PATH,
    US_DATA_PATH,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Date range
# ---------------------------------------------------------------------------

START_DATE = '2010-01-04'
END_DATE = '2024-12-31'

# ---------------------------------------------------------------------------
# Ticker universes
# ---------------------------------------------------------------------------

US_TICKERS: list[str] = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'BRK-B', 'LLY',
    'AVGO', 'JPM', 'XOM', 'TSLA', 'UNH', 'V', 'MA', 'JNJ', 'PG', 'HD',
    'MRK', 'ABBV', 'CVX', 'COST', 'BAC', 'NFLX', 'KO', 'PEP', 'ADBE',
    'CRM', 'WMT', 'TMO', 'ABT', 'MCD', 'ORCL', 'CSCO', 'ACN', 'AMD',
    'LIN', 'PM', 'NKE', 'NEE', 'DHR', 'TXN', 'VZ', 'INTC', 'AMGN',
    'RTX', 'BMY', 'HON', 'QCOM', 'IBM',
]

CHINA_TICKERS: list[str] = [
    '600519.SS', '601318.SS', '600036.SS', '601166.SS', '601288.SS',
    '601398.SS', '600900.SS', '601988.SS', '600276.SS', '600887.SS',
    '600028.SS', '601628.SS', '601601.SS', '600016.SS', '601336.SS',
    '601688.SS', '600050.SS', '600030.SS', '601881.SS', '601328.SS',
    '600309.SS', '600085.SS', '603288.SS', '600763.SS', '601012.SS',
    '600196.SS', '300059.SZ', '000858.SZ', '000333.SZ', '002415.SZ',
    '000651.SZ', '000725.SZ', '002594.SZ', '300750.SZ', '002352.SZ',
    '300760.SZ', '002304.SZ', '000568.SZ', '002142.SZ', '300015.SZ',
    '002714.SZ', '002507.SZ', '000002.SZ', '000001.SZ', '600000.SS',
    '600009.SS', '600010.SS', '600011.SS', '600012.SS', '600015.SS',
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_data_dir(path: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _fetch_via_massive(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Attempt to fetch OHLCV from the massive REST API using list_aggs.

    Returns a tidy DataFrame with MultiIndex (date, ticker) and columns
    [open, high, low, close, volume], or raises RuntimeError on failure.
    """
    token = os.getenv('MASSIVE_TOKEN')
    if not token:
        raise RuntimeError('MASSIVE_TOKEN env var not set')

    from massive import RESTClient  # type: ignore[import]

    client = RESTClient(api_key=token)
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        try:
            rows = []
            for agg in client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan='day',
                from_=start,
                to=end,
                limit=50000,
            ):
                rows.append({
                    'date':   pd.Timestamp(agg.timestamp, unit='ms', tz='UTC').normalize().tz_localize(None),
                    'open':   agg.open,
                    'high':   agg.high,
                    'low':    agg.low,
                    'close':  agg.close,
                    'volume': agg.volume,
                })
            if rows:
                df = pd.DataFrame(rows)
                df['ticker'] = ticker
                frames.append(df)
        except Exception as exc:
            logger.warning('massive: failed for %s – %s', ticker, exc)

    if not frames:
        raise RuntimeError('massive returned no data for any ticker')

    combined = pd.concat(frames, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])
    combined = combined.set_index(['date', 'ticker'])
    combined = combined[['open', 'high', 'low', 'close', 'volume']]
    combined = combined.sort_index()
    return combined


def _fetch_via_yfinance(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance via yfinance.

    Downloads all tickers in a single batch call, returns MultiIndex
    (date, ticker) DataFrame with columns [open, high, low, close, volume].
    """
    try:
        import yfinance as yf  # type: ignore[import]
    except ModuleNotFoundError as exc:
        raise RuntimeError('yfinance is not installed') from exc

    logger.info('yfinance: downloading %d tickers …', len(tickers))
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise RuntimeError('yfinance returned empty DataFrame')

    # yfinance returns a wide MultiIndex columns DataFrame when >1 ticker
    # Level-0: price field  (Open / High / Low / Close / Volume)
    # Level-1: ticker symbol
    col_map = {'Open': 'open', 'High': 'high', 'Low': 'low',
               'Close': 'close', 'Volume': 'volume'}

    if isinstance(raw.columns, pd.MultiIndex):
        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            try:
                sub = raw.xs(ticker, axis=1, level=1)[list(col_map.keys())]
            except KeyError:
                logger.warning('yfinance: no data for %s', ticker)
                continue
            sub = sub.rename(columns=col_map)
            sub.index.name = 'date'
            sub['ticker'] = ticker
            sub = sub.dropna(subset=['close'])
            frames.append(sub)
    else:
        # Single ticker edge-case (should not happen given our universe sizes)
        raw = raw[list(col_map.keys())].rename(columns=col_map)
        raw.index.name = 'date'
        raw['ticker'] = tickers[0]
        raw = raw.dropna(subset=['close'])
        frames = [raw]

    if not frames:
        raise RuntimeError('yfinance: all tickers returned empty data')

    combined = pd.concat(frames)
    combined.index = pd.to_datetime(combined.index)
    combined = combined.reset_index().set_index(['date', 'ticker'])
    combined = combined[['open', 'high', 'low', 'close', 'volume']]
    combined = combined.sort_index()
    return combined


def _download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download OHLCV: try massive first, fall back to yfinance."""
    try:
        logger.info('Trying massive API …')
        return _fetch_via_massive(tickers, start, end)
    except Exception as exc:
        logger.warning('massive API unavailable (%s). Falling back to yfinance.', exc)
        return _fetch_via_yfinance(tickers, start, end)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_us_data() -> pd.DataFrame:
    """Download S&P 500 representative daily OHLCV for 2010-01-04 to 2024-12-31.

    Returns
    -------
    pd.DataFrame
        MultiIndex (date, ticker), columns [open, high, low, close, volume].
    """
    logger.info('Downloading US market data (%d tickers) …', len(US_TICKERS))
    df = _download(US_TICKERS, START_DATE, END_DATE)
    path = _ensure_data_dir(US_DATA_PATH)
    df.to_parquet(path)
    logger.info('US data saved → %s  shape=%s', path, df.shape)
    return df


def download_china_data() -> pd.DataFrame:
    """Download CSI 300 representative daily OHLCV for 2010-01-04 to 2024-12-31.

    Returns
    -------
    pd.DataFrame
        MultiIndex (date, ticker), columns [open, high, low, close, volume].
    """
    logger.info('Downloading China market data (%d tickers) …', len(CHINA_TICKERS))
    df = _download(CHINA_TICKERS, START_DATE, END_DATE)
    path = _ensure_data_dir(CHINA_DATA_PATH)
    df.to_parquet(path)
    logger.info('China data saved → %s  shape=%s', path, df.shape)
    return df


def load_market_data(market: str, force_download: bool = False) -> pd.DataFrame:
    """Load OHLCV panel for *market*, downloading if necessary.

    Parameters
    ----------
    market:
        ``'US'`` or ``'China'``.
    force_download:
        If ``True``, re-download even when the parquet cache exists.

    Returns
    -------
    pd.DataFrame
        MultiIndex (date, ticker), columns [open, high, low, close, volume].
        ``date`` is a ``pd.DatetimeIndex``; all OHLCV values are ``float64``
        except volume which may be ``int64``.

    Raises
    ------
    ValueError
        If *market* is not ``'US'`` or ``'China'``.
    """
    if market not in ('US', 'China'):
        raise ValueError(f"market must be 'US' or 'China', got {market!r}")

    data_path = US_DATA_PATH if market == 'US' else CHINA_DATA_PATH
    path = Path(data_path)

    if path.exists() and not force_download:
        logger.info('Loading cached %s data from %s …', market, path)
        df = pd.read_parquet(path)
        df.index.names = ['date', 'ticker']
        return df

    if market == 'US':
        return download_us_data()
    return download_china_data()
