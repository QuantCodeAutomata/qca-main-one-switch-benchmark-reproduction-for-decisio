# Repository: Decision-Time Leakage Benchmark

## Purpose
Reproduce the paper's core benchmark for decision-time leakage in financial backtests.
Three experiments:
- **exp_1**: Main one-switch benchmark (144 config cells: 2 markets × 6 models × 2 horizons × 6 protocols)
- **exp_2**: Year-level stability analysis (2016–2024 walk-forward)
- **exp_3**: Transaction-cost sensitivity for Ridge h=5

## Structure
```
data/           # OHLCV panels (downloaded via massive API)
src/            # Core modules
  data_loader.py    # Market data loading via massive API
  features.py       # Protocol-aware feature engineering
  graph.py          # Monthly correlation graph construction
  models.py         # All 6 model families
  backtest.py       # Execution-semantic-aware backtester
  metrics.py        # SR, RankIC, AUC, LG computation
  config.py         # Protocol/model/grid configuration
exp/            # Experiment runners
  exp_1_benchmark.py
  exp_2_yearly_stability.py
  exp_3_cost_sensitivity.py
tests/          # pytest tests
results/        # Outputs + RESULTS.md
```

## Key Design Decisions

### Protocols (6 variants)
- **CLEAN**: trailing features, train-only normalization, trailing graph, open-t+1 execution
- **TEMP_CENTER**: future_days=3 shift in rolling stats (R_hat_w[x]_t uses x_{·+3})
- **NORM_GLOBAL**: normalization uses full panel stats (not just train split)
- **STRUCT_GRAPH**: symmetric 252-day window centered at month anchor τ_m
- **EXEC_CLOSE**: label = log(C_{t+h}/C_t), trade return = C_{t+1}/C_t - 1
- **EXEC_OPEN**: label = log(O_{t+h}/O_t), trade return = O_{t+1}/O_t - 1 (same-day leakage)

### Labels
- CLEAN: log(O_{t+h+1} / O_{t+1})
- EXEC_CLOSE: log(C_{t+h} / C_t)
- EXEC_OPEN: log(O_{t+h} / O_t)
- TEMP_CENTER / NORM_GLOBAL / STRUCT_GRAPH: same as CLEAN

### Execution Returns
- CLEAN/TEMP_CENTER/NORM_GLOBAL/STRUCT_GRAPH: O_{t+2}/O_{t+1} - 1
- EXEC_CLOSE: C_{t+1}/C_t - 1
- EXEC_OPEN: O_{t+1}/O_t - 1

### Models
- Momentum: score = ret_20 (no training)
- Ridge: alpha=1.0
- LightGBM: num_leaves=31, max_depth=6, lr=0.05, n_estimators=150
- Graph MLP: base 32→32, relation 32→32, head 32→16→1
- LSTM: lookback=60, hidden=32, layers=1, max_train=12000
- Transformer: lookback=60, d_model=32, heads=4, layers=1

### Walk-forward
- Train/val/test yearly splits for test years 2016–2024
- Train = all years before test year (minus last year = val)

### Libraries Used
- massive: data download (MASSIVE_TOKEN env var)
- sklearn: Ridge, scaling, AUC
- lightgbm: LightGBM regressor
- torch: Graph MLP, LSTM, Transformer
- scipy.stats: Spearman correlation, Wilcoxon test
- numpy/pandas: panel manipulation

## Environment
- MASSIVE_TOKEN: API key for massive data download
- Data saved to data/us_ohlcv.parquet and data/china_ohlcv.parquet

## Implementation Notes

### src/config.py
- All grid constants + 4 hyperparam dicts: `LGBM_PARAMS`, `LSTM_PARAMS`, `TRANSFORMER_PARAMS`, `GRAPH_MLP_PARAMS`
- `FUTURE_DAYS=3`, `LOOKBACK=60`, `GRAPH_TRAILING_DAYS=252`, `GRAPH_TOP_PEERS=5`

### src/data_loader.py
- 50 US tickers (`US_TICKERS`) + 50 China tickers (`CHINA_TICKERS`) hardcoded
- `_fetch_via_massive` tries massive API first; `_fetch_via_yfinance` is fallback
- yfinance 1.3.x: columns are MultiIndex (price_field, ticker) – use `xs(ticker, level=1)`
- `load_market_data(market, force_download)` caches to parquet; parquet preserves datetime index

### src/features.py
- All rolling ops done on wide (dates × tickers) DataFrames, then stacked via `stack(future_stack=True)`
- TEMP_CENTER: shift=-future_days applied to all source series before rolling
- Normalize: `stds.where(stds > 1e-8, other=1.0)` guards near-zero stds (e.g. constant synthetic features)
- Labels: CLEAN=log(O_{t+h+1}/O_{t+1}), EXEC_CLOSE=log(C_{t+h}/C_t), EXEC_OPEN=log(O_{t+h}/O_t)

### src/graph.py
- `get_monthly_anchors`: groups DatetimeIndex by `to_period('M')` and takes first per group
- `build_monthly_graph`: CLEAN uses trailing window `< anchor_date`; STRUCT_GRAPH uses symmetric ±half
- Full-corr matrix fill uses `np.ix_` vectorised indexing (not Python loops)
- `compute_relation_features`: date-by-date loop, `W @ feat_matrix` with NaN→0 fill

### src/models.py
- `_TORCH_AVAILABLE` flag set at import time via `try/except ImportError`; all four private `nn.Module` classes (`_GraphMLPNet`, `_LSTMNet`, `_PositionalEncoding`, `_TransformerNet`) are defined inside that try block so they are only evaluated when torch is present
- PyTorch model constructors raise `ImportError` with `pip install torch` hint when torch is absent
- `prepare_sequences(X, y, lookback, max_samples)` returns the **last** `max_samples` windows (most recent) when capping
- Sequence model `predict` returns a `float('nan')` padded array of shape `(len(X),)` – the first `lookback-1` positions are NaN (insufficient history)
- `GraphMLPModel` `fit`/`predict` receive the horizontally concatenated `[X_base | X_rel]` matrix and split internally at `base_features_dim`
- `get_model(name)` factory handles `momentum`, `ridge`, `lightgbm` (case-insensitive); raises `ValueError` with instructions for `graph_mlp`, `lstm`, `transformer` (which require `n_features` at construction)

### src/backtest.py
- `form_portfolio`: uses `math.ceil(top_decile * len(valid))` for portfolio size after dropping NaN scores.
- `compute_trade_returns`: pivots ohlcv once with `.unstack()` (dates × tickers) before the per-date loop; uses calendar index for O_{t+1}/O_{t+2} look-up.
- `compute_turnover`: one-way turnover = |curr_set - prev_set| / len(curr_set); first date = 1.0.
- `run_backtest`: mutable-default-argument anti-pattern avoided by using `costs_bps=None` with an inner default.

### src/metrics.py
- `bootstrap_ci`: vectorised with `rng.integers(0, n, (n_bootstrap, n))` + `.mean(axis=1)` to avoid slow Python loops.
- `wilcoxon_one_sided`: wraps `scipy.stats.wilcoxon(alternative='greater')`; returns `(nan, nan)` on `ValueError`.
- `compute_auc`: skips dates where binary label is all-0 or all-1 (single class).
- `compute_all_metrics`: max_drawdown uses 5 bps net returns (always present in default costs_bps).
