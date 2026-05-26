"""
config.py – Global constants and hyperparameter dicts for the decision-time
leakage benchmark.
"""

# ---------------------------------------------------------------------------
# Experimental grid
# ---------------------------------------------------------------------------

MARKETS = ['US', 'China']

MODELS = ['Momentum', 'Ridge', 'LightGBM', 'GraphMLP', 'LSTM', 'Transformer']

HORIZONS = [5, 20]

PROTOCOLS = ['CLEAN', 'TEMP_CENTER', 'NORM_GLOBAL', 'STRUCT_GRAPH', 'EXEC_CLOSE', 'EXEC_OPEN']

# Walk-forward test years
TEST_YEARS = list(range(2016, 2025))  # 2016-2024 inclusive

# Transaction-cost grid (basis points)
COSTS_BPS = [0, 5, 10, 25, 50]

# ---------------------------------------------------------------------------
# Feature-engineering parameters
# ---------------------------------------------------------------------------

# TEMP_CENTER look-ahead shift in days
FUTURE_DAYS = 3

# Return windows used in base features
RET_WINDOWS = [5, 10, 20]

# Volatility / range windows
VOL_WINDOW = 20          # slow vol – also used for vol_ratio denominator
HL_RANGE_WINDOW = 5

# Momentum look-back
MOM_LOOKBACK = 60

# Sequence model look-back (LSTM / Transformer)
LOOKBACK = 60

# ---------------------------------------------------------------------------
# Graph parameters
# ---------------------------------------------------------------------------

GRAPH_TRAILING_DAYS = 252   # look-back for CLEAN graph
GRAPH_TOP_PEERS = 5         # keep top-k correlations per row

# Feature columns averaged over graph neighbours
RELATION_FEATURES = ['ret_5', 'ret_20', 'vol_ratio_20', 'hl_range_5_mean']

# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

TOP_DECILE = 0.1   # fraction selected long / short

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

US_DATA_PATH = 'data/us_ohlcv.parquet'
CHINA_DATA_PATH = 'data/china_ohlcv.parquet'

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

RESULTS_DIR = 'results'

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------

LGBM_PARAMS: dict = {
    # Paper-specified hyperparameters (methodology section)
    'num_leaves': 31,
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 150,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1,
}

LSTM_PARAMS: dict = {
    # Paper-specified hyperparameters
    'lookback': LOOKBACK,        # 60
    'hidden_size': 32,
    'num_layers': 1,
    'dropout': 0.0,
    'lr': 1e-3,
    'batch_size': 512,
    'epochs': 1,
    'max_train_samples': 12_000,
    'max_val_samples': 3_000,
}

TRANSFORMER_PARAMS: dict = {
    # Paper-specified hyperparameters
    'lookback': LOOKBACK,        # 60
    'd_model': 32,
    'nhead': 4,
    'num_layers': 1,
    'dim_feedforward': 64,
    'dropout': 0.1,
    'lr': 1e-3,
    'batch_size': 512,
    'epochs': 1,
    'max_train_samples': 12_000,
    'max_val_samples': 3_000,
}

GRAPH_MLP_PARAMS: dict = {
    # Paper-specified hyperparameters
    # Base-feature branch:   in → 32 → 32
    'base_hidden': 32,
    'base_out': 32,
    # Relation-feature branch: in → 32 → 32
    'rel_hidden': 32,
    'rel_out': 32,
    # Prediction head:       64 → 32 → 16 → 1
    'head_hidden': 16,
    'lr': 1e-3,
    'batch_size': 1024,
    'epochs': 1,
}
