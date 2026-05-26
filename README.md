# Decision-Time Leakage Benchmark

A reproducible benchmark that quantifies **decision-time leakage** – the
artificial performance inflation caused by using information that is not yet
available at the moment a trading decision is made – across six protocol
variants, six model families, two forecast horizons, and two equity markets.

---

## Overview

When a practitioner inadvertently uses future information during feature
engineering, normalisation, graph construction, or execution, their backtested
Sharpe ratio (SR) can look deceptively high.  This repository reproduces the
"one-switch" benchmark framework: every experiment swaps a single design choice
from **CLEAN** (no leakage) to a leaky variant and measures the resulting
leakage gain (LG) in annualised SR.

### Three experiments

| # | Name | What it measures |
|---|------|-----------------|
| 1 | Main benchmark | LG-SR@5 bps for all 144 configurations |
| 2 | Year-level stability | Consistency of LG across 2016–2024 |
| 3 | Cost sensitivity | How LG degrades with transaction costs |

---

## Repository Structure

```
project/
├── data/                          # OHLCV parquet files (auto-downloaded)
│   ├── us_ohlcv.parquet
│   └── china_ohlcv.parquet
├── src/                           # Core library modules
│   ├── config.py                  # Global constants and hyperparameters
│   ├── data_loader.py             # Market data loading (massive API / yfinance)
│   ├── features.py                # Protocol-aware feature engineering
│   ├── graph.py                   # Monthly correlation graph construction
│   ├── models.py                  # All six model families
│   ├── backtest.py                # Execution-semantic-aware backtester
│   ├── metrics.py                 # SR, RankIC, AUC, LG, CI, Wilcoxon
│   └── runner.py                  # Walk-forward experiment runner
├── exp/                           # Experiment scripts
│   ├── exp_1_benchmark.py         # Main one-switch benchmark
│   ├── exp_2_yearly_stability.py  # Year-level stability analysis
│   └── exp_3_cost_sensitivity.py  # Transaction-cost sensitivity
├── tests/                         # pytest test suite
│   ├── test_features.py
│   ├── test_backtest.py
│   ├── test_metrics.py
│   └── test_models.py
├── results/                       # Output directory
│   └── RESULTS.md                 # Filled after running experiments
├── README.md
└── requirements.txt
```

---

## Protocols

Six protocol variants are compared against the **CLEAN** baseline:

| Protocol | Description | Leakage source |
|----------|-------------|----------------|
| `CLEAN` | Trailing features, train-only normalisation, trailing graph, open-*t+1* execution | None |
| `TEMP_CENTER` | Rolling windows shifted +3 days forward | Feature engineering |
| `NORM_GLOBAL` | Normalisation uses full-panel statistics | Normalisation |
| `STRUCT_GRAPH` | Graph estimation window centred at the month anchor | Graph construction |
| `EXEC_CLOSE` | Label = log(C_{t+h}/C_t); execution at today's close | Execution semantics |
| `EXEC_OPEN` | Label = log(O_{t+h}/O_t); execution at today's open | Execution semantics |

---

## Models

| Model | Description |
|-------|-------------|
| `Momentum` | Score = first feature column (ret signal), no training |
| `Ridge` | sklearn Ridge regression, α = 1.0 |
| `LightGBM` | LGBMRegressor, num\_leaves=31, max\_depth=6, lr=0.05, n=150 |
| `GraphMLP` | Two-branch MLP with base (10→32→32) and relation (4→32→32) branches |
| `LSTM` | Single-layer LSTM, hidden=32, lookback=60 |
| `Transformer` | Single-layer Transformer encoder, d\_model=32, heads=4, lookback=60 |

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd project

# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

---

## Data Requirements

Market data is downloaded automatically on first run.  Two sources are tried:

1. **massive API** (preferred): requires the environment variable `MASSIVE_TOKEN`
   to be set to a valid API key.

   ```bash
   export MASSIVE_TOKEN="your_api_key_here"
   ```

2. **Yahoo Finance** (fallback, no API key required): used automatically when
   `MASSIVE_TOKEN` is not set or the massive API is unavailable.

Data is cached to `data/us_ohlcv.parquet` and `data/china_ohlcv.parquet` and
reused on subsequent runs.

---

## Usage

### Running experiments

```bash
# Experiment 1 – main benchmark (all 144 configurations)
python exp/exp_1_benchmark.py

# Experiment 2 – year-level stability analysis
python exp/exp_2_yearly_stability.py

# Experiment 3 – transaction-cost sensitivity (Ridge h=5)
python exp/exp_3_cost_sensitivity.py
```

Experiments 2 and 3 reuse cached results from Experiment 1 if the
`results/benchmark_results.parquet` file already exists.

### Running the test suite

```bash
pytest tests/ -v
```

Individual test modules:

```bash
pytest tests/test_features.py -v   # feature engineering tests
pytest tests/test_backtest.py  -v  # backtester tests
pytest tests/test_metrics.py   -v  # metrics tests
pytest tests/test_models.py    -v  # model interface tests
```

### Using the runner programmatically

```python
from src.runner import run_benchmark_grid, run_protocol_configuration
from src.data_loader import load_market_data

# Run a single configuration
ohlcv = load_market_data('US')
result = run_protocol_configuration(
    ohlcv=ohlcv,
    market='US',
    model_name='Ridge',
    horizon=5,
    protocol='TEMP_CENTER',
    test_years=list(range(2016, 2025)),
    costs_bps=[0, 5, 10],
)
print(result['metrics'])

# Run the full grid
results_df = run_benchmark_grid(
    markets=['US'],
    models=['Ridge', 'LightGBM'],
    horizons=[5, 20],
    protocols=['CLEAN', 'TEMP_CENTER', 'EXEC_OPEN'],
    costs_bps=[0, 5, 10],
    save_dir='results',
)
print(results_df.head())
```

---

## Output Files

After running the experiments the following files are created in `results/`:

| File | Contents |
|------|---------|
| `benchmark_results.parquet` | Full per-year metric panel (all configurations) |
| `benchmark_results.csv` | Same as above in CSV format |
| `exp1_benchmark_table.csv` | Mean SR@5 bps and LG-SR@5 bps per protocol |
| `exp1_leakage_heatmap.png` | Heatmap of LG-SR@5 bps across configurations |
| `exp1_metrics.csv` | Full per-year metrics for Experiment 1 |
| `exp2_yearly_stability.csv` | Stability statistics (mean LG, CI, Wilcoxon) |
| `exp2_yearly_stability.png` | Bar plot of yearly LG for dominant variants |
| `exp3_cost_sensitivity.csv` | SR and LG vs cost for Ridge h=5 |
| `exp3_cost_sensitivity.png` | Line plot of SR vs cost per protocol |

---

## Expected Results

### Experiment 1 (main benchmark)

The leakage variants are expected to show materially higher backtested SR
compared to the CLEAN baseline.  Typical LG magnitudes:

- **TEMP_CENTER**: moderate positive LG (feature look-ahead)
- **NORM_GLOBAL**: small positive LG (distributional look-ahead)
- **STRUCT_GRAPH**: small-to-moderate positive LG (graph construction)
- **EXEC_CLOSE**: large positive LG (close-to-close return alignment)
- **EXEC_OPEN**: largest positive LG (same-day open execution)

### Experiment 2 (year-level stability)

LG should be positive in the majority (>5/9) of years for TEMP_CENTER and
EXEC_OPEN, with Wilcoxon one-sided p-values below 0.10 for most
(market, model, horizon) combinations.

### Experiment 3 (cost sensitivity)

The leakage gain shrinks as costs increase.  At 25–50 bps the advantage of
EXEC_OPEN disappears or reverses in realistic trading environments.

---

## Citation

If you use this benchmark, please cite the original paper that introduced the
decision-time leakage framework.

---

## License

MIT License – see `LICENSE` for details.
