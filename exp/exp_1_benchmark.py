#!/usr/bin/env python3
"""Experiment 1: Main one-switch benchmark reproduction.

Runs all 144 (2 markets × 6 models × 2 horizons × 6 protocols) configurations
for costs ∈ {0, 5, 10} bps and computes the leakage-gain (LG) in annualised
Sharpe ratio at 5 bps versus the CLEAN baseline.

Outputs
-------
results/exp1_benchmark_table.csv   – LG-SR@5bps table (model × variant)
results/exp1_leakage_heatmap.png   – heatmap of LG-SR@5bps
results/exp1_metrics.csv           – full per-year metric panel
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import COSTS_BPS, HORIZONS, MARKETS, MODELS, PROTOCOLS
from src.runner import run_benchmark_grid

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = "results"
LEAKY_PROTOCOLS = ["TEMP_CENTER", "NORM_GLOBAL", "STRUCT_GRAPH", "EXEC_CLOSE", "EXEC_OPEN"]
COST_LEVEL = 5  # reference cost level for main table
SR_COL = f"SR@{COST_LEVEL}bps"


# ---------------------------------------------------------------------------
# Load or compute benchmark results
# ---------------------------------------------------------------------------

def load_or_run(save_dir: str = RESULTS_DIR) -> pd.DataFrame:
    parquet_path = os.path.join(save_dir, "benchmark_results.parquet")
    if os.path.exists(parquet_path):
        print(f"Loading cached results from {parquet_path}")
        return pd.read_parquet(parquet_path)

    print("Running benchmark grid (this may take a while) …")
    return run_benchmark_grid(
        markets=MARKETS,
        models=MODELS,
        horizons=HORIZONS,
        protocols=PROTOCOLS,
        costs_bps=[0, 5, 10],
        save_dir=save_dir,
    )


# ---------------------------------------------------------------------------
# Leakage-gain computation
# ---------------------------------------------------------------------------

def compute_lg_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean SR@5bps and LG-SR@5bps for every protocol vs CLEAN.

    Returns a DataFrame indexed by (market, model, horizon) with columns
    CLEAN_SR, {variant}_SR, and LG_{variant} for each leaky protocol.
    """
    # Average across test years
    agg = (
        df.groupby(["market", "model", "horizon", "protocol"])[SR_COL]
        .mean()
        .reset_index()
    )
    pivot = agg.pivot_table(
        index=["market", "model", "horizon"],
        columns="protocol",
        values=SR_COL,
    )

    table_cols = {"CLEAN_SR": pivot.get("CLEAN", np.nan)}
    for variant in LEAKY_PROTOCOLS:
        if variant in pivot.columns:
            table_cols[f"{variant}_SR"] = pivot[variant]
            table_cols[f"LG_{variant}"] = pivot[variant] - pivot.get("CLEAN", np.nan)

    return pd.DataFrame(table_cols)


# ---------------------------------------------------------------------------
# Visualisation – leakage heatmap
# ---------------------------------------------------------------------------

def plot_leakage_heatmap(table: pd.DataFrame, save_path: str) -> None:
    lg_cols = [c for c in table.columns if c.startswith("LG_")]
    if not lg_cols:
        print("No LG columns found – skipping heatmap.")
        return

    # Build a single heatmap matrix: rows = (market, model, horizon), cols = variant
    heatmap_data = table[lg_cols].copy()
    heatmap_data.columns = [c.replace("LG_", "") for c in heatmap_data.columns]
    heatmap_data.index = [
        f"{m} | {mod} | h{h}"
        for m, mod, h in heatmap_data.index
    ]

    fig_height = max(6, int(len(heatmap_data) * 0.4))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    sns.heatmap(
        heatmap_data.astype(float),
        ax=ax,
        center=0,
        cmap="RdYlGn",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "LG-SR@5bps"},
    )
    ax.set_title("Leakage Gain in Annualised Sharpe Ratio (vs CLEAN, @5 bps)")
    ax.set_xlabel("Protocol Variant")
    ax.set_ylabel("Market | Model | Horizon")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Heatmap saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_or_run(RESULTS_DIR)
    if df.empty:
        print("No results available – aborting.")
        return

    # Save full per-year metrics
    metrics_path = os.path.join(RESULTS_DIR, "exp1_metrics.csv")
    df.to_csv(metrics_path, index=False)
    print(f"Full metrics saved → {metrics_path}")

    # Build benchmark table
    table = compute_lg_table(df)

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 1 – MAIN ONE-SWITCH BENCHMARK  (SR@5bps & LG-SR@5bps)")
    print("=" * 70)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(table.to_string())

    # Save table
    table_path = os.path.join(RESULTS_DIR, "exp1_benchmark_table.csv")
    table.to_csv(table_path)
    print(f"\nBenchmark table saved → {table_path}")

    # Heatmap
    heatmap_path = os.path.join(RESULTS_DIR, "exp1_leakage_heatmap.png")
    plot_leakage_heatmap(table, heatmap_path)

    # Summary statistics across all configurations
    lg_cols = [c for c in table.columns if c.startswith("LG_")]
    if lg_cols:
        print("\n--- Mean LG-SR@5bps per variant (averaged over all configurations) ---")
        mean_lg = table[lg_cols].mean().rename(
            lambda c: c.replace("LG_", "")
        )
        print(mean_lg.to_string())

        print("\n--- Fraction of configurations with LG > 0 ---")
        frac_pos = (table[lg_cols] > 0).mean().rename(
            lambda c: c.replace("LG_", "")
        )
        print(frac_pos.to_string())


if __name__ == "__main__":
    main()
