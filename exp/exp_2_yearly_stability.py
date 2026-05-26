#!/usr/bin/env python3
"""Experiment 2: Year-level stability analysis of leakage gain.

For the two dominant leaky variants (TEMP_CENTER and EXEC_OPEN) this script
computes the per-year LG-SR@5bps across all (market, model, horizon)
configurations and produces:
  • A stability summary table with mean LG, positive-year count, 95 % bootstrap
    CI, and a one-sided Wilcoxon signed-rank p-value.
  • A bar plot of yearly LG values for each variant.

Outputs
-------
results/exp2_yearly_stability.csv   – stability summary table
results/exp2_yearly_stability.png   – bar plot of yearly LG
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import HORIZONS, MARKETS, MODELS, PROTOCOLS, TEST_YEARS
from src.metrics import (
    compute_yearly_leakage_gain,
    compute_yearly_sharpe,
    summarize_yearly_lg,
)
from src.runner import run_benchmark_grid

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = "results"
DOMINANT_VARIANTS = ["TEMP_CENTER", "EXEC_OPEN"]
COST_LEVEL = 5
SR_COL = f"SR@{COST_LEVEL}bps"


# ---------------------------------------------------------------------------
# Load or compute benchmark results
# ---------------------------------------------------------------------------

def load_or_run(save_dir: str = RESULTS_DIR) -> pd.DataFrame:
    parquet_path = os.path.join(save_dir, "benchmark_results.parquet")
    if os.path.exists(parquet_path):
        print(f"Loading cached results from {parquet_path}")
        return pd.read_parquet(parquet_path)

    print("No cached results found – running benchmark grid for dominant variants …")
    return run_benchmark_grid(
        markets=MARKETS,
        models=MODELS,
        horizons=HORIZONS,
        protocols=["CLEAN"] + DOMINANT_VARIANTS,
        costs_bps=[0, 5, 10],
        save_dir=save_dir,
    )


# ---------------------------------------------------------------------------
# Stability computation
# ---------------------------------------------------------------------------

def compute_stability(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-configuration stability statistics for each dominant variant.

    For each (market, model, horizon, variant) combination:
    - Extract the 9 yearly SR@5bps values for CLEAN and the variant.
    - Compute yearly LG = SR(variant)_year - SR(CLEAN)_year.
    - Summarise via mean, positive_years, CI, Wilcoxon.

    Returns
    -------
    pd.DataFrame with columns:
        market, model, horizon, variant, mean_lg, positive_years,
        ci_lower, ci_upper, wilcoxon_stat, wilcoxon_pval
    """
    rows = []
    grouped = df.groupby(["market", "model", "horizon", "protocol", "year"])[SR_COL].mean()

    for market in df["market"].unique():
        for model in df["model"].unique():
            for horizon in df["horizon"].unique():
                try:
                    clean_sr = grouped.loc[(market, model, horizon, "CLEAN")]
                except KeyError:
                    continue
                clean_yearly = clean_sr.to_dict() if hasattr(clean_sr, "to_dict") else {clean_sr.name: float(clean_sr)}

                for variant in DOMINANT_VARIANTS:
                    try:
                        var_sr = grouped.loc[(market, model, horizon, variant)]
                    except KeyError:
                        continue
                    var_yearly = var_sr.to_dict() if hasattr(var_sr, "to_dict") else {var_sr.name: float(var_sr)}

                    lg_arr = compute_yearly_leakage_gain(var_yearly, clean_yearly)
                    if len(lg_arr) == 0:
                        continue

                    summary = summarize_yearly_lg(lg_arr)
                    rows.append({
                        "market": market,
                        "model": model,
                        "horizon": horizon,
                        "variant": variant,
                        **summary,
                    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bar plot of yearly LG
# ---------------------------------------------------------------------------

def plot_yearly_lg(df_raw: pd.DataFrame, save_path: str) -> None:
    """Bar plot: x = year, y = mean LG-SR@5bps across configurations."""
    fig, axes = plt.subplots(
        1, len(DOMINANT_VARIANTS), figsize=(6 * len(DOMINANT_VARIANTS), 5), sharey=True
    )
    if len(DOMINANT_VARIANTS) == 1:
        axes = [axes]

    for ax, variant in zip(axes, DOMINANT_VARIANTS):
        # Compute per-year mean LG across all (market, model, horizon)
        grouped = df_raw.groupby(["market", "model", "horizon", "protocol", "year"])[SR_COL].mean()
        yearly_lgs: dict = {}
        for year in sorted(df_raw["year"].unique()):
            clean_vals = []
            var_vals = []
            for market in df_raw["market"].unique():
                for model in df_raw["model"].unique():
                    for horizon in df_raw["horizon"].unique():
                        try:
                            c_sr = float(grouped.loc[(market, model, horizon, "CLEAN", year)])
                            v_sr = float(grouped.loc[(market, model, horizon, variant, year)])
                            clean_vals.append(c_sr)
                            var_vals.append(v_sr)
                        except (KeyError, TypeError):
                            continue
            if clean_vals:
                lg = np.mean(var_vals) - np.mean(clean_vals)
                yearly_lgs[year] = lg

        if not yearly_lgs:
            ax.set_title(f"{variant} – no data")
            continue

        years = sorted(yearly_lgs.keys())
        values = [yearly_lgs[y] for y in years]
        colors = ["steelblue" if v >= 0 else "tomato" for v in values]
        ax.bar(years, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"LG-SR@5bps – {variant}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean LG-SR@5bps")
        ax.set_xticks(years)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Year-Level Stability of Leakage Gain in Sharpe Ratio", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Yearly stability plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_or_run(RESULTS_DIR)
    if df.empty:
        print("No results available – aborting.")
        return

    if "year" not in df.columns:
        print("Results DataFrame missing 'year' column – aborting.")
        return

    stability = compute_stability(df)

    print("\n" + "=" * 70)
    print("EXPERIMENT 2 – YEAR-LEVEL STABILITY OF LEAKAGE GAIN")
    print("=" * 70)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(stability.to_string(index=False))

    stab_path = os.path.join(RESULTS_DIR, "exp2_yearly_stability.csv")
    stability.to_csv(stab_path, index=False)
    print(f"\nStability table saved → {stab_path}")

    plot_path = os.path.join(RESULTS_DIR, "exp2_yearly_stability.png")
    plot_yearly_lg(df, plot_path)

    # High-level summary
    if not stability.empty:
        print("\n--- Mean LG by variant (averaged across all configurations) ---")
        summary = stability.groupby("variant")[["mean_lg", "positive_years"]].mean()
        print(summary.to_string())

        print("\n--- Proportion with Wilcoxon p-value < 0.10 ---")
        sig = stability.groupby("variant")["wilcoxon_pval"].apply(
            lambda s: (s < 0.10).mean()
        )
        print(sig.to_string())


if __name__ == "__main__":
    main()
