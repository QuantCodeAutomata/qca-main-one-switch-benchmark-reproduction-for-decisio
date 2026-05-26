#!/usr/bin/env python3
"""Experiment 3: Transaction-cost sensitivity for Ridge h=5.

Runs the Ridge model with h=5 for CLEAN, TEMP_CENTER, and EXEC_OPEN protocols
across costs ∈ {0, 5, 10, 25, 50} bps.  Reports how the annualised Sharpe ratio
and leakage gain degrade as costs increase.

Outputs
-------
results/exp3_cost_sensitivity.csv   – cost-sensitivity table
results/exp3_cost_sensitivity.png   – line plot SR vs cost, one line per protocol
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import MARKETS, TEST_YEARS
from src.runner import run_benchmark_grid

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = "results"
COST_LEVELS = [0, 5, 10, 25, 50]
PROTOCOLS = ["CLEAN", "TEMP_CENTER", "EXEC_OPEN"]
MODEL = "Ridge"
HORIZON = 5
PROTOCOL_COLORS = {
    "CLEAN":       "steelblue",
    "TEMP_CENTER": "darkorange",
    "EXEC_OPEN":   "firebrick",
}
PROTOCOL_MARKERS = {
    "CLEAN":       "o",
    "TEMP_CENTER": "s",
    "EXEC_OPEN":   "^",
}


# ---------------------------------------------------------------------------
# Load or compute results
# ---------------------------------------------------------------------------

def load_or_run(save_dir: str = RESULTS_DIR) -> pd.DataFrame:
    cache_path = os.path.join(save_dir, "exp3_raw_results.parquet")
    if os.path.exists(cache_path):
        print(f"Loading cached Exp-3 results from {cache_path}")
        return pd.read_parquet(cache_path)

    # Use a dedicated subdirectory so we never overwrite exp_1's benchmark_results.parquet
    exp3_tmp = os.path.join(save_dir, "exp3_tmp")
    os.makedirs(exp3_tmp, exist_ok=True)
    print("Running Ridge h=5 cost-sensitivity experiment …")
    df = run_benchmark_grid(
        markets=MARKETS,
        models=[MODEL],
        horizons=[HORIZON],
        protocols=PROTOCOLS,
        costs_bps=COST_LEVELS,
        save_dir=exp3_tmp,
    )

    if not df.empty:
        df.to_parquet(cache_path, index=False)

    return df


# ---------------------------------------------------------------------------
# Build cost-sensitivity table
# ---------------------------------------------------------------------------

def build_cost_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean SR@{c}bps and LG-SR@{c}bps for each (market, cost, protocol).

    Returns
    -------
    pd.DataFrame with columns:
        market, protocol, cost_bps, mean_SR, LG_SR
    """
    rows = []
    for market in df["market"].unique():
        mdf = df[df["market"] == market]
        for cost in COST_LEVELS:
            sr_col = f"SR@{cost}bps"
            if sr_col not in mdf.columns:
                continue
            for protocol in PROTOCOLS:
                pdf = mdf[mdf["protocol"] == protocol]
                if pdf.empty:
                    continue
                mean_sr = float(pdf[sr_col].mean(skipna=True))
                rows.append({
                    "market":   market,
                    "protocol": protocol,
                    "cost_bps": cost,
                    "mean_SR":  mean_sr,
                })

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)

    # Add LG vs CLEAN
    clean_sr = (
        table[table["protocol"] == "CLEAN"]
        .set_index(["market", "cost_bps"])["mean_SR"]
    )
    table = table.merge(
        clean_sr.rename("clean_SR"),
        on=["market", "cost_bps"],
        how="left",
    )
    table["LG_SR"] = table["mean_SR"] - table["clean_SR"]
    table = table.drop(columns=["clean_SR"])

    return table.sort_values(["market", "cost_bps", "protocol"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Line plot: SR vs cost
# ---------------------------------------------------------------------------

def plot_cost_sensitivity(table: pd.DataFrame, save_path: str) -> None:
    markets = sorted(table["market"].unique())
    n_markets = len(markets)
    fig, axes = plt.subplots(1, n_markets, figsize=(7 * n_markets, 5), sharey=False)
    if n_markets == 1:
        axes = [axes]

    for ax, market in zip(axes, markets):
        mdf = table[table["market"] == market]
        for protocol in PROTOCOLS:
            pdf = mdf[mdf["protocol"] == protocol].sort_values("cost_bps")
            if pdf.empty:
                continue
            ax.plot(
                pdf["cost_bps"],
                pdf["mean_SR"],
                label=protocol,
                color=PROTOCOL_COLORS.get(protocol, "gray"),
                marker=PROTOCOL_MARKERS.get(protocol, "o"),
                linewidth=1.8,
                markersize=6,
            )

        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_title(f"{market} Market – Ridge h=5")
        ax.set_xlabel("Transaction Cost (bps)")
        ax.set_ylabel("Mean Annualised Sharpe Ratio")
        ax.set_xticks(COST_LEVELS)
        ax.legend(title="Protocol")
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Transaction-Cost Sensitivity of SR (Ridge, h=5)", fontsize=13
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Cost-sensitivity plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_or_run(RESULTS_DIR)
    if df.empty:
        print("No results available – aborting.")
        return

    table = build_cost_table(df)
    if table.empty:
        print("Cost table is empty – aborting.")
        return

    # Print tables per market
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 – TRANSACTION-COST SENSITIVITY  (Ridge h=5)")
    print("=" * 70)
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 120)

    for market in sorted(table["market"].unique()):
        print(f"\n--- {market} ---")
        mdf = table[table["market"] == market]
        pivot = mdf.pivot_table(
            index="cost_bps", columns="protocol", values=["mean_SR", "LG_SR"]
        )
        print(pivot.to_string())

    # Save
    csv_path = os.path.join(RESULTS_DIR, "exp3_cost_sensitivity.csv")
    table.to_csv(csv_path, index=False)
    print(f"\nCost-sensitivity table saved → {csv_path}")

    # Plot
    plot_path = os.path.join(RESULTS_DIR, "exp3_cost_sensitivity.png")
    plot_cost_sensitivity(table, plot_path)

    # Break-even analysis
    print("\n--- Break-even costs (cost at which SR@cost = 0) ---")
    for market in sorted(table["market"].unique()):
        for protocol in PROTOCOLS:
            sub = table[
                (table["market"] == market) & (table["protocol"] == protocol)
            ].sort_values("cost_bps")
            if sub.empty:
                continue
            # Find first crossing below 0
            positive = sub[sub["mean_SR"] > 0]["cost_bps"]
            if len(positive) == len(sub):
                note = f"> {COST_LEVELS[-1]} bps"
            elif positive.empty:
                note = f"< {COST_LEVELS[0]} bps"
            else:
                note = f"~{int(positive.max())} bps"
            print(f"  {market:8s} {protocol:15s}: {note}")


if __name__ == "__main__":
    main()
