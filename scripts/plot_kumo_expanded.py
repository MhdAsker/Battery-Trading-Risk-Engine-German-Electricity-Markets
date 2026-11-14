"""Render the frozen 24-origin Kumo benchmark figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results/kumo_development_24")
ASSETS = Path("docs/assets")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    daily = pd.read_csv(RESULTS / "daily_comparison.csv", parse_dates=["delivery_day"])
    forecasts = pd.read_parquet(RESULTS / "forecasts.parquet")
    dispatch = pd.read_parquet(RESULTS / "dispatch.parquet")

    methods = [
        "persistence",
        "hour_of_week",
        "linear",
        "quantile_gbm",
        "kumo_direct",
        "regime_kumo",
    ]
    labels = ["Persistence", "Hour/week", "Linear", "Quantile GBM", "Kumo", "Regime Kumo"]
    colors = ["#94a3b8", "#94a3b8", "#7c3aed", "#c026d3", "#2563eb", "#0f766e"]
    maes = [metrics["forecast_metrics"][name]["mae_eur_mwh"] for name in methods]
    revenues = [metrics["strategy_metrics"][name]["net_revenue_eur"] for name in methods]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(labels, maes, color=colors)
    axes[0].set_ylabel("MAE (EUR/MWh; lower is better)")
    axes[0].set_title("Point-forecast accuracy")
    axes[1].bar(labels, revenues, color=colors)
    axes[1].set_ylabel("Net revenue (EUR; 24 days)")
    axes[1].set_title("Degradation-adjusted battery value")
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Frozen 24-origin Kumo development benchmark")
    fig.tight_layout()
    fig.savefig(ASSETS / "kumo_expanded_summary.png", dpi=170)
    plt.close(fig)

    high_risk = daily.issue_time_regimes.str.contains("negative_price_risk|high_volatility")
    colors_by_day = np.where(high_risk, "#0f766e", "#94a3b8")
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].bar(daily.delivery_day, daily.mae_improvement_eur_mwh, color=colors_by_day)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Direct − retrieval MAE\n(EUR/MWh)")
    axes[1].bar(daily.delivery_day, daily.revenue_improvement_eur, color=colors_by_day)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Retrieval − direct\nnet revenue (EUR)")
    axes[1].tick_params(axis="x", rotation=60)
    fig.suptitle("Retrieval helps mainly in issue-time risk regimes (green)")
    fig.tight_layout()
    fig.savefig(ASSETS / "kumo_expanded_daily_differences.png", dpi=170)
    plt.close(fig)

    selected_days = [pd.Timestamp("2023-05-30"), pd.Timestamp("2023-11-07")]
    actual = (
        dispatch.loc[dispatch.strategy == "kumo_direct", "realized_price_eur_mwh"]
        .groupby(level=0)
        .first()
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for axis, day in zip(axes, selected_days, strict=True):
        local = forecasts.index.tz_convert("Europe/Berlin")
        mask = local.date == day.date()
        x = local[mask]
        axis.plot(
            x,
            actual.reindex(forecasts.index[mask]),
            color="black",
            linewidth=2,
            label="Actual",
        )
        axis.plot(x, forecasts.loc[mask, "kumo_direct"], label="Kumo direct")
        axis.plot(x, forecasts.loc[mask, "regime_kumo"], label="Regime Kumo")
        axis.axhline(0, color="grey", linewidth=0.7)
        axis.set_title(str(day.date()))
        axis.set_ylabel("EUR/MWh")
        axis.tick_params(axis="x", rotation=30)
    handles, plot_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, plot_labels, loc="lower center", ncol=3)
    fig.suptitle("Retrieval success case and failure case")
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(ASSETS / "kumo_expanded_cases.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
