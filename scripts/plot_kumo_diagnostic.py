"""Render the committed Kumo development diagnostic figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("results/kumo_development_diagnostic")
ASSETS = Path("docs/assets")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    forecasts = pd.read_parquet(RESULTS / "forecasts.parquet")
    dispatch = pd.read_parquet(RESULTS / "dispatch.parquet")
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))

    actual = (
        dispatch.loc[dispatch.strategy == "kumo_direct", "realized_price_eur_mwh"]
        .groupby(level=0)
        .first()
        .reindex(forecasts.index)
    )
    local_days = forecasts.index.tz_convert("Europe/Berlin").date
    days = list(dict.fromkeys(local_days))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=False)
    for axis, day in zip(axes.flat, days, strict=True):
        mask = local_days == day
        x = forecasts.index[mask].tz_convert("Europe/Berlin")
        axis.plot(x, actual.loc[mask], color="black", linewidth=2.2, label="Actual")
        axis.plot(x, forecasts.loc[mask, "kumo_direct"], label="Kumo direct")
        axis.plot(x, forecasts.loc[mask, "regime_kumo"], label="Regime retrieval")
        axis.plot(x, forecasts.loc[mask, "persistence"], linestyle="--", label="Persistence")
        axis.axhline(0, color="grey", linewidth=0.7)
        axis.set_title(str(day))
        axis.set_ylabel("EUR/MWh")
        axis.tick_params(axis="x", rotation=25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle("Kumo development diagnostic: forecasts and realized prices")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(ASSETS / "kumo_development_forecasts.png", dpi=170)
    plt.close(fig)

    methods = ["persistence", "hour_of_week", "kumo_direct", "regime_kumo"]
    maes = [metrics["forecast_metrics"][name]["mae_eur_mwh"] for name in methods]
    revenues = [metrics["strategy_metrics"][name]["net_revenue_eur"] for name in methods]
    labels = ["Persistence", "Hour of week", "Kumo direct", "Regime Kumo"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].bar(labels, maes, color=["#94a3b8", "#94a3b8", "#2563eb", "#0f766e"])
    axes[0].set_ylabel("MAE (EUR/MWh; lower is better)")
    axes[0].set_title("Point-forecast accuracy")
    axes[1].bar(labels, revenues, color=["#94a3b8", "#94a3b8", "#2563eb", "#0f766e"])
    axes[1].set_ylabel("Net revenue (EUR; four days)")
    axes[1].set_title("Degradation-adjusted dispatch value")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Preliminary diagnostic only — n=4 development days")
    fig.tight_layout()
    fig.savefig(ASSETS / "kumo_development_summary.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
