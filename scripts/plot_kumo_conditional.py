"""Render conditional Kumo development and retrospective-validation figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results/kumo_conditional_113")
ASSETS = Path("docs/assets")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    validation = metrics["validation"]
    daily = pd.read_csv(RESULTS / "validation_daily.csv", parse_dates=["delivery_day"])

    methods = [
        "persistence",
        "linear",
        "quantile_gbm",
        "kumo_direct",
        "regime_kumo",
        "conditional_regime_kumo",
    ]
    labels = ["Persistence", "Linear", "Quantile GBM", "Kumo", "Always-on", "Conditional"]
    colors = ["#94a3b8", "#7c3aed", "#c026d3", "#2563eb", "#0f766e", "#ea580c"]
    maes = [validation["forecast_metrics"][name]["mae_eur_mwh"] for name in methods]
    revenues = [validation["strategy_metrics"][name]["net_revenue_eur"] for name in methods]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(labels, maes, color=colors)
    axes[0].set_ylabel("MAE (EUR/MWh; lower is better)")
    axes[0].set_title("Retrospective-validation forecasts")
    axes[1].bar(labels, revenues, color=colors)
    axes[1].set_ylabel("Net revenue (EUR; 44 origins)")
    axes[1].set_title("Identical-asset battery value")
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Conditional Kumo: 2024 retrospective validation")
    fig.tight_layout()
    fig.savefig(ASSETS / "kumo_conditional_validation_summary.png", dpi=170)
    plt.close(fig)

    conditional_revenue = (
        daily.conditional_regime_kumo_revenue_eur - daily.kumo_direct_revenue_eur
    )
    always_revenue = daily.regime_kumo_revenue_eur - daily.kumo_direct_revenue_eur
    x = np.arange(len(daily))
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].bar(
        x,
        daily.kumo_direct_mae_eur_mwh - daily.conditional_regime_kumo_mae_eur_mwh,
        color=np.where(daily.retrieval_activated, "#ea580c", "#94a3b8"),
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Direct − conditional MAE\n(EUR/MWh)")
    axes[1].plot(x, always_revenue, color="#0f766e", label="Always-on − direct")
    axes[1].plot(x, conditional_revenue, color="#ea580c", label="Conditional − direct")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Daily net-revenue difference (EUR)")
    axes[1].set_xticks(x[::3], daily.delivery_day.dt.strftime("%Y-%m-%d")[::3], rotation=60)
    axes[1].legend()
    fig.suptitle("Conditional activation does not dominate always-on retrieval")
    fig.tight_layout()
    fig.savefig(ASSETS / "kumo_conditional_daily_validation.png", dpi=170)
    plt.close(fig)

    regime = pd.DataFrame(validation["regime_summary"]).T.loc[
        ["high_volatility", "negative_price_risk", "ordinary"]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    regime[["always_mae_improvement_eur_mwh", "conditional_mae_improvement_eur_mwh"]].plot.bar(
        ax=axes[0], color=["#0f766e", "#ea580c"]
    )
    regime[["always_revenue_improvement_eur", "conditional_revenue_improvement_eur"]].plot.bar(
        ax=axes[1], color=["#0f766e", "#ea580c"]
    )
    axes[0].set_ylabel("MAE improvement vs direct (EUR/MWh)")
    axes[1].set_ylabel("Revenue improvement vs direct (EUR/day)")
    for axis in axes:
        axis.axhline(0, color="black", linewidth=0.8)
        axis.tick_params(axis="x", rotation=20)
        axis.legend(["Always-on", "Conditional"])
    fig.suptitle("Validation performance by issuance-time regime")
    fig.tight_layout()
    fig.savefig(ASSETS / "kumo_conditional_regimes.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
