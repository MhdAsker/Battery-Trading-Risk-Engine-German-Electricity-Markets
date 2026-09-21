"""Create publication-ready README figures from checked-in benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "navy": "#153B5B",
    "blue": "#2389DA",
    "cyan": "#5BC0EB",
    "orange": "#F18F01",
    "green": "#2A9D8F",
    "red": "#D1495B",
    "grey": "#667085",
}
LABELS = {
    "perfect_foresight": "Perfect foresight",
    "persistence": "Persistence",
    "hour_of_week": "Hour of week",
    "ml_p50": "ML P50",
    "quantile_robust": "Quantile robust",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F8FAFC",
            "axes.edgecolor": "#D0D5DD",
            "axes.grid": True,
            "grid.color": "#E4E7EC",
            "grid.linewidth": 0.7,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def strategy_overview(metrics: dict[str, object], output: Path) -> None:
    values = metrics["strategy_metrics"]
    order = [
        "perfect_foresight",
        "persistence",
        "ml_p50",
        "quantile_robust",
        "hour_of_week",
    ]
    revenue = [values[name]["net_revenue_eur"] for name in order]
    colors = [COLORS["grey"], COLORS["orange"], COLORS["blue"], COLORS["green"], COLORS["cyan"]]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.bar([LABELS[name] for name in order], revenue, color=colors, width=0.68)
    ax.bar_label(bars, labels=[f"€{value:,.0f}" for value in revenue], padding=4, fontweight="bold")
    ax.set_ylabel("Net revenue (€)")
    ax.set_title("Battery trading revenue by strategy — November 2024 benchmark")
    ax.set_ylim(0, max(revenue) * 1.18)
    ax.tick_params(axis="x", rotation=12)
    ax.text(
        0.99,
        0.98,
        "1 MW / 2 MWh · 27 complete days · €5/MWh degradation",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=COLORS["grey"],
    )
    fig.tight_layout()
    save(fig, output, "strategy_net_revenue.png")


def forecast_accuracy(metrics: dict[str, object], output: Path) -> None:
    values = metrics["forecast_metrics"]
    names = ["persistence", "hour_of_week", "p50"]
    labels = ["Persistence", "Hour of week", "Quantile GBM P50"]
    x = np.arange(len(names))
    width = 0.34
    mae = [values[name]["mae"] for name in names]
    rmse = [values[name]["rmse"] for name in names]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    mae_bars = ax.bar(x - width / 2, mae, width, label="MAE", color=COLORS["blue"])
    rmse_bars = ax.bar(x + width / 2, rmse, width, label="RMSE", color=COLORS["cyan"])
    ax.bar_label(mae_bars, fmt="%.1f", padding=3)
    ax.bar_label(rmse_bars, fmt="%.1f", padding=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Forecast error (€/MWh)")
    ax.set_title("Out-of-sample day-ahead forecast accuracy")
    ax.set_ylim(0, max(rmse) * 1.2)
    ax.legend()
    fig.tight_layout()
    save(fig, output, "forecast_accuracy.png")


def cumulative_revenue(dispatch: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    palette = {
        "perfect_foresight": COLORS["grey"],
        "persistence": COLORS["orange"],
        "hour_of_week": COLORS["cyan"],
        "ml_p50": COLORS["blue"],
        "quantile_robust": COLORS["green"],
    }
    for strategy, group in dispatch.groupby("strategy"):
        hourly = group.sort_index().net_revenue_eur.cumsum()
        ax.plot(hourly.index, hourly, label=LABELS[strategy], color=palette[strategy], linewidth=2)
    ax.set_title("Cumulative realized net revenue")
    ax.set_ylabel("Cumulative net revenue (€)")
    ax.set_xlabel("Delivery date (UTC)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    save(fig, output, "cumulative_net_revenue.png")


def risk_return(metrics: dict[str, object], output: Path) -> None:
    values = metrics["strategy_metrics"]
    strategies = ["persistence", "hour_of_week", "ml_p50", "quantile_robust"]
    palette = [COLORS["orange"], COLORS["cyan"], COLORS["blue"], COLORS["green"]]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for strategy, color in zip(strategies, palette, strict=True):
        row = values[strategy]
        ax.scatter(
            row["cvar_loss_eur"],
            row["net_revenue_eur"],
            s=70 + row["throughput_mwh"] * 1.8,
            color=color,
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )
        ax.annotate(
            LABELS[strategy],
            (row["cvar_loss_eur"], row["net_revenue_eur"]),
            xytext=(7, 6),
            textcoords="offset points",
        )
    ax.set_title("Risk–return trade-off for realistic strategies")
    ax.set_xlabel("CVaR tail loss (€ per hourly interval; lower is better)")
    ax.set_ylabel("Net revenue (€; higher is better)")
    ax.text(
        0.02,
        0.02,
        "Bubble size represents battery throughput",
        transform=ax.transAxes,
        color=COLORS["grey"],
    )
    fig.tight_layout()
    save(fig, output, "risk_return.png")


def representative_day(dispatch: pd.DataFrame, output: Path) -> None:
    robust = dispatch[dispatch.strategy == "quantile_robust"].sort_index()
    daily_range = robust.realized_price_eur_mwh.groupby(robust.index.floor("D")).agg(
        lambda values: values.max() - values.min()
    )
    selected_day = daily_range.idxmax()
    day = robust[robust.index.floor("D") == selected_day]
    hours = np.arange(len(day))

    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.fill_between(hours, day.p10, day.p90, color=COLORS["cyan"], alpha=0.25, label="P10–P90")
    ax.plot(hours, day.p50, color=COLORS["blue"], linewidth=2, label="P50 forecast")
    ax.plot(
        hours,
        day.realized_price_eur_mwh,
        color=COLORS["navy"],
        linewidth=2.3,
        label="Realized price",
    )
    ax.axhline(0, color=COLORS["grey"], linewidth=0.8)
    ax.set_title(f"Forecast uncertainty on the highest-spread test day — {selected_day:%d %B %Y}")
    ax.set_xlabel("Delivery hour (UTC)")
    ax.set_ylabel("Day-ahead price (€/MWh)")
    ax.set_xticks(np.arange(0, 24, 2))
    ax.legend(ncol=3)
    fig.tight_layout()
    save(fig, output, "forecast_uncertainty_day.png")

    fig, (power_ax, soc_ax) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]}
    )
    net_power = day.discharge_mw - day.charge_mw
    power_ax.bar(
        hours,
        net_power,
        color=np.where(net_power >= 0, COLORS["green"], COLORS["orange"]),
        width=0.8,
    )
    power_ax.axhline(0, color=COLORS["grey"], linewidth=0.8)
    power_ax.set_ylabel("Grid power (MW)")
    power_ax.set_title("Quantile-robust battery dispatch on the highest-spread test day")
    power_ax.text(
        0.01, 0.92, "Discharge + / Charge −", transform=power_ax.transAxes, color=COLORS["grey"]
    )
    soc_ax.plot(hours, day.soc_mwh, color=COLORS["blue"], linewidth=2.5)
    soc_ax.fill_between(hours, 0.2, day.soc_mwh, color=COLORS["blue"], alpha=0.12)
    soc_ax.axhline(0.2, linestyle="--", color=COLORS["grey"], linewidth=1, label="10% SOC limit")
    soc_ax.axhline(1.8, linestyle="--", color=COLORS["grey"], linewidth=1, label="90% SOC limit")
    soc_ax.set_ylim(0, 2.05)
    soc_ax.set_ylabel("State of charge (MWh)")
    soc_ax.set_xlabel("Delivery hour (UTC)")
    soc_ax.set_xticks(np.arange(0, 24, 2))
    soc_ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    save(fig, output, "battery_dispatch_day.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/energy_benchmark_2024"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    configure_style()
    dispatch = pd.read_parquet(args.results / "dispatch.parquet")
    metrics = json.loads((args.results / "metrics.json").read_text(encoding="utf-8"))
    strategy_overview(metrics, args.output)
    forecast_accuracy(metrics, args.output)
    cumulative_revenue(dispatch, args.output)
    risk_return(metrics, args.output)
    representative_day(dispatch, args.output)


if __name__ == "__main__":
    main()
