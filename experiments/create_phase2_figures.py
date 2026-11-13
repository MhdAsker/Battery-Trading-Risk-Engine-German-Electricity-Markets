"""Create README-ready Phase 2 result figures from immutable artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("results/phase2_2024")
ASSETS = Path("docs/assets")
ASSETS.mkdir(parents=True, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

frontier = pd.read_csv(RESULTS / "efficient_frontier.csv")
fig, ax = plt.subplots(figsize=(7.2, 4.4))
points = ax.scatter(
    frontier.cvar_loss_eur,
    frontier.expected_revenue_eur,
    c=frontier.risk_aversion,
    s=80,
    cmap="viridis",
)
for row in frontier.itertuples():
    ax.annotate(f"lambda={row.risk_aversion:g}", (row.cvar_loss_eur, row.expected_revenue_eur))
ax.set(xlabel="Scenario CVaR loss (EUR)", ylabel="Expected scenario revenue (EUR)")
fig.colorbar(points, ax=ax, label="Risk aversion")
fig.tight_layout()
fig.savefig(ASSETS / "phase2_efficient_frontier.png", dpi=180)
plt.close(fig)

metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
coverage = metrics["probabilistic"]["conditional_coverage"]
labels = ["Overall", "Negative price", "Price spike", "High volatility"]
values = [
    metrics["probabilistic"]["p10_p90_coverage"],
    coverage["negative_price"]["coverage"],
    coverage["price_spike"]["coverage"],
    coverage["high_volatility"]["coverage"],
]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
bars = ax.bar(labels, values, color=["#2563eb", "#0f766e", "#dc2626", "#7c3aed"])
ax.axhline(0.8, color="black", linestyle="--", label="Nominal 80%")
ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3)
ax.set(ylabel="P10-P90 empirical coverage", ylim=(0, 1), title="Calibration varies by regime")
ax.legend()
fig.tight_layout()
fig.savefig(ASSETS / "phase2_regime_coverage.png", dpi=180)
plt.close(fig)

pnl = pd.read_csv(RESULTS / "daily_pnl.csv", index_col=0)
fig, ax = plt.subplots(figsize=(7.2, 4.4))
pnl.cumsum().plot(ax=ax, color=["#64748b", "#2563eb"])
ax.set(xlabel="German delivery day", ylabel="Cumulative net revenue (EUR)")
ax.legend(["P50 deterministic", "Stochastic CVaR"])
fig.tight_layout()
fig.savefig(ASSETS / "phase2_cumulative_revenue.png", dpi=180)
plt.close(fig)
