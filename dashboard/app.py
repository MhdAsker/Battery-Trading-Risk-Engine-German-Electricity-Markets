"""Streamlit explorer for saved forecast and dispatch artifacts."""

from pathlib import Path

import pandas as pd
import streamlit as st

from bess_trading.visualization.dashboard_data import (
    cumulative_revenue,
    select_strategy,
    strategies,
    strategy_revenue,
)

st.set_page_config(page_title="German BESS Trading", layout="wide")
st.title("German BESS Trading & Risk Engine")
page = st.sidebar.radio(
    "Page",
    [
        "Market Overview",
        "Forecast Explorer",
        "Battery Dispatch",
        "Strategy Comparison",
        "Revenue & Risk",
        "Sensitivity Analysis",
    ],
)
phase1_artifact = Path("results/phase1_2024/dispatch.parquet")
default_artifact = (
    phase1_artifact
    if phase1_artifact.exists()
    else Path("results/energy_benchmark_2024/dispatch.parquet")
)
artifact = Path(st.sidebar.text_input("Artifact parquet", str(default_artifact)))

if not artifact.exists():
    st.info(
        "No experiment artifact found. Run a backtest and save its dispatch parquet, "
        "or enter a path in the sidebar."
    )
    st.stop()

data = pd.read_parquet(artifact)
if not isinstance(data.index, pd.DatetimeIndex):
    st.error("Artifact must use a DatetimeIndex.")
    st.stop()
strategy = st.sidebar.selectbox("Strategy", strategies(data))
selected_data = select_strategy(data, strategy)
days = (
    selected_data.index.tz_convert("Europe/Berlin").date
    if selected_data.index.tz is not None
    else selected_data.index.date
)
selected = st.sidebar.selectbox("Delivery day", sorted(set(days)))
day = selected_data[pd.Series(days == selected, index=selected_data.index)]

if page == "Market Overview":
    cols = [
        c
        for c in ["realized_price_eur_mwh", "forecast_price_eur_mwh", "p10", "p50", "p90"]
        if c in day
    ]
    st.line_chart(day[cols])
    if "realized_price_eur_mwh" in selected_data:
        st.metric(
            "Negative-price intervals",
            int((selected_data.realized_price_eur_mwh < 0).sum()),
        )
elif page == "Forecast Explorer":
    st.line_chart(
        day[
            [
                c
                for c in ["realized_price_eur_mwh", "p10", "p50", "p90", "forecast_price_eur_mwh"]
                if c in day
            ]
        ]
    )
elif page == "Battery Dispatch":
    st.line_chart(day[[c for c in ["charge_mw", "discharge_mw", "soc_mwh"] if c in day]])
elif page == "Strategy Comparison":
    if "strategy" in data and "net_revenue_eur" in data:
        st.bar_chart(strategy_revenue(data))
    else:
        st.info("A multi-strategy artifact requires `strategy` and `net_revenue_eur` columns.")
elif page == "Revenue & Risk":
    if "net_revenue_eur" in data:
        st.line_chart(cumulative_revenue(selected_data))
        st.metric("Net revenue", f"€{selected_data.net_revenue_eur.sum():,.0f}")
    else:
        st.info("This artifact has not been settled against realized prices.")
else:
    if "scenario" in data and "net_revenue_eur" in data:
        st.bar_chart(data.groupby("scenario").net_revenue_eur.sum())
    else:
        st.info("Sensitivity artifacts require `scenario` and `net_revenue_eur` columns.")

st.dataframe(day)
