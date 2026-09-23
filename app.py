"""
FoodSight Nigeria — Streamlit Dashboard
A causal forecasting system for Nigerian food price volatility.

Three views:
  1. Forecast View — historical prices + 6-month-ahead forecast with confidence band
  2. Driver View   — SHAP-based driver attribution, in plain language
  3. Alert View     — traffic-light indicator against historical spike thresholds

Data expected in ./data/ (see README.md for the file list and where each comes from).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FoodSight Nigeria",
    page_icon="🌾",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"

COMMODITY_LABELS = {
    "Rice_local": "Rice (Local)",
    "Rice_imported": "Rice (Imported)",
    "Gari_white": "Gari (White)",
    "Beans_brown": "Beans (Brown)",
    "Palm_oil": "Palm Oil",
    "Groundnut_oil": "Groundnut Oil",
    "Vegetable_oil": "Vegetable Oil",
    "Yam": "Yam",
    "Maize_white": "Maize (White)",
}
UNIT_LABELS = {
    "Palm_oil": "NGN / litre", "Groundnut_oil": "NGN / litre", "Vegetable_oil": "NGN / litre",
}


# ---------------------------------------------------------------------------
# Data loading (cached — reload only when files change)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    merged = pd.read_csv(DATA_DIR / "FoodSight_national_merged.csv", parse_dates=["date"])
    forecasts = pd.read_csv(DATA_DIR / "future_forecasts_6mo.csv", parse_dates=["date"])
    mape_summary = pd.read_csv(DATA_DIR / "all_commodities_mape_summary.csv", index_col=0)
    shap_pct = pd.read_csv(DATA_DIR / "all_commodities_shap_pct.csv", index_col=0)
    thresholds = pd.read_csv(DATA_DIR / "alert_thresholds.csv", index_col=0)
    break_summary = pd.read_csv(DATA_DIR / "structural_break_summary.csv", index_col=0)
    return merged, forecasts, mape_summary, shap_pct, thresholds, break_summary


try:
    merged, forecasts, mape_summary, shap_pct, thresholds, break_summary = load_data()
    DATA_OK = True
except FileNotFoundError as e:
    DATA_OK = False
    MISSING_FILE = str(e)

COMMODITIES = list(COMMODITY_LABELS.keys())


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🌾 FoodSight Nigeria")
st.sidebar.caption("A causal forecasting system for Nigerian food price volatility")

if not DATA_OK:
    st.error(
        f"Data file not found: {MISSING_FILE}\n\n"
        "Place the required CSVs in the `data/` folder next to this script. "
        "See README.md for the full list and generation source."
    )
    st.stop()

selected_commodity = st.sidebar.selectbox(
    "Select a commodity",
    COMMODITIES,
    format_func=lambda c: COMMODITY_LABELS[c],
)
unit = UNIT_LABELS.get(selected_commodity, "NGN / kg")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About this system**\n\n"
    "Three models are compared under walk-forward validation for each commodity: "
    "ARIMA (univariate baseline), ARIMAX (multivariate extension), and LightGBM "
    "(gradient-boosted ML benchmark). Forecasts shown here use each commodity's "
    "best-performing model by walk-forward MAPE."
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: NBS (food prices, CPI, fuel), CHIRPS (rainfall), Investing.com (exchange rate). "
    "National level, monthly, Jan 2016 – May 2026."
)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title(f"{COMMODITY_LABELS[selected_commodity]}")
best_model = mape_summary.loc[selected_commodity, "best_model"]
best_mape = mape_summary.loc[selected_commodity, best_model]
latest_price = merged[selected_commodity].iloc[-1]
latest_date = merged["date"].iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Latest price", f"₦{latest_price:,.2f}", help=f"As of {latest_date.strftime('%B %Y')}")
col2.metric("Best model", best_model, help="Selected by lowest walk-forward MAPE")
col3.metric("Walk-forward MAPE", f"{best_mape:.2f}%", help="Out-of-sample accuracy, 1-month-ahead")

tab1, tab2, tab3 = st.tabs(["📈 Forecast View", "🔍 Driver View", "🚦 Alert View"])


# ---------------------------------------------------------------------------
# TAB 1 — Forecast View
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Historical Prices and 6-Month Forecast")

    hist = merged[["date", selected_commodity]].rename(columns={selected_commodity: "price"})
    fc = forecasts[forecasts["commodity"] == selected_commodity].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["price"], mode="lines", name="Historical",
        line=dict(color="#1f77b4", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["forecast"], mode="lines+markers", name="Forecast",
        line=dict(color="#d62728", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([fc["date"], fc["date"][::-1]]),
        y=pd.concat([fc["upper_80"], fc["lower_80"][::-1]]),
        fill="toself", fillcolor="rgba(214,39,40,0.15)",
        line=dict(color="rgba(255,255,255,0)"), name="80% interval", showlegend=True,
    ))
    fig.add_vline(x=pd.Timestamp("2020-03-01"), line_dash="dot", line_color="orange",
                  annotation_text="COVID-19", annotation_position="top")
    fig.add_vline(x=pd.Timestamp("2023-05-01"), line_dash="dot", line_color="red",
                  annotation_text="Subsidy removal", annotation_position="top")
    fig.update_layout(
        height=450, hovermode="x unified",
        yaxis_title=f"Price ({unit})", xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=10),
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown(f"**Next 6 months ({fc['date'].min().strftime('%b %Y')} – {fc['date'].max().strftime('%b %Y')})**")
    display_fc = fc[["date", "forecast", "lower_80", "upper_80"]].copy()
    display_fc["date"] = display_fc["date"].dt.strftime("%b %Y")
    display_fc.columns = ["Month", "Forecast (₦)", "Lower (80%)", "Upper (80%)"]
    st.dataframe(display_fc.set_index("Month"), width='stretch')

    with st.expander("Model comparison for this commodity (walk-forward MAPE)"):
        row = mape_summary.loc[[selected_commodity], ["ARIMA", "ARIMAX", "LightGBM"]]
        st.dataframe(row, width='stretch')
        st.caption(
            "Lower is better. ARIMA is the best-performing model for every commodity in this "
            "study — external variables did not improve on the univariate baseline at this "
            "sample size and monthly resolution, a genuine finding under walk-forward evaluation "
            "rather than a modelling shortfall."
        )


# ---------------------------------------------------------------------------
# TAB 2 — Driver View
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("What's Driving This Forecast")
    st.caption(
        "SHAP-based attribution from the LightGBM model, showing each factor's relative "
        "contribution to price movements for this commodity."
    )

    row = shap_pct.loc[selected_commodity].sort_values(ascending=True)
    driver_labels = {
        "own_price_lags": "Recent price momentum",
        "exchange_rate": "Exchange rate (NGN/USD)",
        "all_items_cpi": "General inflation (CPI)",
        "food_cpi": "Food inflation (Food CPI)",
        "fuel_price_ngn_per_litre": "Fuel price",
        "rainfall_anomaly": "Rainfall anomaly",
        "post_subsidy_removal": "Post-subsidy period",
    }
    row.index = [driver_labels.get(i, i) for i in row.index]

    fig2 = go.Figure(go.Bar(
        x=row.values, y=row.index, orientation="h",
        marker_color="#2ca02c",
        text=[f"{v:.1f}%" for v in row.values], textposition="outside",
    ))
    fig2.update_layout(
        height=350, xaxis_title="Share of driver influence (%)",
        margin=dict(t=10, b=10), yaxis=dict(automargin=True),
    )
    st.plotly_chart(fig2, width='stretch')

    top_external = row.drop("Recent price momentum", errors="ignore").idxmax()
    top_external_pct = row.drop("Recent price momentum", errors="ignore").max()
    st.info(
        f"**In plain terms:** for {COMMODITY_LABELS[selected_commodity]}, the single largest "
        f"external factor is **{top_external}**, accounting for roughly **{top_external_pct:.0f}%** "
        f"of the model's driver-based influence. The remainder is explained by the commodity's "
        f"own recent price trend, which typically carries the largest single share overall."
    )

    with st.expander("How to read this"):
        st.markdown(
            "- Bars show each factor's *relative* contribution to the model's predictions for "
            "this commodity, not a causal effect size.\n"
            "- \"Recent price momentum\" captures the tendency for prices to continue their "
            "recent trend — a real and often dominant signal, not a data artefact.\n"
            "- These percentages come from the SHAP values of the LightGBM model specifically; "
            "since ARIMA (a model without engineered features) is the most accurate forecaster "
            "here, treat this view as explaining *what correlates with* price movements, not as "
            "the mechanism behind the forecast shown in the Forecast View."
        )


# ---------------------------------------------------------------------------
# TAB 3 — Alert View
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Price Alert Status")

    t = thresholds.loc[selected_commodity]
    latest = t["latest_price"]
    amber = t["rolling24_amber_p75"]
    red = t["rolling24_red_p90"]

    if latest >= red:
        status, color, emoji = "RED — Above recent spike threshold", "#d62728", "🔴"
    elif latest >= amber:
        status, color, emoji = "AMBER — Approaching recent spike threshold", "#ff7f0e", "🟠"
    else:
        status, color, emoji = "GREEN — Within recent normal range", "#2ca02c", "🟢"

    st.markdown(
        f"<div style='background-color:{color}22; border-left:6px solid {color}; "
        f"padding:1.2rem; border-radius:6px;'>"
        f"<h3 style='margin:0;color:{color};'>{emoji} {status}</h3>"
        f"<p style='margin:0.5rem 0 0 0;'>Current price: <b>₦{latest:,.2f}</b> "
        f"(as of {t['latest_date']})</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown(
        "**Thresholds are based on a rolling 24-month recent window**, not a fixed pre-2023 "
        "baseline — food prices have risen structurally since the May 2023 subsidy removal, so "
        "comparing today's price against pre-2023 levels would flag every commodity as red "
        "permanently and convey no new information. The rolling window instead flags genuine "
        "deviations from the *current* price regime."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Amber threshold (recent 75th pct.)", f"₦{amber:,.2f}")
        st.metric("Red threshold (recent 90th pct.)", f"₦{red:,.2f}")
    with col_b:
        st.metric("Pre-subsidy amber (75th pct., for reference)", f"₦{t['amber_threshold_p75']:,.2f}")
        st.metric("Pre-subsidy red (90th pct., for reference)", f"₦{t['red_threshold_p90']:,.2f}")

    with st.expander("Structural break context for this commodity"):
        if selected_commodity in break_summary.index:
            b = break_summary.loc[selected_commodity]
            st.write(
                f"Pre-subsidy average monthly growth: **{b['pre_mean_pct_change']:.2f}%** "
                f"→ post-subsidy: **{b['post_mean_pct_change']:.2f}%**"
            )
            st.write(
                f"Volatility (std. dev. of monthly growth): **{b['pre_std']:.2f}%** "
                f"→ **{b['post_std']:.2f}%**"
            )
            sig = "statistically significant" if b["significant_at_5pct"] else "not statistically significant at 5%"
            st.write(f"Welch's t-test on the mean shift: **{sig}** (p = {b['p_value']:.4f}).")


st.markdown("---")
st.caption(
    "FoodSight Nigeria — MSc Information Technology project, University of Ilorin. "
    "For research and educational use; not a substitute for official NBS or CBN price data."
)
