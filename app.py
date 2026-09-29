"""
FoodSight Nigeria — Streamlit Dashboard (v2, restructured per the desktop UX audit)

Information architecture:
  Overview      — what is happening (KPIs, forecast, top drivers, price status)
  Forecast      — full chart, chart controls, monthly table
  Drivers       — SHAP-based driver attribution in plain language
  Price Status  — normal / watch / high state against recent-window thresholds
  Methodology   — models, accuracy, thresholds, structural breaks, data sources

Data expected in ./data/ (see README.md).
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="FoodSight Nigeria", page_icon="🌾", layout="wide")

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
CATEGORIES = {
    "Rice_local": "Grains & tubers", "Rice_imported": "Grains & tubers",
    "Gari_white": "Grains & tubers", "Yam": "Grains & tubers", "Maize_white": "Grains & tubers",
    "Beans_brown": "Legumes",
    "Palm_oil": "Oils", "Groundnut_oil": "Oils", "Vegetable_oil": "Oils",
}
UNIT_LABELS = {"Palm_oil": "NGN / litre", "Groundnut_oil": "NGN / litre", "Vegetable_oil": "NGN / litre"}
DRIVER_LABELS = {
    "own_price_lags": "Recent price momentum",
    "exchange_rate": "Exchange rate (NGN/USD)",
    "all_items_cpi": "General inflation (CPI)",
    "food_cpi": "Food inflation (Food CPI)",
    "fuel_price_ngn_per_litre": "Fuel price",
    "rainfall_anomaly": "Rainfall anomaly",
    "post_subsidy_removal": "Post-subsidy period",
}
PAGES = ["Overview", "Forecast", "Drivers", "Price Status", "Methodology"]

# ---------------------------------------------------------------------------
# Design tokens (single source of truth for colour + a small CSS layer)
# ---------------------------------------------------------------------------
C = dict(
    bg="#0e1117", surface="#151b26", border="#232c3b",
    text="#e6eaf2", muted="#8b95a7",
    accent="#4da3ff", forecast="#f5a524",
    green="#34c38f", amber="#f5a524", red="#f0524f", neutral="#8b95a7",
)
STATUS = {  # key -> (label, colour, plain-language description)
    "normal": ("Normal", C["green"], "within its recent normal range"),
    "watch": ("Watch", C["amber"], "approaching its recent spike threshold"),
    "high": ("High", C["red"], "above its recent spike threshold"),
}

st.markdown(
    f"""
<style>
  .block-container {{ padding-top: 2rem; max-width: 1400px; }}
  .fs-eyebrow {{ color:{C['muted']}; font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; margin:0; }}
  .fs-title {{ font-size:2rem; font-weight:700; margin:.1rem 0 1rem 0; line-height:1.15; }}
  .fs-card {{ background:{C['surface']}; border:1px solid {C['border']}; border-radius:10px;
              padding:1rem 1.1rem; height:100%; }}
  .fs-label {{ color:{C['muted']}; font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; margin:0; }}
  .fs-value {{ font-size:1.9rem; font-weight:700; line-height:1.2; margin:.25rem 0 .1rem 0; }}
  .fs-value.sm {{ font-size:1.5rem; }}
  .fs-sub {{ color:{C['muted']}; font-size:.82rem; margin:0; }}
  .fs-insight {{ background:{C['surface']}; border-left:4px solid {C['accent']}; border-radius:6px;
                 padding:.8rem 1rem; margin:.5rem 0 1rem 0; font-size:.98rem; }}
  .fs-section {{ font-size:1.05rem; font-weight:600; margin:1.4rem 0 .4rem 0; }}
  .fs-bar-row {{ margin:.55rem 0; }}
  .fs-bar-head {{ display:flex; justify-content:space-between; font-size:.88rem; margin-bottom:.2rem; }}
  .fs-bar-track {{ background:{C['border']}; border-radius:4px; height:8px; }}
  .fs-bar-fill {{ border-radius:4px; height:8px; }}
  .fs-status-banner {{ border-radius:10px; padding:1.2rem 1.3rem; }}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data loading
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
# Sidebar: brand → commodity selector → navigation → footer
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🌾 FoodSight Nigeria")
st.sidebar.caption("Forecasting Nigerian food price volatility")

if not DATA_OK:
    st.error(
        f"Data file not found: {MISSING_FILE}\n\n"
        "Place the required CSVs in the `data/` folder next to this script. "
        "See README.md for the full list and generation source."
    )
    st.stop()

selected = st.sidebar.selectbox(
    "Commodity",
    COMMODITIES,
    key="commodity",
    format_func=lambda c: f"{COMMODITY_LABELS[c]}  ·  {CATEGORIES[c]}",
    help="Click and type to search.",
)
page = st.sidebar.radio("View", PAGES, key="page")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: NBS, CHIRPS, Investing.com · National level, monthly, "
    "Jan 2016 – " + merged["date"].iloc[-1].strftime("%b %Y")
)


# ---------------------------------------------------------------------------
# Derived values for the selected commodity
# ---------------------------------------------------------------------------
label = COMMODITY_LABELS[selected]
unit = UNIT_LABELS.get(selected, "NGN / kg")
hist = merged[["date", selected]].rename(columns={selected: "price"})
fc = forecasts[forecasts["commodity"] == selected].sort_values("date").copy()
latest_price = float(hist["price"].iloc[-1])
latest_date = hist["date"].iloc[-1]

best_model = mape_summary.loc[selected, "best_model"]
best_mape = float(mape_summary.loc[selected, best_model])

t = thresholds.loc[selected]
amber, red = float(t["rolling24_amber_p75"]), float(t["rolling24_red_p90"])


def status_key(price: float) -> str:
    return "high" if price >= red else "watch" if price >= amber else "normal"


status = status_key(latest_price)
status_label, status_color, status_desc = STATUS[status]

end_fc = fc.iloc[-1]
outlook_pct = (end_fc["forecast"] / latest_price - 1) * 100
if outlook_pct > 2:
    outlook_word, outlook_arrow, outlook_color = "Rising", "▲", C["amber"]
elif outlook_pct < -2:
    outlook_word, outlook_arrow, outlook_color = "Falling", "▼", C["green"]
else:
    outlook_word, outlook_arrow, outlook_color = "Stable", "►", C["neutral"]
peak_status = status_key(float(fc["forecast"].max()))

drivers = shap_pct.loc[selected].rename(index=DRIVER_LABELS).sort_values(ascending=False)
top_driver, top_driver_pct = drivers.index[0], float(drivers.iloc[0])
externals = drivers.drop("Recent price momentum", errors="ignore")
top_ext, top_ext_pct = externals.index[0], float(externals.iloc[0])


def naira(x: float, dp: int = 0) -> str:
    return f"₦{x:,.{dp}f}"


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def kpi_card(title, value, sub, color=None, small=False):
    style = f" style='color:{color}'" if color else ""
    cls = "fs-value sm" if small else "fs-value"
    return (f"<div class='fs-card'><p class='fs-label'>{title}</p>"
            f"<p class='{cls}'{style}>{value}</p><p class='fs-sub'>{sub}</p></div>")


def insight(text_html):
    st.markdown(f"<div class='fs-insight'>{text_html}</div>", unsafe_allow_html=True)


def section(text):
    st.markdown(f"<p class='fs-section'>{text}</p>", unsafe_allow_html=True)


def page_header(eyebrow):
    st.markdown(f"<p class='fs-eyebrow'>{eyebrow}</p><p class='fs-title'>{label}</p>",
                unsafe_allow_html=True)


def kpi_row():
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Current price", naira(latest_price, 2),
                         f"{unit} · {latest_date.strftime('%B %Y')}"), unsafe_allow_html=True)
    c2.markdown(kpi_card("6-month outlook", f"{outlook_arrow} {outlook_pct:+.1f}%",
                         f"{outlook_word} · {naira(end_fc['forecast'])} by {end_fc['date'].strftime('%b %Y')}",
                         outlook_color), unsafe_allow_html=True)
    c3.markdown(kpi_card("Price status", status_label,
                         f"Vs. recent 24-month range", status_color), unsafe_allow_html=True)
    c4.markdown(kpi_card("Forecast accuracy", f"{best_mape:.2f}% avg. error",
                         f"1-month-ahead · {best_model}", small=True), unsafe_allow_html=True)


def forecast_chart(height=420, years=5, show_events=False, show_thresholds=False):
    """Layered by priority: history + forecast + band first; events/thresholds optional."""
    last = hist.iloc[[-1]]
    start = latest_date - pd.DateOffset(years=years) if years else hist["date"].min()
    h = hist[hist["date"] >= start]

    band_x = pd.concat([last["date"], fc["date"], fc["date"][::-1], last["date"]])
    band_y = pd.concat([last["price"], fc["upper_80"], fc["lower_80"][::-1], last["price"]])
    line_x = pd.concat([last["date"], fc["date"]])
    line_y = pd.concat([last["price"], fc["forecast"]])

    fig = go.Figure()
    fig.add_shape(type="rect", x0=latest_date, x1=fc["date"].max() + pd.Timedelta(days=20),
                  y0=0, y1=1, yref="paper", fillcolor="rgba(255,255,255,0.03)", line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=band_x, y=band_y, fill="toself", fillcolor="rgba(245,165,36,0.16)",
                             line=dict(width=0), name="80% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=h["date"], y=h["price"], mode="lines", name="Historical",
                             line=dict(color=C["accent"], width=2.5),
                             hovertemplate="₦%{y:,.0f}<extra>Historical</extra>"))
    fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines+markers", name="Forecast",
                             line=dict(color=C["forecast"], width=2.5, dash="dash"), marker=dict(size=6),
                             hovertemplate="₦%{y:,.0f}<extra>Forecast</extra>"))

    if show_thresholds:
        for val, col, nm in [(amber, C["amber"], "Watch threshold"), (red, C["red"], "High threshold")]:
            fig.add_hline(y=val, line_dash="dot", line_color=col, opacity=0.8,
                          annotation_text=nm, annotation_position="top left",
                          annotation_font_color=col)

    # Forecast boundary (always shown, explicitly labelled)
    fig.add_shape(type="line", x0=latest_date, x1=latest_date, y0=0, y1=1, yref="paper",
                  line=dict(color=C["muted"], width=1, dash="dot"))
    fig.add_annotation(x=latest_date, y=1, yref="paper", text="Forecast starts →", showarrow=False,
                       xanchor="right", yanchor="bottom", font=dict(size=11, color=C["muted"]))

    if show_events:
        for d, txt in [("2020-03-01", "COVID-19"), ("2023-05-01", "Subsidy removal")]:
            ts = pd.Timestamp(d)
            if ts >= start:
                fig.add_shape(type="line", x0=ts, x1=ts, y0=0, y1=1, yref="paper",
                              line=dict(color=C["muted"], width=1, dash="dot"))
                fig.add_annotation(x=ts, y=0, yref="paper", text=txt, showarrow=False,
                                   xanchor="left", yanchor="bottom", textangle=-90,
                                   font=dict(size=10, color=C["muted"]))

    fig.update_layout(
        template="plotly_dark", height=height, hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title=f"Price ({unit})", xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=40, b=10, l=10, r=10),
    )
    fig.update_xaxes(tickformat="%b %Y", showgrid=False)
    fig.update_yaxes(gridcolor=C["border"], tickprefix="₦", tickformat=",")
    return fig


def driver_bars(n=None, height=None):
    """Compact HTML bars, leading driver highlighted."""
    rows = drivers.head(n) if n else drivers
    mx = float(rows.max()) or 1
    html = ""
    for i, (name, v) in enumerate(rows.items()):
        col = C["accent"] if i == 0 else "#3a4a63"
        html += (f"<div class='fs-bar-row'><div class='fs-bar-head'><span>{name}</span>"
                 f"<span style='color:{C['muted']}'>{v:.1f}%</span></div>"
                 f"<div class='fs-bar-track'><div class='fs-bar-fill' "
                 f"style='width:{v / mx * 100:.0f}%;background:{col}'></div></div></div>")
    return html


def status_banner():
    st.markdown(
        f"<div class='fs-status-banner' style='background:{status_color}1f;border:1px solid {status_color}66;'>"
        f"<p class='fs-label' style='color:{status_color}'>Price status</p>"
        f"<p class='fs-value' style='color:{status_color};margin:.15rem 0'>{status_label}</p>"
        f"<p class='fs-sub' style='color:{C['text']}'>{label} is {naira(latest_price)}, {status_desc}.</p></div>",
        unsafe_allow_html=True)


def overview_insight():
    proj = ""
    if peak_status != status:
        proj = (f" The forecast peaks at <b>{STATUS[peak_status][0].lower()}</b> levels "
                f"({naira(fc['forecast'].max())}).")
    insight(
        f"<b>{label}</b> costs <b>{naira(latest_price)}</b> and is <b>{status_desc}</b>. "
        f"The model expects prices to be <b>{outlook_word.lower()}</b> "
        f"(<b>{outlook_pct:+.1f}%</b>) to about <b>{naira(end_fc['forecast'])}</b> by "
        f"{end_fc['date'].strftime('%B %Y')}, with an 80% range of "
        f"{naira(end_fc['lower_80'])}–{naira(end_fc['upper_80'])}.{proj} "
        f"<br><span style='color:{C['muted']}'>Watch: <b>{top_driver}</b> is the leading driver "
        f"({top_driver_pct:.0f}%)"
        + (f"; the largest external factor is <b>{top_ext}</b> ({top_ext_pct:.0f}%)."
           if top_driver != top_ext else ".") + "</span>"
    )


# ===========================================================================
# PAGES
# ===========================================================================
if page == "Overview":
    page_header("Overview · What is happening")
    kpi_row()
    section("Price history and 6-month forecast")
    st.plotly_chart(forecast_chart(height=400), width="stretch")
    st.caption("Shaded band = 80% range: the model expects the actual price to land inside it "
               "about 8 times out of 10. Everything right of the dotted line is a forecast.")
    overview_insight()

    left, right = st.columns([3, 2])
    with left:
        section("Key drivers")
        st.markdown(f"<div class='fs-card'>{driver_bars(n=3)}"
                    f"<p class='fs-sub' style='margin-top:.6rem'>Share of driver influence. "
                    f"See the Drivers view for all factors.</p></div>", unsafe_allow_html=True)
    with right:
        section("Price status")
        status_banner()

elif page == "Forecast":
    page_header("Forecast · Projections and historical trend")
    kpi_row()
    section("Price history and forecast")
    ctl1, ctl2, ctl3 = st.columns([2, 2, 2])
    rng = ctl1.radio("Range", ["2Y", "5Y", "All"], index=1, horizontal=True, key="range")
    show_events = ctl2.toggle("Show events", value=False, help="COVID-19 and subsidy removal markers")
    show_thr = ctl3.toggle("Show status thresholds", value=False)
    years = {"2Y": 2, "5Y": 5, "All": None}[rng]
    st.plotly_chart(forecast_chart(height=460, years=years, show_events=show_events,
                                   show_thresholds=show_thr), width="stretch")
    st.caption("Dashed line = forecast; shaded band = 80% range (the actual price should fall "
               "inside it about 8 in 10 times).")
    overview_insight()

    section(f"Monthly forecast · {fc['date'].min().strftime('%b %Y')} – {fc['date'].max().strftime('%b %Y')}")
    d = fc[["date", "forecast", "lower_80", "upper_80"]].copy()
    d["vs_now"] = (d["forecast"] / latest_price - 1) * 100
    d["Month"] = d["date"].dt.strftime("%b %Y")
    d = d.set_index("Month")[["forecast", "vs_now", "lower_80", "upper_80"]]
    d.columns = ["Forecast (₦)", "Change vs now", "Lower (80%)", "Upper (80%)"]
    st.dataframe(
        d.style.format({"Forecast (₦)": "₦{:,.2f}", "Change vs now": "{:+.1f}%",
                        "Lower (80%)": "₦{:,.2f}", "Upper (80%)": "₦{:,.2f}"})
         .set_properties(subset=["Forecast (₦)"], **{"font-weight": "700"})
         .set_properties(subset=["Lower (80%)", "Upper (80%)"], **{"color": C["muted"]}),
        width="stretch")

elif page == "Drivers":
    page_header("Drivers · Factors influencing price")
    insight(
        f"For <b>{label}</b>, the leading driver is <b>{top_driver}</b> ({top_driver_pct:.0f}% of the "
        f"model's driver-based influence)."
        + (f" The largest <i>external</i> factor is <b>{top_ext}</b> at <b>{top_ext_pct:.0f}%</b>; "
           f"the rest comes from the commodity's own recent price trend and other factors."
           if top_driver != top_ext else
           f" The next largest factor is <b>{drivers.index[1]}</b> ({drivers.iloc[1]:.0f}%).")
    )
    d = drivers.sort_values(ascending=True)
    colors = [C["accent"] if n == top_driver else "#3a4a63" for n in d.index]
    fig2 = go.Figure(go.Bar(x=d.values, y=d.index, orientation="h", marker_color=colors,
                            text=[f"{v:.1f}%" for v in d.values], textposition="outside",
                            hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig2.update_layout(template="plotly_dark", height=380, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Share of driver influence (%)",
                       margin=dict(t=10, b=10, l=10, r=40), yaxis=dict(automargin=True))
    fig2.update_xaxes(gridcolor=C["border"], range=[0, max(d.values) * 1.15])
    st.plotly_chart(fig2, width="stretch")

    with st.expander("How to read this"):
        st.markdown(
            "- Bars show each factor's *relative* contribution to the model's predictions for this "
            "commodity, not a causal effect size.\n"
            "- \"Recent price momentum\" captures the tendency for prices to continue their recent "
            "trend — a real signal, not a data artefact.\n"
            "- Percentages come from SHAP values of the LightGBM model. Since ARIMA (no engineered "
            "features) is the most accurate forecaster here, treat this view as explaining what "
            "*correlates with* price movements, not the mechanism behind the forecast shown."
        )

elif page == "Price Status":
    page_header("Price Status · Normal, watch or high")
    top, side = st.columns([3, 2])
    with top:
        status_banner()
    with side:
        gap_amber = (amber / latest_price - 1) * 100
        gap_red = (red / latest_price - 1) * 100
        st.markdown(kpi_card(
            "Distance to thresholds",
            f"{gap_amber:+.1f}% to Watch" if status == "normal" else
            (f"{gap_red:+.1f}% to High" if status == "watch" else "Above High"),
            f"Watch at {naira(amber)} · High at {naira(red)}", small=True), unsafe_allow_html=True)

    proj_txt = (f"Looking ahead, the forecast stays <b>{STATUS[peak_status][0].lower()}</b>-level through "
                f"{end_fc['date'].strftime('%B %Y')}." if peak_status == status else
                f"Looking ahead, the forecast reaches <b>{STATUS[peak_status][0].lower()}</b> levels "
                f"({naira(fc['forecast'].max())}) within the next 6 months.")
    insight(f"{label} is <b>{status_label.lower()}</b> today. {proj_txt}")

    section("Price against status thresholds")
    st.plotly_chart(forecast_chart(height=380, years=3, show_thresholds=True), width="stretch")

    with st.expander("How thresholds work"):
        st.markdown(
            "Thresholds use a **rolling 24-month window** (75th percentile = Watch, 90th = High), not "
            "a fixed pre-2023 baseline. Food prices rose structurally after the May 2023 subsidy "
            "removal, so a pre-2023 baseline would flag every commodity as high permanently. The "
            "rolling window flags genuine deviations from the *current* price regime."
        )
        a, b = st.columns(2)
        a.metric("Watch (recent 75th pct.)", naira(amber, 2))
        a.metric("High (recent 90th pct.)", naira(red, 2))
        b.metric("Pre-subsidy 75th pct. (reference)", naira(float(t["amber_threshold_p75"]), 2))
        b.metric("Pre-subsidy 90th pct. (reference)", naira(float(t["red_threshold_p90"]), 2))

elif page == "Methodology":
    page_header("Methodology · Models, accuracy and data")
    st.markdown(
        "Three models are compared under walk-forward validation for each commodity: **ARIMA** "
        "(univariate baseline), **ARIMAX** (multivariate extension) and **LightGBM** (gradient-boosted "
        "ML benchmark). Forecasts use each commodity's best model by walk-forward MAPE."
    )
    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi_card("Selected model", best_model, "Lowest walk-forward MAPE", small=True),
                unsafe_allow_html=True)
    c2.markdown(kpi_card("Average error", f"{best_mape:.2f}%",
                         "MAPE, 1-month-ahead, out-of-sample", small=True), unsafe_allow_html=True)
    c3.markdown(kpi_card("ARIMA order", str(fc["arima_order"].iloc[0]),
                         "(p, d, q) fitted on full series", small=True), unsafe_allow_html=True)

    section("Model comparison (walk-forward MAPE, lower is better)")
    row = mape_summary.loc[[selected], ["ARIMA", "ARIMAX", "LightGBM"]]
    st.dataframe(row.style.format("{:.2f}%"), width="stretch")
    st.caption(
        "ARIMA is the best-performing model for every commodity in this study — external variables "
        "did not improve on the univariate baseline at this sample size and monthly resolution, a "
        "genuine finding under walk-forward evaluation rather than a modelling shortfall."
    )
    with st.expander("What is MAPE?"):
        st.markdown("Mean Absolute Percentage Error: on average, how far the forecast landed from the "
                    "actual price, as a percentage of the actual price. Walk-forward means each "
                    "forecast used only data available at the time.")

    section("Structural break around the May 2023 subsidy removal")
    if selected in break_summary.index:
        b = break_summary.loc[selected]
        sig = "statistically significant" if b["significant_at_5pct"] else "not statistically significant at 5%"
        st.write(f"Average monthly growth: **{b['pre_mean_pct_change']:.2f}%** → **{b['post_mean_pct_change']:.2f}%** "
                 f"(before → after). Volatility (std. dev.): **{b['pre_std']:.2f}%** → **{b['post_std']:.2f}%**. "
                 f"Welch's t-test on the mean shift: **{sig}** (p = {b['p_value']:.4f}).")

    section("Thresholds")
    st.write(f"Rolling 24-month percentiles: Watch (75th) **{naira(amber, 2)}**, High (90th) **{naira(red, 2)}**. "
             f"Pre-subsidy reference: {naira(float(t['amber_threshold_p75']), 2)} / "
             f"{naira(float(t['red_threshold_p90']), 2)}.")

    section("Data sources")
    st.write("NBS (food prices, CPI, fuel), CHIRPS (rainfall), Investing.com (exchange rate). "
             "National level, monthly, Jan 2016 – " + latest_date.strftime("%b %Y") + ".")

st.markdown("---")
st.caption("FoodSight Nigeria — MSc Information Technology project, University of Ilorin. "
           "For research and educational use; not a substitute for official NBS or CBN price data.")
