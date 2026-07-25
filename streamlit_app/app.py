import streamlit as st
import plotly.graph_objects as go
from filters import render_sidebar
from uac_metrics import summary_stats


def fmt(value, spec, suffix=""):
    """Format a possibly-NaN metric value; shows an em dash instead of literal 'nan'."""
    if value is None or (isinstance(value, float) and value != value):
        return "\u2014"
    return f"{value:{spec}}{suffix}"


st.set_page_config(
    page_title="UAC Care Transition Efficiency",
    page_icon="\U0001F9ED",
    layout="wide",
)

st.title("Care Transition Efficiency & Placement Outcome Analytics")
st.caption(
    "Unaccompanied Alien Children (UAC) Program \u2014 reframing the dataset from a capacity-monitoring "
    "lens to a **process efficiency** lens: how fast children move through the pipeline, where "
    "backlogs form, and whether outcomes are improving over time."
)

df, full_df, z_thresh, metric_mode = render_sidebar()

if df.empty:
    st.warning("No data in the selected date range. Widen your date range in the sidebar.")
    st.stop()

stats = summary_stats(df)

st.markdown("### Key Performance Indicators \u2014 selected date range")
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric(
        "Transfer Efficiency Ratio",
        fmt(stats['avg_transfer_efficiency_30d'], ".2f"),
        help="Transfers \u00f7 CBP Custody (stock), 30-session avg. Higher = faster CBP\u2192HHS turnover.",
    )
with c2:
    st.metric(
        "Discharge Effectiveness",
        fmt(stats['avg_discharge_effectiveness_30d'], ".3f"),
        help="Discharges \u00f7 HHS Care (stock), 30-session avg. Higher = faster sponsor placement turnover.",
    )
with c3:
    st.metric(
        "Pipeline Throughput Rate",
        fmt(stats['avg_pipeline_throughput_30d'], ".2f", "x"),
        help="Total exits \u00f7 Total entries, rolling 30-session window. >1.0 = system draining faster than it fills.",
    )
with c4:
    bar = stats['avg_backlog_accumulation_30d']
    st.metric(
        "Backlog Accumulation Rate",
        fmt(bar, "+.1f", " /session"),
        help="Rolling 14-session avg of (new entries \u2212 final exits). Positive = backlog growing; negative = draining.",
    )
with c5:
    st.metric(
        "Outcome Stability Score",
        fmt(stats['avg_outcome_stability_30d'], ".2f"),
        help="1 \u2212 coefficient of variation of Discharge Effectiveness (30-session window). Closer to 1 = more consistent placement outcomes.",
    )

c6, c7, c8, c9 = st.columns(4)
with c6:
    st.metric("Children currently in CBP custody (latest)", f"{int(stats['latest_cbp_stock']):,}")
with c7:
    st.metric("Children currently in HHS care (latest)", f"{int(stats['latest_hhs_stock']):,}")
with c8:
    total_bn = (stats['total_bottleneck_days_cbp'] or 0) + (stats['total_bottleneck_days_hhs'] or 0)
    st.metric("Bottleneck sessions flagged", f"{total_bn}")
with c9:
    st.metric("Stagnation sessions flagged", f"{stats['total_stagnation_days'] or 0}")

# Threshold-based visual alerts
latest = df.iloc[-1]
alerts = []
if latest.get("cbp_bottleneck"):
    alerts.append("\U0001F534 **CBP custody** is currently in a flagged bottleneck state (stock elevated & still growing).")
if latest.get("hhs_bottleneck"):
    alerts.append("\U0001F534 **HHS care** is currently in a flagged bottleneck state (stock elevated & still growing).")
if latest.get("is_stagnation"):
    alerts.append("\U0001F7E1 Discharge Effectiveness has been in a **sustained stagnation run** \u2014 exits are structurally slower than the recent baseline.")
if stats['avg_backlog_accumulation_30d'] and stats['avg_backlog_accumulation_30d'] > 0:
    alerts.append("\U0001F7E0 Backlog Accumulation Rate has been **positive** on average over the last 30 sessions \u2014 the pipeline is net-adding unresolved cases.")
if stats['avg_pipeline_throughput_30d'] and stats['avg_pipeline_throughput_30d'] < 1:
    alerts.append("\U0001F7E0 Pipeline Throughput Rate is **below 1.0** over the last 30 sessions \u2014 exits are not keeping pace with entries.")

if alerts:
    for a in alerts:
        st.warning(a)
else:
    st.success("\u2705 No active bottleneck, stagnation, or backlog alerts as of the latest reporting session in range.")

st.markdown("---")
st.markdown("### Pipeline snapshot")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["date"], y=df["cbp_stock"], name="CBP custody", line=dict(color="#EF4444", width=1.3)))
fig.add_trace(go.Scatter(x=df["date"], y=df["hhs_stock"], name="HHS care", yaxis="y2", line=dict(color="#2563EB", width=1.3)))
fig.update_layout(
    height=420,
    margin=dict(l=10, r=10, t=30, b=10),
    yaxis=dict(title="Children in CBP custody", color="#EF4444"),
    yaxis2=dict(title="Children in HHS care", overlaying="y", side="right", color="#2563EB"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified",
)
st.plotly_chart(fig, width="stretch")

st.markdown("### ML Operations Outlook")
st.info(
    "Use **ML Forecast & Risk Outlook** in the sidebar for seven-session discharge/backlog forecasts "
    "and CBP/HHS bottleneck-risk estimates. Forecasts are evaluated chronologically against a "
    "persistence baseline and, where available, a PyTorch LSTM benchmark."
)

st.info(
    "Use the pages in the left sidebar to dig into **Care Pipeline Flow**, "
    "**Transfer & Discharge Efficiency**, **Bottleneck Detection**, **Outcome Trend Analysis**, "
    "and the **ML Forecast & Risk Outlook**."
)
