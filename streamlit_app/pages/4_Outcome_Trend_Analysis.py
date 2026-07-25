import streamlit as st
import plotly.graph_objects as go
from filters import render_sidebar
from uac_metrics import weekday_weekend_summary, month_over_month_summary

st.set_page_config(page_title="Outcome Trend Analysis", page_icon="\U0001F4C8", layout="wide")
st.title("\U0001F4C8 Outcome Trend Analysis")
st.caption("Are placement outcomes improving or deteriorating over time? Includes era, weekday/weekend, and month-over-month views.")

df, full_df, z_thresh, metric_mode = render_sidebar()
if df.empty:
    st.warning("No data in the selected date range.")
    st.stop()

st.markdown("### Outcome Stability Score \u2014 consistency of placement outcomes")
fig_oss = go.Figure()
fig_oss.add_trace(go.Scatter(x=df["date"], y=df["outcome_stability_score"], name="Outcome Stability Score",
                              line=dict(color="#DB2777", width=1.6)))
fig_oss.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Score (1 = most stable)")
st.plotly_chart(fig_oss, width="stretch")
st.caption("1 \u2212 coefficient of variation of Discharge Effectiveness over a rolling 30-session window. Sharp drops indicate sudden inconsistency in reunification success.")

st.markdown("### Estimated HHS length-of-stay trend")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["date"], y=df["hhs_los_days_est"], name="Est. HHS length of stay",
                          line=dict(color="#7C3AED", width=1.6)))
fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Estimated days in care")
st.plotly_chart(fig, width="stretch")

st.markdown("---")
st.markdown("### Weekday vs. weekend transition speed")
ww = weekday_weekend_summary(df)
st.dataframe(ww, width="stretch")
if "Weekend" in ww.index and "Weekday" in ww.index:
    wk = ww.loc["Weekday"]
    we = ww.loc["Weekend"]
    st.caption(
        f"Weekday avg Transfer Efficiency Ratio: {wk['avg_transfer_efficiency']:.2f} vs. "
        f"weekend: {we['avg_transfer_efficiency']:.2f}. Weekday avg Discharge Effectiveness: "
        f"{wk['avg_discharge_effectiveness']:.3f} vs. weekend: {we['avg_discharge_effectiveness']:.3f}. "
        f"(Note: HHS reports far fewer Friday/Saturday sessions in this dataset \u2014 'weekend' here is mostly Sunday.)"
    )

st.markdown("---")
st.markdown("### Month-over-month placement trends")
mom = month_over_month_summary(df)
st.dataframe(mom, width="stretch", height=320)

c1, c2 = st.columns(2)
with c1:
    figm = go.Figure()
    figm.add_trace(go.Bar(x=mom.index, y=mom["total_discharged"], marker_color="#2563EB"))
    figm.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), title="Total discharged (successful placements) per month")
    st.plotly_chart(figm, width="stretch")
with c2:
    figm2 = go.Figure()
    figm2.add_hline(y=1.0, line_dash="dash", line_color="gray")
    figm2.add_trace(go.Bar(x=mom.index, y=mom["avg_pipeline_throughput"], marker_color="#7C3AED"))
    figm2.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), title="Avg. Pipeline Throughput Rate per month")
    st.plotly_chart(figm2, width="stretch")

st.markdown("---")
st.markdown("### Era-over-era comparison (6-month windows)")
df_era = df.copy()
df_era["era"] = df_era["date"].dt.to_period("6M").astype(str)
era_summary = df_era.groupby("era").agg(
    avg_transfer_efficiency=("transfer_efficiency_ratio", "mean"),
    avg_discharge_effectiveness=("discharge_effectiveness", "mean"),
    avg_pipeline_throughput=("pipeline_throughput_rate", "mean"),
    avg_outcome_stability=("outcome_stability_score", "mean"),
    avg_hhs_los_days=("hhs_los_days_est", "mean"),
    bottleneck_days_hhs=("hhs_bottleneck", "sum"),
    stagnation_days=("is_stagnation", "sum"),
    sessions=("date", "count"),
).round(3)
st.dataframe(era_summary, width="stretch")

st.markdown("### Download enriched dataset")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered data as CSV", csv, "uac_enriched_filtered.csv", "text/csv")
