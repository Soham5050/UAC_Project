import streamlit as st
import plotly.graph_objects as go
from filters import render_sidebar, ratio_col

st.set_page_config(page_title="Transfer & Discharge Efficiency", page_icon="\u2696\ufe0f", layout="wide")
st.title("\u2696\ufe0f Transfer & Discharge Efficiency Panels")
st.caption(
    "**Transfer Efficiency Ratio** = Transfers \u00f7 CBP Custody (stock) \u2014 what share of the current "
    "CBP load moves to HHS each session. **Discharge Effectiveness** = Discharges \u00f7 HHS Care (stock) "
    "\u2014 what share of the current HHS load is placed with sponsors each session."
)

df, full_df, z_thresh, metric_mode = render_sidebar()
if df.empty:
    st.warning("No data in the selected date range.")
    st.stop()

ter_col = ratio_col("transfer_efficiency_ratio", metric_mode)
de_col = ratio_col("discharge_effectiveness", metric_mode)
thr_col = "pipeline_throughput_rate"
bar_col = "backlog_accumulation_rate"

avg_ter = df[ter_col].mean()
avg_de = df[de_col].mean()
avg_thr = df[thr_col].mean()
avg_bar = df[bar_col].mean()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Avg. Transfer Efficiency Ratio", f"{avg_ter:.2f}")
with c2:
    st.metric("Avg. Discharge Effectiveness", f"{avg_de:.3f}")
with c3:
    delta_color = "normal" if avg_thr >= 1 else "inverse"
    st.metric("Avg. Pipeline Throughput Rate", f"{avg_thr:.2f}x",
              delta=f"{'above' if avg_thr>=1 else 'below'} parity", delta_color=delta_color)
with c4:
    delta_color = "inverse" if avg_bar > 0 else "normal"
    st.metric("Avg. Backlog Accumulation Rate", f"{avg_bar:+.1f} /session",
              delta=f"{'growing' if avg_bar>0 else 'draining'}", delta_color=delta_color)

st.markdown(f"### Transfer Efficiency Ratio \u2014 showing **{metric_mode}**")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["date"], y=df[ter_col], name="Transfer Efficiency Ratio", line=dict(color="#EA580C", width=1.8)))
fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), title="Transfers \u00f7 CBP Custody (stock)")
st.plotly_chart(fig, width="stretch")

st.markdown(f"### Discharge Effectiveness \u2014 showing **{metric_mode}**")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df["date"], y=df[de_col], name="Discharge Effectiveness", line=dict(color="#059669", width=1.8)))
fig2.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), title="Discharges \u00f7 HHS Care (stock)")
st.plotly_chart(fig2, width="stretch")

st.markdown("### Pipeline Throughput Rate (rolling 30-session: total exits \u00f7 total entries)")
fig3 = go.Figure()
fig3.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="parity (ratio = 1)")
fig3.add_trace(go.Scatter(x=df["date"], y=df[thr_col], name="Pipeline Throughput Rate", line=dict(color="#7C3AED", width=1.8)))
fig3.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig3, width="stretch")
st.caption("Total HHS discharges \u00f7 total new CBP apprehensions over a rolling 30-session window \u2014 measures whether the whole pipeline nets children out as fast as they come in, end to end.")

st.markdown("### Backlog Accumulation Rate (rolling 14-session average)")
fig4 = go.Figure()
fig4.add_hline(y=0, line_color="gray")
colors = ["#EF4444" if v and v > 0 else "#10B981" for v in df[bar_col]]
fig4.add_trace(go.Bar(x=df["date"], y=df[bar_col], marker_color=colors, name="Backlog Accumulation Rate"))
fig4.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig4, width="stretch")
st.caption("Rolling 14-session average of (new entries \u2212 final exits) across the whole pipeline. Red = backlog growing that period; green = draining.")
