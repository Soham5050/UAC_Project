import streamlit as st
import plotly.graph_objects as go
from filters import render_sidebar

st.set_page_config(page_title="Care Pipeline Flow", page_icon="\U0001F9ED", layout="wide")
st.title("\U0001F9ED Care Pipeline Flow Visualization")
st.caption("How children move from CBP apprehension \u2192 CBP custody \u2192 transfer to HHS \u2192 HHS care \u2192 discharge/placement.")

df, full_df, z_thresh, metric_mode = render_sidebar()
if df.empty:
    st.warning("No data in the selected date range.")
    st.stop()

st.markdown("### Daily flow volumes")
fig = go.Figure()
fig.add_trace(go.Bar(x=df["date"], y=df["cbp_inflow"], name="New CBP apprehensions", marker_color="#F59E0B"))
fig.add_trace(go.Bar(x=df["date"], y=df["cbp_to_hhs"], name="Transferred CBP\u2192HHS", marker_color="#EF4444"))
fig.add_trace(go.Bar(x=df["date"], y=df["hhs_discharged"], name="Discharged from HHS", marker_color="#2563EB"))
fig.update_layout(barmode="group", height=420, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified",
                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
st.plotly_chart(fig, width="stretch")

st.markdown("### Custody stock over time")
col1, col2 = st.columns(2)
with col1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df["date"], y=df["cbp_stock"], name="Daily", line=dict(color="#FCA5A5", width=1)))
    fig1.add_trace(go.Scatter(x=df["date"], y=df["cbp_stock_roll14"], name="14-session avg", line=dict(color="#B91C1C", width=2.2)))
    fig1.update_layout(title="Children in CBP custody", height=360, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig1, width="stretch")
with col2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["date"], y=df["hhs_stock"], name="Daily", line=dict(color="#93C5FD", width=1)))
    fig2.add_trace(go.Scatter(x=df["date"], y=df["hhs_stock_roll14"], name="14-session avg", line=dict(color="#1D4ED8", width=2.2)))
    fig2.update_layout(title="Children in HHS care", height=360, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig2, width="stretch")

st.markdown("### Pipeline stage snapshot (latest session in range)")
latest = df.iloc[-1]
sankey_labels = ["New CBP Apprehensions", "CBP Custody", "Transferred to HHS", "HHS Care", "Discharged"]
fig3 = go.Figure(data=[go.Sankey(
    node=dict(
        pad=20, thickness=18,
        line=dict(color="black", width=0.4),
        label=sankey_labels,
        color=["#F59E0B", "#FCA5A5", "#EF4444", "#93C5FD", "#2563EB"],
    ),
    link=dict(
        source=[0, 1, 2, 3],
        target=[1, 2, 3, 4],
        value=[
            max(latest["cbp_inflow"], 0.1),
            max(latest["cbp_to_hhs"], 0.1),
            max(latest["cbp_to_hhs"], 0.1),
            max(latest["hhs_discharged"], 0.1),
        ],
        color=["#FDE68A", "#FCA5A5", "#BFDBFE", "#93C5FD"],
    ),
)])
fig3.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10),
                    title=f"Flow snapshot for {latest['date'].date()}")
st.plotly_chart(fig3, width="stretch")
st.caption(
    "Sankey widths are illustrative of a single reporting session's flow volumes, not cumulative stock. "
    "CBP custody and HHS care stocks (shown above) reflect the accumulated population, not just that day's flow."
)

with st.expander("View underlying data table"):
    st.dataframe(
        df[["date", "cbp_inflow", "cbp_stock", "cbp_to_hhs", "hhs_stock", "hhs_discharged"]]
        .rename(columns={
            "date": "Date", "cbp_inflow": "New CBP Apprehensions", "cbp_stock": "CBP Custody (stock)",
            "cbp_to_hhs": "Transferred CBP\u2192HHS", "hhs_stock": "HHS Care (stock)", "hhs_discharged": "Discharged from HHS",
        }).sort_values("Date", ascending=False),
        width="stretch", height=350,
    )
