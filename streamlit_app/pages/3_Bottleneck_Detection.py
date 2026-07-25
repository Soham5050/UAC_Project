import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from filters import render_sidebar


def fmt(value, spec):
    if value is None or (isinstance(value, float) and value != value):
        return "n/a (warm-up period)"
    return f"{value:{spec}}"


st.set_page_config(page_title="Bottleneck Detection", page_icon="\U0001F6A8", layout="wide")
st.title("\U0001F6A8 Bottleneck & Stagnation Detection")
st.caption(
    "Two complementary signals: **Bottlenecks** (level-based) flag sessions where custody stock is "
    "unusually high and still growing. **Stagnation** (flow-based) flags sustained runs where "
    "Discharge Effectiveness sits below its own recent baseline \u2014 a structural slowdown in exits, "
    "even if the stock level itself looks normal."
)

df, full_df, z_thresh, metric_mode = render_sidebar()
if df.empty:
    st.warning("No data in the selected date range.")
    st.stop()

latest = df.iloc[-1]
alert_cols = st.columns(3)
with alert_cols[0]:
    if latest["cbp_bottleneck"]:
        st.error(f"\U0001F534 CBP bottleneck as of {latest['date'].date()} (z={fmt(latest['cbp_stock_z'], '.2f')}, thr={z_thresh})")
    else:
        st.success(f"\u2705 CBP not flagged (z={fmt(latest['cbp_stock_z'], '.2f')}, thr={z_thresh})")
with alert_cols[1]:
    if latest["hhs_bottleneck"]:
        st.error(f"\U0001F534 HHS bottleneck as of {latest['date'].date()} (z={fmt(latest['hhs_stock_z'], '.2f')}, thr={z_thresh})")
    else:
        st.success(f"\u2705 HHS not flagged (z={fmt(latest['hhs_stock_z'], '.2f')}, thr={z_thresh})")
with alert_cols[2]:
    if latest["is_stagnation"]:
        st.warning(f"\U0001F7E1 Discharge stagnation active as of {latest['date'].date()}")
    else:
        st.success("\u2705 No active discharge stagnation")

st.markdown("---")
st.markdown("## Level-based: Bottleneck Detection")

st.markdown("### CBP custody \u2014 flagged bottleneck days")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["date"], y=df["cbp_stock"], name="CBP custody", line=dict(color="#EF4444", width=1)))
bn = df[df["cbp_bottleneck"]]
fig.add_trace(go.Scatter(x=bn["date"], y=bn["cbp_stock"], mode="markers", name="Flagged bottleneck",
                          marker=dict(color="black", size=7, symbol="x")))
fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig, width="stretch")

st.markdown("### HHS care \u2014 flagged bottleneck days")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df["date"], y=df["hhs_stock"], name="HHS care", line=dict(color="#2563EB", width=1)))
bn2 = df[df["hhs_bottleneck"]]
fig2.add_trace(go.Scatter(x=bn2["date"], y=bn2["hhs_stock"], mode="markers", name="Flagged bottleneck",
                           marker=dict(color="black", size=7, symbol="x")))
fig2.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig2, width="stretch")


def episodes(frame, flag_col, gap_days=7):
    flagged = frame[frame[flag_col]].copy()
    if flagged.empty:
        return pd.DataFrame(columns=["Start", "End", "Sessions flagged"])
    flagged["gap"] = flagged["date"].diff().dt.days.fillna(99)
    flagged["episode"] = (flagged["gap"] > gap_days).cumsum()
    ep = flagged.groupby("episode").agg(Start=("date", "min"), End=("date", "max"), **{"Sessions flagged": ("date", "count")})
    return ep.sort_values("Start", ascending=False).reset_index(drop=True)


col1, col2 = st.columns(2)
with col1:
    st.markdown("#### CBP bottleneck episodes")
    st.dataframe(episodes(df, "cbp_bottleneck"), width="stretch", height=260)
with col2:
    st.markdown("#### HHS bottleneck episodes")
    st.dataframe(episodes(df, "hhs_bottleneck"), width="stretch", height=260)

st.caption(
    f"Total flagged sessions in range: {int(df['cbp_bottleneck'].sum())} (CBP), "
    f"{int(df['hhs_bottleneck'].sum())} (HHS) out of {len(df)}."
)

st.markdown("---")
st.markdown("## Flow-based: Discharge Stagnation")
st.caption(
    "Discharge Effectiveness below its own trailing 90-session median, sustained for 5+ consecutive "
    "reporting sessions \u2014 identifies prolonged stagnation periods distinct from stock-level bottlenecks."
)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df["date"], y=df["discharge_effectiveness"], name="Discharge Effectiveness",
                           line=dict(color="#059669", width=1)))
stag = df[df["is_stagnation"]]
fig3.add_trace(go.Scatter(x=stag["date"], y=stag["discharge_effectiveness"], mode="markers", name="Stagnation session",
                           marker=dict(color="#B45309", size=6, symbol="circle")))
fig3.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig3, width="stretch")

st.markdown("#### Stagnation episodes")
st.dataframe(episodes(df, "is_stagnation"), width="stretch", height=260)
st.caption(f"Total flagged stagnation sessions in range: {int(df['is_stagnation'].sum())} out of {len(df)}.")
