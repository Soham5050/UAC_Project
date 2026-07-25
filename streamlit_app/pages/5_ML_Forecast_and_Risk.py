import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from filters import render_sidebar
from uac_ml import run_ml_pipeline

st.set_page_config(page_title="ML Forecast & Risk Outlook", page_icon="\U0001F916", layout="wide")
st.title("\U0001F916 ML Forecast & Bottleneck Risk Outlook")
st.caption(
    "Seven-session aggregate operational outlook. These are decision-support estimates about the "
    "overall pipeline \u2014 not predictions about individual children, sponsor suitability, or policy "
    "causality, and not a guarantee of future outcomes."
)

df, full_df, z_thresh, metric_mode = render_sidebar()

if len(full_df) < 120:
    st.error(
        "The ML outlook needs at least 120 reporting sessions of history to build leakage-safe "
        "features. Widen the date range or use the full dataset."
    )
    st.stop()


@st.cache_resource(show_spinner="Training the ML benchmark and preparing the seven-session outlook\u2026")
def cached_suite(fingerprint: str, _data: pd.DataFrame):
    return run_ml_pipeline(_data)


fingerprint = f"{full_df['date'].min()}-{full_df['date'].max()}-{len(full_df)}-{z_thresh}"
suite = cached_suite(fingerprint, full_df)

st.markdown(
    f"**Held-out test period:** {suite.test_start.date()} \u2013 {suite.test_end.date()} "
    "(final chronological reporting sessions, untouched during model selection)"
)
st.caption(
    f"\u2139\ufe0f Bottleneck-risk models below are trained using the **bottleneck alert sensitivity "
    f"= {z_thresh}** currently set in the sidebar. Changing that slider on any page redefines what "
    f"counts as a 'bottleneck' and retrains the risk models the next time this page loads."
)

st.markdown("### Model comparison \u2014 persistence baseline vs. tabular ML vs. LSTM")
st.dataframe(suite.comparison.round(3), width="stretch")
if not suite.lstm_available:
    st.info(f"LSTM benchmark: {suite.lstm_reason}")
st.caption(
    "The **Active** column marks the model used for the live forecast below, selected per target by "
    "the lower validation MAE."
)

st.markdown("---")
st.markdown("### Seven-session forecast")

titles = {
    "hhs_discharged": "HHS discharges (children exiting to sponsors)",
    "system_net_flow": "System net flow (entries \u2212 final exits)",
}
for target in ("hhs_discharged", "system_net_flow"):
    result = suite.forecasts[target]
    forecast = result.values
    recent = full_df.tail(30)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent["date"], y=recent[target], name="Recent actual",
        line=dict(color="#2563EB", width=1.6),
    ))
    fig.add_trace(go.Scatter(
        x=forecast["session"], y=forecast["upper_80"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=forecast["session"], y=forecast["lower_80"], mode="lines", fill="tonexty",
        fillcolor="rgba(124,58,237,0.18)", line=dict(width=0), name="80% interval",
    ))
    fig.add_trace(go.Scatter(
        x=forecast["session"], y=forecast["prediction"], mode="lines+markers",
        line=dict(color="#7C3AED", width=3), name=f"{result.model_name} forecast",
    ))
    fig.update_layout(
        title=titles[target], height=380, hovermode="x unified",
        margin=dict(l=10, r=10, t=45, b=10),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Active model: **{result.model_name}** \u00b7 validation MAE {result.validation_mae:.2f} "
        f"\u00b7 test MAE {result.test_mae:.2f} \u00b7 test RMSE {result.test_rmse:.2f}"
    )

st.markdown("---")
st.markdown("### Seven-session bottleneck risk")
left, right = st.columns(2)
labels = {"cbp_risk": "CBP", "hhs_risk": "HHS"}
for column, key in ((left, "cbp_risk"), (right, "hhs_risk")):
    risk = suite.risks[key]
    with column:
        st.markdown(f"#### {labels[key]} bottleneck risk")
        if risk.available:
            st.metric("Probability, next 7 sessions", f"{risk.probability:.0%}")
            if risk.probability >= 0.60:
                st.warning("Elevated risk \u2014 above the 60% alert threshold.")
            else:
                st.success("Below the 60% alert threshold.")
            st.dataframe(pd.DataFrame([risk.metrics]).round(3), width="stretch")
        else:
            st.info(risk.reason)

available_risks = [r for r in suite.risks.values() if r.available and not r.feature_importance.empty]
if available_risks:
    st.markdown("---")
    top = available_risks[0]
    importance = top.feature_importance.head(10).sort_values("importance")
    fig_imp = go.Figure(go.Bar(
        x=importance["importance"], y=importance["feature"],
        orientation="h", marker_color="#059669",
    ))
    fig_imp.update_layout(
        title=f"Top operational signals \u2014 {labels[top.label]} risk model",
        height=380, margin=dict(l=10, r=10, t=45, b=10),
    )
    st.plotly_chart(fig_imp, width="stretch")

st.markdown("---")
st.markdown("### Download current outlook")
st.download_button(
    "Download model comparison as CSV",
    suite.comparison.to_csv(index=False).encode("utf-8"),
    "uac_ml_outlook.csv",
    "text/csv",
)

with st.expander("Methodology and limitations"):
    st.markdown(
        "- Every feature is built only from reporting sessions at or before the prediction origin; "
        "rolling features are shifted by one session before fitting to prevent target leakage.\n"
        "- Models are evaluated on a held-out **chronological** test period \u2014 the final reporting "
        "sessions in the dataset \u2014 never seen during training or model selection.\n"
        "- The active forecast model is chosen per target by the lower validation MAE, compared "
        "against a persistence baseline (repeats the latest observed value) and, when available, a "
        "compact PyTorch **LSTM** benchmark.\n"
        "- Bottleneck risk is a classification probability for **any** CBP/HHS bottleneck occurring "
        "in the next seven reporting sessions, using the project's existing level-and-net-flow "
        "bottleneck definition. A probability of 60% or higher is treated as elevated risk.\n"
        "- These are aggregate, **operational decision-support signals only** \u2014 not individual-"
        "level predictions, sponsor assessments, causal claims, or guarantees of future outcomes."
    )
