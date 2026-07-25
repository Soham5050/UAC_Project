"""Shared sidebar controls + cached data loader used by every page of the app."""

import streamlit as st
import pandas as pd
from pathlib import Path
from uac_metrics import build_full_dataset

DATA_PATH = Path(__file__).parent / "data" / "HHS_Unaccompanied_Alien_Children_Program.csv"


@st.cache_data
def get_data(z_thresh: float = 1.5, stagnation_min_run: int = 5):
    return build_full_dataset(str(DATA_PATH), z_thresh=z_thresh, stagnation_min_run=stagnation_min_run)


def render_sidebar():
    """Renders shared date-range, z-threshold, and ratio-toggle controls.
    Uses session_state so selections persist as the user moves between pages."""

    st.sidebar.markdown("## Filters")

    z_thresh = st.sidebar.slider(
        "Bottleneck alert sensitivity (z-score threshold)",
        min_value=1.0, max_value=3.0, value=1.5, step=0.1,
        help="Lower = more days flagged as bottlenecks. A day is flagged when custody stock "
             "is this many rolling standard deviations above its 60-session baseline AND still net-accumulating.",
        key="z_thresh",
    )

    full_df = get_data(z_thresh=z_thresh)
    min_d, max_d = full_df["date"].min().date(), full_df["date"].max().date()

    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_d, max_d),
        min_value=min_d, max_value=max_d,
        key="date_range",
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
    else:
        start, end = min_d, max_d

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Metric view")
    metric_mode = st.sidebar.radio(
        "Show ratio-based metrics as:",
        options=["Raw (daily)", "14-session rolling average"],
        index=1,
        key="metric_mode",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Data source: HHS Unaccompanied Alien Children Program daily report "
        f"({min_d} \u2013 {max_d}, {len(full_df)} reporting sessions)."
    )

    mask = (full_df["date"].dt.date >= start) & (full_df["date"].dt.date <= end)
    filtered = full_df.loc[mask].reset_index(drop=True)

    return filtered, full_df, z_thresh, metric_mode


def ratio_col(base_col: str, metric_mode: str) -> str:
    """Return the correct column name (raw vs rolling) based on sidebar toggle."""
    if metric_mode == "14-session rolling average":
        return f"{base_col}_roll14"
    return base_col
