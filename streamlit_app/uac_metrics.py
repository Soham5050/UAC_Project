"""
uac_metrics.py
Shared data-loading and metrics logic for the UAC Care Transition Efficiency project.
Used by both the analysis notebook and the Streamlit app so the two stay in sync.

KPI definitions follow the official project brief:
  - Transfer Efficiency Ratio   = Transfers (CBP->HHS) / CBP Custody (stock)
  - Discharge Effectiveness     = Discharges / HHS Care (stock)
  - Pipeline Throughput Rate    = Total exits / Total entries (rolling window)
  - Backlog Accumulation Rate   = net daily accumulation across the full pipeline (rolling avg)
  - Outcome Stability Score     = 1 - coefficient of variation of Discharge Effectiveness (rolling)
"""

import pandas as pd
import numpy as np

RAW_COLS = {
    "Date": "date",
    "Children apprehended and placed in CBP custody*": "cbp_inflow",
    "Children in CBP custody": "cbp_stock",
    "Children transferred out of CBP custody": "cbp_to_hhs",
    "Children in HHS Care": "hhs_stock",
    "Children discharged from HHS Care": "hhs_discharged",
}


def load_data(path: str) -> pd.DataFrame:
    """Load and clean the raw HHS UAC CSV export."""
    df = pd.read_csv(path, thousands=",")
    df = df.rename(columns=RAW_COLS)
    df = df.dropna(subset=["date"])
    df["date"] = pd.to_datetime(df["date"], format="%B %d, %Y", errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    numeric_cols = ["cbp_inflow", "cbp_stock", "cbp_to_hhs", "hhs_stock", "hhs_discharged"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Report data as published (business days / weekdays, with gaps on weekends & holidays).
    # We keep the native reporting cadence rather than reindexing to every calendar day,
    # since HHS does not report on non-business days.
    df = df.drop_duplicates(subset="date").reset_index(drop=True)
    return df


def compute_core_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived KPI, flow, and backlog metrics per the official KPI definitions."""
    df = df.copy()

    # --- KPI 1: Transfer Efficiency Ratio = Transfers / CBP Custody (stock) ---
    # What share of the CURRENT CBP custody load got moved to HHS that session.
    # This is a turnover/velocity measure, not a vs-inflow measure.
    df["transfer_efficiency_ratio"] = df["cbp_to_hhs"] / df["cbp_stock"].replace(0, np.nan)

    # --- KPI 2: Discharge Effectiveness = Discharges / HHS Care (stock) ---
    df["discharge_effectiveness"] = df["hhs_discharged"] / df["hhs_stock"].replace(0, np.nan)

    # --- Net flow (kept for backlog/bottleneck diagnostics) ---
    df["cbp_net_flow"] = df["cbp_inflow"] - df["cbp_to_hhs"]
    df["hhs_net_flow"] = df["cbp_to_hhs"] - df["hhs_discharged"]
    df["system_net_flow"] = df["cbp_inflow"] - df["hhs_discharged"]  # whole-pipeline entries vs. final exits

    # --- Rolling smoothing (14-session ~ roughly 3 reporting weeks) ---
    roll_base_cols = [
        "cbp_inflow", "cbp_to_hhs", "hhs_discharged", "cbp_stock", "hhs_stock",
        "transfer_efficiency_ratio", "discharge_effectiveness",
        "cbp_net_flow", "hhs_net_flow", "system_net_flow",
    ]
    for col in roll_base_cols:
        df[f"{col}_roll14"] = df[col].rolling(14, min_periods=5).mean()

    # --- KPI 3: Pipeline Throughput Rate = Total exits / Total entries (rolling window) ---
    # Rolling-30-session sum of final HHS discharges vs. rolling-30-session sum of new CBP entries.
    entries_30 = df["cbp_inflow"].rolling(30, min_periods=10).sum()
    exits_30 = df["hhs_discharged"].rolling(30, min_periods=10).sum()
    df["pipeline_throughput_rate"] = exits_30 / entries_30.replace(0, np.nan)

    # --- KPI 4: Backlog Accumulation Rate ---
    # Rolling-14-session average of (entries - final exits): children/day net-added to the
    # unresolved pipeline (CBP + HHS combined). Positive = backlog growing; negative = draining.
    df["backlog_accumulation_rate"] = df["system_net_flow"].rolling(14, min_periods=5).mean()

    # --- KPI 5: Outcome Stability Score = 1 - CV(Discharge Effectiveness), rolling 30-session ---
    roll_mean_de = df["discharge_effectiveness"].rolling(30, min_periods=10).mean()
    roll_std_de = df["discharge_effectiveness"].rolling(30, min_periods=10).std()
    cv = roll_std_de / roll_mean_de.replace(0, np.nan)
    df["outcome_stability_score"] = (1 - cv).clip(lower=0, upper=1)

    # --- Estimated average Length of Stay (Little's Law: LOS = Stock / Throughput rate) ---
    cbp_outflow_30 = df["cbp_to_hhs"].rolling(30, min_periods=10).mean()
    hhs_outflow_30 = df["hhs_discharged"].rolling(30, min_periods=10).mean()
    df["cbp_los_days_est"] = df["cbp_stock"] / cbp_outflow_30.replace(0, np.nan)
    df["hhs_los_days_est"] = df["hhs_stock"] / hhs_outflow_30.replace(0, np.nan)

    # --- Calendar features for temporal analysis ---
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"])
    df["month"] = df["date"].dt.to_period("M").astype(str)

    return df


def detect_bottlenecks(df: pd.DataFrame, z_thresh: float = 1.5) -> pd.DataFrame:
    """
    Flag bottleneck days using rolling z-scores on custody stocks and net flows.
    A bottleneck day = stock is unusually high AND net flow is positive (still filling up)
    relative to the trailing 60-session baseline. (Level-based: "is the system overloaded right now")
    """
    df = df.copy()

    for stock_col, net_col, prefix in [
        ("cbp_stock", "cbp_net_flow", "cbp"),
        ("hhs_stock", "hhs_net_flow", "hhs"),
    ]:
        roll_mean = df[stock_col].rolling(60, min_periods=15).mean()
        roll_std = df[stock_col].rolling(60, min_periods=15).std()
        z = (df[stock_col] - roll_mean) / roll_std.replace(0, np.nan)
        df[f"{prefix}_stock_z"] = z
        df[f"{prefix}_bottleneck"] = (z > z_thresh) & (df[net_col] > 0)

    return df


def detect_stagnation(df: pd.DataFrame, min_run: int = 5) -> pd.DataFrame:
    """
    Flag stagnation days: Discharge Effectiveness below its own trailing 90-session median,
    sustained for at least `min_run` consecutive reporting sessions. This is a FLOW-based
    signal (structural slowdown in exits), distinct from the LEVEL-based bottleneck flag.
    """
    df = df.copy()
    roll_median = df["discharge_effectiveness"].rolling(90, min_periods=20).median()
    below = df["discharge_effectiveness"] < roll_median

    # identify runs of consecutive True values >= min_run
    run_id = (below != below.shift()).cumsum()
    run_lengths = below.groupby(run_id).transform("size")
    df["is_stagnation"] = below & (run_lengths >= min_run)
    return df


def weekday_weekend_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compare transition speed on weekday vs. weekend reporting sessions."""
    g = df.groupby(df["is_weekend"].map({True: "Weekend", False: "Weekday"})).agg(
        avg_transfer_efficiency=("transfer_efficiency_ratio", "mean"),
        avg_discharge_effectiveness=("discharge_effectiveness", "mean"),
        avg_cbp_inflow=("cbp_inflow", "mean"),
        avg_cbp_to_hhs=("cbp_to_hhs", "mean"),
        avg_hhs_discharged=("hhs_discharged", "mean"),
        sessions=("date", "count"),
    ).round(3)
    return g


def month_over_month_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly aggregation of placement / throughput trends."""
    g = df.groupby("month").agg(
        total_discharged=("hhs_discharged", "sum"),
        total_transferred=("cbp_to_hhs", "sum"),
        total_apprehended=("cbp_inflow", "sum"),
        avg_transfer_efficiency=("transfer_efficiency_ratio", "mean"),
        avg_discharge_effectiveness=("discharge_effectiveness", "mean"),
        avg_pipeline_throughput=("pipeline_throughput_rate", "mean"),
        avg_outcome_stability=("outcome_stability_score", "mean"),
        sessions=("date", "count"),
    ).round(3)
    return g


def summary_stats(df: pd.DataFrame) -> dict:
    """High-level summary numbers for headline reporting / dashboard KPIs."""
    latest = df.iloc[-1]
    last_30 = df.tail(30)
    last_90 = df.tail(90)

    return {
        "date_range": (df["date"].min(), df["date"].max()),
        "n_reporting_days": len(df),
        "latest_date": latest["date"],
        "latest_cbp_stock": latest["cbp_stock"],
        "latest_hhs_stock": latest["hhs_stock"],
        "avg_transfer_efficiency_30d": last_30["transfer_efficiency_ratio"].mean(),
        "avg_discharge_effectiveness_30d": last_30["discharge_effectiveness"].mean(),
        "avg_pipeline_throughput_30d": last_30["pipeline_throughput_rate"].mean(),
        "avg_backlog_accumulation_30d": last_30["backlog_accumulation_rate"].mean(),
        "avg_outcome_stability_30d": last_30["outcome_stability_score"].mean(),
        "avg_transfer_efficiency_90d": last_90["transfer_efficiency_ratio"].mean(),
        "avg_discharge_effectiveness_90d": last_90["discharge_effectiveness"].mean(),
        "avg_cbp_los_days_30d": last_30["cbp_los_days_est"].mean(),
        "avg_hhs_los_days_30d": last_30["hhs_los_days_est"].mean(),
        "total_bottleneck_days_cbp": int(df["cbp_bottleneck"].sum()) if "cbp_bottleneck" in df else None,
        "total_bottleneck_days_hhs": int(df["hhs_bottleneck"].sum()) if "hhs_bottleneck" in df else None,
        "total_stagnation_days": int(df["is_stagnation"].sum()) if "is_stagnation" in df else None,
    }


def build_full_dataset(path: str, z_thresh: float = 1.5, stagnation_min_run: int = 5) -> pd.DataFrame:
    df = load_data(path)
    df = compute_core_metrics(df)
    df = detect_bottlenecks(df, z_thresh=z_thresh)
    df = detect_stagnation(df, min_run=stagnation_min_run)
    return df


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "HHS_Unaccompanied_Alien_Children_Program.csv"
    df = build_full_dataset(path)
    print(df.shape)
    print(df.tail())
    stats = summary_stats(df)
    for k, v in stats.items():
        print(k, ":", v)

