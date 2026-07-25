# UAC Care Transition Efficiency & Placement Outcome Analytics

Multi-page Streamlit app for the "Machine Learning Intern" project brief: reframes the HHS
Unaccompanied Alien Children dataset from capacity monitoring to **process efficiency**.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Structure

- `app.py` — Home page: headline KPIs, active bottleneck alerts, pipeline snapshot chart.
- `pages/1_Care_Pipeline_Flow.py` — Daily flow volumes, custody stock trends, Sankey snapshot.
- `pages/2_Transfer_Discharge_Efficiency.py` — CBP→HHS transfer ratio & HHS discharge ratio panels, raw/rolling toggle.
- `pages/3_Bottleneck_Detection.py` — Z-score based bottleneck flagging, adjustable sensitivity, episode tables.
- `pages/4_Outcome_Trend_Analysis.py` — Era-over-era (6-month) trend comparison, length-of-stay trend, CSV export.
- `pages/5_ML_Forecast_and_Risk.py` — ML Forecast & Risk Outlook: seven-session forecasts and CBP/HHS bottleneck-risk estimates.
- `filters.py` — Shared sidebar controls (date range, bottleneck sensitivity, ratio display mode) + cached data loader.
- `uac_metrics.py` — Core metrics logic (also used by the companion analysis notebook, kept in sync).
- `uac_ml.py` — Leakage-safe ML pipeline: feature/target construction, chronological splits, tabular and LSTM forecasts, bottleneck-risk classifiers.
- `data/` — Source CSV.

## ML Forecast & Risk Outlook

The ML page forecasts aggregate HHS discharges and system net flow for the next **seven reporting sessions**, then estimates the chance of a CBP or HHS bottleneck during that window. It compares a
persistence baseline, a direct tabular ML model (HistGradientBoosting), and a compact PyTorch
**LSTM** benchmark, using **chronological** validation and an untouched final test period — the
active model for each target is selected by lower validation MAE, not assumed in advance.

All features are built only from reporting sessions at or before the prediction origin; rolling
features are shifted by one session before fitting to prevent target leakage. Bottleneck-risk
probabilities and forecast intervals (empirical 80% bands from validation residuals) are shown
alongside the evaluation metrics and test period dates.

The system is operational decision support only: it is **not individual**-level prediction, sponsor
assessment, causal inference, or a guarantee of future placement outcomes. If PyTorch is unavailable
or a chronological segment is too short, the page falls back to the tabular model and states the
reason rather than fabricating an LSTM result.

## Methodology notes

- **Transfer/discharge ratios**: `transferred-out ÷ inflow` per stage. >1.0 = draining faster than filling.
- **Length of stay**: Little's Law estimate (stock ÷ 30-session rolling outflow rate).
- **Bottleneck flag**: custody stock >N rolling z-scores (default 1.5, adjustable) above its 60-session
  baseline, and still net-accumulating that session.
- Reporting cadence follows HHS's native publishing schedule (near-daily, skips weekends/holidays) —
  not forward-filled to avoid fabricating flow data on unreported days.
