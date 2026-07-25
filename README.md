# UAC Care Transition Efficiency & Placement Outcome Analytics

Reframes the HHS Unaccompanied Alien Children (UAC) daily reporting dataset from a
**capacity-monitoring** lens to a **process-efficiency and outcome-evaluation** lens — measuring
how fast children move through the CBP → HHS → sponsor pipeline, where backlogs form, and whether
placement outcomes are improving or deteriorating over time.

Built for the "Machine Learning Intern" project brief (Unified Mentor / U.S. Department of Health
and Human Services).

## Contents

| File / folder | What it is |
|---|---|
| `UAC_Research_Paper.docx` | Full research paper — background, methodology, EDA, findings, insights, recommendations, limitations. |
| `UAC_Executive_Summary.docx` | Plain-language summary for non-technical government stakeholders. |
| `UAC_Care_Transition_Analysis.ipynb` | Jupyter notebook — data cleaning, KPI derivation, full EDA, charts, headline findings, CSV export. |
| `streamlit_app/` | Live multi-page dashboard (see `streamlit_app/README.md` for details). |

## Headline finding

Total volume moving through the system fell sharply in 2025 versus 2023–2024, but the estimated
time children spent in HHS care rose over the same period — from roughly 31–34 days on average to
an estimated 156 days. A smaller population in the system did not mean a faster-moving system. This
only becomes visible when discharges are measured against the size of the population still in care,
not from raw custody counts alone.

## KPIs

| KPI | Formula |
|---|---|
| Transfer Efficiency Ratio | Transfers ÷ CBP Custody (stock) |
| Discharge Effectiveness Index | Discharges ÷ HHS Care (stock) |
| Pipeline Throughput Rate | 30-session rolling: total exits ÷ total entries |
| Backlog Accumulation Rate | 14-session rolling avg net inflow |
| Outcome Stability Score | 1 − coefficient of variation of Discharge Effectiveness |

Two supporting length-of-stay estimates (CBP, HHS) are derived using Little's Law
(stock ÷ rolling outflow rate).

## Running the dashboard

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

Includes 5 pages: Home (KPI overview + alerts), Care Pipeline Flow, Transfer & Discharge Efficiency,
Bottleneck Detection, Outcome Trend Analysis, and an ML Forecast & Bottleneck Risk Outlook
(seven-session forecasts + CBP/HHS risk scoring, evaluated chronologically against a persistence
baseline and an optional PyTorch LSTM benchmark).

## Data quality notes

- The dataset contains almost no Friday/Saturday reporting sessions — weekday/weekend comparisons
  are effectively driven by Sunday reporting and are flagged as such throughout.
- Length-of-stay figures are Little's Law approximations, not directly measured per-child durations.
- The first ~10 sessions produce missing values for 30-session rolling KPIs by construction and are
  excluded from era-comparison analysis.

## Data source

HHS Unaccompanied Alien Children Program daily reporting, January 12, 2023 – December 21, 2025
(720 reporting sessions).
