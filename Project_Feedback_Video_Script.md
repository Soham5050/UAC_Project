# Project Feedback Video Script
### UAC Care Transition Efficiency & Placement Outcome Analytics

*Target length: ~2.5–3 minutes. Read naturally, don't rush — pauses are fine.*

---

**[0:00 – 0:20] Intro**

Hi, I'm Soham Deolalikar, and this is my project for the Machine Learning Intern track —
Care Transition Efficiency Analytics for the Unaccompanied Alien Children program, built
using HHS's own daily reporting data.

**[0:20 – 0:55] What the project does**

The brief asked me to move away from just tracking how many children are in custody at a
given moment, and instead measure how *efficiently* they move through the system — from
CBP custody, into HHS care, and finally to a sponsor.

So I built five KPIs directly from the raw daily counts — Transfer Efficiency, Discharge
Effectiveness, Pipeline Throughput, Backlog Accumulation, and Outcome Stability — plus
bottleneck and stagnation detection to catch slowdowns that raw custody counts alone would
hide.

**[0:55 – 1:30] The technical build**

The deliverables are a fully executed analysis notebook, a five-page Streamlit dashboard,
a research paper, and an executive summary. I also went one step further and added a
machine learning layer on top — a seven-session-ahead forecast for discharges and backlog,
plus bottleneck-risk classifiers for CBP and HHS. I made sure it was built leakage-safe —
every feature only uses information available before the prediction point, evaluated
chronologically against a persistence baseline, not just handed a good-looking accuracy
number.

**[1:30 – 2:05] Biggest challenge / what I learned**

The most interesting part wasn't the modeling — it was catching something the raw numbers
were hiding. Total volume through the system dropped sharply in 2025, which on the surface
looks like good news. But when I measured discharge rate against the population still in
care, the estimated time children spent in HHS custody had actually gone *up*, from around
a month to over five months. That's the kind of thing you only find when you stop trusting
headline counts and actually build the process-level metrics — which was the whole point
of this project.

It also pushed me to think carefully about honesty in reporting — flagging data quality
issues like the missing Friday/Saturday sessions instead of quietly smoothing over them,
and being upfront that the LSTM benchmark should say "unavailable" rather than fake a
result when it can't run properly.

**[2:05 – 2:35] Reflection**

This project stretched me beyond just writing an ML pipeline — I had to think like an
analyst asking "is this number actually true, or just convenient," and like an engineer
making sure the whole thing — notebook, dashboard, and docs — stayed consistent and
testable end to end.

**[2:35 – 2:50] Closing**

Thanks for reviewing my submission — I'm happy to walk through any part of the codebase,
the KPI methodology, or the ML pipeline in more detail if that's useful.

---

### Notes before recording
- Swap "Hi, I'm Soham Deolalikar" for however you want to open — first name only is fine too.
- If you want it shorter, the safest cut is the whole "[1:30–2:05]" middle paragraph on data quality — keep the length-of-stay finding, it's your strongest point.
- Say the length-of-stay numbers slowly (31–34 days → ~156 days) — it's the one stat worth landing clearly.
- Record a test take first to check your total time fits whatever limit the form implies.
