# Corrected Video Transcript — UAC Care Transition Efficiency Project
*Cleaned for accuracy and clarity, aligned to your original timeline*

---

**00:00**
Hi, I'm Soham Deolalikar, and this is my project for the Machine Learning Intern track.

**00:08**
Tracking Care Transition Efficiency Analytics for the Unaccompanied Alien Children program, built using HHS's own daily reporting data.

**00:18**
What the project actually does —

**00:22**
this is the Care Pipeline Flow visualization.

**00:28**
The brief asked me to move away from just tracking how many children are in custody at a given moment, and instead measure how efficiently they move through the system — from CBP custody, into HHS care, and finally to a sponsor.

**00:47**
So I built five KPIs directly from the raw daily counts.

**00:54**
As you can see — Care Pipeline Flow, Transfer & Discharge Efficiency, Bottleneck Detection, Outcome Trend Analysis, and ML Forecast & Risk.

**01:05**
Transfer Efficiency, Discharge Effectiveness, Pipeline Throughput, Backlog Accumulation, and Outcome Stability — plus bottleneck and stagnation detection, to catch slowdowns that raw custody counts alone would hide.

**01:20**
The technical build — the deliverables are a fully executed analysis notebook, a five-page Streamlit dashboard, as you can see here,

**01:28**
and a research paper and executive summary.

**01:30**
I also went one step further and added a machine learning layer on top of that —

**01:37**
a seven-session-ahead forecast for discharges and backlog.

**01:40**
You can see it here —

**01:44**
and also the bottleneck risk classifiers for CBP and HHS.

**01:54**
I made sure it was built leakage-safe.

**01:59**
Every feature only uses information available before the prediction point, evaluated chronologically against a persistence baseline — not just handed a good-looking accuracy number.

**02:10**
The biggest challenge — what I actually learned — the most interesting part wasn't the modeling.

**02:15**
It was catching something the raw numbers were hiding.

**02:17**
Total volume through the system dropped sharply in 2025, which on the surface looks like good news. But when I measured discharge rate against the population still in care,

**02:33**
the estimated time children spent in HHS care had actually gone up — from around a month to about five months.

**02:42**
That's the kind of thing you only find when you stop trusting the headline counts and actually build the process-level metrics, which was the whole point of this project.

**02:54**
It also pushed me to think carefully about honesty in reporting — flagging data quality issues like the missing Friday and Saturday sessions,

**03:06**
instead of quietly smoothing over them, and being upfront that the LSTM benchmark should say "unavailable" rather than fake a result when it can't run properly.

**03:17**
This project stretched me beyond just writing an ML pipeline.

**03:23**
I had to think like an analyst asking, "is this number actually true, or just convenient" — and like an engineer, making sure

**03:35**
the whole thing — notebook, dashboard, docs — stayed consistent and tested end to end.

**03:50**
Thanks for reviewing my submission.

**03:52**
I'm happy to walk through any part of the codebase, the KPI methodology, or the ML pipeline in more detail, if that's useful.

**04:02**
And yes, that's all — thank you for watching.

---

### What was fixed
- "Soham del Oligarch" → **Soham Deolalikar**
- "CVV custody" → **CBP custody**
- "Nature's first study" → **HHS care**
- "double for" → **seven-session-ahead forecast**
- Removed filler duplicates ("uh, you can see it here" x2, stray "uh"s) and false starts, without changing anything you actually said in substance
- Tightened a couple of run-on sentences for readability — the spoken content is unchanged

### Note
This is a **transcript cleanup for your records** (e.g. if the form or a reviewer asks for captions/subtitles). It doesn't change your actual video audio — if you want the *audio* itself cleaner (fixing "CBP" mispronunciation, removing your name stumble), that still needs a short re-record of just those two spots, not a transcript edit.
