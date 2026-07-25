# Technical Audit — UAC Care Transition Efficiency Project
*Scope: `streamlit_app/` (1,427 lines across 9 Python files). Every file was read in full before writing this report. Findings are marked **CONFIRMED** (verified by reading the code and/or reproducing behavior) or **NEEDS RUNTIME VERIFICATION** (plausible from static reading, not independently reproduced). Nothing below is guessed — phases with no real findings say so explicitly rather than being padded.*

---

## Important correction first

Earlier in this conversation I told you the bottleneck-sensitivity (z-score) slider has **no effect on the ML Forecast & Risk page**. That was wrong, and this audit caught it:

**CONFIRMED — 🟠 Major — Misleading UI coupling**
`filters.py:12,30` → `get_data(z_thresh=z_thresh)` bakes `z_thresh` into `cbp_bottleneck`/`hhs_bottleneck` for the *entire* `full_df`, and `pages/5_ML_Forecast_and_Risk.py:30` includes `z_thresh` in the ML cache fingerprint. Since bottleneck flags are the actual training target for the risk classifiers (`uac_ml.py:136-137`) and a feature for every model (`uac_ml.py:112-113`), **moving the sensitivity slider on any page silently changes what "elevated risk" means on the ML page, and forces a full model retrain** next time that page runs.
- **Why it happens:** the slider was designed as a shared, global control (reasonable for the Bottleneck Detection page), but it was never surfaced as also controlling ML target definitions — nothing on the ML page tells the user their risk score depends on a control they set three pages ago.
- **Fix:** either (a) display the current `z_thresh` value directly on the ML page next to the risk metrics ("risk model trained using bottleneck threshold = 1.5"), or (b) decouple the ML page's bottleneck definition from the sidebar slider entirely and fix it at a documented default.
- **Est. fix time:** 15–30 min for (a).

---

## PHASE 1 — Architecture

| Module | Score /10 | Notes |
|---|---|---|
| `uac_metrics.py` | 8 | Clean single-responsibility module — load → compute → detect. Well-commented, matches the brief's KPI formulas exactly. No circular imports. |
| `uac_ml.py` | 6 | Does too much in one file (feature engineering + 3 model types + risk classification + suite orchestration = 441 lines). Not currently a maintenance problem at this size, but the next feature added here should trigger a split (e.g. `uac_ml_features.py`, `uac_ml_models.py`). |
| `filters.py` | 7 | Good shared-sidebar pattern, correctly uses `st.cache_data`. Docstring undersells what `z_thresh` actually controls (see correction above). |
| `app.py` + `pages/*.py` | 7 | Consistent structure across all 5 pages (render_sidebar → guard empty → render). Some duplicated chart-building boilerplate (bottleneck marker overlay logic appears near-identically in `pages/3_Bottleneck_Detection.py:42-57`) that could be a shared helper — not urgent, but a DRY violation worth naming.

**Dependency graph:** clean and acyclic — `uac_metrics.py` has zero internal dependencies; `uac_ml.py` depends only on the enriched DataFrame shape (not on `uac_metrics.py` directly, coupling is implicit via column-name contract, see Phase 2); `filters.py` depends on `uac_metrics.py`; every page depends on `filters.py` and, for page 5, `uac_ml.py`. No circular imports found.

**Dead code:** `filters.py:4` imports `pandas as pd` — genuinely unused (confirmed via grep, zero `pd.` usages in the file). 🔵 trivial cleanup.

---

## PHASE 2 — Code Review

**CONFIRMED — 🟡 Minor — Implicit column-name contract, no schema validation at the boundary**
`uac_ml.py:87-98` (`_require_columns`) checks that required columns *exist*, but never validates dtypes (e.g. that `cbp_bottleneck`/`hhs_bottleneck` are actually boolean, or that `date` is actually datetime64). If `uac_metrics.py` ever changes a column's dtype, `uac_ml.py` fails downstream with a confusing sklearn error instead of a clear one at the boundary.
- **Fix:** assert dtypes, not just presence, in `_require_columns`.
- **Est. fix time:** 15 min.

**CONFIRMED — 🟡 Minor — Hardcoded date format is a silent-data-loss risk**
`uac_metrics.py:32` → `pd.to_datetime(df["date"], format="%B %d, %Y", errors="coerce")`. If HHS ever changes their CSV's date format (e.g. `"2026-01-01"` instead of `"January 1, 2026"`), every row silently becomes `NaT` and gets dropped by `dropna(subset=["date"])` on the next line — no error, no warning, just an empty or truncated dataset.
- **Reproduce:** feed a CSV with ISO dates through `load_data()` — it returns 0 rows, no exception raised.
- **Fix:** after the coerce step, assert `df["date"].notna().mean() > 0.9` (or similar) and raise a clear `ValueError` naming the expected format if the drop rate is high.
- **Est. fix time:** 15 min.

**CONFIRMED — 🟡 Minor — Silent conflict resolution in deduplication**
`uac_metrics.py:43` → `df.drop_duplicates(subset="date")` silently keeps the first row for any duplicate date without logging how many rows were dropped or whether the discarded rows had conflicting values. For a government dataset, a silently-discarded conflicting row is worth a log line at minimum.
- **Fix:** `dupes = df.duplicated(subset="date").sum(); if dupes: print(f"Dropped {dupes} duplicate-date rows")` before the drop.
- **Est. fix time:** 10 min.

**CONFIRMED — 🔴 Critical (now fixed) — Active-model selection excluded the baseline from competition**
Already found and fixed earlier in this conversation: `uac_ml.py:418` (pre-fix) built the candidate pool as `[tabular[target]] + ([lstm[target]] if available else [])` — **the persistence baseline was computed and displayed but never allowed to win**, contradicting the pipeline's own stated design principle ("always report a persistence baseline... select active forecast by lower validation MAE"). Verified the effect was real: Tabular ML validation MAE was ~10x worse than the baseline (115 vs 11 for HHS discharges) and was still marked Active. Fixed by including `baseline[target]` in the candidate list; verified the baseline now correctly wins and the live deployed app reflects it. Documenting here for completeness since a code audit should have caught this itself, not relied on the user asking "did we build the right model."

**CONFIRMED — 🟡 Minor — No test coverage for the exact class of bug above**
`tests/test_uac_ml.py` never asserts that the model with the lowest validation MAE is the one actually marked `Active`, or that persistence can win. This is why the bug shipped despite "all tests passing" — the tests checked structural correctness (leakage safety, output shape, bounded probabilities) but never checked *selection correctness*.
- **Suggested new test:**
```python
def test_active_model_is_the_lowest_validation_mae_candidate(self):
    result = run_ml_pipeline(synthetic_enriched(210), lstm_epochs=2)
    for target, forecast in result.forecasts.items():
        rows = result.comparison[result.comparison["Target"] == target.replace("_", " ").title()]
        self.assertEqual(forecast.model_name, rows.loc[rows["Validation MAE"].idxmin(), "Model"])
```
- **Est. fix time:** 20 min.

**CONFIRMED — 🟡 Minor — Missing type hints on several public functions**
`uac_metrics.py`: `weekday_weekend_summary`, `month_over_month_summary` are typed; `episodes()` inside `pages/3_Bottleneck_Detection.py:60` has zero type hints and is a module-level function nested in a page script — not a bug, but inconsistent with the typed style used everywhere else.

**CONFIRMED — 🔵 Enhancement — No logging module used anywhere**
All 9 files use zero `logging` calls — any diagnostic output would need `print()` (which is what the `__main__` block in `uac_metrics.py:205-213` does). Fine for a portfolio project; would need `logging` with proper levels before any real deployment where you'd want to distinguish info/warning/error in production logs.

**No mutable default arguments found** — checked every function signature; none use `def f(x=[])` or `def f(x={})` patterns. ✅

**No bare `except:` clauses found** — checked every file; no exception handling exists at all in the core logic (which is arguably the opposite problem — see Phase 8).

---

## PHASE 3 — Streamlit Review

**CONFIRMED — 🟠 Major — NaN renders as literal text on KPI cards for narrow/early date ranges**
Reproduced directly: filtering to a 3-day window near the start of the dataset (2023-01-25 to 2023-01-27, within the ~10-session warm-up period for 30-session rolling KPIs) produces `NaN` for `avg_pipeline_throughput_30d` and `avg_outcome_stability_30d`. Because `app.py:45` formats these as `f"{stats['avg_pipeline_throughput_30d']:.2f}x"`, the rendered KPI card literally reads **`nanx`**, and the backlog card reads **`+nan /session`**. This is not a crash, but it's the kind of thing a reviewer notices immediately and it undermines confidence in the whole dashboard.
- **Reproduced with actual data**, not assumed.
- **Same pattern exists in `pages/3_Bottleneck_Detection.py:24,26,29`** (`z={latest['cbp_stock_z']:.2f}`) — the z-score is also NaN during the same warm-up window.
- **Fix:** wrap every such f-string with a NaN guard, e.g. a small helper: `def fmt(x, spec, suffix=""): return "—" if pd.isna(x) else f"{x:{spec}}{suffix}"`.
- **Est. fix time:** 30–45 min to apply consistently across `app.py` and `pages/3_Bottleneck_Detection.py`.

**CONFIRMED — 🟡 Minor — Empty-dataframe guard exists but doesn't cover the "too-small" case**
`app.py:21` and equivalent lines in pages 1–4 correctly guard `if df.empty: st.warning(...); st.stop()`. But as shown above, a 1–3 row dataframe is *not* empty and sails past this guard straight into the NaN-rendering issue. The guard should arguably be `if len(df) < 10` for pages that rely on rolling-window KPIs, not just `df.empty`.

**CONFIRMED — 🟢 Working as designed, not a bug — ML page ignores the date-range filter**
Verified in `pages/5_ML_Forecast_and_Risk.py:15,26-31`: the page deliberately uses `full_df`, not the filtered `df`, for both training and the "Recent actual" chart — the date-range sidebar control genuinely has no effect on this page's forecast, by design (the model always forecasts from the end of the full dataset forward). This matches what I told you earlier and holds up under this audit. The z-score slider issue above is the real, previously-uncaught problem — the date range was correctly identified as inert.

**NEEDS RUNTIME VERIFICATION — Caching cost accumulation**
`pages/5_ML_Forecast_and_Risk.py:25` uses `st.cache_resource` keyed by a fingerprint that includes `z_thresh` at 0.1 increments across a 1.0–3.0 range (21 possible values). Each distinct value the user drags through triggers a full retrain (28 GBM fits + 2 risk classifiers + optional LSTM), and `cache_resource` keeps every distinct cached result in memory for the life of the server process. In a long-running shared deployment (not a fresh session each time), this could accumulate meaningfully — I did not run a live multi-user session to confirm actual memory growth, so this is flagged for verification rather than asserted as fact.
- **Mitigation if confirmed:** round `z_thresh` to fewer effective buckets before building the fingerprint, or switch to `st.cache_resource(max_entries=...)`.

**No `st.session_state` used anywhere outside widget `key=` bindings** — confirmed via grep. This is actually fine here (Streamlit's automatic widget-state persistence via `key=` is sufficient for this app's needs), just noting there's no custom session-state logic to audit for the usual rerun-related bugs.

**Mobile layout / dark mode / accessibility** — **NEEDS RUNTIME VERIFICATION.** All pages use `layout="wide"` and Plotly with `width="stretch"`, which is the correct modern pattern, but I cannot verify actual rendering on a phone-width viewport or a dark-mode OS setting without a live browser session. Flagging as unverified rather than claiming either way.

---

## PHASE 4 — Machine Learning Audit

**Data leakage: CONFIRMED CLEAN.** Traced every `.shift()` call:
- Feature lags/rolling windows (`uac_ml.py:108-111`): all use `.shift(1)` after rolling, or plain lag `shift(lag)` with `lag >= 1` — strictly historical.
- Targets (`uac_ml.py:130,133,136-137`): all use `.shift(-step)` for `step in 1..horizon` — strictly future, and only ever assigned to `targets`/`dates`/`risk_targets`, never to `features`.
- Train/validation/test splits (`make_temporal_splits`, `uac_ml.py:154-163`): strictly chronological slicing of a sorted index, `train < validation < test`, no shuffling anywhere in the pipeline. Verified no `sklearn.model_selection.train_test_split` (which would shuffle by default) is used anywhere.
- Scalers for the LSTM path (`uac_ml.py:286-287`) are `.fit()` on `splits.train_index` only, then `.transform()`'d onto validation/test — correct, no fit-on-test leakage.

**CONFIRMED — 🟡 Minor — Final production models are refit on 100% of the data, including the test period**
`uac_ml.py:212-214` (tabular) and `uac_ml.py:337-341` (LSTM): after evaluating on the held-out test split, the actual forecast that gets shown to the user comes from a *second* model refit on the entire dataset (train+validation+test combined). **This is standard, defensible ML practice** — you evaluate honestly on held-out data, then use all available data for the model that actually ships — but it does mean the displayed validation/test MAE numbers describe a *different, slightly less-informed* model than the one producing the actual forecast shown on the page. This should be stated explicitly in the page's methodology expander (it currently isn't) so a technical reviewer doesn't mistake it for leakage.
- **Fix:** one sentence in the methodology expander: "Displayed accuracy metrics come from a model trained only on data before the test period; the live forecast above is produced by a separate model refit on the complete dataset for maximum recency."
- **Est. fix time:** 5 min.

**Reproducibility: CONFIRMED, mostly.** `random_state=42` is set for both `HistGradientBoostingRegressor` and `HistGradientBoostingClassifier` (`uac_ml.py:177,371`), and `torch.manual_seed(seed)` / `np.random.seed(seed)` / `random.seed(seed)` are all set before LSTM training (`uac_ml.py:283-285`). **One gap:** `permutation_importance` (`uac_ml.py:388-395`) does set `random_state=42`, so that's also reproducible — confirmed clean on closer read.

**CONFIRMED — 🟡 Minor — Recursive error accumulation is not applicable here, but worth stating why**
The forecast is **direct multi-horizon** (`uac_ml.py:204`, one independently-trained model per horizon step 1–7), not recursive (predict step 1, feed it back in to predict step 2, etc.). This is actually the *safer* choice — it avoids the classic recursive-forecast error-compounding problem entirely. Worth noting as a deliberate strength, not a gap, since the audit brief specifically asked about this.

**CONFIRMED — 🟡 Minor — Missing value handling is a single global strategy**
`SimpleImputer(strategy="median")` (`uac_ml.py:169,363`) is applied uniformly to every feature, including the cyclical `weekday_sin/cos`, `month_sin/cos` features and the boolean-lag features. Median imputation is a reasonable default but isn't feature-aware (e.g. a missing boolean lag might be more sensibly imputed as `0`/False than as a median of 0.02). Low-impact given how few NaNs survive the `build_supervised_frame` eligibility filter, but worth naming.

**Risk score calibration: NEEDS RUNTIME VERIFICATION.** `HistGradientBoostingClassifier.predict_proba` outputs are used directly as "probability" without any calibration step (e.g. Platt scaling or isotonic regression via `CalibratedClassifierCV`). Tree-ensemble probabilities are frequently not well-calibrated out of the box. I did not run a reliability diagram against this specific dataset to confirm the miscalibration is actually present at a meaningful magnitude — flagging as a methodological gap to check, not a confirmed defect.

**SHAP readiness:** not implemented (permutation importance is used instead, which is a valid and arguably more robust alternative for this model type). Not a defect — just noting SHAP specifically wasn't used, per the audit brief's checklist.

---

## PHASE 5 — Data Audit

**CONFIRMED clean, previously established in this conversation:** no duplicate dates (`drop_duplicates(subset="date")` enforces this structurally), dates parsed and sorted correctly, no timezone handling needed (single daily cadence, no timestamps). **Known and already-documented gap:** near-total absence of Friday/Saturday reporting sessions (2 Friday, 0 Saturday out of 720) — this was caught and flagged in the research paper and dashboard captions already, not a new finding.

**CONFIRMED — no `.merge()`, `.join()`, or `.pivot()` calls anywhere in the codebase** — the data model is simple enough (single time-indexed table) that none are needed. Nothing to audit here.

**CONFIRMED — `.groupby()` usage is straightforward and correct** in `weekday_weekend_summary` and `month_over_month_summary` (`uac_metrics.py:144,157`) — no unintended row multiplication or silent NaN groups found on inspection.

**CONFIRMED — `.rolling()` windows use explicit `min_periods`** everywhere (never the pandas default of requiring a full window), which is the correct choice to get partial results during the warm-up period instead of a wall of NaN — though as shown in Phase 3, those legitimate early NaNs do surface as raw "nan" text in the UI if a user filters into that window.

---

## PHASE 6 — Performance Audit

**CONFIRMED — 🟠 Major — Redundant double-fit pattern trains ~28 gradient-boosting models per pipeline run**
`_fit_direct_forecast` (`uac_ml.py:202-223`) trains **two** `HistGradientBoostingRegressor` instances per horizon step — one on `train_index` for evaluation, one on the full dataset for the actual forecast — across 7 horizons × 2 targets = 28 total regressor fits per pipeline run, plus 2 more for the risk classifiers' final refit, plus the LSTM's two-stage training. This is architecturally sound (see Phase 4) but is the main driver of the page's multi-second load time, and it's not incremental — every distinct `(date_range, z_thresh)` combination reruns all of it from scratch.
- **Optimization:** none needed for correctness, but if load time becomes a real UX problem, `HistGradientBoostingRegressor` supports `warm_start` for incremental fits, or the 7 horizons could share more computation via a native multi-output regressor.
- **Est. fix time:** 2–4 hours if pursued (moderate refactor, not urgent at current data size of 720 rows).

**CONFIRMED — no repeated/uncached expensive dataframe recomputation found in the metrics pipeline.** `filters.py:11` correctly uses `@st.cache_data` on `get_data()`, so `build_full_dataset` (which does the CSV load + all rolling computations) only reruns when `z_thresh`/`stagnation_min_run` actually change — not on every widget interaction.

**No large object copies or obvious memory leaks found** in `uac_metrics.py` — `.copy()` is used deliberately (not excessively) to avoid mutating caller data, standard pandas practice.

**NEEDS RUNTIME VERIFICATION — Plotly rendering cost.** All charts render the full dataset (up to 720 points) as `go.Scatter` line traces — this is well within Plotly's comfortable range and unlikely to be a real bottleneck, but I did not profile actual browser render time.

---

## PHASE 7 — Security Audit

**CONFIRMED clean across the board** — this is a low-risk surface given the app's design:
- No hardcoded secrets, API keys, or credentials anywhere (grepped all files).
- No `pickle`, `eval()`, `exec()`, `os.system()`, or `subprocess` calls in application code (the one `subprocess` usage is in the test suite, running a Python compatibility check, not user-facing).
- No file upload widgets (`st.file_uploader`) anywhere — the CSV is a fixed bundled file (`filters.py:8`), so there's no user-controlled file input surface, no CSV-injection risk from external uploads, and no path-traversal risk.
- No `os.environ` reads for secrets, nothing logged that could contain sensitive data (the dataset itself is aggregate public HHS statistics, not individual-level PII).
- **Dependency vulnerabilities:** not checked — I did not run `pip-audit` or an equivalent CVE scan against the pinned/unpinned versions in `requirements.txt`. Flagging as unverified rather than claiming clean.

---

## PHASE 8 — Testing Audit

**Current coverage: 4 tests total** — 2 unit tests on feature/target construction (`test_uac_ml.py`), 1 end-to-end pipeline sanity test, 1 Streamlit runtime smoke test covering all 5 pages. All confirmed passing as of the last run in this conversation.

**CONFIRMED gaps** (each is something the test suite does *not* currently check, verified by reading `tests/test_uac_ml.py` and `tests/test_streamlit_runtime.py` in full):

| Missing test | Why it matters |
|---|---|
| Active-model selection correctness (see Phase 2) | This exact gap let the persistence-baseline bug ship silently. Highest-priority test to add. |
| Empty dataframe passed to `build_supervised_frame` | Currently raises `ValueError` (confirmed by reading `uac_ml.py:142-143`) — behavior is correct, but untested, so a future refactor could silently break it. |
| One-row / few-row dataframe (below the 120-session minimum) | `make_temporal_splits` raises `ValueError` (`uac_ml.py:156-157`) — correct behavior, but also untested. |
| Duplicate timestamps reaching `uac_metrics.load_data` | Structurally prevented by `drop_duplicates` (Phase 5), but no test asserts this holds if that line is ever removed/changed. |
| Constant target column (e.g. a stretch of days with identical `hhs_discharged`) | Not tested — HistGradientBoostingRegressor handles zero-variance targets fine mechanically, but MAE/RMSE become uninformative in that regime; no test documents expected behavior. |
| NaN-rendering on narrow date-range filters (Phase 3 finding) | Zero UI-level tests exist for this at all — the Streamlit runtime test only checks "no exception," not "no literal nan text." |
| Corrupted/malformed CSV input | `load_data`'s hardcoded date-format risk (Phase 2) has no test coverage. |

---

## PHASE 9 — Deployment Audit

**Honest scope note:** there is no Dockerfile, no CI/CD configuration (no `.github/workflows/`), and no environment/config management beyond `requirements.txt` and `.streamlit/config.toml` anywhere in this repository. This is **appropriate for the project's actual stage** (a portfolio/submission project deployed via Streamlit Community Cloud) — I'm not marking these as defects, since building out Docker/CI infrastructure for a project at this stage would be over-engineering, not a fix. Listing what exists and what would be needed if this ever became something bigger:

- **`requirements.txt`** — 🟡 Minor: all dependencies use `>=` with no upper bound and no lockfile (`requirements.txt:1-6`). Fine for a single-deploy portfolio project; would be a real reproducibility risk for a team project or long-lived production service, since a future `pip install` could silently pull a breaking major version of `scikit-learn` or `streamlit`.
- **`.streamlit/config.toml`** — exists, theme-only configuration (confirmed present, not reviewed line-by-line since it's cosmetic and out of scope for a bug audit).
- **No health-check endpoint, no structured logging, no monitoring** — not applicable at this deployment tier (Streamlit Community Cloud handles process supervision itself); would matter if self-hosting on raw infrastructure.

---

## PHASE 10 — UI/UX Audit

Based on the screenshots you've shared earlier in this conversation plus static review of the layout code:

- **Consistent hierarchy across pages** — every page follows title → caption → alerts/metrics → charts → detail tables, which reads as a deliberate, professional pattern rather than ad-hoc.
- **KPI cards use `help=` tooltips consistently** (`app.py:34,40,46,53,59`) — good practice, not something every student project bothers with.
- **The NaN-text issue (Phase 3) is the single biggest UI polish gap** — it's the kind of thing that, cosmetically, makes an otherwise professional dashboard look unfinished the moment someone picks an unusual date range.
- **Dark mode / mobile layout** — genuinely unverified (see Phase 3), not claiming either way.
- I won't manufacture "redesign suggestions" (spacing, typography, icons, color palette) beyond what I can see in your screenshots — those earlier screenshots show a clean, readable default Streamlit theme with a purposeful teal/navy accent; I don't have enough visual evidence to responsibly critique spacing/typography choices I haven't seen rendered.

---

## Scores

| Category | Score | Basis |
|---|---|---|
| Architecture | 7/10 | Clean separation, minor DRY violation, no circular imports |
| ML methodology | 8/10 | Leakage-safe, correctly chronological, direct (non-recursive) forecasting, reproducible seeds — docked for the now-fixed selection bug and the untested calibration question |
| Security | 9/10 | Minimal attack surface, nothing hardcoded, no unsafe patterns found — docked only for unverified dependency CVEs |
| UI/UX | 6/10 | Solid structure, docked hard for the confirmed NaN-rendering defect and the z-threshold coupling being invisible to the user |
| Maintainability | 7/10 | Well-commented, consistent style, `uac_ml.py` approaching the size where it should split |
| Production readiness | 5/10 | Appropriate for its actual stage (portfolio/submission), genuinely not production-hardened (no dependency pinning, no logging, no dependency scanning) — this is not a criticism of the project's goals, just an honest label |

---

## Top fixes, ranked by actual impact

1. 🔴→✅ Model-selection bug — **already fixed and deployed**, verified live.
2. 🟠 NaN-text on KPI cards for narrow date ranges — confirmed reproducible, ~30–45 min fix.
3. 🟠 Z-threshold silently changing ML risk definitions with no on-page indication — ~15–30 min fix.
4. 🟡 Add a test asserting active-model selection is actually correct — ~20 min, closes the exact gap that let #1 ship.
5. 🟡 Hardcoded date-format parser with silent data loss on format change — ~15 min guard.
6. 🟡 Pin dependency versions for reproducibility — ~10 min, trivial but real.
7. 🔵 Remove unused `pandas` import in `filters.py` — 1 min.

## Bottom line

The core engineering — the leakage-safe ML pipeline, the KPI methodology, the chronological evaluation discipline — is genuinely solid and holds up under a real line-by-line read, not just "tests pass." The defects that exist are real but shallow: a selection-logic bug (now fixed), a display-formatting gap on an edge case, and one piece of UI behavior (the z-slider coupling) that needed to be surfaced more honestly to the user — which includes the correction I owed you at the top of this report.

**Suitability:**
- ✅ College/Diploma project — well above bar.
- ✅ Internship portfolio piece — strong, especially with the fixes above applied and documented (the fact that you caught and fixed a real selection bug is a *better* portfolio story than if it had been perfect from the start).
- ✅ Open source — viable as-is, would benefit from the dependency pinning and a couple more tests before wide publication.
- 🟡 Startup MVP — the engineering discipline is there; would need the production-readiness items (logging, dependency scanning, error handling around the date-format assumption) before trusting it with real operational decisions.
- ❌ Enterprise deployment — not because the code is bad, but because no project at this stage (no CI/CD, no monitoring, single-file CSV data source) should go straight to enterprise without that surrounding infrastructure — this is true of essentially any student/portfolio project, not a specific criticism of this one.
