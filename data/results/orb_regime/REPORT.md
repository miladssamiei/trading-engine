# HMM Regime Filter on ORB — Negative Result Report

**US500-tick · regime labels applied 2022-01-03 → 2026-04-30 (common window) · research-only, Python**

- **Report date:** 2026-08-24
- **Question:** does trading the validated ORB breakout only in HMM-inferred "high-volatility/trending" regimes improve it, out-of-sample, with zero look-ahead?
- **Answer: NO.** The pre-declared acceptance gate failed on all three criteria (0/3). The intended filter made ORB strictly worse, the *inverse* filter did better than the intended one, and no sensitivity variant beat the unfiltered baseline. Per the study plan's stop rule, no MT5 implementation was attempted and the validated `ORB_Strategy.mq5` remains untouched.

## 1. Design (pre-declared before any result was computed)

### 1.1 Anti-lookahead constraints — all enforced and verified BEFORE performance was computed

1. **No smoothing at decision time.** Every day-T label comes from a hand-rolled log-space scaled FORWARD filter (alpha recursion) only, evaluated at the position of day T's *preceding* observation day. Viterbi (`model.predict`) and forward-backward (`model.predict_proba`) are never called for labels anywhere in `orb_regime_hmm.py`; `model.score` is used only as an independent cross-check of the recursion (max |Δ| = 2.8e-11, requirement < 1e-4).
2. **No full-history in-sample fit.** Walk-forward: fit Gaussian HMM on a trailing window of observation days, freeze parameters (incl. feature scaler stats) for one application month, roll forward and refit. Documented per-month frozen parameters: `fit_blocks.json`.
3. **Prior-session-only observations.** Features for day D use only D's own RTH bars (09:30–15:55 NY); day T's label uses only observations with NY date < T (strictly before T's 09:30 decision cutoff). This mirrors the DPO_20 leak discipline: the leak check was built and run BEFORE any performance number existed.

### 1.2 Observation features (causal), states, walk-forward

- f1 = ln(RV+1e-12), RV = Σ squared log close-to-close returns of the day's RTH closes
- f2 = opening-range width % = (max high − min low of the 6 OR bars)/first OR open ×100
- f3 = |session return| % = |last RTH close − first OR open|/first OR open ×100
- Observation day = session day with exactly 6 OR bars and ≥2 RTH closes (1,369 of 1,370 session days qualify; 2023-09-14 carried).
- Variants: **w252_k2 PRIMARY** (trailing ~12mo fit, monthly refit, 2 states), sensitivities w126_k2 (6mo), w504_k2 (24mo, first applied month 2023-01 due to warmup), w252_k3 (3 states). Canonical state mapping by ascending emission mean of standardized f1 → LOWVOL(/MIDVOL)/HIGHVOL. hmmlearn 0.3.3, seed 42, 0 non-converged EM blocks.
- Labels: w252_k2 → 690 LOWVOL / 423 HIGHVOL days (1,113 labeled session days).

## 2. Leak validation (Step 3, run FIRST) — LEAKCHECK.md: ALL PASS

- **(a)** 13 sample days (incl. first/last applied, month boundaries, DST-adjacent 2022-03-14 / 2022-11-07, seeded-random): exact bar range printed; every last-observation bar is the prior session's 15:55 NY bar.
- **(b)** GLOBAL assertion over all 4 variants × all labeled days (4,194 rows): `last_obs_bar_ts_utc < decision_cutoff_utc` — **0 violations**, tightest margin 17h35m. PASS.
- **(c)** Feature-bar containment for 5 sampled days: recomputing f1 from bars < cutoff matches stored value exactly; including bars ≥ cutoff changes it (the audit bites). PASS.
- **(d)** Filtering invariance for 5 sampled days: label/p_highvol from obs[..T−1] equals the value with +5 future observations appended, |Δp| = 0.0e+00 — future data provably cannot move a label. PASS.
- **(e)** Structural statement: forward-only inference path. PASS.
- Determinism: sha256(labels) = `05cd4e3dbb0f6018…`, byte-identical on re-run (`--verify-determinism` PASS).
- Session-day definition cross-checked against `py_days.csv` (or_bars>0): exact set equality, 1,370 days.

## 3. Backtest method (Step 2)

The validated ORB baseline is strictly per-day independent (one trade/day max; entry/SL/TP/session-end exit all from day D's bars; the only cross-day input — the gap filter — is already encoded in `py_days.csv`). A day-gate regime filter therefore yields exactly the subset of baseline trades whose day-label matches; `orb_regime_backtest.py` computes variants as label-joined subsets of the verified `py_trades.csv` (442 trades, +$531.13 full-range), with metrics replicating `orb_stats.py` verbatim (daily PnL zero-filled over session days; Sharpe = mean/std(ddof=1)·√252; maxDD on cumulative PnL; CAGR ACT/365.25; return/DD). Common window 2022-01-03 → 2026-04-30 (4.32y, 1,058 session days; 368 baseline trades, +5.48%, Sharpe 0.460 — the 2022+ window is a better baseline period than full-range).

### Pre-declared gate (fixed in the script docstring before results existed)

- **G1:** PRIMARY (w252_k2 HIGHVOL-only) Sharpe ≥ baseline+0.15 AND return > baseline.
- **G2:** INVERSE (w252_k2 LOWVOL-only) strictly worse than PRIMARY on both.
- **G3:** ≥2 of 3 sensitivity HIGHVOL variants beat baseline on both Sharpe and return.

## 4. Results — gate FAILS 0/3

| variant | trades | winrate% | exposure% | PnL | ret% | CAGR% | maxDD% | Sharpe | ret/DD |
|---|---|---|---|---|---|---|---|---|---|
| baseline (window) | 368 | 52.2 | 33.1 | +547.91 | +5.48 | 1.24 | 4.99 | **0.460** | 1.10 |
| **PRIMARY** w252_k2 HIGHVOL | 170 | 54.7 | 15.3 | +160.22 | +1.60 | 0.37 | 4.12 | **0.177** | 0.39 |
| INVERSE w252_k2 LOWVOL | 198 | 50.0 | 17.8 | +387.69 | +3.88 | 0.88 | 3.75 | **0.500** | 1.03 |
| sens w126_k2 HIGHVOL | 189 | 55.6 | 17.0 | +345.65 | +3.46 | 0.79 | 3.03 | 0.368 | 1.14 |
| sens w504_k2 HIGHVOL | 88 | 52.3 | 7.9 | −4.36 | −0.04 | −0.01 | 3.60 | −0.006 | −0.01 |
| sens w252_k3 HIGHVOL | 125 | 56.0 | 11.2 | +205.55 | +2.06 | 0.47 | 2.46 | 0.261 | 0.84 |

- Buy&hold same window (ex-financing, $10k/1-lot): +24.38% (price +50.93%), Sharpe 0.621 — still far above every ORB variant.
- **G1 FAIL:** PRIMARY Sharpe 0.177 vs required ≥0.610; return 1.60% vs 5.48%.
- **G2 FAIL:** the inverse (LOWVOL-only, Sharpe 0.500, +3.88%) is BETTER than the intended filter — the study plan's own sanity criterion for "the regime labels aren't capturing what you think they are."
- **G3 FAIL:** 0 of 3 sensitivity variants beat baseline on both metrics (w126_k2 closest, still below on both; w504_k2 outright negative).
- Monthly stability: PRIMARY beat baseline in only 9 of 52 months (worse in 25, equal 18 — sticky multi-week regime runs mean the filter blocks whole months at a time).

## 5. Interpretation (reported, not acted on)

- The win rate on HIGHVOL days is slightly HIGHER (54.7% vs 52.2%) yet PnL is far LOWER: the strategy's payoff is right-skewed, and its large winners historically occurred on days the prior-session-volatility HMM classified as LOWVOL. Filtering for "high-vol/trending" regimes cut exactly the trades that carried the edge.
- The inverse filter's Sharpe 0.500 vs baseline 0.460 is a **post-hoc observation, not an adoptable result**: selecting the inverse hypothesis after seeing the primary fail is precisely the in-sample search this study was designed to forbid (and the margin is well within regime-labeling noise: the same LOWVOL filter under w126/w504 windows was not evaluated as an inverse candidate and the primary's own numbers collapsed under w504). No further variants, windows, state counts, or feature sets were tried, per the stop rule.

## 6. Consequence per the stop rule

- **No MT5 implementation.** Step 4 options (static daily regime file / native forward-filter) are moot; `ORB_Strategy.mq5` is unchanged and remains the validated artifact. No Python↔MT5 dependency of any kind was introduced at any point.
- Artifacts: `evaluate/orb_regime_hmm.py`, `evaluate/orb_regime_backtest.py`, `regime_labels.csv`, `fit_blocks.json`, `LEAKCHECK.md`, `regime_backtest.json`, `regime_backtest_summary.md` (52-row monthly matrix).
