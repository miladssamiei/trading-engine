# GATE.md — Phase 1 signal-quality gate, pre-declared BEFORE any Phase 1 analysis ran

Written 2026-08-26, before `evaluate/vwap_trend_signal_quality.py` was executed
(the aggregation/verification data-prep step had not run either). Nothing in
this file may be changed after Phase 1 numbers are seen. This mirrors the
discipline of `data/results/vwap_rr_exploration/GATE.md`.

## What Phase 1 measures

Every qualifying pullback-entry signal per the Step 0 rule (full ambiguity
table in REPORT.md), on US500-tick M5 session bars, 2021-01-01 → 2026-05-01,
with NO exit rule applied. Per signal, in the TREND direction, in % of the
entry price (entry = open of the entry bar):

- **MFE** — best favorable excursion: LONG max(high)/entry−1, SHORT 1−min(low)/entry
- **MAE** — worst adverse excursion: LONG 1−min(low)/entry, SHORT max(high)/entry−1

Walk is UNCENSORED (no stop/target truncation): from the entry bar inclusive
(its full high/low range is achievable from its open) through the last
session bar OPENING before 15:55 NY (the 15:50 bar, which completes at
15:55), plus the 15:55 bar's OPEN as one final price point (the strategy's
life ends at 15:55:00; that bar's later high/low is not achievable).

## Data basis (declared)

M1 export (already verified full-range: 1,791,504 bars, 64/64 months, 0
multi-day gaps, date set == the verified M5 reference) aggregated to M5 with
tick_volume = Σ M1 tick_volume. BEFORE any signal work, the aggregated OHLC
must be verified bar-for-bar against ORB's terminal-native M5 export
(364,330 bars) at tolerance 1e-6. **Abort Phase 1 if the OHLC mismatch rate
exceeds 0.05% of common bars** (would mean the aggregation is not faithful;
investigate before proceeding).

## Gates (both must pass to proceed to Step 3 exit design)

- **G1 (excursion dominance):** count(MFE ≥ t) > count(MAE ≥ t) at ALL of
  t = 0.10%, 0.15%, 0.20% (threshold granularity mirrors the VWAP MR
  diagnostic whose table showed the reverse — MAE dominating — for that
  strategy's entries).
- **G2 (sample size):** ≥ 150 qualifying signals over the full range.

If either fails: write the null result and STOP — no exit-rule design, no
stop/target grid, no MT5 EA. A confirmed "continuation does not beat
reversion at these entry points on this instrument/timeframe" is the
deliverable, exactly as the sticky-HMM regime experiment stopped at its gate.

## Honesty notes (declared up front, before seeing numbers)

- Signals are treated as INDEPENDENT EVENTS: no position occupancy modeling.
  Two signals from different drift episodes on the same day may overlap in
  book terms; a later full re-simulation (Step 3, only if gated through)
  resolves path dependence properly — the same documented-approximation
  pattern as the VWAP R:R Phase 1.
- The gate uses RAW signal counts; per-day-deduplicated counts are reported
  alongside as context only.
- Informational only (non-gating): mean/median/p90 of MFE and MAE, per-year
  signal counts, direction split, and a first-crossing-order breakdown.
- The 0.10/0.15/0.20 thresholds and the ≥150 minimum were fixed by analogy to
  the two prior strategies' diagnostics before this strategy's first number
  was computed.
