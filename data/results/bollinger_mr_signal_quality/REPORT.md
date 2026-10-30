# Phase 1 REPORT — Bollinger Band mean reversion (Strategy 6) on US500-tick M15: NULL RESULT, stopped

**Run:** 2026-08-28 · Spec/gates pre-declared in `SPEC.md` / `GATE.md`
BEFORE any code or run · Data: M15 aggregated from the verified M5
export (2021-01..2026-04). Hypothesis #9, tested as an independent
mechanism from VWAP MR (band extremity vs a 20-period MA on M15, not
VWAP deviation on M1).

## Verdict

**FAIL → STOP. The pooled signal set fails G2 (dominance): mean MFE
0.1440% vs mean MAE 0.1444% — mean(MFE−MAE) = −0.0004%, as flat as a
null can be — and count-dominance fails at 0.10% (92 vs 100) and 0.15%
(63 vs 68), passing only at 0.20% (46 vs 43, noise).** G1 passed
barely (186 ≥ 150; the pre-declared floor was not lowered). G3's
bootstrap number (P(mean≤0) = 0.284) is NOT a pass in substance: the
90% CI [−0.025, +0.052] straddles zero — the data cannot distinguish
the signal from no edge, which is precisely what G2 already caught. No
Step 3, no exploratory grid (pre-declared as Step-3-only and unable to
rescue a Phase-1 entry failure), no EA.

## Data path (Step 0 finding, resolved before any signal was produced)

The script's verification gate aborted the first attempt: an M1→M15
aggregation has 13,883 partial buckets because the verified M1 export
SKIPS NO-TICK MINUTES; the M5 export (byte-exact vs the terminal in
prior studies) produces only 107 partial M15 buckets (0.09%) and agrees
with the M1-derived aggregation on OHLC for ALL 107,603 buckets
complete in both inputs (0 mismatches). Primary = M5→M15; decision and
evidence recorded in SPEC.md. 286 calendar dates skipped as
evening-only Sundays/holidays (VWAP MR convention).

## Filter funnel (pre-declared attribution view)

| stage | long | short | total |
|---|---|---|---|
| F0 raw band closes (in-session, warmup, BW>0) | 495 | 507 | 1,002 |
| F1 after range-day (rolling 0.6% net / 1.0% H-L) | 183 | 228 | 411 |
| F2 after max-move (0.42% = 30 ES pts conversion) | 134 | 173 | 307 |
| F3 after entry window (signal ≤15:15, entry ≤15:30) | 80 | 106 | 186 |

The range-day filter is the dominant cut (−59%): band closes concentrate
on NON-range days — trending days are what reach the bands, which is the
structural reason this signal family and the range-day context fight
each other. The 0.42% max-move binds strictly inside the 0.6% net-move
condition as SPEC predicted (documented redundancy, kept verbatim).
186 signals on 118 days; all filled (no missing next bars).

## Results (direction-native, % of entry, M5 path to 15:55 forced exit)

| config | n | mean MFE | mean MAE | mean diff | ≥0.10 | ≥0.15 | ≥0.20 |
|---|---|---|---|---|---|---|---|
| pooled (gated) | 186 | 0.1440 | 0.1444 | **−0.0004** | 92/100 | 63/68 | 46/43 |
| long (info) | 80 | 0.1564 | 0.1627 | −0.0063 | — | — | — |
| short (info) | 106 | 0.1347 | 0.1306 | +0.0041 | — | — | — |
| episode-first (info) | 110 | — | — | −0.0235 | — | — | — |

Long slightly negative, short slightly positive — opposite signs, both
≈ 0: no direction carries the mechanism. Episode-level (first bar of
each band-touch run, the non-clustering view) is WORSE, killing the
"overlapping signals diluted a real edge" alternative. Yearly mean
diffs mixed 3/6 negative (2022 worst at −0.174) — no regime in which
this worked.

## Standalone finding vs VWAP MR (the same family question)

VWAP MR's Phase-1 signature was MAE-DOMINATED (adverse excursions beat
favorable at every threshold through 0.20%); this one is FLAT-ZERO
(mean diff −0.0004%, mixed threshold counts). Different signatures,
identical conclusion: the ENTRIES have no reversion edge. The
"does US500 mean-revert intraday from an intraday extremum under
range-day conditions" question is now 0-for-2 across two mechanically
distinct mean-reversion signals (VWAP-deviation M1, band-extremity
M15). Exit geometry could not have rescued this regardless: at the
signals, mean band width is 0.64% of price, so the published 1.5×BW
stop risks ~0.97% against a mid-band target ~0.32% away — an ~1:3
risk/reward against a flat coin, before the 0.8pt cost. Correlation
check not run (no P&L stream exists — stopped at Phase 1, same as every
prior null).

## Scope honesty

Tests the published rule set as resolved in SPEC.md (chart-convention
BB(20,2σ) on the continuous tape, VWAP-MR range-day/max-move/session
conventions reused verbatim, M15 signal granularity with M5 path
resolution) on ONE instrument (US500 CFD) over 5.3 years. It does not
refute Bollinger-based discretionary trading on other instruments,
timeframes, or with regime context.

## Artifacts

`SPEC.md`, `GATE.md` (pre-declared) · `phase1_funnel.csv` ·
`phase1_buckets.csv` (pooled/long/short/episode-first) ·
`phase1_yearly.csv` · `phase1_signals.csv` (186 signals with BW%,
beyond-band%, episode flags) · `phase1_stats.json` ·
`evaluate/bollinger_mr_phase1.py`
