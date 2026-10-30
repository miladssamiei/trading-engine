# Phase 1 REPORT — Intraday multi-horizon trend alignment (project-original hypothesis)

**Run:** 2026-08-27 · Data: verified US500-tick M5 export (364,330 bars, 2021-01..2026-04)
· Spec/gates pre-declared in `SPEC.md` / `GATE.md` BEFORE the run.

**Provenance label:** this is a NEW, project-original hypothesis (intraday
multi-horizon trend alignment, flat by session end). It is NOT a test of
Lempérière et al. (2014) and NOT a test of the video rubric; nothing here
replicates or refutes either source.

## Verdict

**Pre-declared gate: score6 (all 6 horizons incl. 12 h / 24 h) FAILS; score4
(fast horizons 30 m / 1 h / 2 h / 4 h) passes every pre-declared criterion —
by its letter.** Robustness cuts (informational, below) show the pass is
carried by overlapping forward windows clustered on trend days: the
day-unweighted bootstrap is significantly NEGATIVE. Per the pre-declared
pass logic, Step 3 proceeds with score4 — with the fragility documented here
and a day-level bootstrap gate (G4) pre-declared for Step 3 to answer it
directly.

## Step 0 decisions (per SPEC.md)

M5 bars (all horizons are 5-min multiples; verified ORB/trend-pullback
export). Calendar-time trailing anchors (12 h / 24 h span the overnight
close — Monday's 24 h anchor is Friday's close); 12 h / 24 h kept as slow
context, score4 pre-declared as the fallback set. NY session primary
(09:30..15:55, last effective signal ~15:25 given the ≥30-min forward
window), ALL24 secondary (information only). Flat by session end — no
swap/financing by design. Score sampled at every eligible M5 bar
(100,681 NY / 354,105 ALL24 eligible bars; 1,968 bars excluded for horizon
ties, ~0.5%).

## Gate evaluation (GATE.md, pre-declared)

| gate | score6 | score4 |
|---|---|---|
| G1 counts (≥50,000; every bucket ≥500) | PASS | PASS |
| G2 dominance at 0.10/0.15/0.20% + mean(MFE−MAE)>0 at top buckets | **FAIL** (mean diff −0.0004 / −0.0020 at \|s\|=4/6) | PASS (all counts dominate; mean diff +0.0058) |
| G3 ordering of mean diff 0→2→4(→6) | pass w/ one tolerated tie | PASS (strict: −0.0127 < +0.0012 < +0.0058) |
| G4 top bucket 0.15%-dominance ≥5/6 years | **FAIL** (4/6) | PASS (5/6; 2025 fails counts, +0.0008 mean) |

NY buckets (direction-adjusted, % of entry, 90-min uncensored window):

| set | \|s\ | n | mean MFE | mean MAE | mean diff | ≥0.15%: MFE vs MAE | first-touch win |
|---|---|---|---|---|---|---|---|
| score6 | 0 | 13,344 | 0.2326 | 0.2521 | −0.0195 | 6,900 vs 7,078 | 0.408 |
| score6 | 2 | 26,027 | 0.2414 | 0.2427 | −0.0013 | 13,844 vs 13,501 | 0.409 |
| score6 | 4 | 26,414 | 0.2414 | 0.2418 | −0.0004 | 14,006 vs 13,707 | 0.410 |
| score6 | 6 | 34,896 | 0.2333 | 0.2353 | −0.0020 | 18,562 vs 17,685 | 0.427 |
| score4 | 0 | 17,394 | 0.2309 | 0.2436 | −0.0127 | 8,979 vs 9,085 | — |
| score4 | 2 | 34,092 | 0.2376 | 0.2364 | +0.0012 | 17,770 vs 17,538 | — |
| score4 | 4 | 49,195 | 0.2446 | 0.2387 | **+0.0058** | 26,532 vs 25,379 (53.9% vs 51.6%) | — |

Key attribution: **adding the 12 h / 24 h horizons INVERTS the effect**
(score4 top bucket +0.0058 → score6 top bucket −0.0020): intraday
continuation after multi-day-aligned trends does not exist here — if
anything the slow horizons are mildly anti-predictive intraday. The
pre-declared fallback (score4) is therefore not just a rescue; it is the
empirical answer to the SPEC's open question about 12 h / 24 h usefulness:
they hurt.

ALL24 context (information only): same pattern, weaker (score4 \|s\|=4
+0.0018; score6 \|s\|=6 count-dominant but mean diff −0.0010) — the NY
window is not hiding anything material.

## Robustness cuts (informational — NOT gate criteria, run after the gate)

1. **Day-block bootstrap (1,369 days, 5k resamples, day-unweighted):**
   mean diff/day **−0.0156**, 95% CI [−0.0249, −0.0062] — significantly
   negative; hit-margin/day −0.0269, CI [−0.0436, −0.0097]. The POOLED
   +0.0058 is therefore produced by day-size weighting: trend days peg
   \|s4\|=4 for hours and contribute dozens of OVERLAPPING 90-min windows,
   each re-counting the same continuation move. The typical day is slightly
   negative.
2. **Direction split (top bucket):** longs mean diff −0.0023 (hit +0.030),
   shorts +0.0163 (hit +0.015) — the pooled edge leans on the short side;
   long-agreement bars are mean-reverting.
3. **Hour profile:** negative in the 09:00–10:59 morning, positive
   13:00–14:59 (+0.017..+0.019) — time-concentrated, opposite of the
   morning-momentum structure ORB exploits.

## Decision (per the pre-declared GATE.md pass logic)

score6 fails G2/G4 → score4 passes G1–G4 → **Step 3 proceeds with score4**
(TH4), despite the thin margins, exactly as pre-declared — the same
situation as trend-pullback (Phase 1 passed thin; the Step-3 WF gate made
the final call). One additional gate is pre-declared for Step 3 BEFORE that
run (see `../momentum_intraday_rr/GATE-STEP3.md`): a day-block bootstrap of
WF net P&L (5th percentile > 0), targeting precisely the overlap-fragility
diagnosed above. If Step 3 fails, the honest stop-and-report is the
deliverable.

## Artifacts

- `phase1_buckets.csv` (both arms, both score sets, all thresholds)
- `phase1_yearly.csv` (top-bucket dominance per year)
- `phase1_signals_ny.csv` (100,681 per-bar NY records for Step 3)
- `phase1_stats.json` (gate evaluation) · `SPEC.md` · `GATE.md`
- `evaluate/momentum_intraday_phase1.py`
