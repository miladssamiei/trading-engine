# GATE — pre-declared BEFORE the Phase-1 run (2026-08-28)

No Phase-1 output existed when this was written. Project-convention
thresholds (0.10 / 0.15 / 0.20% MFE/MAE; SP2L G1 form; day-block
bootstrap as a first-class gate). Continuation family is 0-for-4 — the
gate is held to the SAME standard as every prior test; nothing is
relaxed for "one more chance", and nothing is tightened either.

## Phase 1 gates (gated unit = POOLED both-direction signal set)

### G1 — sample size
Pooled filled eligible signals >= 150. Set before the funnel is seen;
NOT lowered afterward.

### G2 — dominance
Pooled, direction-adjusted: count(MFE >= t) > count(MAE >= t) for ALL
t in {0.10, 0.15, 0.20}% AND mean(MFE − MAE) > 0.

### G3 — day-block bootstrap
10,000 resamples of per-day mean MFE−MAE: P(mean <= 0) <= 0.40
(meaningfully below 0.5). p05 reported alongside.

### PASS
G1 AND G2 AND G3 → Step 3. Any failure → STOP: null REPORT.md with the
full funnel and the five-family comparison; no Step 3, no EA. Honest
stop-and-report is a complete deliverable.

## Step 3 gates (pre-declared; evaluated only if Phase 1 passes)
- S1: pooled WF net expectancy > 0 AND net Sharpe > 0 (net of 0.8 pt
  round trip, $1/pt/lot, ex-financing).
- S2: total WF trades >= 150.
- S3: >= 4 of 5 full years net-positive AND no year > 60% of total net
  P&L.
- S4: net-of-spread profitability (S1 restated as the explicit cost
  gate).
Any failure → STOP: null REPORT.md, no EA. No exploratory exit grid
exists for this study (exit is fully specified by the source: swing SL,
fixed 1:1 TP).

## Informational only (cannot gate)
Long/short splits; yearly sign agreement; funnel per stage; risk-band
rejection counts; signals/day distribution and 2/day-cap impact;
multiple-zone-same-bar clustering counts; staleness expiry tallies;
adjacent-timeframe robustness arms (only if Phase 1 passes, clearly
labeled as context, never a rescue).

## On an unexpected PASS
Treat with MORE skepticism, not less (family track record): verify the
M15/M5 pairing is not a lucky accident via the pre-declared adjacent
pairs before any Step 4 work is trusted.
