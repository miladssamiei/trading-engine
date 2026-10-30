# GATE — pre-declared BEFORE the Phase-1 run (2026-08-28)

No Phase-1 output existed when this was written. Thresholds mirror the
project convention (0.10 / 0.15 / 0.20% MFE/MAE; SP2L G1 form; the
intraday-momentum bootstrap lesson is a FIRST-CLASS gate from the start,
per the brief). The gated unit is the POOLED signal set (both
directions — the strategy trades both); direction splits are
informational.

## Phase 1 gates

### G1 — sample size
Pooled filled signals >= 150 (project convention). Floor set before the
funnel is seen; NOT lowered afterward.

### G2 — dominance
Pooled, direction-adjusted: count(MFE >= t) > count(MAE >= t) for ALL
t in {0.10, 0.15, 0.20}% AND mean(MFE − MAE) > 0.

### G3 — day-block bootstrap
10,000 resamples of per-day mean MFE−MAE (signals within a day are
related observations — consecutive band-touch bars are exactly the
clustering the intraday-momentum study exposed): P(mean <= 0) <= 0.40,
meaningfully below 0.5. p05 reported alongside.

### PASS
G1 AND G2 AND G3 → Step 3. Any failure → STOP: null REPORT.md with the
full funnel, no Step 3, no EA. Honest stop-and-report is a complete
deliverable.

## Step 3 gates (pre-declared; evaluated only if Phase 1 passes)

Primary = published exit exactly as specified (1.5×BW stop, mid-band
target), occupancy re-simulation, WF stability folds:

- S1: pooled WF net expectancy > 0 AND net Sharpe > 0 (net = after
  0.8 pt round-trip, $1/pt/lot, ex-financing).
- S2: total WF trades >= 150.
- S3: >= 4 of 5 full years net-positive AND no single year > 60% of
  total net P&L.
- S4: net-of-spread profitability (S1 restated as the explicit cost
  gate).

If S1–S4 pass on the published exit → Step 4 (EA build). If ANY fails:
the pre-declared exploratory neighborhood grid (stop multiplier
{1.0, 1.25, 1.75, 2.0}×BW; 1.5 already measured) runs in-sample with
an explicit EXPLORATORY label; a rescue attempt (WF, per-fold
trailing-expectancy selection across the 5 cells) must independently
pass S1–S4 or the study stops at null. The grid cannot rescue a
Phase-1-level entry failure (G1–G3 gate first).

## Informational only (cannot gate)
Long/short splits; episode-level (first-bar-of-run) statistics; yearly
sign agreement; BW% and 1.5×BW% risk-context distributions; funnel
counts per filter stage; VWAP-MR comparison narrative; halted-day
truncation tallies.
