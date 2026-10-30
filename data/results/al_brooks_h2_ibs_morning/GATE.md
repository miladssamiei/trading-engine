# GATE — pre-declared BEFORE the Phase-1 run (2026-08-27)

No Phase-1 output existed when this was written. Thresholds mirror the
project convention (0.10 / 0.15 / 0.20% MFE/MAE; SP2L G1 form). The gated
unit is the COMBINED H2+L2 signal set (the strategy trades both mirrored
directions); per-direction splits are informational.

## G1 — sample size
Combined filled eligible signals >= 150 AND >= 80 distinct
signal-bearing day-blocks. This floor is set BEFORE seeing the funnel and
will NOT be lowered afterward: this variant stacks six filters, and if
the funnel cannot deliver 150 clean signals over 5.3 years, the study
FAILS on sample insufficiency — that is a legitimate null-outcome, not a
reason to re-open the gate.

## G2 — dominance
Combined, direction-adjusted pooled: count(MFE >= t) > count(MAE >= t)
for ALL t in {0.10, 0.15, 0.20}% AND mean(MFE − MAE) > 0.

## G3 — independence-aware bootstrap (the intraday-momentum lesson)
Day-block bootstrap, 10,000 resamples of per-day mean MFE−MAE (signals
within a day are related observations; days are the exchangeable unit):
P(mean <= 0) <= 0.40 — meaningfully below 0.5, not just nominally
positive. p05 is reported alongside (p05 > 0 noted as the strong form,
as in study #7, but the pre-declared bar is the P(mean<=0) condition).

## PASS condition
G1 AND G2 AND G3 on the combined set → Step 3. Any failure → STOP:
null REPORT.md, no Step 3, no EA. Honest stop-and-report is a complete
deliverable (project convention).

## Step 3 gates (pre-declared now; evaluated only if Phase 1 passes)
- S1: pooled walk-forward net expectancy > 0 AND net Sharpe > 0
  (net = after 0.8 pt round-trip spread, $1/pt/lot, ex-financing).
- S2: total walk-forward trades >= 150 (project convention; consistent
  with G1 — not lowered after the fact).
- S3: temporal concentration: >= 4 of 5 full years net-positive AND no
  single year > 60% of total net P&L.
- S4: net-of-spread profitability confirmed (S1 is already net; S4
  re-states it as the explicit cost-survival gate, per project form).
Any S-gate failure → STOP: null REPORT.md, no EA.

## Informational only (cannot gate)
Per-direction (H2 bull / L2 bear) buckets; yearly sign agreement; filter
funnel; risk_pct distribution and 1R descriptive counts; first-touch win
rates; signals/day distribution and 3/day cap impact; comparison vs
study #7's unfiltered full-session null; recommendation (yes/no) on a
follow-up exploratory fixed-R:R check.

## On failure
Write the null REPORT.md including the full filter funnel and the
standalone comparison against study #7 (fourth independent continuation-
family variant), and STOP.
