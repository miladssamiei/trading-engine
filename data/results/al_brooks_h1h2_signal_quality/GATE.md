# GATE — pre-declared BEFORE the Phase-1 run (2026-08-27)

No Phase-1 output existed when this was written. Thresholds mirror the
project convention (0.10 / 0.15 / 0.20% MFE/MAE; SP2L G1 form).

## G1 — sample size
Each of the three configurations (H1, H2, H1+H2), NY arm, direction-adjusted
pooled: >= 150 filled eligible signals. (Bull/bear splits and H3+ are
reported but not gated.)

## G2 — dominance
Per configuration (NY arm, pooled direction-adjusted):
count(MFE >= t) > count(MAE >= t) for ALL t in {0.10, 0.15, 0.20}%
AND mean(MFE − MAE) > 0.

## G3 — at least one configuration passes
G1 AND G2 must hold for at least one of {H1, H2, H1+H2}; otherwise STOP
(null result, no Step 3).

## G4 — independence-aware robustness (lesson from intraday momentum)
For each configuration that passes G2: day-block bootstrap (10,000
resamples, resampling DAYS with replacement over per-day mean MFE−MAE of
filled signals): 5th percentile of the resampled mean > 0. At least one
G2-passing configuration must also pass G4 for Step 3 to proceed; per-config
results all reported. (Day blocks subsume same-pullback clustering within a
day; signals from the same pullback are never treated as independent
observations in this statistic.)

## H1 vs H2 — explicitly NOT a gate
Brooks' claim (H2 > H1) is the thing being measured; the comparison is
reported as a standalone finding whichever way it lands. It cannot cause a
stop by itself and cannot rescue a config that fails G1/G2/G4.

## Informational only (cannot gate)
ALL24 arm, bull/bear split, H3+, yearly sign agreement, first-touch win
rates, risk_brooks_pct distribution, abandoned-setup tallies.

## On failure
Write the null REPORT.md and STOP: no Step 3 exit design, no EA. Honest
stop-and-report is a complete deliverable (project convention).
