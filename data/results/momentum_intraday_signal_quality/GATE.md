# GATE — pre-declared BEFORE the Phase-1 run (2026-08-27)

No Phase-1 output existed when this file was written. Nothing below may be
re-negotiated after seeing results. Mirror of the SP2L / trend-pullback
Phase-1 gates (0.10/0.15/0.20% MFE/MAE convention).

## Eligibility (G1 — sample size)
- NY arm: >= 50,000 eligible signal bars over 2021-01..2026-04.
- Every reported |score| bucket (each score_set, each arm that is reported
  as primary/secondary): >= 500 eligible bars.

## Dominance (G2 — the core signal-quality gate)
For BOTH |score|=4 and |score|=6 buckets of the primary score6 set
(NY arm), direction-adjusted:
- count(MFE >= t) > count(MAE >= t) for ALL t in {0.10, 0.15, 0.20}%
  (project G1 form), AND
- mean(MFE - MAE) > 0.

## Ordering (G3 — does magnitude matter?)
mean(MFE - MAE) strictly increasing across |score| = 0 -> 2 -> 4 -> 6
(score6) and across 0 -> 2 -> 4 (score4). One adjacent-pair tie is
tolerated if the gap is < 0.005 percentage points; anything else fails.

## Temporal stability (G4)
Top bucket (|score| = 6, NY arm): count(MFE >= 0.15%) > count(MAE >= 0.15%)
in >= 5 of the 6 calendar years (2021..2026 partial).

## Pass/fail logic (pre-declared)
- score6 passes G1-G4  -> proceed to Step 3 with score6.
- score6 fails any of G2/G3/G4 while score4 passes ALL of G1-G4
  (with its own buckets 0/2/4, top = |score|=4) -> proceed to Step 3 with
  score4 and document that the 12 h / 24 h horizons were the failure point.
- Both fail -> STOP. Write the null-result REPORT.md; no Step 3, no exit
  design, no sizing, no EA.

## Explicitly NOT gates (reported for information only)
- ALL24 arm buckets (window-choice context).
- First-touch win rates, mean MFE / mean MAE levels, long/short asymmetry.
- Any horizon-level single-horizon predictive table (context only).
