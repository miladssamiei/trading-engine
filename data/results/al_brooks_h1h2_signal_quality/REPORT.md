# Phase 1 REPORT — Al Brooks H1/H2 pullback on US500-tick M5: NULL RESULT, stopped

**Run:** 2026-08-27 · Spec/gates pre-declared in `SPEC.md` / `GATE.md` BEFORE
any code or run · Data: verified US500-tick M5 export (364,330 bars,
2021-01..2026-04).

## Verdict

**FAIL → STOP. All three configurations (H1-only, H2-only, H1+H2) fail the
pre-declared G2 dominance gate: mean(MFE − MAE) is NEGATIVE for every one
(nightly-block bootstrap agrees: P(mean ≤ 0) ≈ 1.0). No configuration
passes, so G3 ("at least one passes") fails → no Step 3, no exit design, no
EA.** The standalone Brooks-claim finding: **H2 is NOT measurably better
than H1 on this instrument — the day-level H2−H1 bootstrap difference is
−0.0013 with 90% CI [−0.018, +0.015] and P(H2 ≤ H1) = 0.55: no difference,
and both are negative.** "Markets often need two legs to correct" is not
visible in this tape's forward-excursion quality.

## Implementation (SPEC.md, mechanical, per Brooks)

Trend = close above EMA(60, M5 = 5 h) with positive 1-hour slope (SP2L
estimator convention + explicit slope; downtrend mirrored). Pullback begins
at the first bar-to-bar high failure; each trigger (high above prior bar's
high, after ≥1 lower-high bar) increments the leg counter — H1, H2, H3+;
new trend high or trend-filter loss resets. Entry: stop at the trigger
bar's high, 6-bar (30-min) timeout (flagged PROJECT choice — Brooks
silent); gap-through fills at the open. 90-min uncensored forward
excursions, SP2L E-XTM first-bar convention, NY 09:30–15:30 entry window,
session-end-capped. Brooks stop context (below pullback low) recorded per
signal — mean distance 0.12–0.23%, structurally sane.

Machine plausibility: 49,094 triggers over 5.3y (~36/day on the 23 h tape),
12,181 in the NY window, 9,636 filled (2,440 = 20% abandoned on timeout,
105 forward-window-ineligible) — a working, non-starved, non-degenerate
implementation.

## Results (direction-adjusted, % of entry, pooled over bull+bear)

| config | n | mean MFE | mean MAE | mean diff | ≥0.15%: MFE vs MAE | first-touch win | day-bootstrap p05 | P(mean≤0) |
|---|---|---|---|---|---|---|---|---|
| H1 | 4,541 | 0.2361 | 0.2433 | **−0.0072** | 2,438 vs 2,383 | 0.400 | −0.044 | 1.00 |
| H2 | 2,135 | 0.2359 | 0.2422 | **−0.0063** | 1,123 vs 1,101 | 0.403 | −0.043 | 0.99 |
| H1+H2 | 6,676 | 0.2360 | 0.2429 | **−0.0069** | 3,561 vs 3,484 | 0.401 | −0.044 | 1.00 |
| H3+ (info) | 2,898 | 0.2233 | 0.2208 | +0.0025 | 1,466 vs 1,403 | 0.404 | — | — |

Splits (informational): bull side is the worst (H1 bull −0.0154, H2 bull
−0.0083); bear side ≈ flat (H1 +0.0033 but count-dominance FAILS at 0.15%:
1,169 vs 1,224; H3+ bear +0.0208 is the only positive cell and was
pre-declared informational — not a rescue). Yearly mean diffs: negative in
5 of 6 years for H1 and H1+H2 (2024 the exception).

Honest note on the count/mean divergence: pooled count-dominance at 0.15%
slightly favors MFE in all pooled configs while means and the day-level
bootstrap say the opposite — the same "more frequent small wins, heavier
adverse tails" structure the intraday-momentum study exposed. G2 required
both, and the day-bootstrap (the specific lesson applied from that study)
settles it: after an H1/H2 breakout fills, the average forward excursion
AGAINST the entry exceeds the excursion in favor on this instrument.

## What this does and does not say

- It does NOT refute Brooks' books, which teach a discretionary, contextual
  method (with trend strength assessment, bar-body quality, measured-move
  targets and active management) on multiple instruments and timeframes.
  What is tested here is the exact mechanical core (bar-count pullback +
  prior-bar-high breakout entry, stop-entry, flat forward window) on ONE
  instrument (US500 CFD M5) over 5.3 years — and that mechanical core has
  no forward-continination quality here.
- The H1-vs-H2 comparison is a direct test of a specific, named claim from
  the source material and is reported as a standalone finding: NOT
  SUPPORTED — no measurable difference on this tape.
- Correlation vs the six prior hypotheses (Step 5 item) is not computed:
  the study stopped at Phase 1 per the pre-declared gate logic, so no
  trade/P&L stream exists to correlate. Family context only: this is a
  continuation-family setup, like trend-pullback and intraday momentum
  (both also null) — the continuation family is now 0-for-3 on this
  instrument.

## Artifacts

- `SPEC.md`, `GATE.md` (pre-declared) · `phase1_buckets.csv` (all configs,
  splits, thresholds) · `phase1_yearly.csv` · `phase1_signals.csv` (9,636
  filled signals with Brooks-stop distances) · `phase1_stats.json` (gates,
  H1-vs-H2 bootstrap, tallies)
- `evaluate/albrooks_phase1.py`
