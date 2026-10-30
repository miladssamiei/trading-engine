# GATE.md — pre-declared before Phase 2 runs (written 2026-08-26, after Phase 1, before any walk-forward execution)

This file declares the Phase 2 protocol and the pass/fail gates BEFORE the
walk-forward script is executed. Nothing below may be re-negotiated after
Phase 2 numbers are seen. Phase 1 output (phase1_grid.csv) is exploratory /
in-sample and may NOT be reported as a result.

## Shortlist (Step 3 rationale)

Grid (phase1_grid.csv) strict neighborhood rule — cell expectancy > 0 AND all
orthogonal neighbors > 0 — is passed by ZERO cells. The only structurally
consistent pattern in the grid is the **125%-of-VWAP target column**: it is
the best cell in every stop row (expectancy monotonically increasing in
target fraction at all four stops). The only two positive-expectancy cells
sit in that column:

- **Candidate A**: stop 0.10%, target 125% of the way back to VWAP (in-sample
  expectancy +0.152/trade; neighbors −0.231 / −0.224 → flagged isolated)
- **Candidate B**: stop 0.25%, target 125% (in-sample +0.047; neighbors
  −0.345 / −0.417 → flagged isolated)
- **Baseline C** (original rule): stop 0.20%, target 100% — included in the
  selection set so the walk-forward may honestly conclude "no candidate beats
  the original out-of-sample".

Both candidates are shortlisted DESPITE failing the strict neighborhood rule,
because the column-level structure is the only defensible hypothesis the grid
produced. This is recorded as a known weakness of the shortlist.

## Phase 2 protocol (fixed)

- Full re-simulation per candidate (entries AND exits re-run together) via a
  parameterized copy of the Python reference engine — resolves Phase 1's
  fixed-entries approximation. Baseline config run must reproduce the original
  281 trades / −$202.42 exactly (engine equivalence check).
- Folds: burn-in = calendar 2021; then 9 non-overlapping half-year folds
  2022H1, 2022H2, ..., 2026H1 (2026H1 = Jan–Apr).
- Selection at each fold boundary: trailing expectancy per trade on ALL data
  strictly before the fold; a candidate is eligible only with ≥ 20 trailing
  trades; if none eligible → baseline C. (Simulations are causal and trades
  never span days, so trailing stats are exact subsets of the full-range run.)
- Walk-forward result = the stitched sequence of per-fold SELECTED trades.
  This is the only number that may be called a result.

## Gates (declared now)

- **G1 (edge)**: walk-forward expectancy > $0.00/trade AND walk-forward daily
  Sharpe ≥ 0.00 — i.e. an improvement over the Python baseline's −0.486 by at
  least +0.486. A rescue that still lands at negative Sharpe is not a rescue.
  (Informational practical bar, not gating: expectancy ≥ +$0.50 and
  Sharpe ≥ +0.25.)
- **G2 (sample)**: walk-forward stitched trade count ≥ 150 over the range.
- **G3 (concentration)**: walk-forward PnL positive in ≥ 4 of the 5 calendar
  years 2022–2026(p) AND the best single year < 60% of the total walk-forward
  profit. (There is no second engine here, so the original report's
  month-sign-agreement bar is operationalized as a temporal-concentration
  guard.)

All three must pass for any candidate to be called confirmed. The MT5 EA is
NOT touched in this pass; a rebuild would be a follow-up step using
walk-forward-confirmed parameters only.

## Honesty note (inherent limitation of this design)

The candidate set itself was chosen by looking at Phase 1 on this same
2021–2026 history. Walk-forward discipline guards the SELECTION WITHIN the
set, not the set's genesis. A true out-of-sample test of the 125%-overshoot
hypothesis requires data after 2026-04 (not present in this terminal).
