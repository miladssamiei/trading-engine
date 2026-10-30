# GATE.md — Step 3 (stop/target) protocol + gates, pre-declared BEFORE any Step 3 analysis ran

Written 2026-08-26, after Phase 1 signal-quality PASSED its gate, and
BEFORE `evaluate/vwap_trend_rr_grid.py` / `evaluate/vwap_trend_rr_walkforward.py`
were executed. Nothing here may be re-negotiated after Step 3 numbers are
seen. Structure mirrors `data/results/vwap_rr_exploration/GATE.md`.

## Grid (exploratory Phase 1 of Step 3)

Stops {0.10, 0.15, 0.20, 0.30}% × targets {0.15, 0.20, 0.30, 0.45}% = 16
cells. Chosen from THIS study's own Phase 1 excursion distribution (median
MFE/MAE ≈ 0.23/0.22, means ≈ 0.36/0.35, dominance emerging at ≥ 0.15%,
p90 ≈ 0.80) — NOT from the source video (whose point values are excluded
by the task definition).

Reconstruction per signal on the Phase 1 walk (entry-bar inclusive; first
crossing order decides; same-bar both-level ambiguity via the engines'
close≥open path heuristic; bar-open gap-through checks on later bars; time
exit at the 15:55 bar open; early-close days exit at last bar close). KNOWN
APPROXIMATION: fixed entries, no occupancy (declared, same pattern as the
VWAP R:R Phase 1).

Validation invariants asserted on every cell/signal: stop_loss outcome ⇒
MAE ≥ stop; win ⇒ MFE ≥ target; time exit ⇒ neither level crossed on the
walk.

## Shortlist rule (declared before the grid runs)

Strict neighborhood rule: a cell is shortlisted only if its expectancy > $0
AND all orthogonal neighbor cells' expectancy > $0. If ZERO cells pass,
shortlisting may fall back to the grid's dominant structural pattern (as in
the VWAP R:R exploration) with the weakness recorded explicitly — max 3
candidates.

## Phase 2 protocol (the only result)

Occupancy-modeled full re-simulation per candidate (entries only while
flat; signals during holding are skipped, not queued; server-style exits
with 0.01-grid stops/targets; 15:55 force exit; early-close carry to next
day's first bar open — identical conventions to the validated reference
engines). Folds: burn-in 2021, then 9 non-overlapping half-year folds
2022H1..2026H1. Selection at each fold boundary: best trailing
expectancy/trade on data strictly before the fold; eligibility ≥ 20
trailing trades; if none eligible → default candidate (stop 0.20%,
target 0.30% — the grid center, fixed a priori). Walk-forward result =
stitched selected folds.

## Gates (all must pass to proceed to Step 4 EA build)

- **G1 (edge):** walk-forward expectancy > $0.00/trade AND walk-forward
  daily Sharpe > 0.
- **G2 (sample):** walk-forward stitched trades ≥ 150.
- **G3 (concentration):** positive PnL in ≥ 4 of the 5 WF years AND best
  single year < 60% of total WF profit (when total > 0).

If any gate fails: null result, STOP — no EA, no parameter adoption.

## Honesty notes (declared up front)

- Candidate genesis is in-sample on this 2021–2026 history (the grid and
  the shortlist rule see the same data); walk-forward guards the selection
  WITHIN the set only. True out-of-sample requires post-2026-04 data.
- Phase 1's tilt was thin (first-cross dead heat at 0.10%); the expected
  outcome is failure, and a clean negative at this gate is the deliverable.
