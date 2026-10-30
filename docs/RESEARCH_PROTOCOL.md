# Research protocol

## Order of work

1. Write a **plain-text SPEC before implementing a new hypothesis**. Declare
   the mechanism, data and sample, price units, clocks, feature availability,
   entry/exit state machine, occupancy, costs, risk controls and stopping rules.
2. Create a **GATE file containing every gate verbatim**, before the first run.
3. Write and execute Phase 1. Save full local ledgers, diagnostics and a REPORT,
   including when the hypothesis fails.
4. Advance only passing candidates to separately specified Phase 2 execution tests.
5. Freeze the records. Corrections, new cost assumptions and remeasurements get
   new names/directories. Never rewrite a failed historical result into a pass.

The original working repository's running log is append-only. The public
release indexes frozen records instead of exposing machine-local notebooks,
credentials or terminal logs.

## Causal controls

- Use only bars completed by the decision timestamp. Higher-timeframe indicators
  must be joined by completion time, not by a future close from the same bucket.
- A pivot requiring right-hand confirmation is unavailable until those bars close.
- Define seeds, warmup and treatment of empty buckets explicitly. Forward filling
  may be required for one study and prohibited in another.
- Synthetic bars can support a declared indicator grid but cannot invent real fills.
- Evaluate opening gaps and stop/target ambiguity pessimistically where specified.
- Track occupied trades, skipped signals and unresolved terminal positions.
- Preserve both realized PnL and floating/marked risk when the claim concerns
  account solvency. A closed-trade equity curve is insufficient for that claim.

## Statistical controls

Report the sample size, expectancy in native units and/or risk units, win rate,
drawdown and Sharpe definition. Daily annualized Sharpe and unannualized
per-trade Sharpe cannot be compared without their sampling context.

Most studies use a chronological moving-block bootstrap with frozen seed, block
length and resample count. Report whether the implementation resamples raw
returns or tests a centered null; a recorded `P(mean <= 0)` is not automatically
an exact hypothesis-test p-value. Dependence, overlap and selection affect its
interpretation. A per-study bootstrap does not correct for this project's many
previous hypotheses or parameter grids.

## ML controls

Freeze features and model families before outcomes. Use causal walk-forward
folds, fit preprocessing only on training data and prevent overlapping label
information from reaching test decisions. Meta-label features such as recent
win rate must use outcomes actually known at entry. Post-hoc model swaps or
threshold searches require a new declared experiment.

## Language of evidence

An exploratory gate pass is not live validation. The project requires independent
walk-forward confirmation, positive after-cost expectancy, adequate sample size
(normally at least 150 trades), year-sign agreement and separately declared
execution validation before stronger claims. A historical owner-requested
100-trade exploratory threshold does not relax that standard.

Negative findings are complete deliverables. Do not optimize a failed frozen
artifact until it appears successful, discard losing years or substitute a
better validation parameter after development selection.
