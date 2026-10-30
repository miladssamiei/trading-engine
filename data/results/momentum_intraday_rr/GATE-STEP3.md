# GATE-STEP3 — pre-declared BEFORE the Step-3 run (2026-08-27)

Written after Phase 1 completed, BEFORE any Step-3 output exists. Protocol
mirrors SP2L / VWAP-RR / trend-pullback Step 3. Nothing below may be
re-negotiated after seeing Step-3 results.

## Signal (from Phase 1)

score4 = sum of sign(close - close_{h ago}) over h in {30 m, 1 h, 2 h, 4 h},
M5 bars, calendar-time trailing anchors, NY session. Direction = sign at
the state change INTO the threshold.

## Trading rule (occupancy sim, 1 lot, gross for gates)

- Entry state change (evaluated on completed M5 bar close):
  TH4 (PRIMARY): |score4| crosses from <4 to =4. Direction = sign(score4).
  TH2 (neighborhood arm, run for robustness only, never selected over TH4):
  |score4| crosses from <2 to >=2.
- Fill conventions: E-NEXT (PRIMARY — entry at next bar's open; the
  EA-honest convention SP2L confirmed with) and E-CLS (entry at signal-bar
  close; optimistic; informational).
- Last entry: signal bar open-minute <= 15:30 NY (matching SP2L/VWAP).
- Exits, in pessimistic priority within a bar (SL checked before TP):
  1. stop: SL% below/above entry (long/short);
  2. target: TP_R x SL in favorable direction;
  3. signal exit: score4 magnitude leaves the entry threshold
     (TH4: |score4| < 4; TH2: |score4| < 2) -> exit at next bar open
     (E-NEXT) / bar close (E-CLS);
  4. forced flat at 15:55 NY (first bar open-minute >= 15:55 -> exit at
     its open; tape end -> last close).
- Occupancy: max 1 position; re-entry allowed immediately after any exit.
- No spread in gate metrics (gross, SP2L convention); the ~0.8 pt
  round-trip tester-real spread (~0.013% at 6000) is reported alongside as
  information.

## Grid (in-sample, clearly labeled exploratory)

SL in {0.15, 0.25, 0.40, 0.60}% x TP_R in {0.5, 0.75, 1.0, 1.5, 2.0}.
2021 is warmup/trailing-start; the walk-forward evaluates 2022-01..2026-04.

## Walk-forward

9 half-year folds (SP2L FOLD_BOUNDS: 2022H1..2026H1+). Per fold: select
(SL, TP_R) by highest trailing expectancy over ALL prior simulated trades
of that config (require >= 20 trailing trades, else defaults
SL=0.25% / TP_R=1.0); trades assigned to folds by entry date. Neighbor
robustness (informational): the selected cell's 4 grid neighbors must not
flip sign of expectancy (SP2L "neighborhood, not lone cell" discipline).

## Gates (ALL must hold; E-NEXT, gross, WF 2022-01..2026-04)

- G1: net expectancy > 0 AND annualized Sharpe > 0.
- G2: >= 150 WF trades.
- G3: positive P&L in >= 4 of 5 calendar years AND no single year > 60% of
  total WF profit.
- G4 (targets the Phase-1 overlap fragility directly): day-block bootstrap
  (10,000 resamples of per-day WF net P&L, resampling days with
  replacement): 5th percentile of resampled total net P&L > 0.
- G5 (convention honesty): E-CLS must not be required for the pass — if
  E-NEXT fails any gate, the hypothesis FAILS regardless of E-CLS.

## On failure

Write the null REPORT.md in this directory and STOP: no EA, no sizing, no
deployment. A stopped gate is a complete deliverable (project convention).
