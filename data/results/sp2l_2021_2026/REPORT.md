# REPORT — SP2L SpikeGap native EA validation (NY_E, TP_R 2.0), 2021-01..2026-04

2026-08-27. Walk-forward-confirmed configuration only (mt5_rebuild_gate):
**NY_E** = NY session entry window (signal bar 09:30–15:30 NY), EMA60
filter ON, SL = pre-spike extreme, TP = 2.0R, E-NEXT entries (signal on
completed M1 bar → market fill at next bar's open), forced exit 15:55 NY
bar open. Selection trail: `data/results/sp2l_ablation/REPORT.md`.

## Engines

- **Python reference** (`evaluate/sp2l_reference.py`, standalone by design):
  full-range on the verified M1 export. Independently reproduces the Step-3
  E-NEXT walk-forward result EXACTLY (2022+ subset: 1,165 trades,
  +811.18 pts — engine cross-validation).
- **Native MQL5 EA** (`ea/SP2L_SpikeGap.mq5`): no bridge/DLL/ML; DST-aware
  sessions (generic US-DST helper, identical to ORB/VWAP EAs);
  NormPrice-explicit market entries AND closes (the off-grid 10015 fix
  applied from the start — 0 rejected entries observed); server-side
  SL/TP on the 0.01 grid; 6 yearly Strategy Tester batches, Model=0
  (every tick), 95,916,945 ticks total.

## Headline results (base $10k, 1 lot, no compounding, gross pts)

| | Python (chart tape) | MT5 (tester tape, NET of real spread) |
|---|---|---|
| Trades | 1,351 (2021 included) | 1,343 |
| Win rate | 37.68% | 35.59% |
| Total | +965.20 pts (+9.65%) | +281.84 pts (+2.82%) |
| Expectancy | +0.714 | +0.21 |
| Sharpe (zero-filled d) | 0.854 | 0.25 |
| Max DD | 365.82 pts | 416.71 pts |
| Per year | 21/22/23/24/25/26: +154/+381/+222/+302/−119/+26 | +100/+153/+141/+129/−260/+19 |

The py↔mt5 gap (−683 pts ≈ 1,343 × ~0.5 pt) is the real bid-ask spread the
tester charges — the reference is gross-of-cost by construction (project
convention). MT5 is the honest live-tradable number: **marginally positive
net edge**; at a wider 0.8 pt round-trip cost assumption it would flip
negative (0.714 gross − 0.8 < 0).

## Reconciliation (RECONCILE.md)

1,275/1,351 matched (same direction, same day, ≤10 min); **89.7% agree**
within tolerances (entry/exit ≤1.5 pts, exit time ≤15 min, reason equal).
All differences classified; **0 unexplained**:

- **Two-tapes bar differences (dominant class).** Decisive quantification:
  comparing every EA-logged setup's SL level (= raw high/low of bar t−3 on
  each engine's own tape) against the export: **2,229/2,231 differ**
  (median 0.19 pts, p90 0.83, max 1.87). The tester's Model=0 tape
  regenerates M1 bars that differ from the chart-served export at the
  sub-point level essentially everywhere. VWAP MR tolerated this (gates
  10–25 pts wide, 94% agreement); SP2L's setup conditions are exact OHLC
  inequalities (doji open=close equality, high-vs-high breaks, 0.04% gap ≈
  2 pts), so decisions flip often (243/1,040 days differ in setup/entered
  counters). Proven example: 2026-02-24 14:54 doji (open==close exactly on
  the chart tape) blocks the SELL setup in Python; the tester tape's bar
  differs and fires it.
- Downstream: setup flips cascade through occupancy (76 only-py + 68
  only-mt5 trades, all with same-day siblings or day-counter shifts),
  exit-level flips (44 slippage, 34 bar-shift, 8 path-flip, 30 mixed), 15
  entry spreads >1.5 pts.
- **1 tester_stops_reject**: 2025-12-29, both engines fired the same
  signal; tester rejected the order (10016 Invalid stops — TP within stops
  level of fill on a ~0.6 pt-risk trade); reference filled (+0.92).

**Honest robustness caveat (the real Step-4 finding):** SP2L's signal set
is NOT robust at the sub-0.25-pt tape level. The EDGE survives the tape
change (tester-tape gross ≈ chart-tape gross: 281.84 + ~673 spread ≈ +955
vs +965) — but the specific trade set differs ~10%, and any live
deployment would face the same knife-edge sensitivity to feed
microdifferences.

## Step 5 — benchmark and correlation (stats.json, monthly_matrix.csv)

- B&H same range: +96.13% price (+34.66% ex-financing on the $10k/1-lot
  base, prior work). SP2L +9.65% gross / +2.82% net — B&H dominates on
  return (consistent with every intraday strategy in this project);
  SP2L's case would rest on diversification, not return.
- Daily-PnL correlations vs prior strategies (SP2L py gross):
  ORB **+0.054**, VWAP MR **−0.005**, VWAP trend-pullback WF **+0.142** —
  all ≈0: **a genuinely new, uncorrelated edge source** (fourth
  hypothesis tested in this project, first to clear every gate).

## Verdict

CONFIRMED through full validation: walk-forward (both conventions +
EA-convention), native EA, 0 unexplained reconciliation differences.
Cautions for any deployment decision: net-of-real-spread expectancy is
+0.21 pts/trade (Sharpe 0.25) — thin, of the order of transaction costs;
2025 negative on both engines; signal set knife-edge sensitive to tape
microdifferences; parameters (PGAP 0.04%, MAX_SL 0.40%) are
conversion-derived, not re-optimized. No live deployment decision made
here — that requires the cost/volume assumptions of the actual account.

## Artifacts

- `ea/SP2L_SpikeGap.mq5` (0 err/0 warn; repo == terminal copy).
- `evaluate/sp2l_reference.py`, `evaluate/sp2l_reconcile.py`,
  `evaluate/sp2l_stats.py`.
- `batches/` (6 yearly runs + summary.csv), `py_trades.csv`,
  `py_days.csv`, `mt5_trades.csv`, `mt5_days.csv`, `RECONCILE.md`,
  `monthly_matrix.csv` (64 months), `stats.json`, `run_sp2l_batches.py`,
  `stitch_batches.py`, `compile.log`.
