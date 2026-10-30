# Step 3 REPORT — intraday multi-horizon trend alignment: NULL RESULT, stopped

**Run:** 2026-08-27 · Protocol pre-declared in `GATE-STEP3.md` BEFORE the run.
**Provenance label:** project-original hypothesis — NOT a test of Lempérière
et al. (2014) or the video rubric.

## Verdict

**FAIL → STOP. The pre-declared primary configuration (TH4, E-NEXT) fails
gate G3: 2026 is negative (−327.1 pts) and 2022 contributes 63.1% of total
WF profit (cap 60%). No EA, no sizing, no deployment. The robustness arm
(TH2) passes every gross gate with stability — but is economically dead net
of the realistic spread (expectancy −0.48 pts/trade). This report is the
complete deliverable, per project convention.**

## Walk-forward results (2022-01..2026-04, 9 half-year folds, trailing-expectancy selection, gross)

| arm | n | total pts | expt | expt net of 0.8pt spread | Sharpe | per-year pts | G1 | G2 | G3 | G4 (day bootstrap) |
|---|---|---|---|---|---|---|---|---|---|---|
| **TH4 / E-NEXT (PRIMARY)** | 7,849 | +1,700.3 | +0.217 | **−0.583** | 1.13 | 2022 +1,072 · 2023 +75 · 2024 +300 · 2025 +581 · **2026 −327** | ✓ | ✓ | **✗** (63% in 2022) | ✓ (p05 +528) |
| TH4 / E-CLS (info) | 7,849 | +1,822.2 | +0.232 | −0.568 | 1.21 | same shape, 2026 −322 | ✓ | ✓ | ✗ | ✓ |
| TH2 / E-NEXT (robustness arm, not selectable) | 6,761 | +2,186.6 | +0.323 | **−0.477** | 1.36 | all 5 years + (858/573/350/279/127) | ✓ | ✓ | ✓ | ✓ (p05 +899) |
| TH2 / E-CLS (info) | 6,761 | +2,203.2 | +0.326 | −0.474 | 1.43 | all 5 years + | ✓ | ✓ | ✓ | ✓ |

Fold-selection stability (anti-overfit signature check): TH4 selects
SL 0.40%/TP_R 1.0 in 8 of 9 folds (SL 0.60/0.75 in the first, data-poor
fold); TH2 selects SL 0.25%/TP_R 2.0 in 8 of 9 folds — stable, not
oscillating. Exit mix (TH4 WF): signal-exit 6,900, TP 720, forced-flat 109,
SL 108, session-end 12 — the stop is a distant backstop; almost all exits
are score-state exits.

## Verification

Trade 0 (2022-01-03) hand-verified bar-by-bar against the raw M5 export with
an independent score recomputation: 09:45 crossing into score4 = −4 →
E-NEXT entry 4,766.436 (exact open match); 10:15 score leaves −4 → exit at
next open 4,779.251 (exact); pnl −12.815 pts (exact). The 08:55 pre-window
crossing correctly did NOT enter (entry window 09:30+, state-change
semantics). Simulation machinery is correct; the result is not a bug.

## Why this failed — the honest reading

1. **The primary configuration's edge is regime-concentrated.** TH4's gross
   +1,700 pts is essentially a 2022 phenomenon (the bear year: short-side
   continuation) plus a mild 2024-25 recovery; 2023 (+75) and 2026 (−327)
   show it does not survive quiet grind regimes — exactly the temporal
   instability the gate exists to catch. This mirrors the trend-pullback
   stop (passed Phase 1 thin, died at the WF gate).
2. **Gross-positive ≠ deployable.** Even the best arm (TH2: +0.323
   pts/trade gross across all 5 positive years, day-bootstrap p05 +899)
   loses −0.48 pts/trade net of the 0.8-pt realistic round-trip spread —
   over the WF window that is −3,222 pts of net bleed. SP2L was confirmed
   at +0.21 NET and was already flagged as thin; this is strictly worse
   than that floor. The instrument's costs exceed the signal's gross edge
   in every configuration (same structural verdict as VWAP MR).
3. **Phase-1 fragility carried through as predicted.** The day-unweighted
   bootstrap in Phase 1 was significantly negative while pooled stats
   passed; the WF G4 bootstrap (on per-day trade P&L, a different, milder
   statistic) passes — but the regime concentration the Phase-1 analysis
   pointed at is precisely what kills TH4 at G3. The two views are
   consistent: a real-but-thin continuation tendency concentrated in
   trending days/regimes, too thin to pay costs.
4. **Not a new edge family (informational, computed at stop stage):**
   daily-P&L correlation of the TH4 WF stream vs the four validated
   intraday strategies: ORB +0.17, VWAP MR −0.06, **trend-pullback +0.37**,
   SP2L +0.15. The highest overlap is with trend-pullback — the same
   intraday-continuation thesis that already failed its own Step-3 gate.
   Even if this signal had cleared its gates, it would largely have been a
   re-test of that family, not a diversifier.

## Provenance restated

This was a project-original hypothesis (intraday multi-horizon trend
alignment). It was NOT a replication attempt of Lempérière et al. (2014) or
the video rubric; this null result refutes neither.

## Artifacts

- `GATE-STEP3.md` (pre-declared protocol) · `grid_insample.csv` (80 configs,
  in-sample, exploratory) · `wf_trades.csv` (TH4/E-NEXT WF trades)
  · `wf_summary.json` (gate evaluation, fold logs)
- `evaluate/momentum_intraday_step3.py`
- Phase 1: `../momentum_intraday_signal_quality/` (REPORT.md, SPEC.md,
  GATE.md, phase1_*.csv/json, `evaluate/momentum_intraday_phase1.py`)
