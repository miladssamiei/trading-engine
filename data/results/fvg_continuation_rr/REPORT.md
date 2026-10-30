# Step 3 REPORT — FVG continuation, fixed-rule occupancy walk-forward: FAIL → study NULL

**Run:** 2026-08-28 · Gates S1–S4 pre-declared in
`../fvg_continuation_signal_quality/GATE.md` BEFORE the run. No
parameter selection existed or was performed (swing SL, fixed 1:1 TP,
2/day cap, session window all fixed by SPEC; the WF folds are a
stability check).

## Result

251 trades (occupancy: one position at a time; 13 signals skipped
in-position; 2/day cap never binding — max 2 signals/day; 45
holds > 4h incl. overnight carries, ex-financing headline).

| | gross | net of 0.8 pt |
|---|---|---|
| total | +128.59 pts | **−72.21 pts** |
| expectancy | +0.512 | **−0.288** |
| Sharpe (√252, zero-filled RTH days) | 0.204 | **−0.114** |
| win rate | 53.4% (net) | |

Exit mix: 134 TP (mean +15.36 pts) vs 117 SL (mean −16.41) at 1:1 —
the slight win-rate edge is more than eaten by the larger average stop
distance and the spread.

Yearly net: 2021 −62.51 · 2022 −24.30 · 2023 +92.26 · 2024 −60.89 ·
2025 +16.46 · 2026 (4m) −33.23 → **2/5 full years positive** (S3
FAIL). Fold detail in `wf_folds.csv`; no fold-selection exists
(rules fixed).

## Gates

- S1 FAIL (net expectancy < 0, net Sharpe < 0)
- S2 PASS (251 ≥ 150)
- S3 FAIL (2/5 years positive)
- S4 FAIL (net-of-spread: gross edge +0.512 ≈ 64% of the 0.8 pt cost)

**VERDICT: FAIL → STOP. No EA, no parameter adoption.** Per the
pre-declared GATE there is no exploratory exit grid for this study (the
source fully specifies the exit); nothing was searched.

## Interpretation

The Phase-1 pass was thin (G3 at 0.3866 vs the 0.40 limit, long side
negative, 3/6 signal-years negative) and the honest trade simulation
with the source's own exit confirms it does not survive contact with
costs: this is the same edge-smaller-than-costs class as VWAP MR
(−0.075 net expt) and intraday-momentum TH2 (−0.48 net). The
pre-declared adjacent-timeframe robustness arms are not run: their
purpose was stress-testing a PASS before an EA build, and GATE.md bars
them from rescuing a failure.

**Continuation family final count: 0-for-5 on this instrument**
(trend-pullback, intraday momentum, unfiltered H1/H2, filtered
H2-morning, FVG retracement-continuation) — across moving averages, leg
counting, and fair-value-gap markers, continuation-after-pullback has
no exploitable net edge at these timeframes. Daily-P&L correlations
(from this study's stream): ORB +0.114, VWAP MR +0.055, trend-pullback
+0.155, SP2L +0.042, intraday momentum +0.164, daily momentum −0.014.

## Artifacts

`wf_trades.csv` (251 trades, entry/exit/reason/gross/net) ·
`wf_yearly.csv` · `wf_folds.csv` · `wf_summary.json` ·
`correlations.csv` · `evaluate/fvg_continuation_step3.py`,
`evaluate/fvg_continuation_correlations.py`
