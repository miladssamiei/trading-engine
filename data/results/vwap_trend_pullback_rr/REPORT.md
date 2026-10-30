# VWAP Trend-Pullback — Step 3 Stop/Target Report (exploratory grid + walk-forward) — GATE FAILED

**US500-tick · M5 · 2021-01-01 → 2026-05-01 · Python only (no EA built — gate failed)**

- **Report date:** 2026-08-26
- **Protocol:** `GATE.md` in this directory declared the grid, shortlist rule, walk-forward protocol and gates BEFORE either Step-3 script ran. Continues `../vwap_trend_pullback_signal_quality/REPORT.md` (Phase 1 gate PASSED, thin).

## Headline verdict

**NOT CONFIRMED — the pre-declared gate fails on temporal concentration (G3), so the project stops here: no EA, no parameter adoption.** The walk-forward result itself is striking at first glance — 2,656 trades, +$1,872.70, expectancy +$0.705/trade, daily Sharpe 1.089, maxDD $449 — and G1 (expt > 0, Sharpe > 0) and G2 (≥ 150 trades) both PASS. But G3 requires positive PnL in ≥ 4 of 5 walk-forward years and only 3/5 are positive: **2024 is flat (−$25.83) and 2026 is clearly negative (−$213.21)**, while 2022 alone contributes 56.6% of the total. The edge is real-looking but regime-dependent, and the two structural caveats below say the live-tradeable version of it is thinner than the headline. This is exactly the case the gate was written for: a Phase-1 grid where all 16 cells are positive with Sharpe ≈ 1 in-sample, and a selection process that cannot rescue temporal instability.

## Phase 1 of Step 3 — exploratory reconstruction grid (in-sample, NOT a validated backtest)

Grid stops {0.10, 0.15, 0.20, 0.30}% × targets {0.15, 0.20, 0.30, 0.45}% (chosen from this study's own Phase-1 excursion distribution — median MFE/MAE ≈ 0.23/0.22, means 0.36/0.35, p90 ≈ 0.80 — NOT from the excluded source video). Per-signal first-crossing reconstruction on the 6,606 Phase-1 signals, fixed-entry approximation (no occupancy), engine-style path conventions, all validation invariants asserted and holding (stop ⇒ MAE ≥ stop; win ⇒ MFE ≥ target; time ⇒ neither crossed; 0 violations across 16 cells × 6,606 signals after fixing the walker's 15:55-open gap-check to match the declared invariant — 14 boundary cases).

| stop% \ target% | 0.15 | 0.20 | 0.30 | 0.45 |
|---|---|---|---|---|
| 0.10 | +0.101 | +0.147 | +0.136 | +0.269 |
| 0.15 | +0.165 | +0.214 | +0.224 | +0.397 |
| 0.20 | +0.183 | +0.267 | +0.292 | +0.484 |
| 0.30 | +0.201 | +0.283 | +0.323 | +0.525 |

(expectancy $/trade; win rates 26–64%, full table in `phase1_grid.csv`.)

**Every cell is positive and every cell passes the strict neighborhood rule** — the first time in this project a grid has done that (VWAP MR's grid: 2/16 positive, zero neighborhood-passing). The magnitudes are consistent with Phase 1's thin tilt (mean MFE−MAE ≈ 0.01% ≈ $0.2–0.5 at these index levels): this is the tilt monetized across the whole excursion grid, not a lone spike. Since the shortlist rule anticipated "max 3 candidates" for exactly this case, three diagonal cells spanning the excursion scale were taken rather than the top cell alone: **S** = (0.10%, 0.20%) scalp corner, **D** = (0.20%, 0.30%) the a-priori default/grid center, **L** = (0.30%, 0.45%) deep-excursion corner.

## Phase 2 — occupancy-modeled re-simulation with walk-forward selection (the only result)

Full re-simulation per candidate: entries only while flat (signals during holding skipped), server-style exits (time-exit priority at the 15:55 open — the one deliberate semantic difference from the grid walker's gap-check, ~14 cases — bar-open gap checks, close≥open same-bar heuristic, 0.01-grid stops/targets, early-close carry). This resolves the fixed-entries approximation: occupancy cuts 6,606 signals to 5,357 / 3,992 / 2,994 trades (S/D/L) — the strategy trades ~2–4 times/day, not 4.9.

In-sample full range (2021-01..2026-04):

| candidate | trades | total | expt/trade | Sharpe | maxDD | expt − 0.4pt | expt − 0.8pt |
|---|---|---|---|---|---|---|---|
| S 0.10/0.20 | 5,357 | +1,376.64 | +0.257 | 1.049 | $395 | −0.143 | −0.543 |
| D 0.20/0.30 | 3,992 | +1,949.90 | +0.488 | 1.130 | $265 | +0.088 | −0.312 |
| L 0.30/0.45 | 2,994 | +2,164.17 | +0.723 | 1.048 | $449 | +0.323 | −0.077 |

Walk-forward (burn-in 2021; folds 2022H1..2026H1; trailing-expectancy selection, ≥ 20 trailing trades):

| fold | selected | trades | PnL |
|---|---|---|---|
| 2022H1 | D | 554 | +381.56 |
| 2022H2 | L | 385 | +677.94 |
| 2023H1 | L | 295 | +194.95 |
| 2023H2 | L | 241 | +98.12 |
| 2024H1 | L | 223 | +40.42 |
| 2024H2 | L | 249 | −66.25 |
| 2025H1 | L | 311 | +528.99 |
| 2025H2 | L | 221 | +230.18 |
| 2026H1 | L | 177 | −213.21 |

**WALK-FORWARD RESULT: 2,656 trades · +$1,872.70 · expectancy +$0.705 · Sharpe 1.089 · maxDD $449.34.** Per-year: 2022 +1,059.50 (939 tr) · 2023 +293.07 (536) · 2024 −25.83 (472) · 2025 +759.17 (532) · 2026 −213.21 (177).

Two honest observations the gate numbers hide:

1. **The selection is nearly static**: trailing expectancy chose L for 8 of 9 folds (D only in 2022H1), so the walk-forward is essentially L's out-of-sample-ordered performance minus one fold — the "selection" added no information, unlike the VWAP R:R case where it actively churned.
2. **The G3 failure is structural, not a selection artifact**: per-year in-sample PnL shows 2024 ≈ 0 and 2026 negative for ALL THREE candidates (S: −165/−172; D: +39/−133; L: −26/−213). The entry rule's edge concentrates in high-volatility years (2022, 2025) and decays to nothing in 2024 and (Jan–Apr) 2026 — the same 2025–2026 regime shift that defeated every VWAP MR exit variant is visible here from the opposite direction (this strategy liked 2025, hates 2026).

### Structural caveats (why even a G3 pass would have warranted caution)

- **Gross-of-cost bias**: all numbers are bid-only M5 with zero spread/commission, matching this project's reference-engine convention. Informational net-of-spread: at the tester's ~0.41 pt spread, one crossing (0.4 pt) leaves WF expectancy ≈ +0.305; two crossings (0.8 pt) make it negative (−0.095). The real edge, if any, is of the same order as transaction costs — an MT5 reconciliation (Step 4) would have been the decisive test, but the gate stops before it.
- **Signal clustering**: ~2–4 trades/day from re-arming drift episodes; trades within a day share the same afternoon path, so 2,656 trades ≪ 2,656 independent observations. The daily-Sharpe calculation handles the daily aggregation correctly, but year-level stability (G3) is the binding constraint, and it failed.
- **Candidate genesis is in-sample** (declared in GATE.md): the grid, thresholds and shortlist all derive from this same 2021–2026 history; true out-of-sample needs post-2026-04 data.

### Gate evaluation

| Gate | Requirement | Observed | Verdict |
|---|---|---|---|
| G1 | WF expt > 0 AND Sharpe > 0 | +0.705 / 1.089 | **PASS** |
| G2 | WF trades ≥ 150 | 2,656 | **PASS** |
| G3 | ≥ 4/5 WF years positive AND best year < 60% of total | 3/5 positive (2024 −25.83, 2026 −213.21); best share 56.6% | **FAIL** |

**Decision: NOT CONFIRMED — stop. No exit-rule adoption, no `ea/VWAP_TrendPullback.mq5` build, no Step 4/5.** Per the task definition and the project's validation discipline, this null result is a complete deliverable.

## What would change this verdict

A revisit is only justified with (a) data beyond 2026-04 to test whether 2026H1 is a drawdown of a live edge or the edge's death, and/or (b) a regime filter that identifies in advance which years pay (the per-year table suggests a volatility/state gate exists in principle — but designing one on this same history would be the same in-sample trap that G3 exists to catch, unless it too is walk-forward-validated on fresh data). The marginal net-of-cost expectancy also means any live version needs institutional-grade costs to matter.

## Artifacts

- `GATE.md` — pre-declared protocol + gates (written before any Step-3 run)
- `phase1_grid.csv` — 16-cell exploratory reconstruction (invariants asserted)
- `wf_folds.csv`, `wf_trades.csv` (per-trade cfg tag), `wf_summary.json`
- Scripts: `evaluate/vwap_trend_common.py` (shared signal engine), `evaluate/vwap_trend_rr_grid.py`, `evaluate/vwap_trend_rr_walkforward.py`
- Phase 1 (signal quality): `../vwap_trend_pullback_signal_quality/` (REPORT.md, GATE.md, signals_mfe_mae.csv, us500tick_m5_vol.csv)
