# VWAP Stop/Target R:R Exploration — Phase 1 (exploratory) + Phase 2 (walk-forward confirmation)

**US500-tick · 2021-01-01 → 2026-05-01 · Python reference engine only (MT5 EA untouched by design)**

- **Report date:** 2026-08-26
- **Question:** does an asymmetric stop/target with reward:risk > 1 fix the VWAP mean-reversion rule set, whose diagnosed structural flaw is a 0.2% stop tighter than the ~0.25%+ VWAP-touch target (losers cost more than winners pay at a 41% win rate)?
- **Method discipline:** Phase 1 (MFE/MAE reconstruction on the existing 281 trade paths) is EXPLORATORY and IN-SAMPLE — no number from it is a result. Phase 2 (full re-simulation with walk-forward candidate selection, gates pre-declared in `GATE.md` before any Phase 2 run) is the only phase allowed to be called a result.

## Headline verdict

**NOT CONFIRMED — asymmetric R:R does not rescue this rule set as tested.** The walk-forward-selected result is **273 trades, −$20.37, expectancy −$0.075/trade, daily Sharpe −0.048** against pre-declared gates of (expt > 0 AND Sharpe ≥ 0). G1 FAIL, G2 PASS (273 ≥ 150), G3 FAIL (only 2 of 5 years positive; the entire net profit of both candidates was earned 2022–2024, and 2025–2026 lose under every configuration). The in-sample "fix" (+0.25 to +0.28 expectancy, positive Sharpe for both candidates) did not survive out-of-sample selection: the gap between in-sample story and walk-forward result is the curve-fitting this two-phase process was designed to expose. No MT5 EA rebuild is justified.

## Step 1 — MFE/MAE of the 281 existing trades (`../vwap_2021_2026/mfe_mae.csv`)

MFE/MAE walked bar-by-bar from the entry bar to each trade's actual exit bar, in % of entry (the strategy's own terms); a second pair extends the walk to the 15:55 forced-exit bar.

| Statistic | MFE % | MAE % |
|---|---|---|
| mean | 0.179 | 0.172 |
| median | 0.178 | 0.205 |
| p90 | 0.332 | 0.264 |

Excursion reach (of 281 trades): ±0.10% — MFE 180 / MAE 206; ±0.15% — 158 / 185; ±0.20% — 125 / 164; ±0.25% — 88 / 42. Two facts jump out: (a) adverse excursions reach deeper than favorable ones at every threshold through 0.20% — the entry timing itself has negative edge, not just the exit geometry; (b) only 88 trades ever saw +0.25% favorable, so any target requiring a full reversion-to-VWAP-and-beyond is fighting the tape's own distribution.

## Phase 1 — EXPLORATORY reconstruction on existing trade paths, NOT a validated backtest

Grid: stop {0.10, 0.15, 0.20, 0.25}% × target {50, 75, 100, 125}% of the way back to the **moving** VWAP, outcome per cell decided by bar-path order (which level crossed first), exits checked from the bar after entry (reference-engine convention), path truncated at the 15:55 bar. **Entries held fixed at the original 281 — documented approximation:** a different exit rule changes when the strategy is next flat and thus which later entries would have existed; reconstructed trades may overlap in book terms. This is a counterfactual per-trade table, not a re-backtest.

**Engine validation:** the (0.20%, 100%) cell replays the identical original rule on identical paths and reproduces the original outcome EXACTLY — n=281, win rate 41.64%, total −$202.42, expectancy −$0.720.

| stop% | target | n | win% | avgW | avgL | expt/trade | total |
|---|---|---|---|---|---|---|---|
| 0.10 | 50% | 281 | 40.93 | 6.71 | −5.13 | −0.282 | −79.29 |
| 0.10 | 75% | 281 | 34.16 | 9.63 | −5.09 | −0.057 | −16.12 |
| 0.10 | 100% | 281 | 27.40 | 12.55 | −5.05 | −0.231 | −64.96 |
| **0.10** | **125%** | 281 | 25.27 | 15.54 | −5.05 | **+0.152** | +42.64 |
| 0.15 | 50% | 281 | 49.82 | 6.62 | −7.34 | −0.381 | −107.04 |
| 0.15 | 75% | 281 | 41.64 | 9.53 | −7.27 | −0.275 | −77.32 |
| 0.15 | 100% | 281 | 34.16 | 12.15 | −7.23 | −0.609 | −171.17 |
| 0.15 | 125% | 281 | 31.67 | 14.90 | −7.23 | −0.224 | −62.83 |
| 0.20 | 50% | 281 | 56.23 | 6.41 | −9.68 | −0.629 | −176.84 |
| 0.20 | 75% | 281 | 48.40 | 9.20 | −9.58 | −0.494 | −138.81 |
| 0.20 | 100% | 281 | 41.64 | 11.61 | −9.52 | −0.720 | −202.42 |
| 0.20 | 125% | 281 | 38.79 | 14.16 | −9.54 | −0.345 | −97.02 |
| 0.25 | 50% | 281 | 62.99 | 6.23 | −12.08 | −0.547 | −153.71 |
| 0.25 | 75% | 281 | 57.30 | 8.75 | −12.03 | −0.124 | −34.88 |
| 0.25 | 100% | 281 | 50.18 | 11.01 | −11.93 | −0.417 | −117.28 |
| **0.25** | **125%** | 281 | 47.33 | 13.40 | −11.96 | **+0.047** | +13.31 |

Only 2 of 16 cells show positive expectancy, both in the 125%-target column. Reading the grid honestly: tight stops + close targets produce many small wins but the worst expectancies at equal R:R; the ONLY structurally consistent pattern is that expectancy increases monotonically with target fraction at every stop level — "let winners overshoot past VWAP" is the grid's single hypothesis. But both positive cells are grid corners whose orthogonal neighbors are all negative — the strict neighborhood-robustness rule (cell > 0 AND all neighbors > 0) is passed by ZERO cells.

## Step 3 — Shortlist (rationale, not top score)

Shortlisted despite failing the strict neighborhood rule, because the column-level monotonicity is the only defensible signal — recorded as a known weakness:

- **A** = stop 0.10%, target 125% (in-sample +0.152; neighbors −0.231/−0.224)
- **B** = stop 0.25%, target 125% (in-sample +0.047; neighbors −0.345/−0.417)
- **Baseline C** = stop 0.20%, target 100% (original rule), included in the selection set so the walk-forward may honestly conclude "nothing beats the original".

## Pre-declared gate (`GATE.md`, written before Phase 2 ran)

- **G1:** WF expectancy > $0.00 AND WF daily Sharpe ≥ 0.00 (≥ +0.486 improvement over the baseline's −0.486).
- **G2:** WF stitched trades ≥ 150.
- **G3:** positive PnL in ≥ 4 of the 5 WF years AND best year < 60% of total.
- Protocol: burn-in 2021; 9 half-year folds 2022H1..2026H1; per fold select the eligible (≥ 20 trailing trades) candidate with the best trailing expectancy on data strictly before the fold; else baseline.
- Honesty note: the candidate set itself was chosen from in-sample Phase 1 on this same history; WF guards the selection WITHIN the set, not the set's genesis.

## Phase 2 — Full re-simulation + walk-forward (the only result)

Full re-simulation (entries AND exits re-run together — resolves Phase 1's fixed-entries approximation). **Engine equivalence: the baseline config reproduces the original run exactly (281 trades, −$202.42, Sharpe −0.486).** Note the trade-count effect Phase 1 could not see: A's tight stop frees the position sooner → 334 entries; B's wide stop blocks re-entries → 274.

In-sample full range vs walk-forward:

| config | in-sample n | in-sample total | in-sample expt | in-sample Sharpe | role |
|---|---|---|---|---|---|
| A 0.10/125% | 334 | +92.70 | +0.278 | +0.228 | candidate |
| B 0.25/125% | 274 | +69.69 | +0.254 | +0.139 | candidate |
| C 0.20/100% | 281 | −202.42 | −0.720 | −0.486 | baseline |

Fold-by-fold (selection on trailing expectancy):

| fold | selected | trades | PnL |
|---|---|---|---|
| 2022H1 | A | 66 | +71.13 |
| 2022H2 | B | 52 | −90.09 |
| 2023H1 | A | 39 | +44.36 |
| 2023H2 | B | 19 | −37.94 |
| 2024H1 | B | 8 | +25.84 |
| 2024H2 | B | 21 | +126.06 |
| 2025H1 | B | 34 | −70.14 |
| 2025H2 | B | 10 | −15.92 |
| 2026H1 | B | 24 | −73.67 |

**WALK-FORWARD RESULT: 273 trades · −$20.37 total · expectancy −$0.075/trade · daily Sharpe −0.048 · maxDD $211.74.** (Baseline over the same 2022+ window: 258 trades, −$146.07, Sharpe −0.403 — the selection did add value over the original rule, but nowhere near positive territory.)

Per-year walk-forward: 2022 −18.96 · 2023 +6.42 · 2024 +151.90 · 2025 −86.06 · 2026 −73.67. Monthly sign: 26/47 positive (55.3%).

### The in-sample → walk-forward gap, explicitly

- Best in-sample expectancy: **+0.278** (A) → walk-forward: **−0.075**. Sharpe +0.228 → −0.048.
- The gap is the curve-fit tax: Phase 1 + in-sample re-simulation said "125%-overshoot targets win"; out-of-sample selection on trailing data could not harvest it.
- Per-year in-sample decomposition shows where the mirage lived: A earned +41.6/+43.2/+135.0 in 2022/2023/2024 and lost −13.7/−57.4 in 2025/2026; B earned +3.0/+147.1/+151.9 then lost −86.1/−73.7. **The regime that paid overshoot targets was 2022–2024; 2025–2026 punish every configuration including the original.** The selector, entering each fold with only trailing data, rode B into that ending.
- Diagnostic (NOT a result — it is hindsight from the same sample): committing to a single candidate for all of 2022+ would have been positive (A +148.58/308 trades; B +142.25/253), i.e. the trailing-expectancy selection signal itself had no persistence — it flipped A→B after 2022H1 and again in 2023, adding churn but no edge. A fixed-candidate win would still be in-sample of genesis (the candidates exist because we saw the full grid on this history); only new data after 2026-04 could test that honestly.

### Gate evaluation

| Gate | Requirement | Observed | Verdict |
|---|---|---|---|
| G1 | expt > 0 AND Sharpe ≥ 0 | −0.075, −0.048 | **FAIL** |
| G2 | ≥ 150 trades | 273 | PASS |
| G3 | ≥ 4/5 years positive, best year < 60% | 2/5 positive | **FAIL** |

## Final verdict

A confirmed negative: **asymmetric stop/target ratios — including both 125%-overshoot variants — do not rescue the VWAP mean-reversion rule set on US500-tick under walk-forward discipline.** The MFE/MAE diagnostic's deeper message is that the entries, not just the exits, lack edge: adverse excursions dominate favorable ones at every threshold through 0.20%, so no exit geometry built on these entry points produces a robust positive expectancy. Per the pre-declared rules: no MT5 EA rebuild, no parameter adoption. The strategy remains retired as originally validated (MT5 −3.07% / Sharpe −0.75; Python −2.02% / −0.49). Any future revisit should target the entry condition itself (deviation threshold, slope gate, or regime filter) rather than exit geometry, and should be tested on data beyond 2026-04 — the 2025–2026 regime shift that defeated every exit variant here is invisible to any design that only ever sees this one history.

## Artifacts

- `../vwap_2021_2026/mfe_mae.csv` — 281 rows, MFE/MAE to original exit and to 15:55
- `phase1_grid.csv` — 16-cell exploratory reconstruction (validation cell exact)
- `GATE.md` — pre-declared protocol + gates (timestamped before Phase 2)
- `wf_folds.csv`, `wf_trades.csv` (per-trade config tags), `wf_summary.json`
- `peryear_analysis.py` — per-year in-sample decomposition (diagnostic)
- Scripts: `evaluate/vwap_mfe_mae.py` (Phase 1), `evaluate/vwap_rr_walkforward.py` (Phase 2 engine + WF + gates)
