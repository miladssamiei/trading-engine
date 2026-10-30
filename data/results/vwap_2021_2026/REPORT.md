# VWAP Mean-Reversion Strategy — Full-Range Validation Report

**US500-tick · 2021-01-01 → 2026-05-01 (full available range)**

- **Report date:** 2026-08-26
- **Range covered:** 2021-01-03 23:00 UTC → 2026-04-30 23:59 UTC — 1,791,504 M1 bars · 64/64 months · 0 multi-day gaps
- **Engines:** MT5 Strategy Tester (`VWAP_MeanReversion.ex5`, Model=0 every tick, M1) vs Python reference (`evaluate/vwap_reference.py`, M1 bid bars + tick_volume)
- **Base:** USD 10,000 fixed account · 1 lot (contract 1.0 → $1 per price point) · NO compounding (fixed lot) · 1:100 leverage

## Headline verdict

**The VWAP mean-reversion rule set LOSES money on US500-tick over 2021 → 2026-04 on both engines, and the two engines AGREE on that conclusion.** MT5: 280 trades, −3.07% total return (−$307.10), Sharpe −0.75, max DD 3.07%. Python: 281 trades, −2.02% (−$202.42), Sharpe −0.49, max DD 2.06%. Trade-level reconciliation is 94.0% at the ORB tolerances with **zero unexplained residuals** — every disagreement is a documented threshold-straddle / tester-tape artifact. The strategy does not beat buy-and-hold on any framing, is strictly dominated by ORB over the identical window, and its daily P&L is essentially uncorrelated with ORB's (Pearson +0.015) — the two strategies draw on different edge sources, but VWAP's source is negative here, so combining them only dilutes ORB (combined MT5 Sharpe 0.11 vs ORB-alone 0.34). **Do not deploy; do not combine with ORB on current evidence.**

## 0. Strategy specification as implemented (interpretations documented)

The published rule set left several things underspecified; each interpretation is recorded because it materially affects reproduction:

| Rule | Implementation |
|---|---|
| Session | 09:30–16:00 NY (RTH), all logic inside session |
| Session VWAP | Cumulative over **completed M1 bars**: Σ(typical × tick_volume)/Σ(tick_volume), typical=(H+L+C)/3, resets per NY session. **No look-ahead**: the VWAP seen during a forming bar is the value through the last completed bar |
| Range day (rolling) | Evaluated at every signal, not once at mid-session (the strategy must know as the session progresses): \|close/open−1\| ≤ 0.6% AND (high−low)/open ≤ 1.0%, using running session extremes over completed bars |
| Long entry | Signal-bar close ≥ 0.25% below VWAP AND 30-min VWAP slope ≥ 0 (flat-to-slightly-positive ⇒ slope gate at 0; the spec gives no upper bound) |
| Short entry | Mirror: close ≥ 0.25% above VWAP AND slope ≤ 0 |
| 30-min slope | (vwap_now / vwap[t−30 min] − 1)×100, **exact-timestamp reference in the same session**; missing reference bar ⇒ no signal (first possible signal is 10:00 NY) |
| Execution | Signal on completed bar → market order on first tick of the next bar; entries stop when the entry bar would open after 15:30 NY (operational cutoff, guarantees ≥ 25 min before force exit) |
| Stop loss | 0.2% beyond entry away from VWAP; server-side; rounded to the 0.01 trade-tick grid |
| Take profit | **Touch of the MOVING VWAP** — no server-side TP exists; tick-monitored close against the last-completed-bar VWAP (long: bid ≥ VWAP; short: ask ≤ VWAP). trades.csv take_profit = VWAP snapshot at entry (audit) |
| Forced exit | 15:55 NY (five minutes before ORB's 16:00 convention, per the published spec) |
| Max-move filter | **Percentage by default: 0.42%** = 30 ES points / ~7200 current index level. DOCUMENTED SIMPLIFICATION: the published 30-point rule is a fixed-point rule from index-point research at much lower index levels; at 2021 prices (~3800) 0.42% ≈ only 16 points. Percentage was chosen for internal consistency with the strategy's other percentage rules (deviation/stop/range-day). A points mode exists (InpMaxMoveMode=1, InpMaxMovePoints=30) but was NOT used |
| Position sizing | Fixed 1 lot (same convention as the ORB base run) |
| Trades/day | Uncapped (spec states no cap); signals evaluated only while flat |

## 1. Scope and data caveats

### 1.1 Timeframe decision: M1

- Probe (`probe_report.md`, 2026-08-24): the full-range M1 history IS served cleanly — **1,791,504 bars, 2021-01-03 23:00 → 2026-04-30 23:59 UTC, 64/64 months, 0 multi-day weekday gaps**, trading-date set **identical** to the already-verified ORB M5 export (1656 dates), and **tick_volume > 0 on every bar** (real_volume is 0 throughout — tick volume only).
- Terminal `[Charts] MaxBars=3000000` confirmed still in effect (required for M1: ~1.79M bars).
- M1 was therefore chosen, matching the published spec's native 1-minute timeframe. M5 was not needed (its monthly range query even short-served January 2021 in the probe's retry pattern — M1 is the cleaner pull).
- The export `us500tick_m1.csv` (time,open,high,low,close,tick_volume) is the single price source for the Python reference and the B&H benchmark.

### 1.2 Same known caveats as ORB

- **Data ends 2026-04-30** (no 2026-05+); the study covers the full AVAILABLE range only.
- Prices are 3-decimal (0.001 grid) while the symbol trade grid is 0.01 — all order prices rounded to 0.01 (`NormPrice`).
- **KEY FIDELITY CAVEAT — the tester runs on its own synthetic-tick tape** (TenTrade-Server, M1-derived, ~0.41 pt spread), not the real Dukascopy ticks feeding the M1 export/Python reference. This drives the residual entry/exit disagreements classified in section 3.
- US500-tick bar timestamps are raw UTC; NY session boundaries via the generic DST rule (2nd Sun Mar 07:00 UTC → 1st Sun Nov 06:00 UTC), logged at EA startup each batch. Prior ORB verification (7/7 zoneinfo spot checks + data fingerprints) carries over — the identical helper code is reused.

### 1.3 New EA-specific lessons (both fixed in-source)

1. **Market orders need an explicit grid-rounded price.** `CTrade::Buy(price=0.0)` internally sends the raw live ask/bid (4-decimal) and the tester rejects every entry with 10015 "Invalid price" — the same trap ORB hit only in its rare market-entry fallback, but here it would have killed 100% of entries. Fix: pass `NormPrice(ask/bid)` as the price argument (the proven MULTI_CAIRO_EA pattern). First smoke run had 0 trades / all-10015; after the fix, all orders fill.
2. **`CTrade::PositionClose` has the same off-grid trap** (ORB's "force-close failed 10015" retries). Exits are frequent here, so the EA closes via an explicit `MqlTradeRequest` with a grid-rounded price (`ClosePositionGrid`).
3. **Exit-reason classification must read `DEAL_REASON`**: server-side SL fills arrive as deals with no EA flag set; the fallback tolerance also had to widen to 0.5 pt (observed 0.09 pt stop-fill slippage on 2026-01-02) or SL fills get mislabeled `vwap_touch`.

## 2. Backtest execution

### 2.1 MT5 batches (Model=0 every tick, M1, 10k USD, 1:100, 1 lot)

| Batch | Window | Ticks | M1 bars | Final balance | Trades | Day records |
|---|---|---|---|---|---|---|
| 2021 | 2021 | 16,365,123 | 331,742 | 9,942.93 | 23 | 309 |
| 2022 | 2022 | 31,445,792 | 341,672 | 9,905.31 | 107 | 310 |
| 2023 | 2023 | 10,114,662 | 332,140 | 9,975.54 | 53 | 310 |
| 2024 | 2024 | 11,249,506 | 333,742 | 10,073.91 | 26 | 312 |
| 2025 | 2025 | 19,062,659 | 339,136 | 9,834.71 | 46 | 312 |
| 2026 | 2026-01-01 → 04-30 | 7,820,903 | 111,694 | 9,960.50 | 25 | 102 |
| **TOTAL** | | **96,058,645** | — | | **280** | **1,655** |

- Tick totals are byte-identical to the ORB batches (same tester tape); the tester generates its own M1 series (~332k bars/year vs ~278k in the real-tick export — the synthetic tape fills gaps).
- Stitched artifacts: `mt5_trades.csv` (280), `mt5_days.csv` (1,655; the only-py 2021-01-03 Sunday-evening row is the same start-of-data boundary ORB saw).

### 2.2 Python reference

`evaluate/vwap_reference.py`, standalone pandas, M1 bid bars + tick_volume. **Independently spot-checked** (`spotcheck_reference.py`): VWAP-at-signal, 30-min slope, deviation, tp0 snapshot and SL recomputed from the raw CSV with a separate implementation match the reference to 4 decimals on first/middle/last trades.

| Metric | Python | MT5 |
|---|---|---|
| Trades | 281 (147 BUY / 134 SELL) | 280 |
| Win rate | 41.6% | 40.0% |
| Exits | SL 164 · VWAP-touch 115 · time 2 | SL 158 · touch 113 · time 9 |
| Total PnL | **−$202.42 (−2.02%)** | **−$307.10 (−3.07%)** |
| CAGR (5.32y) | −0.38% | −0.58% |
| Max DD | $205.88 / 2.06% | $307.10 / 3.07% |
| Daily Sharpe (√252) | −0.486 | −0.748 |
| Return/DD | −0.98 | −1.00 |

The loss structure is structural, not noise: the 0.2% stop is closer than the ~0.25% VWAP target, so a stopped trade loses ~$13–14 while a winning touch nets ~$10–12; with a 41% win rate the expectancy is negative (−$0.72/trade py, −$1.10 mt5). Best year 2023 (+$25 py), worst 2022 (−$101 py). Both engines agree on sign in 58 of 64 months.

## 3. Reconciliation (RECONCILE.md)

- **276/281 Python trades matched** (same direction, same date, |open| ≤ 10 min) → **5 unmatched py, 4 unmatched mt5**.
- **Overall agreement at tolerances (entry ≤ 1.5 pts, exit ≤ 1.5 pts, exit time ≤ 15 min, equal reason): 264/281 = 94.0%.**
- Matched-pair deltas: entry mean 0.398 pt (max 2.589, ask-vs-bid + tape); exit mean 0.681 pt (max 21.716 — path-flip cases below); exit time mean 0.8 min (max 39); reason 272/276; VWAP-at-entry delta mean 0.333 pt (the two tapes' VWAPs track closely).
- **All 9 unmatched trades classified, 0 unexplained:**
  - `dev_threshold_straddle` ×5 — signal deviation within 0.05% of the 0.25% band (as close as 0.0003%): the two tapes' close/VWAP differ by tenths of a point, so only one side crosses the band.
  - `resignal_cascade_exit_timing` ×3 (all MT5) — e.g. 2025-02-03: both engines took the same 15:27 SELL, but the tester tape hit the 0.2% SL **within the entry minute** while the real-tick bar didn't (py exit 15:29); MT5 was flat again for the next bar's signal and re-entered. Same documented tester-tape class as ORB's 2024-04-19, amplified by the strategy's fast re-signal design.
  - `slope_zero_straddle` ×1 — slope 0.0169% from the slope-0 gate.
- Exit outliers: 6 time (max 39 min), 9 price. The 4 largest price outliers (16–22 pts) are **stop_vs_target_path_flips**: the stop (0.2% one side) and the VWAP target (0.25%+ the other) are both inside the exit window and the two tapes cross them in opposite orders — same trade, opposite exits. The rest are stop-fill slippage / level-cross bar shifts.
- Day-level: 1,655 common dates, session-open delta mean 0.271 pt (tape), 9 days with entry-count mismatches (all traced to the cases above), 0 skipped days on either engine (every session had its 09:30 bar).
- **No systematic strategy-logic divergence found.** The negative P&L is a property of the rule set on this instrument, confirmed independently by two engines.

## 4. Buy-and-hold comparison (identical range, from the M1 export)

- First price 3758.942 (2021-01-03 23:00) → last 7224.954 (2026-04-30 23:59): **index +92.21%, CAGR 13.07%** — reproduces the ORB benchmark numbers exactly (independent recomputation from the M1 source).
- $10k / 1-lot ex-financing: **+34.66% (CAGR 5.75%)**, max DD 32.24%, Sharpe 0.82, return/DD 1.07 — the optimistic upper bound.
- Financing (swap_long −20 pts/day, Wednesday ×5): 1,656 open days → ≈ −$55,360 swap → **with-financing LONG −518.94%**. Present both framings; the ex-financing bound already dominates VWAP MR, so the conclusion is robust to financing assumptions (VWAP MR positions are intraday, zero swap).

## 5. Final comparison vs ORB (the actual decision question)

| metric | VWAP-MT5 | VWAP-Python | ORB-MT5 | ORB-Python | B&H ex-fin | B&H w/ fin |
|---|---|---|---|---|---|---|
| total return | **−3.07%** | −2.02% | +4.71% | +5.31% | +34.66% | −518.94% |
| CAGR | −0.58% | −0.38% | +0.87% | +0.98% | +5.75% | n/a |
| max DD | 3.07% | 2.06% | 5.13% | 4.99% | 32.24% | 32.24% |
| Sharpe | **−0.75** | −0.49 | +0.34 | +0.38 | +0.82 | +0.82 |
| return/DD | −1.00 | −0.98 | 0.92 | 1.06 | 1.07 | −16.09 |

**Complementarity vs redundancy — first pass (daily P&L correlation over the overlapping window):**

- corr(VWAP-MT5, ORB-MT5) over 1,369 common session days: **Pearson +0.015, Spearman +0.044**
- corr(VWAP-Python, ORB-Python) over 1,370 days: **Pearson −0.001, Spearman +0.029**

The two strategies are **statistically uncorrelated — genuinely different edge sources** (ORB = morning breakout momentum; VWAP MR = intraday fading back to value), not redundant. However, correlation alone is not the decision criterion: VWAP MR's edge is **negative** on both engines over this window, so the equal-weight combination destroys rather than adds value — combined MT5 +$163.57 / Sharpe 0.11 / maxDD $602 vs ORB-alone +$470.67 / Sharpe 0.34 / maxDD $513. **The strategies are complementary in risk structure but there is nothing worth diversifying INTO on current evidence.** A future variant of VWAP MR that reached non-negative expectancy would be an interesting diversifier precisely because of the ~0 correlation; the 2024 sub-period (py +$29.5) shows the rule is not uniformly negative.

## 6. Verdict statements

1. VWAP MR does NOT beat buy-and-hold on absolute (−3.07% vs +34.66% ex-fin) or risk-adjusted (Sharpe −0.75 vs 0.82; return/DD −1.00 vs 1.07) return — agreed by both engines, robust to financing assumptions.
2. VWAP MR is strictly dominated by ORB over the identical window on return, Sharpe and return/DD (with lower, but irrelevant, drawdown).
3. Engine agreement is 94.0% with 0 unexplained residuals — the validation pipeline (native EA + standalone reference + classified reconciliation) reproduced the ORB methodology cleanly, including two new custom-symbol execution lessons (grid-priced market orders, grid-priced closes) now fixed in-source.
4. No deployment. No combination with ORB. The uncorrelated P&L stream is noted as a property of the edge source, not an actionable result.

## 7. Deliverables

- `ea/VWAP_MeanReversion.mq5` — native EA (0 errors, 0 warnings); repo copy == terminal Experts copy
- `evaluate/vwap_reference.py` — standalone reference; `evaluate/vwap_reconcile.py` — reconciliation; `evaluate/vwap_stats.py` — matrix/aggregation/B&H/ORB-correlation
- `data/results/vwap_2021_2026/` — this REPORT.md, RECONCILE.md, monthly_matrix.csv (64 months), aggregated.json, py/mt5 trades+days CSVs, us500tick_m1.csv (1,791,504 rows), probe_report.md + probe_vwap_data.py (Step-3 verification), run_vwap_batches.py + stitch_batches.py + batches/ (6 yearly runs, summary.csv, agent logs, meta.json), spotcheck_reference.py (independent signal-math verification), compile.log

## Appendix — Monthly matrix (64 months)

Caption: base $10k / 1 lot, no compounding; PnL in USD; win% = share of winning trades; maxdd = in-month peak-to-trough equity drop (USD); bh = close-to-close price return.
| month | py tdays | py tr | py win% | py pnl | py maxdd | mt5 tdays | mt5 tr | mt5 win% | mt5 pnl | mt5 maxdd | bh month % | bh cum % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2021-01 | 20 | 2 | 0.0 | -15.22 | 15.22 | 19 | 2 | 0.0 | -15.24 | 15.24 | -2.0 | -2.0 |
| 2021-02 | 20 | 3 | 66.7 | 7.41 | 7.76 | 20 | 3 | 66.7 | 7.2 | 7.77 | 4.21 | 2.13 |
| 2021-03 | 23 | 2 | 0.0 | -15.4 | 15.4 | 23 | 2 | 0.0 | -15.46 | 15.46 | 3.7 | 5.91 |
| 2021-04 | 21 | 0 | 0.0 | 0.0 | 0.0 | 21 | 0 | 0.0 | 0.0 | 0.0 | 5.21 | 11.43 |
| 2021-05 | 21 | 4 | 25.0 | -16.01 | 24.87 | 21 | 4 | 25.0 | -16.21 | 24.99 | 0.36 | 11.83 |
| 2021-06 | 22 | 1 | 0.0 | -8.39 | 8.39 | 22 | 1 | 0.0 | -8.52 | 8.52 | 2.39 | 14.5 |
| 2021-07 | 22 | 1 | 100.0 | 10.01 | 0.0 | 22 | 1 | 100.0 | 9.86 | 0.0 | 2.27 | 17.09 |
| 2021-08 | 22 | 0 | 0.0 | 0.0 | 0.0 | 22 | 0 | 0.0 | 0.0 | 0.0 | 2.93 | 20.53 |
| 2021-09 | 22 | 2 | 0.0 | -17.5 | 17.5 | 22 | 2 | 0.0 | -17.69 | 17.69 | -4.73 | 14.83 |
| 2021-10 | 21 | 2 | 100.0 | 21.96 | 0.0 | 21 | 2 | 100.0 | 22.28 | 0.0 | 6.98 | 22.84 |
| 2021-11 | 22 | 3 | 33.3 | -10.26 | 10.26 | 22 | 3 | 33.3 | -10.37 | 10.37 | -0.66 | 22.02 |
| 2021-12 | 21 | 3 | 33.3 | -12.95 | 18.55 | 21 | 3 | 33.3 | -12.92 | 18.63 | 4.24 | 27.2 |
| 2022-01 | 21 | 6 | 33.3 | -8.71 | 28.14 | 21 | 6 | 33.3 | -8.74 | 28.3 | -5.84 | 19.77 |
| 2022-02 | 20 | 9 | 55.6 | 25.12 | 17.56 | 20 | 9 | 55.6 | 24.89 | 17.69 | -2.68 | 16.57 |
| 2022-03 | 23 | 9 | 44.4 | -3.72 | 17.82 | 23 | 9 | 44.4 | -3.92 | 17.85 | 3.79 | 20.98 |
| 2022-04 | 20 | 6 | 33.3 | -8.44 | 16.94 | 20 | 6 | 33.3 | -8.58 | 17.01 | -8.83 | 10.3 |
| 2022-05 | 22 | 7 | 57.1 | 14.77 | 16.09 | 22 | 7 | 57.1 | 14.92 | 16.11 | 0.08 | 10.38 |
| 2022-06 | 22 | 15 | 40.0 | -7.53 | 28.37 | 22 | 15 | 40.0 | -5.18 | 25.98 | -8.72 | 0.76 |
| 2022-07 | 21 | 10 | 50.0 | 17.02 | 15.77 | 21 | 10 | 50.0 | 16.79 | 15.82 | 8.64 | 9.46 |
| 2022-08 | 23 | 5 | 20.0 | -23.02 | 32.85 | 23 | 5 | 20.0 | -23.01 | 32.91 | -4.37 | 4.67 |
| 2022-09 | 22 | 10 | 30.0 | -17.8 | 35.87 | 22 | 9 | 44.4 | 9.68 | 16.32 | -8.52 | -4.24 |
| 2022-10 | 21 | 9 | 22.2 | -32.89 | 35.05 | 21 | 9 | 22.2 | -33.68 | 35.28 | 7.74 | 3.17 |
| 2022-11 | 22 | 9 | 33.3 | -16.3 | 23.88 | 22 | 9 | 22.2 | -34.63 | 34.63 | 5.34 | 8.68 |
| 2022-12 | 21 | 12 | 25.0 | -40.58 | 55.86 | 21 | 13 | 23.1 | -43.23 | 57.75 | -5.88 | 2.29 |
| 2023-01 | 21 | 7 | 57.1 | 15.41 | 8.23 | 21 | 7 | 42.9 | -1.13 | 8.26 | 5.77 | 8.19 |
| 2023-02 | 20 | 6 | 33.3 | -14.96 | 24.69 | 20 | 6 | 33.3 | -15.07 | 24.86 | -2.68 | 5.29 |
| 2023-03 | 23 | 10 | 70.0 | 43.54 | 0.21 | 23 | 10 | 70.0 | 41.74 | 0.22 | 3.82 | 9.31 |
| 2023-04 | 19 | 4 | 25.0 | -14.88 | 16.51 | 19 | 4 | 25.0 | -15.59 | 16.73 | 1.31 | 10.75 |
| 2023-05 | 23 | 4 | 50.0 | 6.13 | 8.37 | 23 | 4 | 50.0 | 6.16 | 8.41 | 0.57 | 11.38 |
| 2023-06 | 22 | 2 | 50.0 | 0.9 | 8.6 | 22 | 2 | 50.0 | 0.49 | 8.64 | 6.14 | 18.23 |
| 2023-07 | 20 | 1 | 100.0 | 11.19 | 0.0 | 20 | 1 | 100.0 | 10.86 | 0.0 | 3.29 | 22.12 |
| 2023-08 | 23 | 5 | 20.0 | -21.3 | 35.63 | 23 | 5 | 20.0 | -21.69 | 35.69 | -1.73 | 20.0 |
| 2023-09 | 21 | 0 | 0.0 | 0.0 | 0.0 | 21 | 0 | 0.0 | 0.0 | 0.0 | -4.85 | 14.18 |
| 2023-10 | 22 | 13 | 38.5 | -20.8 | 44.3 | 22 | 13 | 38.5 | -21.4 | 45.03 | -2.54 | 11.27 |
| 2023-11 | 22 | 1 | 0.0 | -8.75 | 8.75 | 22 | 1 | 0.0 | -8.83 | 8.83 | 9.12 | 21.42 |
| 2023-12 | 20 | 0 | 0.0 | 0.0 | 0.0 | 20 | 0 | 0.0 | 0.0 | 0.0 | 4.45 | 26.82 |
| 2024-01 | 22 | 1 | 100.0 | 12.79 | 0.0 | 22 | 1 | 100.0 | 12.94 | 0.0 | 1.82 | 29.13 |
| 2024-02 | 21 | 1 | 0.0 | -9.94 | 9.94 | 21 | 1 | 0.0 | -10.05 | 10.05 | 4.83 | 35.36 |
| 2024-03 | 20 | 2 | 0.0 | -20.6 | 20.6 | 20 | 2 | 0.0 | -20.86 | 20.86 | 3.67 | 40.32 |
| 2024-04 | 22 | 3 | 33.3 | -8.09 | 10.29 | 22 | 2 | 50.0 | 2.14 | 0.0 | -4.68 | 33.76 |
| 2024-05 | 23 | 0 | 0.0 | 0.0 | 0.0 | 23 | 0 | 0.0 | 0.0 | 0.0 | 5.06 | 40.53 |
| 2024-06 | 20 | 1 | 100.0 | 7.56 | 0.0 | 20 | 1 | 100.0 | 7.18 | 0.0 | 3.51 | 45.46 |
| 2024-07 | 23 | 3 | 66.7 | 11.28 | 11.17 | 23 | 2 | 50.0 | 0.97 | 11.17 | 1.52 | 47.67 |
| 2024-08 | 22 | 11 | 54.5 | 28.39 | 25.86 | 22 | 11 | 54.5 | 31.52 | 21.28 | 1.62 | 50.06 |
| 2024-09 | 21 | 3 | 66.7 | 14.65 | 11.03 | 21 | 3 | 66.7 | 14.24 | 11.08 | 1.99 | 53.05 |
| 2024-10 | 23 | 1 | 100.0 | 5.7 | 0.0 | 23 | 1 | 100.0 | 5.67 | 0.0 | -0.73 | 51.93 |
| 2024-11 | 21 | 2 | 100.0 | 29.5 | 0.0 | 21 | 1 | 100.0 | 15.66 | 0.0 | 5.67 | 60.54 |
| 2024-12 | 20 | 1 | 100.0 | 14.46 | 0.0 | 20 | 1 | 100.0 | 14.5 | 0.0 | -2.22 | 56.98 |
| 2025-01 | 21 | 3 | 66.7 | 12.29 | 12.05 | 21 | 3 | 66.7 | 11.93 | 12.06 | 2.3 | 60.59 |
| 2025-02 | 20 | 3 | 0.0 | -35.82 | 23.89 | 20 | 4 | 0.0 | -48.01 | 24.0 | -1.5 | 58.18 |
| 2025-03 | 21 | 10 | 40.0 | -13.92 | 45.55 | 21 | 10 | 20.0 | -62.22 | 78.23 | -5.94 | 48.78 |
| 2025-04 | 21 | 9 | 44.4 | 11.76 | 10.93 | 21 | 10 | 40.0 | 0.12 | 15.88 | 0.32 | 49.26 |
| 2025-05 | 22 | 7 | 42.9 | -17.72 | 37.44 | 22 | 7 | 42.9 | -19.53 | 37.94 | 5.28 | 57.14 |
| 2025-06 | 21 | 2 | 50.0 | 11.22 | 0.0 | 21 | 2 | 50.0 | 10.79 | 0.0 | 4.97 | 64.95 |
| 2025-07 | 23 | 0 | 0.0 | 0.0 | 0.0 | 23 | 0 | 0.0 | 0.0 | 0.0 | 2.14 | 68.48 |
| 2025-08 | 21 | 0 | 0.0 | 0.0 | 0.0 | 21 | 0 | 0.0 | 0.0 | 0.0 | 2.27 | 72.31 |
| 2025-09 | 22 | 3 | 66.7 | 10.94 | 13.47 | 22 | 3 | 66.7 | 9.99 | 13.29 | 3.01 | 77.49 |
| 2025-10 | 23 | 2 | 0.0 | -26.53 | 26.53 | 23 | 2 | 0.0 | -26.76 | 26.76 | 2.55 | 82.01 |
| 2025-11 | 20 | 2 | 0.0 | -26.63 | 26.63 | 20 | 2 | 0.0 | -26.71 | 26.71 | -0.02 | 81.98 |
| 2025-12 | 22 | 3 | 33.3 | -14.89 | 27.13 | 22 | 3 | 33.3 | -14.89 | 27.15 | 0.14 | 82.23 |
| 2026-01 | 21 | 3 | 66.7 | 14.13 | 0.0 | 21 | 3 | 66.7 | 14.25 | 0.0 | 1.28 | 84.56 |
| 2026-02 | 20 | 5 | 60.0 | 15.55 | 27.25 | 20 | 5 | 60.0 | 15.21 | 27.32 | -0.91 | 82.89 |
| 2026-03 | 22 | 16 | 37.5 | -53.39 | 63.28 | 22 | 16 | 37.5 | -55.6 | 63.87 | -4.77 | 74.16 |
| 2026-04 | 21 | 1 | 0.0 | -13.16 | 13.16 | 21 | 1 | 0.0 | -13.36 | 13.36 | 10.36 | 92.21 |
