# ORB Strategy — Full-Range Validation Report

**US500-tick · 2021-01-01 → 2026-05-01 (full available range)**

- **Report date:** 2026-08-23
- **Range covered:** 2021-01-03 23:00 UTC → 2026-04-30 23:55 UTC — 364,330 M5 bars · 64/64 months · 0 missing months · 0 multi-day gaps
- **Engines:** MT5 Strategy Tester (`ORB_Strategy.ex5`, Model=0 every tick) vs Python reference (`evaluate/orb_reference.py`, M5 bid bars)
- **Base:** USD 10,000 fixed account · 1 lot (contract 1.0 → $1 per price point) · NO compounding (fixed lot) · 1:100 leverage

## Headline conclusion

Over the full available range the ORB strategy was modestly profitable on both engines, but it did not beat a plain long position in the underlying. The MT5 backtest produced 447 trades with +4.71% total return (+$470.67), CAGR 0.87%, max drawdown 5.13% and annualized Sharpe 0.34; the Python reference produced 442 trades with +5.31% (+$531.13), CAGR 0.98%, max drawdown 4.99% and Sharpe 0.38. The two engines agree at trade level to 93.1% (416/447); all residual disagreements except one were traced to documented artifacts (gap-threshold straddles, start-of-data boundary, MT5 market-entry fallback, bid/ask entry straddles, early-close carries, exit-level straddles, 16:00-boundary tick snapshots), and the single unexplained case (2024-04-19) was fully diagnosed as a tape difference between the tester's synthetic-tick data base and the real-tick terminal base. Over the same window US500-tick rose +92.2% (CAGR 13.07%); on the same $10k / 1-lot base and excluding financing the buy-and-hold dollar return was +34.66% (CAGR 5.75%) with a higher Sharpe (0.82) and return/DD (1.07). ORB therefore does NOT beat buy-and-hold on absolute or risk-adjusted return over 2021 → 2026-04 on this instrument, and the conclusion is robust to financing assumptions.

## 1. Scope and data caveats

### 1.1 DATA ENDS 2026-04-30 — NOT "present"

- The dataset ends **2026-04-30 23:55 UTC**. May–August 2026 are absent: the `2026.hcc` history ends at April and no ticks beyond April were imported; the re-import extended the START back to 2021-01-01, but the END was never extended past April 2026.
- The backtest therefore covers the **full AVAILABLE range 2021-01-01 → 2026-05-01** (first bar 2021-01-03 23:00 UTC, last bar 2026-04-30 23:55 UTC). No "present" (2026-05+) claim is made; a fresh tick import for 2026-05+ would be required to extend the study.
- Operational note: the terminal was serving only the most recent ~100,000 bars per timeframe (`[Charts] MaxBars=100000`), which capped M5 at 2024-11-08+. It was raised to 3,000,000 and the terminal restarted; the full 364,330-bar range is now served. Relevant to any future full-range pull.

### 1.2 Price precision — uniform 0.001 grid across all years

- All M5 bars carry 3-decimal prices (e.g. 3758.942), uniform 2021–2026. Raw-tick verification (`copy_ticks_range`) shows 100% of bid and ask prices are exact multiples of 0.001 in every sampled year. There is **NO tick-size / spec discontinuity** over 2021–2026; the probe's "grid fraction not ~1.0" flags for all six years are this same uniform property, not a data-quality regression.
- The symbol spec is `digits=4, point=0.0001, tick grid=0.01`. All order levels are rounded to the symbol's **trade tick grid (0.01)**; unrounded 4-decimal stops are rejected by the terminal ("Invalid price"), matching the prior `orb_reconcile` finding. The tick-grid fix holds: no mass order rejection occurred and every batch placed orders.

### 1.3 KEY CAVEAT — the MT5 tester executes on ITS OWN synthetic-tick data base, not the real tick file

- The MT5 Strategy Tester runs on its own data base (TenTrade-Server, M1-derived **SYNTHETIC ticks, ~0.41-point spread**) for US500-tick — NOT the real Dukascopy-style tick file in the main terminal base (Custom) that feeds the M5 export and the Python reference.
- The 2024-04-19 diagnosis (`diag_20240419/`) proved it: a probe EA showed the tester's synthetic ask never crossed the SELL take-profit (min ask 4954.160 vs TP 4954.12), so MT5 correctly rode to `session_end`, while Python's bar-based check fired on the real-tick M5 bar low (bid side) 4953.513.
- This tester-vs-terminal tape difference drives most residual disagreements (boundary straddles, ask-vs-bid fills, the 8 boundary-tick cases) and explains the ~0.2–0.5 pt level deltas.
- Model=0 every-tick WAS used throughout (no downgrade), but on the tester's synthetic-tick tape, not the real tick file. The two tapes are NOT equivalent; results must be read with this fidelity caveat in mind.

## 2. Step 0 — Import verification (probe_report.md)

- M5 range: **364,330 bars**, first bar 2021-01-03 23:00 UTC, last bar 2026-04-30 23:55 UTC.
- Monthly coverage: **64/64 months** (2021-01 → 2026-04), **0 zero-bar months, 0 multi-day gaps**; only single-day holiday-like closures (17 flagged, e.g. 2021-01-01, 2024-12-25, 2026-04-03).
- Tick presence verified every year (13:00–16:00 UTC samples): bid/ask > 0 for 100% of ticks in 2021–2026; last data day 2026-04-30 full day = 101,389 ticks.
- Price precision: uniform 0.001 across all years (Section 1.2).
- DST: EA rule vs zoneinfo — 7/7 spot checks PASS (Section 3).
- CSV export: `us500tick_m5.csv`, 364,330 rows, strictly increasing 300-s timestamps; first row O=3758.942, last row C=7224.954.

## 3. Step 1 — DST re-verification (2021 / 2023 / 2026)

EA rule: offset = −4 h if t ∈ [2nd Sun Mar 07:00 UTC, 1st Sun Nov 06:00 UTC), else −5 h. Matches `zoneinfo` America/New_York on all 7 spot checks (2021-01-15, 2021-06-15, 2023-01-20, 2023-06-20, 2026-01-15, 2026-06-15, 2026-11-15) — **7/7 PASS**.

Data fingerprint — NY 9:30 cash-open spike window, pre/post DST:

| Year | pre-DST (expect 14:30 UTC bigger) | post-DST (expect 13:30 UTC bigger) | Verdict |
|---|---|---|---|
| 2021 | 14:30 bigger 5/5 | 13:30 bigger 4/5 | PASS |
| 2023 | 14:30 bigger 4/5 | 13:30 bigger 5/5 | PASS |
| 2026 | 14:30 bigger 5/5 | 13:30 bigger 4/5 | PASS |

First batch offset line confirms the EA's server mapping: `ORB NY offset now: -5 h -> server 2021.01.05 00:00 = NY 2021.01.04 19:00`.

## 4. Step 2 — MT5 backtest (batches/)

All 6 batches: **Model=0 (every tick) · M5 · US500-tick · 10,000 USD deposit · 1:100 leverage · 1 lot · no compounding**.

| Batch | Window | Ticks processed | M5 bars | Final balance | Trades | Day records |
|---|---|---|---|---|---|---|
| B2021 | 2021 | 16,365,123 | 68,241 | 9,970.77 | 74 | 309 |
| B2022 | 2022 | 31,445,792 | 68,569 | 10,266.83 | 102 | 310 |
| B2023 | 2023 | 10,114,662 | 68,168 | 10,400.03 | 87 | 310 |
| B2024 | 2024 | 11,249,506 | 68,450 | 9,817.74 | 83 | 312 |
| B2025 | 2025 | 19,062,659 | 68,254 | 9,771.97 | 74 | 312 |
| B2026 | 2026-01-01 → 04-30 | 7,820,903 | 22,369 | 10,243.32 | 27 | 102 |
| **TOTAL** | | **96,058,645** | — | | **447** | **1,655** |

- Total ticks processed across all 6 batches: **96,058,645**.
- Stitched: **447 MT5 trades, 1,655 day records**.
- **B2026 reproduces the prior verified 4-month run exactly** (27 trades, final balance 10,243.32).
- No mass order rejection: the 0.01 tick-grid fix holds; every batch placed orders.

## 5. Step 3 — Python reference (evaluate/orb_reference.py)

- Bar count/range confirmed: **364,330 bars, 2021-01-03 23:00 UTC → 2026-04-30 23:55 UTC** — the same M5 export used for the backtest and the B&H benchmark.
- Reference runs on M5 bid bars from the real-tick export (`us500tick_m5.csv`); 1,656 day records, 1,370 session trading days.

Summary statistics (same $10k / 1-lot, fixed-lot base; % returns = PnL / 10,000):

| Metric | Python | MT5 |
|---|---|---|
| Trades | 442 | 447 |
| Total PnL ($) | 531.13 | 470.67 |
| Total return | 5.31% | 4.71% |
| CAGR | 0.98% | 0.87% |
| Max drawdown ($ / %) | 499.15 / 4.99% | 513.24 / 5.13% |
| Annualized Sharpe (session-day daily × √252) | 0.380 | 0.337 |
| Return/DD | 1.06 | 0.92 |

Margin at 1:100 ≈ $37–72; position notional ≈ $3,700–7,200.

## 6. Step 4 — Reconciliation (RECONCILE.md; machine artifact via evaluate/orb_reconcile.py)

- Matched pairs: **437/442** Python trades matched to MT5 (same type, same open date, |open time| ≤ 10 min); **5 unmatched Python, 10 unmatched MT5** (15 trades on 13 days).
- Direction agreement within matched pairs: 437/437 (100%).
- **Overall agreement at tolerances (entry ≤ 1.5 pts, exit ≤ 1.5 pts, exit time ≤ 15 min, mapped reason): 416/447 = 93.1%.**

Matched-pair delta statistics (|MT5 − Python|):

| Delta | mean | p95 | max | tolerance | within |
|---|---|---|---|---|---|
| entry price (pts) | 0.318 | 1.007 | 1.446 | ≤ 1.5 | 437/437 |
| exit price (pts) | 0.429 | 1.182 | 12.127 | ≤ 1.5 | 427/437 |
| exit time (min) | 14.0 | 6.0 | 3,290 | ≤ 15 | 425/437 |
| mapped exit reason | — | — | — | — | 434/437 |

Per-year trade counts:

| Year | Python | MT5 | Matched | MT5 − Python |
|---|---|---|---|---|
| 2021 | 74 | 74 | 74 | +0 |
| 2022 | 103 | 102 | 101 | −1 |
| 2023 | 85 | 87 | 83 | +2 |
| 2024 | 81 | 83 | 80 | +2 |
| 2025 | 72 | 74 | 72 | +2 |
| 2026 | 27 | 27 | 27 | +0 |

### 6.1 Day-level decision differences — 12 of 1,655 common dates

| Classification | Count | Dates |
|---|---|---|
| entry_direction_diff | 8 | 2023-02-13, 2023-03-30, 2023-06-01, 2023-06-09, 2024-02-14, 2024-04-26, 2025-10-20, 2025-12-11 |
| gap_straddle | 2 | 2022-09-27, 2022-11-01 |
| no_prior_close_boundary | 1 | 2021-01-04 |
| orders_placed_diff | 1 | 2022-10-14 |

(The only-py 2021-01-03 Sunday-evening row is the inert first bar of the dataset, not a trading day.)

### 6.2 Unmatched trades (15 on 13 days)

- Unmatched Python (5): #147 SELL 2022-09-27 · #152 SELL 2022-10-14 · #193 BUY 2023-03-02 · #203 SELL 2023-03-30 · #295 BUY 2024-07-12.
- Unmatched MT5 (10): #157 SELL 2022-11-01 · #188 BUY 2023-02-13 · #193 BUY 2023-03-02 · #219 BUY 2023-06-01 · #221 BUY 2023-06-09 · #273 BUY 2024-02-14 · #288 BUY 2024-04-26 · #298 BUY 2024-07-12 · #408 BUY 2025-10-20 · #417 BUY 2025-12-11.

### 6.3 Early-close carries (2 matched trades)

| Date | Dir | py close | mt5 close | exit-time delta | py close price | mt5 close price | exit-price delta |
|---|---|---|---|---|---|---|---|
| 2024-11-29 (Black Friday) | BUY | 11-29 18:10 | 12-02 01:00 | +3,290 min | 6034.49 | 6028.38 | −6.107 |
| 2024-12-24 (Christmas Eve) | BUY | 12-24 18:10 | 12-25 23:00 | +1,730 min | 6036.62 | 6036.94 | +0.320 |

MT5's 16:00 NY force-close cannot fire on these early-close days (market already closed → retcode 10015/10018), so the position is carried into the next session and closed as `session_end`; Python closes at the day's last bar (`data_end`). These are the only matched pairs whose mapped exit REASON differs.

### 6.4 Exit outliers among matched pairs

- **|exit time| > 15 min — 12 pairs:** the 2 early-close carries above plus **10 exit-level straddles** (SL/TP levels differ slightly across the level-source delta + 0.01-grid rounding; a slow market crosses the level in different M5 bars — exit PRICES still agree within tolerance, only the exit TIMESTAMP moves).
- **|exit price| > 1.5 pts — 10 pairs:** **8 `session_end_boundary_tick`** cases (both sides close the same 16:00 NY bar with the same reason, but the exact MT5 exit tick at the boundary is absent from today's M5 export → ~1.5–1.9-pt price deltas; a tick-data-snapshot difference, not strategy logic), **1 early-close carry**, and **1 `exit_reason_flip`**.

### 6.5 Root causes (full range)

| # | Root cause | Days | Trades | Evidence |
|---|---|---|---|---|
| 1 | gap_filter threshold straddle | 2 | 2 | 2022-09-27 py 0.9964% vs mt5 1.0012% · 2022-11-01 py 1.0063% vs mt5 0.9992% |
| 2 | no_prior_close boundary | 1 | 0 | 2021-01-04, first session of the dataset |
| 3 | MT5 market-entry fallback (retcode 10015) | 1 | 1 | 2022-10-14, off-grid live bid |
| 4 | entry ask-vs-bid / entry straddle | 10 | 10 | 8 entry-direction days + 2 both-filled straddles |
| 5 | early-close carry | 2 | 2 | 2024-11-29, 2024-12-24 |
| 6 | exit-level straddle | — | 10 | matched pairs; exit timestamps move, prices agree |
| 7 | session_end_boundary_tick | — | 8 | 16:00 NY close bar; ~1.5–1.9-pt price deltas |
| 8 | ONE unexplained-then-diagnosed trade | 1 | 1 | 2024-04-19 SELL: py TP 19:45 @ 4954.05 vs mt5 session_end 20:00 @ 4966.18, delta 12.13 pts |

### 6.6 The 2024-04-19 diagnosis (diag_20240419/) — tester-tape finding

The single largest trade-level disagreement in the whole range: **py#284 / mt5#286, 2024-04-19 SELL**. Python fired take_profit at 19:45 (TP 4954.0527; the M5 19:45 bar low is 4953.513, a bid-side value); MT5 carried the same position to 20:00 and closed `session_end` @ 4966.18 (the order's TP was 4954.12). Exit-price delta +12.127 pts, PnL delta +12.13 USD.

Findings of the dedicated investigation:

- A re-run on current data is byte-for-byte identical to the original batch and the stitched row — **NOT a data-snapshot artifact**.
- A probe EA showed the tester's own feed never crosses the TP: the synthetic **ask floor in the window is 4954.160**, just 0.04 ABOVE the TP 4954.12, so MT5's ask-triggered SELL take-profit correctly never fires (confirmed against control case 2024-04-17, where the same mechanism DID fire the TP on the ask at 16:33:03, deal 5017.54).
- The tester's M1-derived synthetic tape (constant ~0.41 spread) smooths the main base's real-tick wick (bid 4953.513 / ask 4954.036 at 19:46:39) into a shallower dip whose ask never hits the TP.
- **Verdict: (b) Python reference bug, data-source component** — Python's bar-low (bid) TP check against the real-tick main base is not reproducible on the tape the EA/tester actually executes on. This tester-vs-terminal tape difference drives most residual disagreements; no systematic strategy-logic deviation was found.

## 7. Steps 5–6 — Monthly matrix and aggregation

- The full **64-month matrix** is embedded in the Appendix (generated programmatically from `monthly_matrix.csv`).
- Conventions: base $10,000 / 1 lot ($1 per price point), no compounding; py/mt5 PnL in USD; pnl % = PnL / 10,000; win rate = % of winning trades; maxdd = in-month peak-to-trough equity drop (USD); B&H columns are **close-to-close PRICE returns** (Section 8).
- Aggregated results (`evaluate/orb_stats.py` → `aggregated.json`): Section 5 table and Section 9.

Months where py/mt5 materially diverge (|pnl| > 20 USD or win-rate > 5 pp) — all seven are traceable to the reconciliation root causes in Section 6:

| Month | py PnL | mt5 PnL | py tr | mt5 tr | py win% | mt5 win% | Root-cause trace |
|---|---|---|---|---|---|---|---|
| 2022-09 | −72.66 | −104.09 | 11 | 10 | 36.4 | 30.0 | gap_straddle_day (09-27) |
| 2022-11 | −31.87 | −2.13 | 9 | 10 | 33.3 | 40.0 | gap_straddle_day (11-01) |
| 2023-02 | −94.37 | −76.63 | 6 | 7 | 16.7 | 28.6 | entry ask-vs-bid (02-13) |
| 2024-02 | −40.23 | −63.41 | 4 | 5 | 25.0 | 20.0 | entry ask-vs-bid (02-14) |
| 2024-04 | +93.72 | +70.95 | 7 | 8 | 57.1 | 50.0 | 04-19 diagnosis + entry straddle (04-26) |
| 2025-10 | +33.16 | +40.71 | 5 | 6 | 60.0 | 66.7 | entry ask-vs-bid (10-20) |
| 2025-12 | +13.47 | +34.94 | 5 | 6 | 60.0 | 66.7 | entry ask-vs-bid (12-11) |

## 8. Step 7 — Buy-and-hold benchmark (US500-tick, identical range)

- First price = open 2021-01-03 23:00 UTC **3758.942**; last price = close 2026-04-30 23:55 UTC **7224.954**.
- **Index price return +92.21%, CAGR 13.07%.**
- On the same $10k / 1-lot base (1 contract → $1 per price point), **ex-financing dollar return +34.66% (CAGR 5.75%)** — this is the **optimistic upper bound** (excludes financing).
- Max drawdown **32.24%** (≈ $3,224 on $10k) · daily Sharpe **0.819** · return/DD **1.07**.

### 8.1 Financing caveat

Symbol documents: `swap_long = −20 points/day`, `swap_short = +2`, `mode = points`, `rollover3days = 5` (Wednesday counts 5). A **5.3-year LONG hold accrues ≈ −$55,360 of swap** (1,656 open days + 278 Wednesdays × 4 extra points) → **with-financing long total return = −518.94%** on the $10k base. In other words, financing dwarfs the price gain. The raw price-difference return is explicitly the **optimistic upper bound** and must NOT be presented as the realistic cost of actually holding the position for 5 years.

ORB positions are **intraday-only** (force-closed by 16:00 NY) → **zero swap** for the strategy.

## 9. Step 8 — Final comparison and verdict

| metric | ORB-MT5 | ORB-Python | B&H ex-fin ($/10k) | B&H w/ fin long |
|---|---|---|---|---|
| total_return_pct | 4.71 | 5.31 | 34.66 | −518.94 |
| cagr_pct | 0.87 | 0.98 | 5.75 | n/a |
| max_dd_pct | 5.13 | 4.99 | 32.24 | 32.24 |
| annualized_sharpe | 0.34 | 0.38 | 0.82 | 0.82 |
| return_dd_ratio | 0.92 | 1.06 | 1.07 | −16.09 |

### Verdict

The two engines **AGREE** here (MT5 and Python reach the same conclusion on every line):

1. **Absolute return:** ORB MT5 4.71% vs B&H ex-financing 34.66% → **B&H higher on absolute return.** (ORB Python 5.31% — same: below 34.66%.)
2. **Risk-adjusted:** ORB Sharpe 0.34 vs B&H 0.82, and ORB return/DD 0.92 vs B&H 1.07 → **B&H higher on BOTH risk-adjusted measures.** (ORB Python: Sharpe 0.38 vs 0.82, return/DD 1.06 vs 1.07 — same.)
3. **Therefore ORB does NOT beat buy-and-hold on absolute OR risk-adjusted return over 2021 → 2026-04 on this instrument (ex-financing).**
4. ORB only "wins" against the financing-loaded B&H, which is a strawman given the symbol's swap rates and that ORB itself trades intraday with zero swap — do NOT use that framing.
5. **Even the optimistic upper bound (no-financing B&H) beats ORB, so the conclusion is robust to financing assumptions.**

## 10. Deliverables

- `REPORT.md` — this final report
- `monthly_matrix.csv` — 64-month matrix (appendix source)
- `RECONCILE.md` — step-4 reconciliation detail (case-by-case classification)
- `RECONCILE/REPORT artifacts` — machine-generated full reconciliation from `evaluate/orb_reconcile.py` (every field, every day, full outlier tables)
- `batches/` — per-year MT5 batch outputs (`summary.csv`, `year_YYYY/{trades, orb_days, agent log, meta}`)
- `us500tick_m5.csv` — full-range M5 export (364,330 rows)
- `py_trades.csv`, `py_days.csv`, `mt5_trades.csv`, `mt5_days.csv` — per-trade and per-day records for both engines
- `diag_20240419/` — the single-trade investigation (rerun trades/agent logs, tester-feed tick probe, main-base tick dump, `DIAGNOSIS.md`)
- `evaluate/orb_stats.py`, `evaluate/orb_reconcile_fullrange.py` — statistics and full-range reconciliation scripts

## Appendix — Monthly matrix (64 months)

Caption: base $10k / 1 lot (contract 1.0 → $1 per price point), no compounding. PnL in USD; pnl % = PnL / 10,000; win% = share of winning trades; maxdd = in-month peak-to-trough equity drop (USD). B&H `bh month %` / `bh cum %` are close-to-close PRICE returns over the same calendar window.
| month | py tdays | py skip | py skip gap | py trades | py win% | py pnl | py pnl % | py maxdd | mt5 tdays | mt5 skip | mt5 skip gap | mt5 trades | mt5 win% | mt5 pnl | mt5 pnl % | mt5 maxdd | bh month % | bh cum % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2021-01 | 20 | 1 | 0 | 8 | 50.0% | -24.67 | -0.25% | 60.11 | 19 | 0 | 0 | 8 | 50.0% | -26.73 | -0.27% | 61.63 | -2.00% | -2.00% |
| 2021-02 | 20 | 0 | 0 | 6 | 50.0% | 53.52 | 0.54% | 19.83 | 20 | 0 | 0 | 6 | 50.0% | 53.51 | 0.54% | 19.92 | 4.21% | 2.13% |
| 2021-03 | 23 | 3 | 3 | 8 | 25.0% | -119.51 | -1.20% | 153.60 | 23 | 3 | 3 | 8 | 25.0% | -121.02 | -1.21% | 154.00 | 3.70% | 5.91% |
| 2021-04 | 21 | 0 | 0 | 4 | 75.0% | 23.83 | 0.24% | 11.61 | 21 | 0 | 0 | 4 | 75.0% | 22.79 | 0.23% | 11.64 | 5.21% | 11.43% |
| 2021-05 | 21 | 2 | 2 | 6 | 33.3% | -8.71 | -0.09% | 40.92 | 21 | 2 | 2 | 6 | 33.3% | -9.58 | -0.10% | 41.08 | 0.36% | 11.83% |
| 2021-06 | 22 | 0 | 0 | 2 | 100.0% | 30.04 | 0.30% | 0.00 | 22 | 0 | 0 | 2 | 100.0% | 29.18 | 0.29% | 0.00 | 2.39% | 14.50% |
| 2021-07 | 22 | 2 | 2 | 5 | 60.0% | 53.45 | 0.53% | 21.29 | 22 | 2 | 2 | 5 | 60.0% | 53.17 | 0.53% | 21.27 | 2.27% | 17.09% |
| 2021-08 | 22 | 0 | 0 | 6 | 100.0% | 38.90 | 0.39% | 0.00 | 22 | 0 | 0 | 6 | 100.0% | 37.40 | 0.37% | 0.00 | 2.93% | 20.53% |
| 2021-09 | 22 | 1 | 1 | 8 | 62.5% | 54.38 | 0.54% | 31.44 | 22 | 1 | 1 | 8 | 62.5% | 52.24 | 0.52% | 31.56 | -4.73% | 14.83% |
| 2021-10 | 21 | 0 | 0 | 9 | 33.3% | -121.34 | -1.21% | 128.42 | 21 | 0 | 0 | 9 | 33.3% | -121.75 | -1.22% | 128.35 | 6.98% | 22.84% |
| 2021-11 | 22 | 2 | 2 | 5 | 20.0% | -67.95 | -0.68% | 51.34 | 22 | 2 | 2 | 5 | 20.0% | -68.84 | -0.69% | 51.97 | -0.66% | 22.02% |
| 2021-12 | 21 | 3 | 3 | 7 | 71.4% | 71.28 | 0.71% | 37.48 | 21 | 3 | 3 | 7 | 71.4% | 70.40 | 0.70% | 37.59 | 4.24% | 27.20% |
| 2022-01 | 21 | 5 | 5 | 6 | 50.0% | -20.02 | -0.20% | 89.46 | 21 | 5 | 5 | 6 | 50.0% | -20.07 | -0.20% | 89.52 | -5.84% | 19.77% |
| 2022-02 | 20 | 5 | 5 | 6 | 50.0% | 16.44 | 0.16% | 50.22 | 20 | 5 | 5 | 6 | 50.0% | 16.30 | 0.16% | 50.34 | -2.68% | 16.57% |
| 2022-03 | 23 | 2 | 2 | 13 | 53.8% | 49.83 | 0.50% | 76.17 | 23 | 2 | 2 | 13 | 53.8% | 48.95 | 0.49% | 76.26 | 3.79% | 20.98% |
| 2022-04 | 20 | 2 | 2 | 8 | 62.5% | 70.24 | 0.70% | 54.92 | 20 | 2 | 2 | 8 | 62.5% | 70.43 | 0.70% | 54.92 | -8.83% | 10.30% |
| 2022-05 | 22 | 7 | 7 | 8 | 62.5% | 53.42 | 0.53% | 39.24 | 22 | 7 | 7 | 8 | 62.5% | 53.18 | 0.53% | 39.25 | 0.08% | 10.38% |
| 2022-06 | 22 | 8 | 8 | 6 | 66.7% | 88.61 | 0.89% | 29.95 | 22 | 8 | 8 | 6 | 66.7% | 89.02 | 0.89% | 30.07 | -8.72% | 0.76% |
| 2022-07 | 21 | 5 | 5 | 7 | 57.1% | 13.59 | 0.14% | 33.27 | 21 | 5 | 5 | 7 | 57.1% | 11.18 | 0.11% | 33.34 | 8.64% | 9.46% |
| 2022-08 | 23 | 3 | 3 | 9 | 44.4% | -18.11 | -0.18% | 46.71 | 23 | 3 | 3 | 9 | 44.4% | -20.07 | -0.20% | 46.83 | -4.37% | 4.67% |
| 2022-09 | 22 | 3 | 3 | 11 | 36.4% | -72.66 | -0.73% | 110.58 | 22 | 4 | 4 | 10 | 30.0% | -104.09 | -1.04% | 112.41 | -8.52% | -4.24% |
| 2022-10 | 21 | 7 | 7 | 10 | 60.0% | 9.40 | 0.09% | 96.35 | 21 | 7 | 7 | 9 | 55.6% | -7.09 | -0.07% | 112.37 | 7.74% | 3.17% |
| 2022-11 | 22 | 5 | 5 | 9 | 33.3% | -31.87 | -0.32% | 51.10 | 22 | 4 | 4 | 10 | 40.0% | -2.13 | -0.02% | 52.22 | 5.34% | 8.68% |
| 2022-12 | 21 | 3 | 3 | 10 | 80.0% | 132.25 | 1.32% | 5.92 | 21 | 3 | 3 | 10 | 80.0% | 131.22 | 1.31% | 5.53 | -5.88% | 2.29% |
| 2023-01 | 21 | 1 | 1 | 8 | 50.0% | 44.63 | 0.45% | 55.76 | 21 | 1 | 1 | 8 | 50.0% | 43.97 | 0.44% | 55.91 | 5.77% | 8.19% |
| 2023-02 | 20 | 4 | 4 | 6 | 16.7% | -94.37 | -0.94% | 106.27 | 20 | 4 | 4 | 7 | 28.6% | -76.63 | -0.77% | 88.34 | -2.68% | 5.29% |
| 2023-03 | 23 | 4 | 4 | 12 | 41.7% | 21.80 | 0.22% | 47.48 | 23 | 4 | 4 | 11 | 45.5% | 32.51 | 0.33% | 48.03 | 3.82% | 9.31% |
| 2023-04 | 19 | 0 | 0 | 7 | 57.1% | 79.91 | 0.80% | 15.10 | 19 | 0 | 0 | 7 | 57.1% | 78.79 | 0.79% | 15.09 | 1.31% | 10.75% |
| 2023-05 | 23 | 0 | 0 | 9 | 66.7% | 65.78 | 0.66% | 29.27 | 23 | 0 | 0 | 9 | 66.7% | 63.58 | 0.64% | 29.43 | 0.57% | 11.38% |
| 2023-06 | 22 | 0 | 0 | 5 | 60.0% | 31.32 | 0.31% | 17.63 | 22 | 0 | 0 | 7 | 57.1% | 43.19 | 0.43% | 20.88 | 6.14% | 18.23% |
| 2023-07 | 20 | 0 | 0 | 1 | 100.0% | 36.73 | 0.37% | 0.00 | 20 | 0 | 0 | 1 | 100.0% | 36.65 | 0.37% | 0.00 | 3.29% | 22.12% |
| 2023-08 | 23 | 0 | 0 | 10 | 60.0% | 37.03 | 0.37% | 25.48 | 23 | 0 | 0 | 10 | 60.0% | 35.68 | 0.36% | 25.57 | -1.73% | 20.00% |
| 2023-09 | 21 | 1 | 0 | 6 | 50.0% | 3.16 | 0.03% | 56.13 | 21 | 1 | 0 | 6 | 50.0% | 1.53 | 0.02% | 56.29 | -4.85% | 14.18% |
| 2023-10 | 22 | 0 | 0 | 12 | 58.3% | 65.86 | 0.66% | 46.61 | 22 | 0 | 0 | 12 | 58.3% | 63.51 | 0.64% | 46.79 | -2.54% | 11.27% |
| 2023-11 | 22 | 1 | 1 | 3 | 100.0% | 43.75 | 0.44% | 0.00 | 22 | 1 | 1 | 3 | 100.0% | 43.31 | 0.43% | 0.00 | 9.12% | 21.42% |
| 2023-12 | 20 | 0 | 0 | 6 | 83.3% | 35.71 | 0.36% | 20.35 | 20 | 0 | 0 | 6 | 83.3% | 33.94 | 0.34% | 20.42 | 4.45% | 26.82% |
| 2024-01 | 22 | 0 | 0 | 6 | 16.7% | -75.76 | -0.76% | 75.76 | 22 | 0 | 0 | 6 | 16.7% | -77.23 | -0.77% | 77.23 | 1.82% | 29.13% |
| 2024-02 | 21 | 2 | 2 | 4 | 25.0% | -40.23 | -0.40% | 40.23 | 21 | 2 | 2 | 5 | 20.0% | -63.41 | -0.63% | 63.41 | 4.83% | 35.36% |
| 2024-03 | 20 | 0 | 0 | 6 | 50.0% | 2.49 | 0.02% | 42.05 | 20 | 0 | 0 | 6 | 50.0% | 0.71 | 0.01% | 42.35 | 3.67% | 40.32% |
| 2024-04 | 22 | 2 | 2 | 7 | 57.1% | 93.72 | 0.94% | 38.87 | 22 | 2 | 2 | 8 | 50.0% | 70.95 | 0.71% | 40.57 | -4.68% | 33.76% |
| 2024-05 | 23 | 1 | 1 | 2 | 0.0% | -64.05 | -0.64% | 64.05 | 23 | 1 | 1 | 2 | 0.0% | -64.30 | -0.64% | 64.30 | 5.06% | 40.53% |
| 2024-06 | 20 | 0 | 0 | 5 | 0.0% | -119.43 | -1.19% | 119.43 | 20 | 0 | 0 | 5 | 0.0% | -119.61 | -1.20% | 119.61 | 3.51% | 45.46% |
| 2024-07 | 23 | 2 | 2 | 8 | 62.5% | 53.93 | 0.54% | 29.82 | 23 | 2 | 2 | 8 | 62.5% | 52.23 | 0.52% | 29.86 | 1.52% | 47.67% |
| 2024-08 | 22 | 4 | 4 | 8 | 87.5% | 179.54 | 1.80% | 17.98 | 22 | 4 | 4 | 8 | 87.5% | 178.24 | 1.78% | 18.04 | 1.62% | 50.06% |
| 2024-09 | 21 | 1 | 1 | 9 | 22.2% | -68.87 | -0.69% | 132.04 | 21 | 1 | 1 | 9 | 22.2% | -69.46 | -0.69% | 132.64 | 1.99% | 53.05% |
| 2024-10 | 23 | 0 | 0 | 11 | 54.5% | -19.01 | -0.19% | 61.42 | 23 | 0 | 0 | 11 | 54.5% | -21.69 | -0.22% | 61.68 | -0.73% | 51.93% |
| 2024-11 | 21 | 1 | 1 | 9 | 44.4% | -146.26 | -1.46% | 122.97 | 21 | 1 | 1 | 9 | 44.4% | -154.09 | -1.54% | 123.92 | 5.67% | 60.54% |
| 2024-12 | 20 | 1 | 1 | 6 | 83.3% | 86.64 | 0.87% | 34.64 | 20 | 1 | 1 | 6 | 83.3% | 85.41 | 0.85% | 34.66 | -2.22% | 56.98% |
| 2025-01 | 21 | 2 | 2 | 6 | 50.0% | 2.83 | 0.03% | 55.37 | 21 | 2 | 2 | 6 | 50.0% | 1.43 | 0.01% | 55.45 | 2.30% | 60.59% |
| 2025-02 | 20 | 2 | 2 | 10 | 60.0% | 89.60 | 0.90% | 44.38 | 20 | 2 | 2 | 10 | 60.0% | 87.35 | 0.87% | 44.69 | -1.50% | 58.18% |
| 2025-03 | 21 | 5 | 5 | 7 | 57.1% | 86.54 | 0.87% | 54.65 | 21 | 5 | 5 | 7 | 57.1% | 86.45 | 0.86% | 54.62 | -5.94% | 48.78% |
| 2025-04 | 21 | 12 | 12 | 8 | 37.5% | -227.43 | -2.27% | 212.10 | 21 | 12 | 12 | 8 | 37.5% | -228.64 | -2.29% | 212.88 | 0.32% | 49.26% |
| 2025-05 | 22 | 7 | 7 | 6 | 33.3% | -45.06 | -0.45% | 50.83 | 22 | 7 | 7 | 6 | 33.3% | -45.84 | -0.46% | 50.64 | 5.28% | 57.14% |
| 2025-06 | 21 | 0 | 0 | 7 | 28.6% | -129.19 | -1.29% | 90.32 | 21 | 0 | 0 | 7 | 28.6% | -129.80 | -1.30% | 90.75 | 4.97% | 64.95% |
| 2025-07 | 23 | 0 | 0 | 2 | 50.0% | 17.70 | 0.18% | 0.94 | 23 | 0 | 0 | 2 | 50.0% | 16.98 | 0.17% | 1.52 | 2.14% | 68.48% |
| 2025-08 | 21 | 0 | 0 | 7 | 42.9% | 25.62 | 0.26% | 73.70 | 21 | 0 | 0 | 7 | 42.9% | 23.23 | 0.23% | 74.17 | 2.27% | 72.31% |
| 2025-09 | 22 | 1 | 1 | 4 | 25.0% | -37.53 | -0.38% | 89.58 | 22 | 1 | 1 | 4 | 25.0% | -38.09 | -0.38% | 90.10 | 3.01% | 77.49% |
| 2025-10 | 23 | 1 | 1 | 5 | 60.0% | 33.16 | 0.33% | 88.42 | 23 | 1 | 1 | 6 | 66.7% | 40.71 | 0.41% | 88.55 | 2.55% | 82.01% |
| 2025-11 | 20 | 2 | 2 | 5 | 40.0% | -76.03 | -0.76% | 135.67 | 20 | 2 | 2 | 5 | 40.0% | -76.75 | -0.77% | 135.70 | -0.02% | 81.98% |
| 2025-12 | 22 | 0 | 0 | 5 | 60.0% | 13.47 | 0.13% | 27.12 | 22 | 0 | 0 | 6 | 66.7% | 34.94 | 0.35% | 27.26 | 0.14% | 82.23% |
| 2026-01 | 21 | 2 | 2 | 5 | 20.0% | 13.26 | 0.13% | 41.56 | 21 | 2 | 2 | 5 | 20.0% | 11.36 | 0.11% | 42.69 | 1.28% | 84.56% |
| 2026-02 | 20 | 0 | 0 | 11 | 81.8% | 143.76 | 1.44% | 14.07 | 20 | 0 | 0 | 11 | 81.8% | 141.23 | 1.41% | 13.85 | -0.91% | 82.89% |
| 2026-03 | 22 | 5 | 5 | 6 | 66.7% | 128.69 | 1.29% | 39.65 | 22 | 5 | 5 | 6 | 66.7% | 127.85 | 1.28% | 40.19 | -4.77% | 74.16% |
| 2026-04 | 21 | 2 | 2 | 5 | 60.0% | -36.62 | -0.37% | 84.11 | 21 | 2 | 2 | 5 | 60.0% | -37.12 | -0.37% | 84.28 | 10.36% | 92.21% |
