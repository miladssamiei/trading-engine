# SPEC — ORB + SP2L combined portfolio exercise (allocation pre-declared)

Status: written 2026-08-27 BEFORE any combined-performance computation.
This is a portfolio-construction exercise on EXISTING validated artifacts —
no new strategy design, no entry-rule changes, no exit tuning. Both legs
are the MT5 net-of-spread runs (live-tradable approximation), NOT the
Python gross references.

## Inputs (exact files)

- ORB: `data/results/orb_2021_present/mt5_trades.csv` (447 trades,
  +470.67 pts documented), `mt5_days.csv` (session-day universe).
- SP2L: `data/results/sp2l_2021_2026/mt5_trades.csv` (1,343 trades,
  +281.84 pts net documented), `mt5_days.csv` (per-day total_pnl,
  cross-check only).
- B&H benchmark row REUSED as documented (no recompute): +34.66%
  ex-financing, CAGR 5.75%, Sharpe 0.82, max DD 32.24%, return/DD 1.07.

## Step 0 — plumbing and reproduction gate (must pass before anything)

- Day universe: ORB session days (`mt5_days` is_session), sorted — the
  same universe orb_stats.py used for the documented ORB numbers.
- Daily P&L per leg: trades grouped by open-time UTC date, zero-filled
  over the universe (project correlation convention). SP2L's own
  `mt5_days.total_pnl` cross-checked against its trades groupby.
- Reproduction tolerances (else ABORT): trade counts exact; totals
  within 0.05 pts; max DD within 0.03 pp of documented (ORB 5.13%, SP2L
  416.71 pts = 4.17%). Sharpe: must match documented within 0.01
  (ORB 0.34), or within 0.03 with the deviation PROVEN to be the
  day-universe convention — diagnosed 2026-08-27 before any combined
  result was computed (first run aborted at this gate): SP2L's
  documented 0.25 was computed by sp2l_stats.py on a 2,605-day
  union-of-strategy-days universe; on the common 1,369-day ORB session
  universe SP2L's trades/total/DD reproduce EXACTLY and Sharpe is 0.275.
  All portfolio comparisons use the common-universe leg stats for
  consistency (standalone rows recomputed on the same universe as the
  combinations). Known-benign bookkeeping note: SP2L mt5_trades books
  one trade day (2025-07-04 UTC, -21.85) that the EA's own days CSV
  attributes to the prior NY date — trade totals unaffected.

## Conventions

$10,000 fixed base, $1/pt/lot, no compounding, static weights over the
full range. Sharpe = mean/std(ddof=1) × √252 on the zero-filled daily
series. Max DD from the true combined cumulative curve. CAGR exponent =
data-range years (first→last M5 bar, /365.25, 5.32y). Return/DD =
total return % / max DD %.

## Step 1 — allocation schemes (PRE-DECLARED; chosen on fragility and
risk logic, NOT on combined backtest performance)

σ_orb, σ_sp2l = full-range daily-P&L std of each MT5 leg on the common
universe. All schemes normalized to Σ(lots) = 2.0 — the equal-notional
total — so book size is constant across schemes and returns comparable.
Fractional lots allowed (static sizing fractions).

- S1 EQUAL NOTIONAL: 1.00 lot ORB + 1.00 lot SP2L (as independently
  backtested; naive baseline).
- S2 EQUAL RISK CONTRIBUTION: w_i ∝ 1/σ_i, Σw = 2 → equal w·σ risk
  contributions at constant book size.
- S3 ORB-WEIGHTED / SP2L-CAPPED at 75/25 risk budget: w_i ∝ b_i/σ_i
  with b = (0.75, 0.25), Σw = 2. PRIMARY scheme. Rationale (fragility,
  not performance): SP2L is documented as tape-sensitive (2,229/2,231
  setup points differ between tester and chart tapes) and parameter-
  fragile to the unverified XAUUSD point-conversion (survivable band
  narrow, per SENSITIVITY_ADDENDUM); ORB's validation is cleaner. 75/25
  chosen over 80/20 as the less aggressive weighting on the fragile leg.

All three computed and reported side by side; none selected post hoc.

## Steps 2-4

- Step 2: per scheme — total return, CAGR, Sharpe, max DD (true combined
  curve), return/DD; vs ORB alone, SP2L alone, B&H. One summary table.
- Step 3: (a) combined DD vs standalone DDs per scheme; (b) combined
  Sharpe vs ORB's 0.34; (c) joint-tail analysis — count and worst cases
  of BOTH-legs-negative days, worst joint day, worst combined DD window
  (dates + depth), plus the mt5-mt5 daily correlation (documented
  py-based value +0.054 noted for context).
- Step 4: year-by-year for S3 (S1 for context): per-year combined P&L,
  best-year share of total profit flagged at >60% (project convention),
  2026 partial year labeled.

## On interpretation

A negative conclusion ("combining doesn't help enough to justify SP2L's
fragility") is as valid a deliverable as a positive one. No EA, no
deployment, no new gates on the legs — this exercise cannot retro-validate
either strategy.
