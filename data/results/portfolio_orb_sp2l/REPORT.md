# REPORT — ORB + SP2L combined portfolio (allocation pre-declared, MT5 net legs)

**Run:** 2026-08-27 · Allocation schemes + conventions pre-declared in
`SPEC.md` BEFORE any combined result was computed (the script's
reproduction gate aborted the first run before any combined number
existed; the gate diagnosis is documented in SPEC Step 0). Inputs: the
two validated MT5 net-of-spread runs — ORB
(`data/results/orb_2021_present/mt5_trades.csv`) and SP2L
(`data/results/sp2l_2021_2026/mt5_trades.csv`). No new strategy logic.

## Step 0 — reproduction (data plumbing, PASSED)

| leg | trades | total pts | Sharpe | max DD |
|---|---|---|---|---|
| ORB | 447 = 447 ✓ | +470.67 = +470.67 ✓ | 0.337 (doc 0.34) ✓ | 5.13% = 5.13% ✓ |
| SP2L | 1,343 = 1,343 ✓ | +281.84 = +281.84 ✓ | 0.275 (doc 0.25) ⚠ | 416.71 pts = 416.71 ✓ |

⚠ The SP2L Sharpe delta is a day-universe convention, proven before any
combined number was computed: the documented 0.25 was computed by
sp2l_stats.py on a 2,605-day union-of-strategy-days universe; on this
exercise's common 1,369-day ORB session-day universe (the universe the
ORB docs used) SP2L's trades/total/DD reproduce EXACTLY and Sharpe is
0.275. All comparisons below use common-universe leg stats so standalone
rows and combinations are computed identically. Benign bookkeeping note:
SP2L's trades file books one day (2025-07-04 UTC, −21.85) that its own
days CSV attributes to the prior NY date; totals unaffected. Daily
correlation (mt5-mt5, zero-filled, common universe): **ρ = +0.069**
(documented py-based value +0.054 — consistent near-zero).

## Step 1 — pre-declared schemes, realized weights (σ over full range)

σ(ORB daily) = 16.217 pts, σ(SP2L daily) = 11.905 pts. All schemes at
constant book size Σ lots = 2.0.

| scheme | ORB lots | SP2L lots | risk split |
|---|---|---|---|
| S1 equal notional | 1.000 | 1.000 | 58/42 |
| S2 equal risk contribution | 0.847 | 1.153 | 50/50 |
| S3 ORB-weighted/SP2L-capped 75/25 | 1.375 | 0.625 | 75/25 |

S3 rationale (pre-declared, fragility not performance): SP2L's documented
tape-sensitivity and parameter fragility (XAUUSD point-conversion,
SENSITIVITY_ADDENDUM) vs ORB's cleaner validation.

## Step 2 — summary (base $10k, no compounding, 2021-01..2026-04, 5.32y)

| portfolio | total ret | CAGR | Sharpe | max DD | ret/DD |
|---|---|---|---|---|---|
| ORB alone (1 lot) | 4.71% | 0.87% | 0.337 | 5.13% | 0.92 |
| SP2L alone (1 lot) | 2.82% | 0.52% | 0.275 | 4.17% | 0.68 |
| 2× ORB (same book size, no diversification) | 9.42% | 1.71% | 0.337 | 10.26% | 0.92 |
| **S1 equal notional** | **7.53%** | **1.37%** | **0.420** | 8.02% | 0.94 |
| S2 equal risk | 7.24% | 1.32% | 0.418 | 7.94% | 0.91 |
| S3 ORB75/SP2L25 | 8.23% | 1.50% | 0.398 | 8.29% | 0.99 |
| B&H ex-financing (documented) | 34.66% | 5.75% | 0.820 | 32.24% | 1.07 |

## Step 3 — does diversification actually help?

**(a) Drawdown — sub-additive, not smaller in absolute terms.** Combined
max DD (8.0–8.3%) is LARGER than either leg's standalone (5.13/4.17%) —
the book is 2 lots, so that is expected; the fair comparisons: per-lot
DD ≈ 4.0% (S1) vs ORB's 5.13%, and vs a same-size undiversified book
(2×ORB = 10.26%) S1's 8.02% is a 22% reduction. Low correlation does
deliver DD sub-additivity — real but modest.

**(b) Sharpe — YES, every scheme beats ORB's 0.337.** S1 0.420 (+0.083,
+25% relative), S2 0.418, S3 0.398. Scheme choice barely matters
(spread 0.022); the naive 1+1 is best because SP2L's diversification
contribution is largest at full weight — but SP2L is precisely the
fragile leg. Even the best combined Sharpe remains half of B&H's 0.82.

**(c) Tails — the near-zero correlation is an average, and it fails
exactly when it matters.** 91 of 1,369 days (6.6%) had BOTH legs lose.
Worst joint days: 2025-04-11 (ORB −85.64 + SP2L −36.22 = −121.86,
−1.22% of base in one day), 2025-03-04 (−94.66), 2022-05-02 (−78.71).
Worst drawdown windows: every scheme AND both standalone legs bottom on
the SAME dates (troughs 2025-12-05/08; S1 peak 2025-04-03 → −802;
S3 peak 2024-09-06 → −829; ORB alone 2024-09-06 → −513; SP2L alone
2025-01-10 → −417). The 2025 losing streak hit both strategies
simultaneously — joint-tail correlation arrived precisely when
diversification was needed. At end-of-data every portfolio remains
underwater from its peak (S1 −382, S3 −339, ORB −208, SP2L −301 from
respective peaks).

## Step 4 — year-by-year stability (S3 primary, S1 context; 2026 = 4 months)

| year | ORB | SP2L | S1 | S3 | S3 share of total |
|---|---|---|---|---|---|
| 2021 | −29.23 | +100.07 | +70.84 | +22.29 | 2.7% |
| 2022 | +266.83 | +153.14 | +419.97 | +462.65 | 56.2% |
| 2023 | +400.03 | +140.91 | +540.94 | +638.23 | **77.5%** |
| 2024 | −182.25 | +129.24 | −53.01 | −169.96 | −20.6% |
| 2025 | −228.03 | −260.25 | −488.28 | −476.18 | −57.8% |
| 2026 (4m) | +243.32 | +18.73 | +262.05 | +346.37 | 42.1% |

**Concentration flag FIRES: best year (2023) = 77.5% of total combined
profit > 60%.** And 2024+2025 are consecutive negative years for the
combined book (2025 jointly negative on BOTH legs — the only year
neither strategy made money). The 2026 partial recovery (+346 S3) is
4 months of data, not a confirmed regime.

## Step 5 — recommendation (honest)

**The diversification is mathematically real but not good enough to
justify adding SP2L: ORB alone remains the better choice.**

What combining delivers on this tape: Sharpe 0.337 → 0.40–0.42, DD
sub-additivity (−22% vs a doubled ORB book), ret/DD up to 0.99, and
2024 would actually have been cushioned (SP2L +129 vs ORB −182).

Why it still fails the bar:
1. **Concentration:** 77.5% of combined profit is 2023 alone — fails the
   project's own 60% convention; the entire diversification benefit was
   earned in 2022–2023.
2. **Joint tail:** the one year that mattered (2025) had both legs lose
   together (−488 S1); all portfolios bottomed the same week in
   Dec-2025 and none had recovered its peak by end-of-data. Average
   correlation ≈ 0 did not protect the joint tail.
3. **Unpriced fragility:** SP2L's +281.84 carries documented
   tape-sensitivity (10% trade-set differences across tapes) and a
   parameter-convention caveat whose survivable band is narrow; any
   realistic haircut to the fragile leg's contribution eats most of the
   +0.06–0.08 Sharpe gain (at S1, zeroing SP2L's 2025 alone drops the
   combined advantage materially; at S3 the satellite already concedes
   most of the Sharpe benefit, keeping mainly the return tilt).
4. **Still dominated:** even the best combined row (S3 ret/DD 0.99,
   Sharpe 0.40) sits below B&H ex-financing (1.07, 0.82) on every
   risk-adjusted metric, on far more operational complexity (two EAs,
   two validation regimes, one fragile data dependency).

If the combined book is nevertheless desired, S3 (ORB 1.375 / SP2L
0.625) is the defensible form — best return and ret/DD, SP2L capped at
25% of risk for fragility reasons stated before the results — accepting
a demonstrated 15-month underwater period and an unrecovered peak. The
naive S1 has the best Sharpe but is the most exposed to SP2L's
fragility; S2 offers no advantage over S1 worth its complexity.

This exercise cannot retro-validate either leg; it only measures their
combination on the already-validated MT5 logs.

## Artifacts

`SPEC.md` (pre-declared) · `combined_daily_pnl.csv` (per-day leg and
scheme P&L) · `combined_equity.csv` (cumulative curves, all schemes +
legs) · `yearly_breakdown.csv` · `portfolio_summary.json` (reproduction
check, weights, metrics, tails, DD windows) ·
`evaluate/portfolio_orb_sp2l.py`
