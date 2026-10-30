# Multi-horizon momentum (daily-close trend following) on US500 — Variants P & R

**Run:** 2026-08-27 · **Data:** US500.c native D1, 2014-06-23 → 2026-08-27 (3,146 bars)
· **Base:** $10,000 fixed, avg-1-lot exposure, no compounding · **References:** `evaluate/momentum_variant_p.py`, `evaluate/momentum_variant_r.py`, `evaluate/momentum_common.py`

## Verdict up front

**Neither variant shows a temporally stable edge on US500, and both are strictly dominated by buy-and-hold on the identical range.** Ex-financing (before any swap), Variant P (paper-faithful, primary n=5 months) made **+$58 (+0.58%, Sharpe 0.01)** over 11.3 years vs B&H **+$5,607 (+56.1%, Sharpe 0.72)**; Variant R (video rubric, normalized) made **+$142 (+1.42%, Sharpe 0.02)** over 12.0 years vs B&H **+$5,724 (+57.2%, Sharpe 0.71)**. Under the project-documented swap convention (below) both variants — and B&H itself — lose catastrophically, every year. No Step-4 build (EA or script) is warranted. This is the honest outcome the task anticipated: *a genuinely new signal source in the literature, correct across the full spectrum of instruments the paper tested, but unconfirmed on this specific single instrument.*

## Step 0 — infrastructure decisions and justification

### D1 data source: US500.c native daily bars (not the tick import)

| candidate | bars | range | quality |
|---|---|---|---|
| **US500.c native D1** (real broker symbol, chosen) | **3,146** | **2014-06-23 → 2026-08-27 (12.2y, to today)** | 3 multi-day gaps, all Christmas/New-Year 2014; ~258-259 bars/yr; tick_volume>0 on all bars |
| US500-tick D1 (verified Dukascopy import) | 1,656 | 2021-01-03 → 2026-04-30 (5.3y) | 0 multi-day gaps, 8 holiday singles |

The native broker history is **2.3× longer** and current. Tape sanity vs the verified import over the 1,374 common dates (probe_report.md): daily-return **correlation 0.969**, **92.8% sign agreement**, median close diff **0.059%** (p90 0.219%) — acceptable divergence for a weekly-rebalanced strategy whose thresholds are not knife-edge (unlike SP2L's 0.04% gap gates, this study has no microstructure-sensitive thresholds). Live execution would trade US500.c, so its tape is the operationally correct one. US500-tick D1 is retained as an anchor only.

**Anchor reproduction (both pass):** US500-tick D1 B&H ex-financing **+34.69%** (documented +34.66%; residual = first-bar open vs close convention) and swap **−$55,360.00 exactly** (−20 × (1,656 + 4×278 Wednesdays)), reproducing the ORB/VWAP financing numbers.

### Costs and financing — and the swap-unit caveat

- P&L basis: **$1 per index point per lot** (contract 1.0), identical to every prior strategy here. Spread cost: **1.0 pt per unit of position change** (US500.c spread reads 100–300 symbol points = 1–3 index pts; 1.0 used, conservative vs the ~0.41 pt documented on the tick import).
- **Financing (primary model, project-documented convention):** `swap_long −20 $/day/lot, swap_short +2 $/day/lot, Wednesday ×5` (`rollover3days=5` read as Wednesday-swap ×5 — the reading behind all prior financing figures, e.g. the 5.3y B&H −$55,360 / −518.94%). Charged on the close-of-day position for the coming night.
- **★ Caveat:** strict MT5 points semantics on US500.c (`point = 0.01`) would instead charge −20 × 0.01 = **−$0.20/day**, i.e. financing **100× smaller**; the custom US500-tick symbol's inherited digits make its strict reading (−$0.002/day) nonsense, which is why the project standardized on the $1/point reading. This report therefore **always shows ex-financing results alongside**, and the conclusions below are checked under both readings (they do not change: see "Robustness to the swap-unit ambiguity").
- User prompt said "Wednesday triple"; the project's established symbol-spec convention is Wednesday ×5. Primary runs use ×5; under either multiplier the financing-dominated conclusions are identical in direction.

### Execution model & rebalance

- **Weekly rebalance** (re-score and resize once per week, per both sources): signals evaluated at the close of the **last trading day of each ISO week**, new position effective at that close. 636 weekly closes total.
- Variant P's inputs are monthly, so within-month weekly re-application is a no-op (positions turn over at the first weekly close after each month end — ~135 changes in 11.3y); Variant R re-scores genuinely weekly (613 changes).
- **Model choice (documented per Step 0, though no MT5 backtest was ultimately warranted):** a daily-close strategy needs no intrabar precision — **Model=1 (OHLC) or Model=2 (open prices)** is appropriate; Model=0 would add nothing except the synthetic-tape knife-edge sensitivity documented in SP2L. Not defaulted to Model=0; not run at all (signal never reached the build gate).
- Fractional lots allowed in the references; US500.c trades 0.1-lot steps live (volume_min 0.1) — rounding impact at weekly cadence is negligible vs spread and noted for any future build.

## Step 2 — single-instrument caveat (stated BEFORE any performance number)

**This test evaluates only whether the signal has any edge on US500 alone, over 12.2 years of daily closes. It is NOT a replication of Lempérière et al. (2014), whose evidence rests on 140–200 years across ~58 markets and whose reported performance comes from cross-market diversification — not from any single index. Nor is it the video's multi-asset portfolio. Nothing here can confirm or refute the paper's cross-asset claim; it can only fail or pass the much narrower single-instrument question. Variant R's rubric has no prior evidence at all; this run is its first.**

Small-sample caveat (Step 3 requirement): weekly-rebalanced trend following makes far fewer independent decisions than the intraday strategies — **636 weekly closes, 135 position changes (P n=5), 628 scores / 613 changes (R)** over 12 years. Any conclusion is correspondingly weak-anchored; a Sharpe of ±0.5 is well inside noise at this n.

## Variant P — paper-faithful signal (monthly EMA crossover, constant-risk sizing)

`s_n(t) = (p(t−1) − EMA_n(t−1)) / vol_n(t−1)` on month-end closes; EMA decay α=1/n months; vol = same-decay EMA of |monthly Δclose|; lots ∝ sign(s)/vol. Free constant fixed so avg |lots| = 1 (exposure-matched to B&H). Full paper horizon range:

| n (months) | first trade | ex-fin P&L $ | ex-fin Sharpe | ex-fin maxDD % | long % | swap $ | net $ |
|---|---|---|---|---|---|---|---|
| 2 | 2014-11-07 | +627 | +0.11 | 17.0 | 74.3 | −86,332 | −85,704 |
| 3 | 2015-01-02 | −902 | −0.16 | 16.2 | 77.4 | −87,336 | −88,239 |
| **5 (paper primary)** | 2015-05-01 | **+58** | **+0.01** | **13.9** | 82.5 | **−91,756** | **−91,697** |
| 7 | 2015-09-04 | +2,391 | +0.43 | 11.1 | 84.9 | −89,036 | −86,645 |
| 10 | 2016-03-04 | +2,019 | +0.36 | 16.5 | 89.7 | −90,647 | −88,628 |
| 15 | 2017-01-06 | +1,922 | +0.34 | 16.8 | 95.0 | −86,394 | −84,473 |
| 20 | 2017-11-03 | +2,916 | +0.48 | 16.8 | 98.3 | −80,890 | −77,975 |

Reading: consistent with the paper's *direction* (positive at 6/7 horizons — longer = better) but the magnitudes are noise-level on this instrument/range: best Sharpe +0.48 vs B&H +0.72. The signal is long 74–98% of the time (S&P bull regime), so with the documented financing it is simply a worse-financed index hold.

## Variant R — video rubric (4 lookbacks, ±4 score, cheat-sheet weight, 30d-vol scaling)

Score distribution over 628 weekly decisions: −4: 78, −2: 93, 0: 92, +2: 114, +4: 251 (long-tilted, as expected in a bull regime). Exposure 58.1% long / 27.2% short / 14.6% flat; median annualized vol 11.5%.

| config | ex-fin P&L $ | ex-fin Sharpe | ex-fin maxDD % | avg / max lots | max leverage | swap $ | net $ |
|---|---|---|---|---|---|---|---|
| **normalized (avg 1 lot, primary)** | **+142** | **+0.02** | **14.0** | 1.00 / 4.05 | 3.2× | **−83,724** | **−83,582** |
| literal target 10% | −521 | −0.04 | 32.1 | 2.08 / 10.19 | 8.0× | −170,578 | −171,099 |
| literal target 20% | −1,042 | −0.04 | 64.2 | 4.16 / 20.39 | 15.9× | −341,156 | −342,199 |
| literal target 30% | −1,564 | −0.04 | 96.2 | 6.24 / 30.58 | 23.9× | −511,734 | −513,298 |

Reading: the rubric's discretization and vol-scaling produce **no edge** on US500 — normalized ≈ 0, and the rubric-literal sizing is strictly negative ex-financing while carrying 8–24× notional leverage (the formula's `(target%/vol%)×portfolio` maps to avg 2–6 lots on a $10k portfolio). **This is the rubric's first recorded evidence and it is negative.**

## Side-by-side vs buy-and-hold (identical ranges, 1 lot, $10k base)

| | Variant P (n=5, 2015-05-01→) | Variant R (norm., 2014-08-22→) | B&H (P range) | B&H (R range) |
|---|---|---|---|---|
| ex-fin total $ | +58 | +142 | +5,607 | +5,724 |
| ex-fin return % | +0.58 | +1.42 | +56.07 | +57.24 |
| ex-fin Sharpe | 0.01 | 0.02 | **0.72** | **0.71** |
| ex-fin maxDD % | 13.9 | 14.0 | **12.1** | **12.1** |
| swap $ (1 lot / strategy) | −91,756 | −83,724 | −105,580 | −111,800 |
| with-financing $ | −91,697 | −83,582 | −99,974 | −106,076 |

Both variants lose to B&H on absolute return, Sharpe AND drawdown, before financing; after financing everything loses (B&H included — the financing convention makes ANY overnight-long US500 position unviable; the strategies merely lose slightly less swap than an always-long hold because of short/flat stints, while giving up nearly all price capture).

**Robustness to the swap-unit ambiguity:** under the strict-points reading (financing ×0.01: swap −$918 / −$837), Variant P nets ≈ **−$860**, Variant R ≈ **−$695** — still negative, still strictly below B&H ex-financing. **The verdict (no edge, dominated by B&H) is unchanged under either financing convention.**

## Year-by-year stability (the ORB/VWAP/trend-pullback/SP2L standard)

- **Net (with documented financing): 0 of 12 years positive for BOTH variants** — fails any G3-style stability gate outright.
- **Ex-financing:** P n=5 positive in 6/12 years (−696, −160, +1,117, +76, +173, +14, +645, −628, −73, +629, −573, −464); R positive in 6/12 (−307, −540, −69, +472, −35, +316, +62, −496, +321, +462, +248, −371, +79) — scattered signs, sub-$1,200 magnitudes: noise around zero, no temporal stability in either variant.

## Correlation vs the four intraday strategies (daily P&L, calendar-day aligned, zero-filled union — sp2l_stats conventions; ex-financing P&L used for the momentum side)

| | ORB | VWAP MR | trend-pullback WF | SP2L | B&H 1-lot price |
|---|---|---|---|---|---|
| Variant P (n=5) | −0.051 | +0.012 | −0.012 | +0.000 | **+0.298** |
| Variant R | +0.055 | −0.012 | +0.033 | −0.038 | −0.012 |

(P vs R: +0.121.) Sanity anchors reproduce prior work: trend-pullback–SP2L 0.143 (documented 0.142), ORB–trend-pullback 0.197. Both momentum variants are ≈0 vs every intraday strategy — the *frequency families are genuinely uncorrelated* — but with no edge to diversify into, and Variant P carries a 0.30 market-beta correlation, so it would ADD risk, not diversify, in any combined book.

## Step 4 — build decision: none (and what would have been chosen)

The gate was "only if a real, temporally stable signal" — not met (ex-fin Sharpe ≈ 0.01–0.02 primary configs, no year-stability, dominated by B&H on every metric). **No EA, no scheduled-execution script is built; there is nothing to automate.** For the record, had a variant been confirmed, the deliberate choice would have been a **weekly scheduled Python script via the MT5 API** (recompute signal on weekly close, place/resize one position, no tick-level logic — one decision per week makes an every-tick native EA unnecessary complexity), with any MT5-side backtest on Model=1/2 per Step 0. That reasoning is recorded here so the default-to-native-EA pattern of the intraday strategies is not silently inherited.

## Conclusion — with the single-instrument caveat restated

1. Variant P's paper carries real, peer-reviewed multi-market evidence; **on US500 alone over 12.2 years its signal shows no exploitable edge** (ex-fin Sharpe ≤ 0.48 across every horizon the paper tests, ~0 at the paper's primary n=5). Consistent with the paper's own construction: the edge it documents is cross-sectional and multi-century, not single-index.
2. Variant R's rubric — an untested retail simplification — **receives its first evidence here and it is negative** (no edge normalized; strictly negative under its own literal sizing with 8–24× leverage).
3. Both are dominated by B&H ex-financing; under the project's documented swap convention every multi-day-holding configuration including B&H itself is deeply unviable, and the strict-points swap reading does not rescue either variant (still negative, still dominated).
4. Uncorrelated with all four validated intraday strategies (|ρ| ≤ 0.06) — a different frequency family, but with nothing to diversify into; adding P to a book would add beta (ρ 0.30 vs B&H), not diversification.
5. This outcome is a **realistic and useful single-instrument result, stated plainly**: no confirmation, no build, no deployment. Re-testing would require multiple instruments (the paper's actual design), which this project's data does not currently cover.

## Artifacts

- `probe_daily.py`, `probe_report.md` — Step 0 source decision
- `export_d1.py`, `us500c_d1.csv`, `us500tick_d1.csv` — daily exports (bar-open UTC epoch, 00:00 day boundary)
- `variant_p_summary.csv`, `variant_p_yearly.csv`, `variant_p_daily_n5.csv`, `variant_p_monthly_n5.csv`, `variant_p_weekly_n5.csv`, `equity_curve_variant_p.png`
- `variant_r_summary.csv`, `variant_r_yearly.csv`, `variant_r_daily.csv`, `variant_r_monthly.csv`, `variant_r_weekly.csv`, `equity_curve_variant_r.png`
- `correlations.csv` (written by `evaluate/momentum_common.py` as the correlation runner)
- Reproduce: `python evaluate/momentum_variant_p.py` · `python evaluate/momentum_variant_r.py` · `python evaluate/momentum_common.py` (env: `reinforce_rl`)
