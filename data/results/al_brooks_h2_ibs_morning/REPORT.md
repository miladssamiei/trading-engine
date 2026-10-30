# Phase 1 REPORT — Al Brooks H2 + IBS>65 + NY-morning variant on US500-tick M5: NULL RESULT, stopped

**Run:** 2026-08-27 · Spec/gates pre-declared in `SPEC.md` / `GATE.md` BEFORE
any code or run · Data: verified US500-tick M5 export (364,330 bars,
2021-01..2026-04). Hypothesis #8, tested as an INDEPENDENT specification
(EMA(20) context, IBS>65 signal-bar strength, doji exclusion, prior-day
high/low context, 1.7×ATR spike exclusion, fixed 1:1 R:R, no management,
3/day cap, morning-only session) — not a re-test of #7.

## Verdict

**FAIL → STOP. The combined H2+L2 signal set fails ALL THREE pre-declared
gates.** The six-filter funnel collapses 7,594 raw leg-2 triggers to 100
qualifying signals (91 filled) — below the pre-declared ≥150 floor (G1,
not lowered after the fact) — and the surviving signals have decisively
NEGATIVE quality: mean MFE 0.1790% vs mean MAE 0.2894%,
mean(MFE−MAE) = **−0.1103%**, count-dominance fails at all three
thresholds (G2), day-block bootstrap **P(mean ≤ 0) = 0.9986** (G3).
No Step 3, no exit simulation, no EA. Under the fixed 1:1 rule the
imbalance is not marginal: within the 90-min window 57 of 91 signals
reach −1R adverse excursion vs 39 that reach +1R favorable — the setup
loses before the 0.8pt cost is even applied. A follow-up exploratory
fixed-R:R check is NOT warranted: at mean risk 0.19% the MFE/MAE gap is
an order of magnitude too wide for a different ratio to plausibly rescue,
and n=91 makes any ratio grid fishing.

## Filter funnel (the pre-declared attribution view)

| stage | bull H2 | bear L2 | total | cumulative keep |
|---|---|---|---|---|
| F0 raw leg-2 triggers | 4,185 | 3,409 | 7,594 | — |
| F1 after IBS (>65 / <35) | 2,077 | 1,663 | 3,740 | 49.2% |
| F2 after doji exclusion | 1,992 | 1,598 | 3,590 | 47.3% |
| F3 after 1.7×ATR spike exclusion | 1,945 | 1,523 | 3,468 | 45.7% |
| F4 after morning window [09:30,11:30) | 119 | 111 | 230 | 3.0% |
| F5 after prior-day high/low context | 59 | 41 | 100 | 1.3% |
| filled (6-bar timeout) | 54 | 37 | 91 | 9 timeout-abandoned |

The two binding filters are the SESSION RESTRICTION (−93.4% at F4: only
6.6% of filtered triggers land in the 2-hour NY morning) and the
PRIOR-DAY CONTEXT (−56.5% at F5: requiring price beyond the prior
session's extreme during a pullback). IBS, doji and ATR exclusions each
remove comparatively little (combined −54%). Machine plausibility is
sound: 37,061 all-leg triggers over 5.3y, leg decay 24,357/7,594/2,878/
2,232 — non-starved, non-degenerate. Cap impact: max 1 signal/day after
filtering — the 3/day cap never binds.

## Results (% of entry, direction-native 90-min excursions)

| config | n | mean MFE | mean MAE | mean diff | ≥0.10: MFE/MAE | ≥0.15 | ≥0.20 | day-boot p05 | P(mean≤0) |
|---|---|---|---|---|---|---|---|---|---|
| combined (gated) | 91 | 0.1790 | 0.2894 | **−0.1103** | 58/65 | 43/51 | 35/41 | −0.177 | **0.9986** |
| bull H2 (info) | 54 | 0.1688 | 0.2404 | −0.0716 | — | — | — | — | — |
| bear L2 (info) | 37 | 0.1939 | 0.3608 | −0.1669 | — | — | — | — | — |

Yearly mean diffs: NEGATIVE IN ALL SIX YEARS (2021 −0.064, 2022 −0.055,
2023 −0.115, 2024 −0.169, 2025 −0.174, 2026 −0.056) — no regime in which
the filtered subset worked. Both directions negative; the bear side
(morning L2 below the prior day's low) is the worst cell, consistent
with adverse-excursion asymmetry against fresh-short entries.

## Standalone finding vs study #7 (unfiltered H1/H2 full-session null)

#7's unfiltered H2 (EMA60+slope, full 09:30–15:30, no bar-quality
filter): n=2,135, mean(MFE−MAE) −0.0063, P(≤0)=0.99. This variant's
heavily-filtered morning H2+L2: n=91, mean −0.1103, P(≤0)=0.9986 —
**an order of magnitude WORSE, not better**. The filter stack did not
expose a hidden high-quality subset; the signals it selects (strong-IBS
bar breaking the prior day's extreme, in the first two NY hours) are
among the WORST performers of the family on this tape — chasing the
prior session's range boundary in the morning after an EMA20-side
pullback buys adverse excursion, not continuation. The SP2L lesson that
"session choice was decisive" does not rescue this family: the morning
window here did most of the funnel's cutting and the surviving subset
is more negative than the population it came from. Caveat: n=91 is
small (G1 failed on count as well as quality); but the point estimate,
the sign consistency across all years, and both directions agree with
the family-wide pattern rather than contradicting it.

**Continuation family on this instrument is now 0-for-4 across
independent variants: trend-pullback, intraday momentum, unfiltered
H1/H2 (study #7), and this filtered H2-morning variant.** Correlation
vs prior hypotheses is not computed (stopped at Phase 1; no P&L stream
exists — same as #7).

## What this does and does not say

- Tests ONE video's mechanical rule set (as resolved in SPEC.md,
  including the flagged interpretations: IBS-on-signal-bar causality,
  10% doji threshold, range-based 1.7×ATR vs prior-bar ATR,
  joint EMA+prior-day reading (a), 09:30–11:30 window, session-unit
  prior day) on ONE instrument (US500 CFD M5) over 5.3 years. It does
  not refute the source's discretionary practice nor speak to other
  instruments/timeframes.
- The gate floor (≥150 signals) was set before the run and held: the
  funnel's 91 is a legitimate sample-insufficiency failure on its own,
  and the quality gates failed decisively regardless.
- Spec discipline: gates and SPEC were written before any code; the
  funnel order and every interpretation are documented in SPEC.md.

## Artifacts

- `SPEC.md`, `GATE.md` (pre-declared) · `phase1_funnel.csv` (6-stage
  funnel per direction) · `phase1_buckets.csv` (combined/bull/bear,
  thresholds, 1R descriptive) · `phase1_yearly.csv` (all six years
  negative) · `phase1_signals.csv` (91 signals with IBS/body/ATR-ratio
  attributes) · `phase1_stats.json` (gates, tallies, cap impact)
- `evaluate/albrooks_h2_morning_phase1.py`
