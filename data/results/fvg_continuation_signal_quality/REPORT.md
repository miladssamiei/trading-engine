# Phase 1 REPORT — FVG retracement-continuation on US500-tick (M15 anchor / M5 trigger): PASSED (thin), Step 3 FAILED → study NULL

**Run:** 2026-08-28 · Spec/gates pre-declared in `SPEC.md` / `GATE.md`
BEFORE any code or run · Data: verified M5 export + M5→M15 aggregation
(121,486 buckets, 107 partial = 0.09%, 0 OHLC mismatches vs M1-derived
on all 107,603 complete-in-both buckets — the Bollinger-MR Step-0
path, re-verified). Hypothesis #10, continuation-family attempt #5.

## Verdict

**Phase 1 PASSES all three pre-declared gates — the FIRST
continuation-family signal set to do so — but with the thinnest
possible G3 margin, and Step 3 (fixed-rule occupancy walk-forward with
the source's own 1:1 exit) then FAILS decisively: net expectancy
−0.288 pts/trade, net Sharpe −0.114, only 2/5 full years positive.**
Overall study verdict: **NULL — no EA, no parameter adoption.** The
continuation family is now 0-for-5. Full Step-3 report:
`../fvg_continuation_rr/REPORT.md`.

## Implementation summary

M15 three-candle FVGs (strict: h[N−2] < l[N]); mitigation state on the
M5 tape (touch / full-fill-inversion, inversion-before-trigger
conservative); staleness bounds (30-day untouched, 576-bar armed);
running-extreme trigger (M5 close beyond the max/min since the impulse
bar); swing stop (touch-through-trigger extreme); session window
[09:30,11:00) on the TRIGGER bar; risk band [0.14,0.56]% (10–40 pt
conversion, ORB-anchor discipline); entry next-M5-open; E-XTM 18-bar
excursions; day-block bootstrap gate. The source's discretionary
timeframe-switching was replaced by the fixed M15/M5 pairing (documented
as the largest deviation). News blackout NOT implemented (no calendar
data — untested simplification; 10:00 ET releases fall inside the
window; results may be optimistic — and the study failed anyway).

## Filter funnel

| stage | bull | bear | total |
|---|---|---|---|
| F0 raw M15 FVGs | 14,513 | 12,232 | 26,745 |
| F1 touched (partial mitigation) | 8,436 | 7,592 | 16,028 |
| F2 trigger fired | 2,605 | 1,983 | 4,588 |
| F3 trigger in [09:30,11:00) | 212 | 179 | 391 |
| F4 risk band → final signals | 139 | 125 | 264 |

Tallies: 10,261 inverted-on-contact, 11,409 inverted while armed, 4,197
triggers out of window, 427 untouched-expired, 30 armed-expired,
~127 risk-band rejections. The session window is the dominant cut
(−91.5% at F3): FVG-continuation triggers concentrate outside the NY
morning. Machine non-degenerate (26,745 zones over 5.3y; all states
exercised). Max 2 signals/day → the 2/day cap never binds.

## Phase 1 results (direction-native, % of entry, 90-min windows)

| config | n | mean MFE | mean MAE | mean diff | ≥0.10 | ≥0.15 | ≥0.20 |
|---|---|---|---|---|---|---|---|
| pooled (gated) | 264 | 0.2640 | 0.2574 | **+0.0066** | 197/173 | 166/145 | 133/122 |
| long (info) | 139 | 0.2398 | 0.2424 | −0.0025 | — | — | — |
| short (info) | 125 | 0.2909 | 0.2741 | +0.0168 | — | — | — |

G1 ✓ (264 ≥ 150). G2 ✓ (dominance at all three thresholds + mean > 0).
G3 ✓ BUT at the wire: day-block bootstrap P(mean≤0) = **0.3866** vs the
≤ 0.40 limit (p05 −0.034, p95 +0.048 — CI straddles zero). Skeptical
flags recorded per GATE.md ("unexpected pass → more scrutiny"): the
pooled edge is carried by the SHORT side only (longs are negative);
signal-yearly is 3/6 years negative (2021 −0.025, 2022 −0.027, 2026
−0.046); the G3 margin is 0.013 from failing. Spot-checks: signal
entry/MFE/MAE reproduced exactly by independent recomputation.

## What Step 3 did with this edge (summary)

Fixed rules only (swing SL, 1:1 TP, 2/day cap, occupancy
one-position-at-a-time, 0.8pt cost): 251 trades, gross +128.59 pts
(expt +0.512, gross Sharpe 0.204, 53.4% net win rate) → net of cost
−72.21 pts (expt −0.288, Sharpe −0.114); years 2021 −62.5 / 2022 −24.3
/ 2023 +92.3 / 2024 −60.9 / 2025 +16.5 / 2026 −33.2. S1 FAIL, S2 pass
(251 ≥ 150), S3 FAIL (2/5 years). Gross edge ≈ 64% of the spread cost —
the same edge-smaller-than-costs class as VWAP MR and intraday-momentum
TH2. Per GATE.md no exploratory exit grid exists for this study (exit
fully specified by the source); the pre-declared adjacent-timeframe
robustness arm is moot at a Step-3 stop (its purpose was validating a
pass before an EA build).

## Correlations (Step 5; computed from the WF stream regardless of outcome)

vs ORB +0.114 (80 common days) · VWAP MR +0.055 (245) · trend-pullback
WF +0.155 (191) · SP2L +0.042 (137) · intraday momentum WF +0.164 (191)
· daily momentum P −0.014 (245). All ≈ 0 with the usual
continuation-family kinship (trend-pullback/intraday-momentum highest)
— an FVG is another marker on the same underlying question. H1/H2,
H2-morning and BB MR have no P&L streams (Phase-1 stops).

## Artifacts

`SPEC.md`, `GATE.md` · `phase1_funnel.csv` · `phase1_buckets.csv` ·
`phase1_yearly.csv` · `phase1_signals.csv` (264 signals with
risk/zone/retrace columns) · `phase1_stats.json` ·
`evaluate/fvg_continuation_phase1.py`; Step 3:
`../fvg_continuation_rr/` (REPORT.md, wf_trades.csv, wf_yearly.csv,
wf_folds.csv, wf_summary.json, correlations.csv) ·
`evaluate/fvg_continuation_step3.py`,
`evaluate/fvg_continuation_correlations.py`
