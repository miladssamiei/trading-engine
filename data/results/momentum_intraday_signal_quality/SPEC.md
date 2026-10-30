# SPEC — Intraday multi-horizon trend alignment (project-original hypothesis)

Status: SPEC + GATE written 2026-08-27 BEFORE any Phase-1 run.

## Provenance label (required in ALL outputs)

This is a NEW, project-original hypothesis: intraday multi-horizon trend
alignment, flat by end of session. It is *inspired by* the general structure
of the daily multi-horizon momentum test (check trend agreement across
several lookback horizons) but is NOT a test of Lempérière et al. (2014)
("Two Centuries of Trend Following") and NOT a test of the video rubric. No
result of this study may be described as replicating or refuting either
source. Rationale for the intraday redesign: the daily test
(data/results/momentum_multihorizon/) was swap-dominated and limited to 628
weekly decisions; the intraday design removes financing (flat by session
end, like every validated intraday strategy here) and raises decision count
to O(10^5) bars.

## Step 0 — rule definition

### Bars: M5
`data/results/orb_2021_present/us500tick_m5.csv` (verified clean: 364,330
bars, 2021-01..2026-04, the ORB/trend-pullback export). Justification: every
lookback in the set is an exact multiple of 5 minutes, so M5 loses nothing vs
M1 for signal granularity; M5 is 5x cheaper and already verified. M1 would
only matter for sub-5-minute horizons, which are not in the set.

### Lookbacks (calendar-time trailing windows)
30 min, 1 h, 2 h, 4 h, 12 h, 1 day (= 1800, 3600, 7200, 14400, 43200,
86400 s). Anchor bar = LAST bar with time <= t - h ("trailing calendar
window"). 12 h / 24 h therefore SPAN the overnight close (and, on Mondays,
the weekend: the 24 h anchor of Monday 10:00 is Friday's close). This is the
explicit decision: the longer horizons measure trend INCLUDING the
overnight/weekend gap — they serve as slow context filters, the same role
the EMA60 daily filter plays in SP2L. Positions still open and exit within
one session, so no financing. A "swing" variant with overnight carry is NOT
tested (it would reintroduce swap, and the documented swap convention makes
any test of it a financing study, not a signal study).
Fallback set (pre-declared ablation, not a tuning knob): score4 = fast
horizons only (30 m, 1 h, 2 h, 4 h) — used to attribute failure if the 12 h
/ 24 h horizons turn out to be the problem.

### Scoring
At each eligible M5 bar close t: dir_h = +1 if close[t] > close[anchor_h]
else -1 (ties impossible on 3-decimal prices; if equal -> 0 excluded — none
expected). score6 = sum over 6 horizons, range -6..+6 in steps of 2.
score4 = sum over the 4 fast horizons, range -4..+4 in steps of 2.

### Session: NY window (primary), ALL24 (secondary arm)
Primary signal/entry window 09:30..15:55 NY (wall minutes 570..955,
matching every validated intraday strategy here; best liquidity; same
session as ORB/SP2L so Step-5 correlation is like-for-like). Secondary
Phase-1 arm ALL24 (all bars) answers "did the window choice hide the edge"
— analysis only, not a tuning grid.

### Exit discipline (design, applied in later steps)
Forced flat 15:55 NY, uniform, regardless of score. Phase 1 forward windows
are capped at the tape session end (gap > 60 min) so no window can cross a
close.

### Sizing (spec only; implemented if and only if Step 3 is reached)
Same vol-normalized family as daily Variant R with an intraday estimator:
rolling mean of |M5 close-to-close return| over the trailing one data-day
(~276 bars), annualized x sqrt(bars_per_year = 276 x 252); target risk %
declared before any Step-3 run. Phase 1 needs no sizing.

### Rebalance cadence
Phase 1: score sampled at EVERY eligible M5 bar (max decision count).
Trading cadence (if reached): re-score every M5 bar; resize throttling
(on |Δscore| >= 2 vs every 30 min) is a Step-3 design decision declared
before that step runs — not silently defaulted.

## Phase 1 — the single most important check

Question: does multi-horizon intraday agreement predict short-term
CONTINUATION? At each eligible bar: direction d = sign(score); entry =
bar close; forward window = next 18 M5 bars (90 min), truncated at tape
session end; minimum 6 forward bars (30 min) else the bar is ineligible.

Direction-adjusted excursions (% of entry, uncensored within the window —
SP2L/trend-pullback conventions):
  MFE = max forward excursion IN the score direction (highs for d=+1,
        lows for d=-1)
  MAE = max forward excursion AGAINST the score direction
Buckets by |score|: {0, 2, 4, 6} for score6; {0, 2, 4} for score4.
|score|=0 has no direction: measured in the up-frame (arbitrary, symmetric
under the null).

Required relationship: higher |score| (more horizons agreeing) -> stronger,
more one-sided forward movement in the score direction. If score magnitude
has NO ordered relation to forward-move quality, STOP (no exit design, no
sizing, no EA) — same discipline as the sticky-HMM and trend-pullback stops.

## Artifacts

- `phase1_buckets.csv` — per (arm, score_set, |score| bucket) metrics
- `phase1_yearly.csv` — top-bucket dominance per calendar year
- `phase1_signals_ny.csv` — per-bar NY-arm records (for Step 3 if passed)
- `REPORT.md` — always produced (null result is a complete deliverable)
