# SPEC — Fair Value Gap retracement-continuation (ICT-style) on US500-tick, signal quality first

Status: SPEC + GATE written 2026-08-28 BEFORE any Phase-1 code or run.
Hypothesis #10, continuation-family attempt #5 (family currently 0-for-4).

## Source and the single biggest deviation

Promotional YouTube setup ("Ryley Trades", paid-course-affiliated,
unsubstantiated win-rate claims): retrace into a Fair Value Gap
(imbalance), enter on break of the most recent local high/low. The
presenter freely switches chart timeframes (30s/1m/2m/3m/15m) by eye.
THIS PROJECT substitutes a FIXED pairing — M15 anchor (gap definition)
/ M5 execution (retracement + trigger) — replacing all discretionary
timeframe scaling. Documented as the largest deviation from the source;
Step 5 (if Phase 1 passes) carries a pre-declared adjacent-pair
robustness check precisely because this pairing was chosen, not derived.

## Step 0 — exact definitions

### Data (reused verified paths)
Execution: verified M5 export
(`data/results/orb_2021_present/us500tick_m5.csv`, 364,330 bars,
2021-01..2026-04). Anchor: M15 aggregated from that same M5 tape
(floor(epoch/900); 121,486 buckets, 107 partial = 0.09%; verified vs
M1-derived aggregation with 0 OHLC mismatches on all complete-in-both
buckets — the Bollinger-MR Step-0 resolution, re-verified at run time).

### FVG definition (M15 anchor, three-candle)
Bullish FVG at anchor bar N: h[N−2] < l[N]. Zone = [h[N−2] (bottom),
l[N] (top)], strictly positive height. Bearish mirror: l[N−2] > h[N],
zone = [h[N] (bottom), l[N−2] (top)]. The FVG is knowable only when N
closes; all monitoring starts with the first M5 bar after N's close
(causal).

### Mitigation state (tracked on the M5 series)
- UNMITIGATED: no M5 bar has entered the zone.
- PARTIALLY MITIGATED ("touched"): an M5 bar entered the zone (bull:
  low ≤ zone top) without fully filling it.
- FULLY FILLED / INVERTED (dead): an M5 bar traded to/through the far
  edge (bull: low ≤ zone bottom; bear: high ≥ zone top). The entire
  zone has been traversed → ineligible from that bar on. Inversion is
  checked BEFORE trigger within a bar (conservative: a bar that both
  fills the gap and closes beyond the trigger level kills the setup).
- A bar that gaps straight through the entire zone on first contact
  (touch and fill in one bar) is inverted-on-contact (dead, tallied).

### Staleness bounds (source silent — explicit mechanical choices)
- UNTOUCHED FVGs expire 30 calendar days after creation (tape time).
- TOUCHED (armed) FVGs die if no trigger within 576 M5 tape bars
  (≈ 2 trading days) after first touch.
Both documented interpretations: a morning-continuation setup with a
zone retraced weeks later, or armed for days without continuation, is
outside any reasonable reading of the setup; the bounds also keep the
active-set bookkeeping finite.

### Retracement, local high/low, trigger (M5, running-extreme form)
Bull (bear mirrored throughout):
- Retracement begins at the first M5 bar entering the zone (touch bar T).
- The structure to break: the running maximum M5 HIGH from the first
  M5 bar inside anchor bar N (the impulse candle) through the bar
  before the trigger bar — this includes any post-gap extension highs
  and any post-touch minor highs, i.e., "the most recent local high
  since the move began" as a causal running extreme.
- TRIGGER: an M5 CLOSE strictly above that running max, while the FVG is
  touched-not-inverted. Entry = market at the NEXT M5 bar's open
  (completed-bar signal convention).

### Stop loss (project "beyond the swing point" convention)
Bull: SL = the retracement's extreme = minimum M5 low from the touch
bar through the trigger bar. Structurally > zone bottom (else inverted
first) and < entry, so risk > 0. No offset beyond the extreme (H1/H2
convention).

### Take profit: fixed 1:1 (hard parameter)
TP distance = SL distance from entry. No exit grid, no management
(Phase 1 is exit-free regardless; Step 3 applies 1:1 exactly).

### Point-range conversion (10–40 points → risk band)
The source's raw 10–40 point range (instrument unspecified; the video
is NASDAQ-flavored but unconfirmed) is converted with the project's
fixed-point discipline (ORB 30 pt = 0.42% at ~7200; SP2L gold-points →
% of price): 10–40 US500-scale index points ≈ 0.14%–0.56% of price.
Mechanical role (interpretation, since the source gives no operational
rule): a RISK VALIDITY BAND — a signal whose (entry − SL)/entry falls
outside [0.14%, 0.56%] is excluded (too-tight = noise, too-wide =
untradeable). The 0.42% ORB anchor sits mid-band, supporting the scale.

### Session: NY 09:30–11:00 ET — TRIGGER-bar window
Signal (trigger) bars: M5 open-minute in [570, 660) — bars 09:30..10:55;
the 11:00 bar is excluded (window is [09:30, 11:00) exactly as
specified; NOT H2-morning's 09:30–11:30). Entry may fill at/after 11:00
(the order follows the signal). FVG creation, mitigation and staleness
run on the continuous 23h tape (overnight imbalances retraced at the NY
open are in-scope for this setup style; documented interpretation).

### Trade cap: 2 per day
Applied in Step 3 (execution constraint): first 2 by trigger-bar time;
same-timestamp bull/bear collision resolved bull-first (arbitrary,
documented). Phase 1 samples all qualifying signals and reports the
signals/day distribution so the cap's impact is visible.

### News blackout — UNTESTED SIMPLIFICATION (explicit)
The source avoids entries within 5 minutes of scheduled news. This
project has NO news-calendar data source; the filter is NOT
implemented. Results may be OPTIMISTIC to the extent real news-driven
volatility spikes (10:00 ET releases fall inside this window) would
have been excluded by the source's rule. Stated, not silently dropped.

## Phase 1 (signal quality — established methodology)

Entry = next M5 open after the trigger close. Direction-native,
uncensored excursions over the H1/H2 window: 18 M5 bars / 90 min,
capped at tape-session end (gap > 60 min), ≥ 6 forward bars required
(else ineligible, tallied). SP2L E-XTM first-bar convention
(fill-bar close/extremes then extremes). Thresholds 0.10/0.15/0.20% of
entry. Non-independence: FVG signals during the same trending stretch
(and multiple zones triggering on one bar) are related observations —
the DAY-BLOCK BOOTSTRAP over per-day mean MFE−MAE is the gate
statistic from the start.

### Filter funnel (per direction + total, cumulative)
F0 raw M15 FVGs (height > 0) → F1 touched (partial mitigation reached)
→ F2 trigger fired (close beyond running extreme, not inverted,
staleness respected) → F3 trigger in session window [570,660) →
F4 risk band [0.14,0.56]% at fill → filled & eligible signals.
Plus tallies: inverted-on-contact, inverted while armed, armed-expired,
untouched-expired, out-of-band risk, no-next-bar.

## Phase 1 question

Does the FVG-retrace-then-break entry have forward continuation quality
(direction-native MFE dominance over MAE at project thresholds,
surviving day-block bootstrapping)?

## Step 3 (only if Phase 1 passes) — pre-declared
Fixed rules only (1:1 TP, swing SL, 2/day cap, session window) — no
parameter search. Occupancy-modeled re-simulation (one position at a
time; simultaneous multi-zone triggers resolved by earliest trigger
time), half-year WF folds (2021 burn-in; 2022H1..2026H1) as a stability
check; costs 0.8 pt round trip, $1/pt/lot, $10k base; positions carry
to SL/TP (no time exit; overnight holds tallied). Gates S1 net expt>0
AND net Sharpe>0; S2 ≥150 trades; S3 ≥4/5 years positive AND no year
>60% of net P&L; S4 net-of-spread (S1 restated). Fail → null, no EA.

## Step 5 (report)
Comparison vs the four prior continuation nulls; if (surprisingly)
PASSING: extra skepticism — re-check the M15/M5 pairing is not a lucky
timeframe accident via a clearly-labeled robustness arm on adjacent
pairs (M30 anchor / M5 trigger; M15 anchor / M15 trigger as boundary),
reported as robustness context, not a new search. Correlation vs the
nine prior hypotheses: computable only if a P&L stream exists (Step 3
reached); at a Phase-1 stop there is no stream (stated honestly, same
as every prior null).

## Artifacts

`data/results/fvg_continuation_signal_quality/`: SPEC.md, GATE.md,
phase1_funnel.csv, phase1_buckets.csv, phase1_yearly.csv,
phase1_signals.csv, phase1_stats.json, REPORT.md. Step 3+ artifacts
only if gates pass.
