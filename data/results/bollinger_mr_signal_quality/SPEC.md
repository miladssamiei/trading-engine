# SPEC — Bollinger Band mean reversion (Strategy 6) on US500-tick, signal quality first

Status: SPEC + GATE written 2026-08-28 BEFORE any Phase-1 code or run.
Hypothesis #9. Mechanically distinct from the failed VWAP MR (#2): the
signal is band-width extremity vs a 20-period MA (not VWAP deviation),
on M15 (not M1). Same skepticism: signal quality is measured BEFORE any
exit design.

## Step 0 — exact definitions (every judgment call flagged)

### Bars / data path (DECISION: aggregate verified M5 → M15)
Primary: M15 aggregated from the verified M5 export
(`data/results/orb_2021_present/us500tick_m5.csv`, 364,330 bars,
byte-exact vs the terminal in prior studies). Buckets keyed by
floor(epoch/900). VERIFICATION (ran 2026-08-28 BEFORE any signal was
produced; the script's abort gate fired on the first attempt):
(1) M5→M15: 121,486 buckets, 107 partial (≠3 M5 bars = 0.09% — kept,
locations tallied; the terminal M15 chart aggregates the same way);
(2) an independent M1→M15 aggregation from the verified M1 export agrees
on OHLC for ALL common buckets where both inputs are complete
(0 mismatches), and the M1 path has 13,883 partial buckets — the M1
export skips no-tick minutes, so M5 is the denser, cleaner verified
grid and is therefore the primary. Path-walking (Phase-1 excursions,
Step-3 exits) uses the same M5 tape for intrabar fidelity and the 15:55
forced exit.

### Bollinger Bands (chart convention, documented)
20-period SMA of M15 closes; band offset 2 population standard
deviations (ddof=0, the standard BB formula); computed over the
CONTINUOUS 23h tape (no per-session reset — matches universal chart
behavior; overnight bars feed the bands). mid[t] = SMA20, up/lo[t] =
mid ± 2σ. Band width BW[t] = up[t] − lo[t]. Signals require ≥20 prior
M15 bars (warmup) and BW > 0 (degenerate σ=0 excluded, tallied).

### Range-day identification (REUSED VERBATIM from VWAP MR)
Session = NY wall [09:30, 16:00) = [570, 960). session_open = first
session bar's open; a day whose first session bar is not 09:30 is
skipped entirely (no_session_open, VWAP MR convention). ROLLING at each
signal bar (VWAP MR's documented interpretation, not a mid-session
snapshot): net_pct = |close/session_open − 1|×100 ≤ 0.6 AND hl_pct =
(intraday high − intraday low so far)/session_open×100 ≤ 1.0, intraday
extremes tracked over session bars 09:30→current (M15 aggregation of
the same tape ⇒ identical extremes to VWAP MR's M1 tracking).

### Max-move filter (REUSED conversion: 30 ES points → 0.42%)
|close − session_open| ≤ session_open × 0.42% — the documented
ORB/VWAP-MR conversion (30 ES points = 30 index points ≈ 0.42% of
US500 at ~7200; VWAP MR MAXMOVE_PCT). NOTE: 0.42% is strictly tighter
than the 0.6% net-move condition on the same quantity — kept verbatim
because the source lists both; the binding constraint is documented in
the funnel.

### Signals and entries (VWAP MR timing conventions, M15 granularity)
- Long signal: M15 close ≤ lower band. Short: close ≥ upper band
  (equality included).
- Signal on COMPLETED bar; entry market at the NEXT bar's open
  (VWAP MR convention), next-bar open ≤ 15:30 (LAST_ENTRY_M=930) ⇒
  signal bars open 09:30..15:15.
- All filters (range-day rolling, max-move, warmup, BW>0) evaluated at
  the signal bar.
- Phase 1 samples EVERY qualifying signal bar (the brief's instruction);
  same-band consecutive touches are related observations — handled by
  the day-block bootstrap gate. Episode-level view (first bar of each
  maximal consecutive band-close run) reported informationally.

### Phase 1 excursions (exit-free, uncensored to forced exit)
Entry = first M5 bar at/after the M15 signal close, at its open.
Direction-native MFE/MAE % of entry, SP2L E-XTM first-bar convention:
fill-bar MFE = max(close−entry, 0) (long), MAE = max(entry−low, 0);
then max/min over M5 bars through the window end = last M5 bar of that
date opening before 15:55 (forced-exit wall 955, VWAP MR). Days whose
tape ends early: window truncated at the last available bar (tallied).
Favorable = toward the middle band (up for long), adverse = further
beyond the band — exactly the direction-native definition.

### Published exit (Step 3 ONLY — not used in Phase 1)
SL = 1.5 × BW (signal-bar upper − lower) beyond entry (long: entry −
1.5·BW; short mirrored). TP = touch of the middle band — a MOVING
target monitored bar-by-bar at M5 resolution against the mid-band value
through the PREVIOUS completed M15 bar (VWAP MR's prev-completed-bar
target convention). Same-bar SL+TP ambiguity → SL-FIRST (conservative,
the stricter project convention — SP2L SPEC §8; VWAP MR's close-vs-open
heuristic is looser and deliberately not reused). Open-gap through a
level → exit at the open. Forced exit 15:55 NY at the first M5 bar's
open (VWAP MR convention); no overnight holds. Occupancy: one position
at a time; signals while in a position are ignored. Prices normalized
to the 0.01 trade grid.

### Step 3 protocol (pre-declared)
Primary test = the published exit EXACTLY as specified, occupancy
re-simulation, no parameter freedom: walk-forward folds are a stability
check (2021 burn-in; folds 2022H1..2026H1). Costs 0.8 pt round trip,
$1/pt/lot, $10k base, no compounding. ONLY IF the published-exit WF
fails its gates: exploratory in-sample grid (clearly labeled) over the
stop multiplier {1.0, 1.25, 1.75, 2.0}×BW around the published 1.5
(published cell already measured); any rescue attempt = walk-forward
with per-fold trailing-expectancy selection across the 5 multiplier
cells (SP2L convention), same S-gates. No other parameter may vary.

### Step 5 (only if validated)
Monthly matrix, B&H comparison, daily-P&L correlation vs all eight
prior hypotheses via the established runner — with specific attention
to VWAP MR (same mean-reversion family; high ρ would mean the same
"does US500 mean-revert intraday" question, not a new mechanism).

## Phase 1 question

Do band-extremity closes under range-day conditions have genuine
reversion quality (direction-native MFE dominance over MAE at project
thresholds, surviving day-block bootstrapping)? This is the test VWAP
MR's entries never passed.

## Artifacts

`data/results/bollinger_mr_signal_quality/`: SPEC.md, GATE.md,
phase1_funnel.csv, phase1_buckets.csv, phase1_yearly.csv,
phase1_signals.csv, phase1_stats.json, REPORT.md.
Step 3+ only if gates pass: `data/results/bollinger_mr_rr/`,
`ea/BollingerMR.mq5`, references/reconciliation,
`data/results/bollinger_mr_2021_2026/`.
