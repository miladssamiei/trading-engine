# SPEC — Al Brooks H2, IBS>65 signal-bar filter, NY-morning session (US500-tick M5)

Status: SPEC + GATE written 2026-08-27 BEFORE any Phase-1 code or run.
This is hypothesis #8 — an INDEPENDENT test, not a re-test of #7
(`../al_brooks_h1h2_signal_quality/`, EMA60+slope, full RTH window, no
bar-quality filter → decisive null). Material differences here: EMA(20)
context, IBS>65 signal-bar strength, doji exclusion, prior-day high/low
context, 1.7×ATR spike exclusion, fixed 1:1 R:R, no management, 3/day cap,
morning-only session.

## Source and scope

Second Al-Brooks-style video specification. The mechanical core
(bar-count pullback, stop-entry at trigger-bar extreme, pullback swing
stop) is shared with study #7 and reused EXACTLY; everything new is a
filter/context/exit specification. Every item the source leaves ambiguous
is resolved below and flagged as an interpretation.

## Step 0 — exact definitions

### Bars
M5, verified US500-tick export
(`data/results/orb_2021_present/us500tick_m5.csv`, 364,330 bars,
2021-01..2026-04) — same tape as all prior studies. Epoch timestamps,
NY-converted for all session logic (project convention).

### Signal: H2 bull + L2 bear mirror
The source is not confirmed long-only, so the bear case is implemented as
the explicit mirror and BOTH directions are tested; the strategy unit is
the COMBINED set (H2+L2). Per-direction results are reported
informationally.

### Leg-counting state machine (reused verbatim from study #7)
TREND state: running `swing_hi` (bull). Pullback begins at the first
bar-to-bar high failure (`h[i] <= h[i-1]`): leg=0, `pullback_low` =
running min of lows from that bar, `lower_high_seen=True`. In pullback,
in order: (a) new trend high (`h[i] > swing_hi`) → reset to TREND;
(b) TRIGGER `h[i] > h[i-1]` AND `lower_high_seen` → leg+=1, bar i is the
H{leg} trigger; `lower_high_seen=False`; consecutive higher-highs do not
retrigger; (c) otherwise `lower_high_seen=True`; (d) `pullback_low`
updated always. Bear mirrored on lows (L{leg}). State resets on trend
-filter loss. THIS STUDY: only leg==2 triggers (H2/L2) are eligible.

### Trend context — EMA(20) (replaces #7's EMA60+12-bar-slope)
Close-based EMA, span 20 M5 bars, `ewm(span=20, adjust=False)` over the
continuous tape (project estimator convention; no per-day reset — matches
chart-platform EMA behavior and every prior study). Convention: bull
state requires `close[i] > EMA20[i]`; bear requires `close[i] < EMA20[i]`.
No slope term: the source lists EMA(20) context only; #7's slope was that
study's own strengthening and is deliberately not carried over.

### IBS>65 signal-bar filter — interpretation resolved
IBS = (close − low) / (high − low) × 100, computed on the H2/L2 SIGNAL
bar — the bar that triggers the setup and hosts the stop order (its high
for H2, low for L2). The user brief says "triggering (entry) bar"; the
fill bar reading is causally impossible under "no management" (a fill-bar
IBS check could only be enforced by exiting at the fill bar's close,
which is management). The signal bar is the only causal reading
consistent with the hard "exits only via SL/TP" rule, and matches Brooks'
signal-bar-strength concept (a bull signal bar closes near its high).
Bull: IBS(signal) > 65. Bear mirror (symmetric, explicit): IBS(signal)
< 35. Zero-range bars: IBS undefined → excluded (subsumed by doji rule).

### Doji exclusion — threshold chosen
Doji = body ≤ 10% of the bar's total range: `|c−o| ≤ 0.10 × (h−l)`,
including the degenerate `h == l` bar. Applied to the SIGNAL bar (same
causality reasoning as IBS: the fill bar cannot be checked at order
placement without introducing management). 10% is THIS PROJECT's choice;
the source gives no number.

### Spike exclusion — 1.7×ATR, range-based
ATR(14) on M5, Wilder RMA (alpha = 1/14) of true range over the
continuous tape — the standard ATR definition. The exclusion checks the
SIGNAL bar's RANGE (`h−l`, not body: a "spike" is an outsized total bar
movement; range is the conventional spike metric), against
ATR14 evaluated at the PRIOR bar (`atr[i−1]`) so the bar does not inflate
the yardstick it is measured against. Exclude if `h[i]−l[i] > 1.7 ×
atr[i−1]`.

### Prior-day high/low context — ambiguity resolved
The source's "EMA / Prior Day Low/High" is read as interpretation (a):
a JOINT trend-continuation context — the EMA(20) side condition above AND
price beyond the prior day's extreme. Because the EMA side already gates
the state machine (it is the engagement filter), the incremental filter
checked at the trigger bar is: bull requires `close[trigger] > prior-day
HIGH`; bear requires `close[trigger] < prior-day LOW`. Reasoning: (a) is
the reading that fits "trend continuation" framing (trading above
yesterday's high establishes upside context); interpretation (b) ("two
independent filters") is either identical to (a) when both are required
or incoherent as an either-or (an EMA-only-or-PD-only context would not
be an "additional filter"). Flagged as a necessary interpretation.
"Prior day" = the previous CFD trading session on this tape — 18:00 ET
through 16:14 ET next calendar day (gap > 60 min segments sessions;
project data-day definition, same unit as the verified D1 tape). Its
high/low over all bars of that session. At a morning trigger the prior
session is always complete (causal).

### Entry, timeout, fills (project convention reuse)
- Stop order at the signal bar's high (H2) / low (L2), no offset —
  exactly study #7's convention.
- Timeout: 6 bars (30 min) — #7's documented project choice, reused for
  consistency (source silent). No fill within 6 bars (bounded by session
  end) → abandoned, tallied.
- Fill: first bar f in (signal, signal+6] touching the stop level;
  gap-through fills at the open (`entry = max(stop, o[f])` bull).
- The fill may occur after 11:30 (order placed in-window stays working
  for its 6-bar life); SL/TP then resolve without any time-based exit.

### Session — NY morning only (necessary interpretation)
Signal-bar open-minute in [09:30, 11:30) NY (bars 09:30..11:25, 24 M5
bars). The source image gives no exact times; 09:30–11:30 is the
conventional "AM session" boundary in comparable price-action material.
Explicitly different from #7's full-RTH [09:30, 15:30] window — the
session restriction is a headline feature of this variant (per SP2L's
finding that session choice was decisive).

### Stop loss — pullback swing point (project convention reuse)
SL beyond the swing point = the machine's `pullback_low` at trigger
time: the running minimum low since the pullback began (the low of the
pullback leg preceding the H2 entry; for a two-leg pullback this is the
deeper of the two lows — "the most recent swing low since the trend's
last high"). Exactly #7's recorded `pullback_low` convention, now
binding. Bear: `pullback_high`. No offset beyond the extreme.
Structurally `pb_ext ≤ l[signal] < h[signal] = stop`, so risk > 0 for
every non-doji signal (zero-risk degeneracy impossible).

### Take profit — fixed 1:1, hard parameter
TP distance = SL distance from entry price (bull: TP = entry +
(entry − SL); bear mirrored). Hard-specified: NO exit grid, no
optimization of TP/SL in this study. Management: NONE — exits are SL or
TP only; no breakeven, trailing, partials, or time exit. Open trades
carry across tape sessions until SL/TP (documented; overnight holds
tallied in Step 3).

### Trade cap
Maximum 3 trades per day, TOTAL across both directions (not per
direction). Tie-break: first 3 by signal-bar time; a same-timestamp
bull/bear collision is resolved bull-first (arbitrary, documented).
Applied in Step 3 (the cap is an execution constraint). Phase 1 samples
all qualifying signals but reports the signals/day distribution so the
cap's impact is visible before Step 3.

## Phase 1 (signal quality — methodology reused from #7)

Same causal excursion methodology: direction-native, uncensored within
an 18-bar/90-min forward window capped at session end, E-XTM first-bar
convention (fill-bar close/extremes), ≥6 forward bars to session end
required (else ineligible, tallied). Thresholds 0.10/0.15/0.20% of
entry. Non-independence discipline: H2/L2 signals within the same day
are related observations — the gate statistic is the day-block bootstrap
over per-day mean MFE−MAE.

### Filter funnel (reported stage by stage, per direction + total)
F0 raw leg-2 triggers (machine output; EMA20-side-gated by construction)
→ F1 after IBS (>65 bull / <35 bear, signal bar) → F2 after doji
exclusion (signal bar) → F3 after ATR spike exclusion (signal bar) →
F4 after morning-session restriction (signal-bar open-minute in window)
→ F5 after prior-day context (close beyond prior-session extreme) →
fills within 6-bar timeout → eligible signals (forward-window rule).
Stage order between F1–F3 is immaterial to the final set (same-bar
properties); counts are cumulative in the stated order.

### Informational (cannot gate)
Per-direction buckets; yearly mean diffs; per-signal risk_pct
((entry−SL)/entry) and the 1R descriptive counts
count(MFE ≥ risk) vs count(MAE ≥ risk); first-touch win rate at 0.15%;
signals/day distribution (cap impact); abandoned tallies. A follow-up
exploratory R:R check (other fixed ratios) may be NOTED in the report as
warranted/not warranted — but is not run and cannot influence gates.

## Step 3 (only if Phase 1 passes) — fixed-rule walk-forward

No parameter selection exists (TP/SL/session/cap all fixed), so
walk-forward is a STABILITY check, not a search: the identical fixed
rule set is simulated on the standard project folds (burn-in 2021;
folds 2022H1..2026H1, 9 half-years) and per-fold + pooled stats
reported. Occupancy-modeled full re-simulation on M5 OHLC with the
project's prior conventions: bar-open gap fills; same-bar SL/TP
ambiguity resolved SL-FIRST (conservative); prices on the 0.01 trade
grid; 3/day cap with first-3-by-time priority; costs 0.8 pt round trip
($1/pt/lot); ex-financing headline with overnight holds tallied.

## Step 4 (only if Step 3 passes)

Native EA `ea/AlBrooksH2Morning.mq5` + Python reference + reconciliation
under standard project architecture and validation discipline.

## Artifacts

`SPEC.md`, `GATE.md`, `phase1_funnel.csv`, `phase1_buckets.csv`,
`phase1_yearly.csv`, `phase1_signals.csv`, `phase1_stats.json`,
`REPORT.md`; Step-3 artifacts only if gates pass.
