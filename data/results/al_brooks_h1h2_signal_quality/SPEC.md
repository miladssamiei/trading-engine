# SPEC — Al Brooks H1/H2 pullback (bar-counting) on US500-tick M5

Status: SPEC + GATE written 2026-08-27 BEFORE any Phase-1 code or run.

## Source and precision

Al Brooks' bar-counting pullback method, from his published price-action
books (secondary sources corroborate the same mechanical core): in an
established uptrend, a pullback begins when a bar fails to make a new
bar-to-bar high; the first later bar whose HIGH exceeds the immediately
prior bar's HIGH is the H1 trigger (the breakout above the prior bar's high
IS the signal); if follow-through fails, the correction continues into a
second leg and the next equivalent trigger is H2; the counter extends
generically (H3, H4, ...). Entry: stop order at the trigger bar's high.
Bear case mirrored (L1/L2, lows). Unlike the YouTube-sourced strategies
(Order Zone, SP2L), the pattern is mechanically unambiguous and is
implemented exactly; the free choices below are flagged as THIS PROJECT's
implementation decisions where Brooks is silent.

## Step 0 — exact definitions

### Bars
M5, verified US500-tick export (`data/results/orb_2021_present/us500tick_m5.csv`,
364,330 bars, 2021-01..2026-04) — same tape as ORB / trend-pullback /
intraday-momentum studies.

### Trend filter (SP2L-convention EMA + explicit slope)
- EMA: `close.ewm(span=60, adjust=False)` — same estimator/alpha convention
  as SP2L's validated EMA60. Period 60 M5 bars = 5 hours of context:
  longer than the pullback structures themselves (~15-60 min), which is the
  property a trend-context filter needs. (SP2L ran EMA60 on M1; the period
  is re-justified for M5 rather than copied blindly.)
- Slope: EMA[i] − EMA[i−12] (12 M5 bars = 1 hour) — added per this study's
  definition of "trend" (SP2L used price-vs-EMA only; the slope condition
  is an explicit strengthening, documented).
- Uptrend at bar i: close[i] > EMA[i] AND EMA[i] > EMA[i−12]. Downtrend
  mirrored. No trend otherwise.

### Leg-counting state machine (bull case; bear mirrored on lows)
State persists across tape gaps (2 h maintenance, weekends); it resets ONLY
on the events below, so overnight context is preserved.

1. TREND state (uptrend, no pullback): `swing_hi` = running max of highs.
2. Pullback begins when h[i] ≤ h[i−1] (first bar-to-bar failure): enter
   PULLBACK, `leg = 0`, `pullback_low` = low of this bar onward,
   `lower_high_seen = True` (this bar is the first lower-high bar).
3. In PULLBACK, evaluated in this order while the uptrend filter holds:
   a. RESET: h[i] > swing_hi (new trend high) → pullback over, counter
      cleared, back to TREND (checked first — a new trend high is not a
      trigger).
   b. TRIGGER: h[i] > h[i−1] AND lower_high_seen → leg += 1, bar i is the
      H{leg} trigger bar; entry stop = h[i]; lower_high_seen = False.
      Consecutive higher-high bars do NOT retrigger (a lower-high bar is
      required between triggers) — this encodes "the correction continues
      into a second leg" exactly: follow-through failure (next bar's high
      below the H1 high) re-arms the counter for H2.
   c. otherwise (h[i] ≤ h[i−1]): lower_high_seen = True.
   d. pullback_low = min(pullback_low, l[i]) always.
4. Trend-filter loss (close vs EMA / slope fails): clear all state; on
   re-engagement the swing high restarts fresh from that bar.

### Entry trigger, timeout, fills
- Entry window (primary arm): trigger bar NY open-minute in
  [09:30, 15:30] (570..930) — project convention. Secondary arm ALL24:
  any trigger bar (information only). The state machine itself runs
  continuously; only ENTRY is window-restricted.
- Buy stop at h[trigger]; timeout N = 6 subsequent bars (30 min) —
  Brooks does not specify a timeout in the sources reviewed; this is a
  flagged PROJECT CHOICE. No fill within N bars (bounded by session end)
  → setup abandoned (tallied, not sampled).
- Fill: first bar f in (trigger, trigger+6] with h[f] ≥ stop →
  entry = max(stop, open[f]) (gap-through fills at the open).
- Sampling eligibility: fill bar must leave ≥ 6 M5 bars to the tape
  session end (gap > 60 min defines session end; window = 18 bars/90 min
  capped at session end) — same forward-window convention as the intraday
  momentum Phase 1.

### Excursions (Phase 1, before any exit design)
Direction-native, uncensored within the window, % of entry (SP2L E-XTM
first-bar convention): fill-bar MFE = max(c[f] − entry, 0), fill-bar MAE =
max(entry − l[f], 0); then max/min of highs/lows over bars f+1..window end.

### Stop-loss context (Brooks convention, recorded per signal)
Brooks' stop is below the pullback low (bull). Recorded per signal as
`risk_brooks_pct` = (entry − pullback_low)/entry × 100 (context now,
binding only in Step 3). No take-profit in Phase 1; TP_R is an open,
flagged PROJECT parameter for Step 3 (SP2L treatment).

### Configurations (tests Brooks' own H2>H1 teaching, does not assume it)
H1-only, H2-only, H1+H2 combined — three separate configurations, both
directions, direction-adjusted pooled + bull/bear split. H3+ counted by the
generic machine and reported informationally only.

## Phase 1 question

Does H1/H2 breakout-entry have forward continuation quality (MFE-over-MAE
dominance at project thresholds), and is H2 better than H1 on this
instrument (Brooks' claim: "markets often need two legs to correct")?
Both reported explicitly regardless of outcome.

## Artifacts

`phase1_buckets.csv`, `phase1_yearly.csv`, `phase1_signals.csv`,
`phase1_stats.json` (incl. H1-vs-H2 comparison and tallies), `REPORT.md`.
