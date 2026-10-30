# SPEC.md — SP2L spike/gap/continuation on US500-tick: Step 0 reconstruction

Written 2026-08-27, BEFORE any Phase 1 signal-quality number was computed
(the only execution before this file: a wall-minute coverage probe of the
M1 export — pure data plumbing, no strategy logic).

## 0. Source and caveats

Source: `SP2L_Advanced_Bot.py` (third party, Alireza Sadabadi, XAUUSD M1 live
bot, v2.0). The `Meta` module (GetRates/run/resume) is an undocumented
execution layer — NOT reproduced; data access uses this project's verified
CSV export, and (only if all gates pass) execution uses a native MQL5 EA.
The bot polls every 10 s and acts on the candle state at poll time; this
port reconstructs the same decisions on CLOSED bars, which is deterministic
and EA-reproducible. Every interpretation below resolves an ambiguity in
the source and is implemented exactly as written here.

## 1. Instrument substitution and unit conversion ★

XAUUSD → US500-tick. Gold-calibrated absolute point thresholds are NOT
portable (ORB's 30-pt→0.42% discipline). Conversions:

| Source param | Gold value | Conversion | US500 value |
|---|---|---|---|
| SPIKE_CANDLE_SIZE | 1.5 | dimensionless body ratio | **1.5** (as-is) |
| PGAP_POINTS | 100 pts (point=0.01 → $1.00) | $1.00 / ~$2,500 gold ≈ 0.04% | **PGAP_PCT = 0.04%** of the gapped-over level |
| MAX_SL_DISTANCE_POINTS | 1000 pts ($10.00) | $10 / ~$2,500 ≈ 0.40% | **MAX_SL_PCT = 0.40%** of entry price |
| EMA_PERIOD | 60 bars | bars (M1) | **60** (as-is) |
| ADX_PERIOD / MIN_ADX | 14 / 20.0 | dimensionless | **14 / 20.0** (as-is) |
| MAX_OPPOSITE_MOVES | 1 | bars | **1** (as-is) |
| TP_R | 1.0 | R-multiple | **1.0** default; Step-3 grid 0.5/0.75/1.0/1.5/2.0 |
| SECOND_ENTRY (vol ×2) | off | — | ablation toggle; Phase 1 = extra signal at entry∓risk/2 |

PGAP_PCT sensitivity {0.02, 0.08}% is reported INFORMATIONALLY on the
no-filter 24h baseline only; gates are evaluated only at PGAP_PCT=0.04%.

## 2. Timeframe

**M1** — the source's native timeframe, and the M1 export is already
verified clean full-range (1,791,504 bars, 64/64 months, 0 multi-day gaps,
trading-date set identical to the verified M5 export; VWAP MR Step 0 probe).
The pattern is defined by adjacent-bar body ratios and a price gap spanning
the spike candle; M5 aggregation would silently redefine both. Bars are the
terminal-served M1 bars (`data/results/vwap_2021_2026/us500tick_m1.csv`,
2021-01-03 23:00 UTC → 2026-04-30 23:59 UTC). Setup/pending machinery runs
on the full ~24 h stream (the instrument has a daily data halt 16:15–17:59
ET; halts/weekends act as natural gaps, no special handling — same as gold's
halt in the source).

## 3. Setup detection (on closed bars, at close of bar t)

Indices relative to t at its close: t-3 = pre-spike, t-2 = spike, t-1 =
post-spike, t = trigger. body(x) = close(x) − open(x) (SIGNED, as source).

BUY setup at t (all 10, exactly the source's buy0..buy7 + p_gap + spike):
1. `low(t) < low(t−1)` — trigger candle breaks the post-spike low
2. `close(t−1) > close(t−2)`; 3. `open(t−1) > open(t−2)`
4. `close(t−2) > close(t−3)`; 5. `open(t−2) > open(t−3)`
6. `close(t−1) > open(t−1)`; 7. `close(t−2) > open(t−2)`; 8. `close(t−3) > open(t−3)`
9. gap: `low(t−1) > high(t−3) · (1 + 0.0004)`
10. spike: `body(t−2) > 1.5·body(t−1)` AND `> 1.5·body(t−3)` AND `> 1.5·body(t)`

SELL is the exact mirror (highs↔lows, signs inverted, gap
`high(t−1) < low(t−3) · (1 − 0.0004)`).

Faithful quirks kept deliberately: (a) the spike ratio compares SIGNED
bodies, so clause 10 is trivially true for the trigger candle t whenever t
closes against the setup direction (usual case — t is a pullback bar); the
ratio only binds on same-direction candles, exactly as the source computes;
(b) the gap spans the spike candle (t−1's low vs t−3's high), so no literal
tick gap is required — the spike may overlap either neighbor; (c) if BUY and
SELL setups fire on the same bar t, NEITHER is taken (source: `buy and not
sell`).

## 4. Pending → entry (the pending-order mechanism)

A setup creates ONE pending: direction, setup bar index t, SL = low(t−3)
(BUY) / high(t−3) (SELL) — the extreme of the candle before the spike.
The pending is genuinely multi-bar (u > t strictly; the trigger candle's own
break in §3.1 is part of the SETUP, not the entry):

- Single pending at a time; no new setup detection while a pending (or, in
  live, a position) exists. Phase 1 models pending-occupancy exactly and
  declares position-occupancy as the usual Phase-1 independent-events
  approximation (Step 3 re-simulation resolves it). Shadow semantics: a
  pending's lifetime ends at its outcome bar (entry bar, INVALID bar, or NY
  date end); setups on bars ≤ the outcome bar are shadowed and the machine
  resumes strictly after it. (The source's same-call re-detection after an
  INVALID on the same bar is deliberately not modeled — slightly
  conservative, uniform across configurations.)
- On each later bar u (same NY calendar date; a pending EXPIRES at its NY
  date's end — deviation from the source's indefinite overnight lifetime,
  required by the day-bounded MFE/MAE censoring; documented):
  1. Break: BUY `low(u) < low(u−1)`; SELL `high(u) > high(u−1)`. No break →
     pending persists.
  2. `risk = low(u) − SL` (BUY). risk ≤ 0 → pending persists (pullback has
     not reached deep enough yet).
  3. `risk > MAX_SL_PCT · entry` → **pending KILLED** (source "INVALID").
  4. Trend-structure filter (if ON): no bar w in (t, u] with a run of
     ≥ 2 consecutive non-higher-highs (BUY) / non-lower-lows (SELL) ending
     at w (source: MAX_OPPOSITE_MOVES=1, counted bar-by-bar t+1..u; fails →
     pending persists).
  5. EMA filter (if ON): close(u) strictly above EMA60(u) (BUY) / below
     (SELL); else persists.
  6. ADX filter (if ON): ADX14(u) ≥ 20; else persists.
  7. Session variant: entry bar's wall-minute inside the variant window and
     ≤ variant last-entry cutoff (below); else persists.
  8. All pass → ENTRY at bar u. Pending clears.

Entry price = the breaking candle's extreme (BUY: low(u); SELL: high(u)) —
the price the source records (`entry = current_low/high`), i.e. the level
the live 10 s-poll sees when it detects the break. This is mildly
optimistic on closed bars (fill at the bar's extreme). Honesty mechanism:
Phase 1 computes MFE/MAE under BOTH conventions — **E-XTM** (entry = bar
extreme, source-faithful; same-bar favorable credit = close(u)−entry only,
same-bar adverse = 0 since the extreme is the bar's limit) and **E-CLS**
(entry = close(u), pessimistic; walk starts at bar u+1, no same-bar credit).
R = risk = entry − SL; TP = entry + TP_R·R (Step 3 only).

## 5. Session variants (DST-aware via zoneinfo, the established helper)

| Variant | Entry window (local wall) | Last entry | Forced exit / walk end |
|---|---|---|---|
| LON | 08:00–16:00 Europe/London | 16:00 (960) | walk last bar ≤16:24 local; final point = 16:25 bar open |
| NY | 09:30–15:30 America/New_York | 15:30 (930) | walk last bar ≤15:54; final point = 15:55 bar open (established ORB/VWAP convention) |
| ALL24 | none (any bar of the NY date) | 23:00 NY (1380) | walk to the NY date's last bar (full range; no extra point) |

Session membership is tested at the ENTRY bar only (source applies its
session filter in `entry_filters_are_valid`). Setups may be detected
outside the session; only entries are gated. LON last-entry 16:00 and
forced-exit 16:25 mirror the NY convention (−30 min / −5 min from the
16:30 close). ALL24 mirrors it against NY midnight. Day key = NY calendar
date throughout (the London window 08:00–16:30 local always falls inside a
single NY date).

## 6. Second entry (ablation toggle)

When ON, each first entry (bar u, price E, risk R, same SL) additionally
spawns a limit at S = E − R/2 (BUY) / E + R/2 (SELL), volume ×2.0 in live.
Phase 1 approximation (documented): the second signal exists iff some later
bar of the same walk window touches S; its entry = S, risk = R/2, and its
MFE/MAE walk starts at the first touch bar (same walk-end rules; under
E-CLS the same-bar credit rule applies to the touch bar as the entry bar).
Second-entry adds no new directional information — it re-weights entries
deeper by R/2; reported so its mechanical effect is visible.

## 7. Phase 1 ablation matrix (Step 1–2)

4 binary toggles (EMA, TREND, ADX, SECOND) × 3 session variants = 48
configurations, all computed from ONE master pending/attempt stream (the
pending machine is re-resolved per configuration because filter choices
change when a pending dies — see §4). Staging: the light-touch single-filter
ablation (no-filter baseline + 4 single-filter-ON, ALL24) is reported as
Stage A; the full 48-grid as Stage B. Both actually run (the master-stream
design makes Stage B nearly free) — disclosed explicitly in REPORT.md.
Ranking is by raw signal quality (MFE/MAE), never backtest P&L. Gates:
GATE.md.

## 8. Step 3 (only for Phase-1 gate-passing configurations)

TP_R grid {0.5, 0.75, 1.0, 1.5, 2.0}. Phase A: first-crossing
reconstruction from the Phase-1 signal set (per-trade SL/TP from §4;
entry-bar same-bar rule as §4; both-level-ambiguity bars resolved
conservatively SL-first). Phase B: occupancy re-simulation + walk-forward
per the pre-declared GATE.md protocol.

## 9. Step 4 (only if Phase 2 confirms)

Native MQL5 EA (no bridge), terminal-served M1 bars, NormPrice-explicit
market orders AND closes (the ORB/VWAP off-grid-price fix from the start),
DST-aware sessions via the established helper, standalone Python reference
(`evaluate/sp2l_reference.py`), trade-by-trade reconciliation at project
tolerances.
