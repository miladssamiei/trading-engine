# REPORT — SP2L spike/gap/continuation on US500-tick: ablation + gated validation

2026-08-27. Source: third-party `SP2L_Advanced_Bot.py` (XAUUSD M1 spike/gap
bot). Ported to US500-tick per SPEC.md (this directory) with documented unit
conversions. Gates pre-declared in GATE.md BEFORE any analysis ran.

## Step 0 — rule reconstruction (SPEC.md)

Full ambiguity-resolution table in SPEC.md. Highlights:
- **M1** (source's native timeframe; the verified clean full-range export).
- Conversions from gold points: PGAP 100 pts→**0.04%** of price
  ($1.00 / ~$2,500), MAX_SL 1000 pts→**0.40%** ($10 / ~$2,500); spike ratio
  1.5, EMA 60, ADX 14/20, MAX_OPPOSITE 1 dimensionless as-is. TP_R kept
  exposed (default 1.0). The `Meta` execution layer was NOT reproduced.
- Pending-order mechanism implemented as genuinely multi-bar (SPEC §4), with
  shadow semantics, per-variant forced-exit censoring, and kill on risk cap.
- Faithful source quirks documented (signed-body spike ratio trivially true
  on counter-direction candles; gap spans the spike candle; BUY+SELL same
  bar → neither).

## Steps 1–2 — Phase 1 signal quality, 48-configuration ablation

Data: verified M1 export (1,791,504 bars, 2021-01..2026-04). Wall-clock
coverage probe (data plumbing only): day runs 18:00→16:14 ET; 1,370 NY
dates have the 09:30 bar; London 08:00–16:30 fully covered; 23:00–23:59 ET
populated (24h variant usable).

Both Stage A (light-touch: ALL24 baseline + 4 single-filter configs) and
Stage B (full 16 filter combos × {LON, NY, ALL24}) actually ran — the
master-stream design made Stage B nearly free. PGAP {0.02, 0.08}%
sensitivity: informational only (declared in GATE.md).

**Results (phase1_ablation.csv):**
- 2,924 setups at PGAP 0.04% (1,344 BUY / 1,580 SELL); 7,122 at 0.02%;
  746 at 0.08% — the gate sits on a steep part of the curve (halving the
  gap more than doubles setups).
- **35/48 configurations passed** the pre-declared Phase 1 gate (MFE>MAE at
  0.10/0.15/0.20% under BOTH entry conventions, ≥150 signals). E-XTM
  (source-style extreme fill) passes everywhere; the pessimistic E-CLS
  (close fill) is the binding constraint — 13 configs fail on it. ALL24
  passes most; LON/ADX-heavy configs fail most.
- Per-year mean MFE>MAE sign agreement holds in all 6 years for the main
  configs (no regime cliff at Phase-1 level).
- Independent spot-check: 3 signals re-derived from raw CSV by separate
  code path — all 10 setup conditions, SL/entry/risk, MFE/MAE verified.

## Step 3 — TP_R grid + walk-forward (Phase 2, the decisive gates)

TP_R ∈ {0.5, 0.75, 1.0, 1.5, 2.0}; occupancy re-simulation (pending machine
live, one position at a time); WF = burn-in 2021, 9 half-year folds
2022H1..2026H1, trailing-expectancy selection (≥20 trades, else 1.0).

Pre-declared confirmation required the WF gates (expt>0 & Sharpe>0, ≥150
trades, ≥4/5 years positive & best year <60%) under BOTH conventions
(E-XTM AND E-CLS):

- **E-XTM passes WF in all 35 configs** — uniformity that itself signals a
  problem: the extreme-fill convention books the bar's extreme as entry,
  guaranteeing the same-bar bounce as profit. Not trusted alone (declared
  in GATE.md multiplicity notes; this is why the both-conventions rule was
  pre-declared).
- **E-CLS fails 31/35** (expt −0.4..+0.1, mostly negative).
- **4 configs pass both: NY_AS, NY_E, NY_EA, NY_ET** — all NY-session.
  NY dominates: every LON/ALL24 config dies under E-CLS.

Post-WF, TIGHTENING-ONLY buildability gate (documented addition, cannot
admit — only disqualify): **E-NEXT** = the project EA convention (signal on
completed bar, market fill at NEXT bar's open), which neither pre-declared
convention simulates. NY_ET (fails) and NY_AS (fails) drop; **NY_E and
NY_EA pass all gates under all three conventions**. NY_E ⊃ NY_EA in signals
(ADX filter only removes trades) and dominates on every metric → **NY_E is
the confirmed configuration**.

WF fold selections for NY_E/E-NEXT: TP_R=**2.0** selected in ALL 9 folds.
WF result: 1,165 trades, +811.18 pts, expt +0.696, Sharpe 0.834, 4/5 years
positive (best 47.0%).

## Verdict

Phase 1 gate: 35/48 pass (thin-to-clear margins; multiplicity disclosed).
Phase 2 walk-forward: **NY_E (NY session, EMA60 filter, TP_R 2.0) CONFIRMED
under both pre-declared conventions AND the EA-convention E-NEXT gate.**
Proceeded to Step 4 (native EA + reconciliation →
`data/results/sp2l_2021_2026/REPORT.md`).

## Post-confirmation sensitivity audit (2026-08-27) — see SENSITIVITY_ADDENDUM.md

- **Stage A attribution**: individual filter/session walk-forwards WERE run
  (all among the 35 passers — no ablation-design gap). NY session alone
  carries the expectancy but fails G3; EMA60 alone is negative under
  unrestricted session; the confirmed NY_E is a genuinely complementary
  combination — neither component suffices.
- **Broker-digit bracket (PGAP/MAX_SL at 10:1, 0.004%→0.4%)**: the WF edge
  survives ONLY at/near the original 0.04% assumption. The 3-decimal-broker
  reading (0.004%) passes Phase 1 but FAILS the walk-forward under both
  honest conventions; 0.02% degrades below the G3 temporal gate; ≥0.2% the
  pattern vanishes (≤44 setups). Regression anchor at 0.04% reproduced the
  original run exactly. **NY_E is therefore PARAMETER-FRAGILE with respect
  to the unverified point-convention assumption in the unit conversion** —
  an added caveat (not a retraction; no alternative value passes gating, so
  the EA and its parameters are unchanged). Full tables and the explicit
  verdict: `SENSITIVITY_ADDENDUM.md`.

## Artifacts

- `SPEC.md`, `GATE.md` (both written before any signal number).
- `phase1_ablation.csv` (48 rows + 2 sensitivity rows), `phase1_stats.json`,
  `signals/<cfg>.csv` (per-signal MFE/MAE, both conventions).
- `step3/`: phaseA_grid.csv, insample_occ.csv, wf_all.csv, wf_folds.csv,
  wf_trades/, step3_summary.json.
- `evaluate/sp2l_phase1.py`, `evaluate/sp2l_step3.py`.
