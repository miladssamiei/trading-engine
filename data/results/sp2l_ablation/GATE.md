# GATE.md — SP2L Phase 1 gate + Phase 2 protocol, pre-declared BEFORE any Phase 1 analysis ran

Written 2026-08-27, after SPEC.md and before `evaluate/sp2l_phase1.py` was
executed. Nothing below may change after Phase 1 numbers are seen. Mirrors
`data/results/vwap_trend_pullback_signal_quality/GATE.md` and
`data/results/vwap_rr_exploration/GATE.md`.

## What Phase 1 measures

Per configuration (16 filter toggles × 3 session variants = 48, plus the
Stage-A light-touch subset), every qualifying SP2L entry signal per the
SPEC.md §3–§6 rule on US500-tick M1, 2021-01 → 2026-04, with NO exit rule.
In the signal direction, % of entry price:

- **MFE** — best favorable excursion; **MAE** — worst adverse excursion.
- Walk UNCENSORED to the variant's forced-exit point (SPEC §5), same-bar
  rules per SPEC §4.
- Computed under BOTH entry conventions — E-XTM (source: entry = breaking
  bar's extreme) and E-CLS (pessimistic: entry = breaking bar's close,
  walk starts next bar).

## Phase 1 gates (a configuration passes only if ALL hold under BOTH E-XTM and E-CLS)

- **G1 (excursion dominance):** count(MFE ≥ t) > count(MAE ≥ t) at ALL of
  t = 0.10%, 0.15%, 0.20%.
- **G2 (sample size):** ≥ 150 first-entry signals over the full range
  (second-entry signals do NOT count toward G2 — they are derivative).

If no configuration passes: write the null result and STOP — no exit-rule
design, no Step 3, no EA.

## Multiplicity honesty (declared up front)

48 configurations are tested; any per-configuration pass suffers
winner's-curse risk. Mitigations, fixed now: (a) G1 must hold under BOTH
entry conventions; (b) Phase 1 passing only ADMITS a configuration to
Step 3 — the decisive gates are Step 3's pre-declared walk-forward gates
below; (c) the Stage-A vs Stage-B distinction is reported explicitly.

## Step 3 protocol (pre-declared; runs only for Phase-1-passing configurations)

- TP_R grid {0.5, 0.75, 1.0, 1.5, 2.0}, SL fixed by the rule (pre-spike
  extreme), no other exit parameter. Phase A = first-crossing
  reconstruction (exploratory, in-sample); Phase B = occupancy
  re-simulation (one position at a time; pending machinery live; signals
  while holding are skipped, not queued) with walk-forward selection:
  burn-in 2021, folds 2022H1..2026H1 (9 half-years), select by trailing
  expectancy (≥ 20 trailing trades, else the a-priori default TP_R=1.0).
- Walk-forward gates (on the walk-forward result, gross-of-cost):
  - **G1:** expectancy > 0 AND Sharpe > 0 (zero-filled daily PnL, √252).
  - **G2:** ≥ 150 walk-forward trades.
  - **G3:** temporal concentration: positive in ≥ 4 of 5 calendar years AND
    no single year > 60% of total profit.
- All three must pass; any failure = NOT CONFIRMED → stop, no EA, no
  parameter adoption.

## Honesty notes (declared before seeing numbers)

- Phase 1 signals are independent events (position occupancy not modeled;
  pending occupancy is modeled). Step 3 Phase B resolves path dependence.
- PGAP_PCT=0.04% / MAX_SL_PCT=0.40% are conversion-derived (SPEC §1), not
  tuned. The {0.02, 0.08}% PGAP sensitivity is informational only and can
  not admit a configuration to Step 3.
- Thresholds 0.10/0.15/0.20% and ≥150 signals were fixed by analogy to the
  two prior strategies' Phase-1 gates before any SP2L number was computed.
- Metrics are gross of transaction costs (bid-only tape), consistent with
  all prior Phase 1 work.
