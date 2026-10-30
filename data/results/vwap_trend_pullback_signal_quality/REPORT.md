# VWAP Trend-Pullback — Phase 1 Signal Quality Report

**US500-tick · M5 · 2021-01-01 → 2026-05-01 · Python only (no EA, no exit rule)**

- **Report date:** 2026-08-26
- **Hypothesis (structural idea only):** trade in the direction of an established VWAP-referenced trend after a small pullback — the third edge-source hypothesis after ORB (breakout momentum) and VWAP MR (fade to value, confirmed negative).
- **Source status:** third-party prop-firm-affiliate video, unverified, self-described by its presenter as likely unprofitable on real capital and optimized for a different objective (passing a prop evaluation). Its stop/target point values and claimed win rate are **deliberately not used anywhere**; only the structural entry concept is tested, with every parameter derived fresh here.
- **Order of operations:** signal quality FIRST, exit design only if gated through (per the sticky-HMM precedent). Gates were pre-declared in `GATE.md` before the analysis script ran.

## Headline verdict

**Phase 1 gate PASSED — proceed to Step 3 — but the dominance is thin, and that is the honest headline.** 6,606 qualifying pullback signals over 1,350 of 1,370 session days; MFE counts exceed MAE counts at all three pre-declared thresholds (0.10%: 4,955 vs 4,893; 0.15%: 4,291 vs 4,141; 0.20%: 3,670 vs 3,513) and G2's sample floor (≥150) is cleared 44×. But the margins are 1.3–4.5% relative, mean MFE 0.361% vs mean MAE 0.351%, and the informational first-crossing order at 0.10% is a statistical dead heat (MFE-first 3,159 vs MAE-first 3,161); genuine ordering advantage only emerges at ≥0.15% excursions (MFE-first 3,107 vs 2,939 at 0.15%; 2,881 vs 2,671 at 0.20%). This is a marginal continuation tilt at these entry points — far weaker than the adverse dominance VWAP MR's entries showed in reverse — and it may not survive the exit-rule stage. The gate's letter is satisfied; its spirit is on notice.

## Step 0 — the rule as implemented (every interpretation documented)

| Ambiguity in source | Resolution (deliberate, fixed for all phases) |
|---|---|
| VWAP basis | Session VWAP on M5 bars: Σ(typ×tick_volume)/Σ(tick_volume), typ=(H+L+C)/3, cumulative from the 09:30 bar, daily reset, completed bars only (no look-ahead). NOT a separately-anchored 15-min VWAP. |
| "15-minute" conditions | The trailing windows live on the M5 grid with exact-timestamp references: 15-min VWAP slope = vwap(i) vs vwap(i−900 s) (3 completed M5 bars); 60-min price change = close(i)/close(i−3600 s)−1 (12 bars). A missing reference bar (12 intra-session M5 gaps exist) ⇒ condition not evaluable ⇒ no drift that bar. The source's 15-min-bar language is operationalized as 3-M5-bar trailing windows — the VWAP itself stays M5-based. |
| Trend direction | Fixed by the 60-min change's sign (≥ +0.1% up / ≤ −0.1% down); conditions (a) close strictly beyond VWAP and (b) VWAP slope agreement must both match that direction. |
| Warm-up (no trading 9:30–10:30) | Enforced by construction: the 60-min lookback needs 12 completed bars, so drift first activates on the bar opening 10:30 and the first possible entry is 10:40 NY; entry bars additionally constrained to open ≥ 10:30. |
| Pullback trigger | While an episode is active, the FIRST M5 bar closing against the trend (close<open in an up-drift; close>open in a down-drift) triggers; entry at the NEXT bar's open, in the TREND's direction. |
| Drift reset mid-day | Drift state D_i is recomputed every bar (not a latch). Conditions breaking ends the episode; re-activation (after a break, or a direct direction flip) starts a NEW episode eligible for one fresh entry. |
| Entries per trend | ONE per episode (the first pullback only) — "first bar that closes against" read literally. Episodes re-arm frequently in practice: 6,606 signals / 1,350 days ≈ 4.9 per day. |
| VWAP cross before entry | Impossible by construction: trigger→entry is atomic (next bar's open, no cancellation window), even if the drift conditions break on the trigger bar itself — which is what a pullback is. |
| Position vs drift deactivation | An open position is NOT closed by drift deactivation; exits own the position (Phase 1 has none — walk to 15:55). |
| Entry cutoff / forced exit | Entry bar open ≤ 15:30 NY; path walked UNCENSORED to 15:55: entry bar inclusive, through the last bar opening before 15:55 (the 15:50 bar), plus the 15:55 bar's open as one final price point (its later high/low is not achievable). |
| Signal independence | KNOWN APPROXIMATION (declared in GATE.md): signals are independent events — no position occupancy; overlapping same-day signals coexist. Step 3's re-simulation resolves this. |
| Data base | ORB's verified M5 export carries no volume column, so the verified full-range M1 export was aggregated to M5 (volume = Σ M1 tick_volume) and validated bar-for-bar against the terminal-native M5 export BEFORE any signal work: **364,330/364,330 bars, 0 only-either-side, 0 OHLC mismatches (max |diff| = 0)** — `us500tick_m5_vol.csv`, `evaluate/vwap_trend_m5_aggregate.py`. |

## Gate (pre-declared, `GATE.md`, written before the run)

- **G1:** count(MFE ≥ t) > count(MAE ≥ t) at ALL of t ∈ {0.10%, 0.15%, 0.20%}
- **G2:** ≥ 150 qualifying signals
- Abort condition on data: M1→M5 aggregation mismatch rate > 0.05% ⇒ Phase 1 aborted (did not trigger: 0.0000%).

## Results (`signals_mfe_mae.csv`, 6,606 rows)

Session days: 1,370 (0 skipped — every day had its 09:30 bar). Signals: 6,606 (3,611 up / 2,995 down) on 1,350 unique days. 12 intra-session M5 gaps (handled by exact-timestamp references).

| Statistic | MFE % | MAE % |
|---|---|---|
| mean | 0.361 | 0.351 |
| median | 0.232 | 0.222 |
| p90 | 0.805 | 0.792 |

| Threshold | MFE ≥ t | MAE ≥ t | Dominant |
|---|---|---|---|
| 0.05% | 5,733 | 5,713 | MFE |
| 0.10% | 4,955 | 4,893 | MFE |
| 0.15% | 4,291 | 4,141 | MFE |
| 0.20% | 3,670 | 3,513 | MFE |
| 0.25% | 3,120 | 3,002 | MFE |
| 0.30% | 2,668 | 2,557 | MFE |

Per-year signal counts and mean excursions:

| Year | n | mean MFE | mean MAE |
|---|---|---|---|
| 2021 | 1,169 | 0.278 | 0.285 |
| 2022 | 1,361 | 0.576 | 0.540 |
| 2023 | 1,258 | 0.315 | 0.316 |
| 2024 | 1,181 | 0.272 | 0.266 |
| 2025 | 1,195 | 0.358 | 0.341 |
| 2026 (Jan–Apr) | 442 | 0.290 | 0.299 |

Informational — first-crossing order (which level a trade could have reached first; NOT part of the gate): 0.10%: MFE-first 3,159 / MAE-first 3,161 / same-bar 149; 0.15%: 3,107 / 2,939 / 54; 0.20%: 2,881 / 2,671 / 20. The ordering edge is zero at 0.10% and modest (~52% of decisive cases) at ≥0.15% — consistent with the count-dominance being driven by the deeper half of the excursion distribution.

### Gate evaluation

| Gate | Requirement | Observed | Verdict |
|---|---|---|---|
| G1 | MFE > MAE counts at 0.10/0.15/0.20% | all three, by 62/150/157 signals | **PASS** |
| G2 | ≥ 150 signals | 6,606 | **PASS** |

**Decision: proceed to Step 3** (stop/target design under the two-phase discipline), with these declared caveats: (1) the tilt is thin and concentrated in deeper excursions — any working exit rule likely needs a target ≥ ~0.15% and will live or die on the 48–52% ordering margin; (2) the 4.9-signals/day rate means the independent-events approximation is stretched far further here than in the VWAP R:R study — the occupancy-modeled re-simulation will cut the effective sample substantially and is the real test; (3) as with every prior phase in this project, no number on this page is a result — only Step 3's walk-forward-confirmed outcome can be.

## Artifacts

- `GATE.md` — pre-declared gates (written before the run)
- `us500tick_m5_vol.csv` — M1-aggregated M5 + volume, verified byte-exact vs ORB's terminal-native M5
- `signals_mfe_mae.csv` — 6,606 signals with trigger/entry times, MFE/MAE, walk lengths
- `evaluate/vwap_trend_m5_aggregate.py`, `evaluate/vwap_trend_signal_quality.py`
