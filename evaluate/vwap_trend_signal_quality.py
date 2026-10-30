# -*- coding: utf-8 -*-
"""
vwap_trend_signal_quality.py — Phase 1 signal-quality diagnostic for the
VWAP trend-pullback hypothesis (US500-tick, M5, 2021-01-01..2026-05-01).

PHASE 1 IS A SIGNAL-QUALITY MEASUREMENT ONLY: no stop, no target, no exit
rule beyond the 15:55 NY walk cutoff. Gates were pre-declared in
data/results/vwap_trend_pullback_signal_quality/GATE.md BEFORE this script
ran. Structural idea only — the source video's parameter values and claimed
win rate are deliberately NOT used anywhere.

RULE (Step 0 — every interpretation deliberate, documented in REPORT.md):

  Session          09:30-16:00 NY (RTH), M5 bars, wall minutes [570, 959].
                   A day needs the 09:30 bar (wall==570) else skipped
                   (same convention as ORB / VWAP MR).
  Session VWAP     cumulative over completed M5 session bars:
                   sum(typical*tick_volume)/sum(tick_volume), typical=(H+L+C)/3,
                   resets daily. NO look-ahead: conditions at bar i use the
                   VWAP through bar i. NOT a separately-anchored 15-min VWAP.
  Warm-up          no trading in the first hour: the 60-min lookback
                   (condition c) requires 12 completed bars, so drift can
                   first activate on the bar opening 10:30 (completes 10:35);
                   first possible entry is 10:40 NY. The warm-up rule is
                   therefore enforced by construction, and additionally
                   entry bars must open >= 10:30.
  Drift conditions (all three, evaluated on each COMPLETED M5 bar i, exact
  timestamp references on the M5 grid, trailing windows — the source's
  "15-minute" language operationalized as 3 completed M5 bars):
    (a) close(i) strictly on the trend side of VWAP:  up: close>vwap(i),
        down: close<vwap(i)
    (b) VWAP slope over the trailing 15 min:  up: vwap(i)>vwap(i-15min),
        down: vwap(i)<vwap(i-15min)  (exact-timestamp reference; missing
        reference bar => condition not evaluable => no drift)
    (c) 60-min price change:  up: close(i)/close(i-60min)-1 >= +0.1%,
        down: <= -0.1%   (exact-timestamp reference, 12 bars)
    Direction is fixed by (c)'s sign; (a),(b) must agree. Missing reference
    => drift state = none on that bar.
  State machine     D_i in {none, up, down} recomputed every bar (NOT a
                   latch). EPISODE = maximal run of same-direction D.
                   - One entry per episode: the FIRST pullback trigger only.
                   - Trigger at bar i (entry at open of i+1) iff the episode
                     was active at bar i-1, the episode has not yet entered,
                     and bar i CLOSES AGAINST the episode direction
                     (up: close(i)<open(i); down: close(i)>open(i)).
                   - The trigger->entry step is ATOMIC: the entry fires at
                     the next bar's open regardless of D_i (conditions may
                     legitimately break on the pullback bar itself — that is
                     what a pullback is). There is no cancellation window.
                   - If D turns none: episode ends (entered flag resets); a
                     later re-activation is a NEW episode eligible for one
                     fresh entry. A direction FLIP (up->down with no none in
                     between) is also a new episode.
                   - An open position is NOT closed by drift deactivation
                     (exits own the position; Phase 1 has no exits — walk
                     to 15:55).
  Entry cutoff     entry bar must open <= 15:30 NY (wall <= 930), matching
                   the two prior strategies' convention.
  Forced exit      15:55 NY: the MFE/MAE walk runs from the entry bar
                   (inclusive) through the last session bar OPENING before
                   15:55 (the 15:50 bar), plus the 15:55 bar's open as one
                   final price point. UNCENSORED: no stop/target truncation.

KNOWN APPROXIMATION (declared in GATE.md): signals are independent events —
no position occupancy. Two signals from different episodes may overlap in
book terms; Step 3's re-simulation (only if gated through) resolves this.

Outputs:
  data/results/vwap_trend_pullback_signal_quality/signals_mfe_mae.csv
  console summary + gate evaluation (report source)
Pure stdlib.
"""

import csv
import datetime as dt
import statistics as st
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\vwap_trend_pullback_signal_quality")
M5_CSV = BASE / "us500tick_m5_vol.csv"

NY = ZoneInfo("America/New_York")
UTC = dt.timezone.utc

SESSION_OPEN_M = 570
SESSION_LAST_M = 959          # inclusive wall range for session bars
FORCE_EXIT_M = 955            # 15:55
LAST_ENTRY_M = 930            # 15:30
SLOPE_REF_S = 900             # 15 min
CHANGE_REF_S = 3600           # 60 min
CHANGE_PCT = 0.1              # +/- 0.1% for condition (c)

GATE_THRESHOLDS = (0.10, 0.15, 0.20)
GATE_MIN_SIGNALS = 150
INFO_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def load_sessions():
    by_date = defaultdict(list)
    with open(M5_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            t = int(row["time"])
            ny = dt.datetime.fromtimestamp(t, tz=UTC).astimezone(NY)
            wall = ny.hour * 60 + ny.minute
            if wall < SESSION_OPEN_M or wall > SESSION_LAST_M:
                continue
            by_date[ny.year * 10000 + ny.month * 100 + ny.day].append({
                "time": t, "open": float(row["open"]),
                "high": float(row["high"]), "low": float(row["low"]),
                "close": float(row["close"]), "vol": int(row["tick_volume"]),
                "wall": wall,
            })
    for d in by_date:
        by_date[d].sort(key=lambda b: b["time"])
    return by_date


def compute_drift(bars, i, vwap_at, close_at):
    """D_i per conditions (a),(b),(c); None when not evaluable."""
    b = bars[i]
    t_slope = b["time"] - SLOPE_REF_S
    t_chg = b["time"] - CHANGE_REF_S
    if t_slope not in vwap_at or t_chg not in close_at:
        return None
    chg = b["close"] / close_at[t_chg] - 1.0
    if chg >= CHANGE_PCT / 100.0:
        if b["close"] > vwap_at[b["time"]] and vwap_at[b["time"]] > vwap_at[t_slope]:
            return "up"
    elif chg <= -CHANGE_PCT / 100.0:
        if b["close"] < vwap_at[b["time"]] and vwap_at[b["time"]] < vwap_at[t_slope]:
            return "down"
    return None


def against_close(bar, direction):
    if direction == "up":
        return bar["close"] < bar["open"]
    return bar["close"] > bar["open"]


def main():
    by_date = load_sessions()
    print(f"session bar-days loaded: {len(by_date)}")

    signals = []
    skipped_no_open = 0
    session_days = 0
    intra_gaps = 0

    for ny_date in sorted(by_date):
        bars = by_date[ny_date]
        if not bars or bars[0]["wall"] != SESSION_OPEN_M:
            skipped_no_open += 1
            continue
        session_days += 1

        # contiguity check (documented; exact-timestamp refs need it)
        for k in range(1, len(bars)):
            if bars[k]["time"] - bars[k - 1]["time"] != 300:
                intra_gaps += 1

        cum_pv = cum_v = 0.0
        vwap_at = {}
        close_at = {}
        drifts = []
        for i, b in enumerate(bars):
            typ = (b["high"] + b["low"] + b["close"]) / 3.0
            if b["vol"] > 0:
                cum_pv += typ * b["vol"]
                cum_v += b["vol"]
            vwap_at[b["time"]] = (cum_pv / cum_v) if cum_v > 0 else None
            close_at[b["time"]] = b["close"]
            drifts.append(compute_drift(bars, i, vwap_at, close_at))

        # last bar index opening before 15:55 (the 15:50 bar), and the 15:55 open
        walk_end = max(i for i, b in enumerate(bars) if b["wall"] < FORCE_EXIT_M)
        idx955 = next((i for i, b in enumerate(bars) if b["wall"] == FORCE_EXIT_M), None)
        pt955 = bars[idx955]["open"] if idx955 is not None else None

        ep_dir = None
        ep_entered = False
        for i, b in enumerate(bars):
            D = drifts[i]
            # pullback trigger uses the episode state as of bar i-1
            if (ep_dir is not None and not ep_entered
                    and i + 1 < len(bars)
                    and bars[i + 1]["wall"] <= LAST_ENTRY_M
                    and bars[i + 1]["wall"] >= SESSION_OPEN_M + 60
                    and against_close(b, ep_dir)):
                e = i + 1
                entry = bars[e]["open"]
                hi = max(x["high"] for x in bars[e:walk_end + 1])
                lo = min(x["low"] for x in bars[e:walk_end + 1])
                if pt955 is not None:
                    hi = max(hi, pt955)
                    lo = min(lo, pt955)
                if ep_dir == "up":
                    mfe = (hi / entry - 1.0) * 100.0
                    mae = (1.0 - lo / entry) * 100.0
                else:
                    mfe = (1.0 - lo / entry) * 100.0
                    mae = (hi / entry - 1.0) * 100.0
                signals.append({
                    "ny_date": ny_date, "dir": ep_dir,
                    "trigger_time": b["time"], "entry_time": bars[e]["time"],
                    "entry": entry, "bars_walked": walk_end - e + 1,
                    "mfe": mfe, "mae": mae,
                    "vwap_at_entry": vwap_at.get(bars[i]["time"]),
                })
                ep_entered = True
            # update episode state with D_i
            if D is None:
                ep_dir, ep_entered = None, False
            elif D == ep_dir:
                pass
            else:
                ep_dir, ep_entered = D, False

    # ---------------- write per-signal CSV ----------------
    def fts(t):
        return dt.datetime.fromtimestamp(t, tz=UTC).strftime("%Y.%m.%d %H:%M")

    cols = ["signal_id", "ny_date", "direction", "trigger_time", "entry_time",
            "entry_price", "bars_walked", "mfe_pct", "mae_pct", "vwap_at_entry",
            "exit_cutoff"]
    with open(BASE / "signals_mfe_mae.csv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(cols)
        for n, s in enumerate(signals, 1):
            w.writerow([n, s["ny_date"], s["dir"], fts(s["trigger_time"]),
                        fts(s["entry_time"]), f"{s['entry']:.4f}",
                        s["bars_walked"], f"{s['mfe']:.4f}", f"{s['mae']:.4f}",
                        f"{s['vwap_at_entry']:.4f}",
                        fts(s["entry_time"] + s["bars_walked"] * 300 - 300)])
    print(f"signals: {len(signals)}  -> signals_mfe_mae.csv")
    print(f"session days: {session_days}  skipped (no 09:30 bar): {skipped_no_open}"
          f"  intra-session M5 gaps: {intra_gaps}")

    # ---------------- summary ----------------
    mfeas = [s["mfe"] for s in signals]
    maes = [s["mae"] for s in signals]
    n = len(signals)
    ups = sum(1 for s in signals if s["dir"] == "up")
    unique_days = len({s["ny_date"] for s in signals})
    per_year = defaultdict(lambda: [0, 0.0, 0.0])
    for s in signals:
        y = s["ny_date"] // 10000
        per_year[y][0] += 1
        per_year[y][1] += s["mfe"]
        per_year[y][2] += s["mae"]

    print()
    print("=" * 78)
    print("Phase 1 — VWAP trend-pullback signal quality (NO exit rule; exploratory)")
    print("=" * 78)
    print(f" signals: {n}   up/down: {ups}/{n - ups}   unique days: {unique_days}")
    print(f" MFE  mean {st.mean(mfeas):.3f}  median {st.median(mfeas):.3f}  "
          f"p90 {sorted(mfeas)[int(0.9 * n)]:.3f}")
    print(f" MAE  mean {st.mean(maes):.3f}  median {st.median(maes):.3f}  "
          f"p90 {sorted(maes)[int(0.9 * n)]:.3f}")
    print()
    print(f" {'threshold':>10} {'MFE>=t':>8} {'MAE>=t':>8} {'dominant':>10}")
    for t in INFO_THRESHOLDS:
        nm = sum(1 for x in mfeas if x >= t)
        na = sum(1 for x in maes if x >= t)
        dom = "MFE" if nm > na else ("MAE" if na > nm else "tie")
        print(f" {t:>9.2%} {nm:>8} {na:>8} {dom:>10}")

    print()
    print(" per-year:  n   meanMFE  meanMAE")
    for y in sorted(per_year):
        c, m, a = per_year[y]
        print(f"   {y}: {c:>4}  {m / c:+8.3f} {a / c:+8.3f}")

    # informational: first-crossing order at gate thresholds
    print()
    print(" informational — first-cross order (recomputed walk):")

    def cross_walk(s, t):
        """Return (mfe_bar, mae_bar) — indices of the first bar whose range
        crosses each level (955-open as final point), else None."""
        bars = by_date[s["ny_date"]]
        e = next(i for i, b in enumerate(bars) if b["time"] == s["entry_time"])
        walk_end = max(i for i, b in enumerate(bars) if b["wall"] < FORCE_EXIT_M)
        idx955 = next((i for i, b in enumerate(bars) if b["wall"] == FORCE_EXIT_M), None)
        entry = s["entry"]
        up = s["dir"] == "up"
        fm = fa = None
        for k in range(e, walk_end + 1):
            mfe_hit = ((bars[k]["high"] / entry - 1.0) * 100.0 >= t) if up \
                else ((1.0 - bars[k]["low"] / entry) * 100.0 >= t)
            mae_hit = ((1.0 - bars[k]["low"] / entry) * 100.0 >= t) if up \
                else ((bars[k]["high"] / entry - 1.0) * 100.0 >= t)
            if fm is None and mfe_hit:
                fm = k
            if fa is None and mae_hit:
                fa = k
            if fm is not None and fa is not None:
                return fm, fa
        if idx955 is not None:
            pv = bars[idx955]["open"]
            mfe955 = ((pv / entry - 1.0) * 100.0 >= t) if up \
                else ((1.0 - pv / entry) * 100.0 >= t)
            mae955 = ((1.0 - pv / entry) * 100.0 >= t) if up \
                else ((pv / entry - 1.0) * 100.0 >= t)
            if fm is None and mfe955:
                fm = idx955
            if fa is None and mae955:
                fa = idx955
        return fm, fa

    for t in GATE_THRESHOLDS:
        first_mfe = first_mae = same_bar = 0
        for s in signals:
            fm, fa = cross_walk(s, t)
            if fm is not None and fa is not None:
                if fm < fa:
                    first_mfe += 1
                elif fa < fm:
                    first_mae += 1
                else:
                    same_bar += 1
            elif fm is not None:
                first_mfe += 1
            elif fa is not None:
                first_mae += 1
        print(f"   t={t:.2%}: MFE-first {first_mfe}  MAE-first {first_mae}  "
              f"same-bar {same_bar}  neither {n - first_mfe - first_mae - same_bar}")

    # ---------------- gate evaluation ----------------
    print()
    print("=" * 78)
    print("GATES (pre-declared in GATE.md)")
    g1 = True
    for t in GATE_THRESHOLDS:
        nm = sum(1 for x in mfeas if x >= t)
        na = sum(1 for x in maes if x >= t)
        ok = nm > na
        g1 = g1 and ok
        print(f"  G1 t={t:.2%}: MFE {nm} vs MAE {na} -> {'PASS' if ok else 'FAIL'}")
    g2 = n >= GATE_MIN_SIGNALS
    print(f"  G2 signals >= {GATE_MIN_SIGNALS}: {n} -> {'PASS' if g2 else 'FAIL'}")
    verdict = "GATE PASSED — proceed to Step 3" if (g1 and g2) else \
              "GATE FAILED — null result, stop (no exit design, no EA)"
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
