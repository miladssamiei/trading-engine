# -*- coding: utf-8 -*-
"""
vwap_trend_rr_grid.py — Step 3, Phase 1 (EXPLORATORY / IN-SAMPLE):
reconstruct stop/target outcomes for the 6,606 Phase-1 signals on a
stops x targets grid, fixed-entry approximation, per GATE.md
(data/results/vwap_trend_pullback_rr/GATE.md). NOT a validated backtest.

Validation invariants asserted per cell/signal (per GATE.md):
stop_loss => MAE >= stop ; win => MFE >= target ; time => neither crossed.
"""

import csv
from pathlib import Path

import vwap_trend_common as tc

OUT = Path(r"C:\Users\Milad\Desktop\RR\data\results\vwap_trend_pullback_rr")
STOPS = [0.10, 0.15, 0.20, 0.30]
TARGETS = [0.15, 0.20, 0.30, 0.45]


def walk_outcome(day, sig, stop_pct, target_pct):
    """First-crossing outcome on the Phase-1 walk. Returns (pnl, reason)."""
    bars = day["bars"]
    e = sig["entry_idx"]
    walk_end = day["walk_end"]
    idx955 = day["idx955"]
    entry = bars[e]["open"]
    up = sig["dir"] == "up"
    sign = 1 if up else -1
    sl = tc.norm_price(entry * (1.0 - sign * stop_pct / 100.0))
    tp = tc.norm_price(entry * (1.0 + sign * target_pct / 100.0))

    def hit_mae(b):
        return ((1.0 - b["low"] / entry) * 100.0 >= stop_pct) if up \
            else ((b["high"] / entry - 1.0) * 100.0 >= stop_pct)

    def hit_mfe(b):
        return ((b["high"] / entry - 1.0) * 100.0 >= target_pct) if up \
            else ((1.0 - b["low"] / entry) * 100.0 >= target_pct)

    # entry bar: full range achievable from its open
    b = bars[e]
    if hit_mae(b) and hit_mfe(b):
        if b["close"] >= b["open"]:
            return round(sign * (sl - entry), 2), "stop_loss"
        return round(sign * (tp - entry), 2), "target"
    if hit_mae(b):
        return round(sign * (sl - entry), 2), "stop_loss"
    if hit_mfe(b):
        return round(sign * (tp - entry), 2), "target"
    for k in range(e + 1, walk_end + 1):
        b = bars[k]
        mae = hit_mae(b)
        mfe = hit_mfe(b)
        if mae and mfe:
            if b["close"] >= b["open"]:
                return round(sign * (sl - entry), 2), "stop_loss"
            return round(sign * (tp - entry), 2), "target"
        if mae:
            return round(sign * (sl - entry), 2), "stop_loss"
        if mfe:
            return round(sign * (tp - entry), 2), "target"
    # neither crossed through the 15:50 bar: final point = 15:55 open.
    # GATE.md invariant: a time exit requires neither level crossed on the
    # walk (which includes this point) -> gap-check the 15:55 open first.
    if idx955 is not None:
        pv = bars[idx955]["open"]
        mae955 = ((1.0 - pv / entry) * 100.0 >= stop_pct) if up \
            else ((pv / entry - 1.0) * 100.0 >= stop_pct)
        mfe955 = ((pv / entry - 1.0) * 100.0 >= target_pct) if up \
            else ((1.0 - pv / entry) * 100.0 >= target_pct)
        if mae955:
            return round(sign * (pv - entry), 2), "stop_loss"
        if mfe955:
            return round(sign * (pv - entry), 2), "target"
        return round(sign * (pv - entry), 2), "time"
    return round(sign * (bars[walk_end]["close"] - entry), 2), "time"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    days = tc.prepare_days()
    signals = [(d, s) for d in days for s in d["signals"]]
    print(f"days: {len(days)}   signals: {len(signals)}")

    # Phase-1 MFE/MAE (recomputed here; must match signals_mfe_mae.csv)
    mfe_mae = []
    for d, s in signals:
        bars = d["bars"]
        e = s["entry_idx"]
        entry = bars[e]["open"]
        hi = max(b["high"] for b in bars[e:d["walk_end"] + 1])
        lo = min(b["low"] for b in bars[e:d["walk_end"] + 1])
        if d["idx955"] is not None:
            pv = bars[d["idx955"]]["open"]
            hi, lo = max(hi, pv), min(lo, pv)
        if s["dir"] == "up":
            mfe_mae.append(((hi / entry - 1) * 100, (1 - lo / entry) * 100))
        else:
            mfe_mae.append(((1 - lo / entry) * 100, (hi / entry - 1) * 100))

    rows = []
    grid = {}
    viol = 0
    for sp in STOPS:
        for tp in TARGETS:
            pnls = []
            reasons = {"stop_loss": 0, "target": 0, "time": 0}
            for n, (d, s) in enumerate(signals):
                pnl, reason = walk_outcome(d, s, sp, tp)
                mfe, mae = mfe_mae[n]
                if reason == "stop_loss":
                    assert mae >= sp - 1e-9, f"invariant MAE {mae} < stop {sp}"
                elif reason == "target":
                    assert mfe >= tp - 1e-9, f"invariant MFE {mfe} < target {tp}"
                else:
                    if mae >= sp - 1e-9 or mfe >= tp - 1e-9:
                        viol += 1
                reasons[reason] += 1
                pnls.append(pnl)
            n = len(pnls)
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            grid[(sp, tp)] = {
                "n": n, "wr": 100.0 * len(wins) / n,
                "aw": sum(wins) / len(wins) if wins else 0.0,
                "al": sum(losses) / len(losses) if losses else 0.0,
                "expt": sum(pnls) / n, "total": sum(pnls), "r": reasons,
            }
            rows.append([f"{sp:.2f}", f"{tp:.2f}", n, len(wins),
                         f"{100.0 * len(wins) / n:.2f}",
                         f"{grid[(sp, tp)]['aw']:.2f}",
                         f"{grid[(sp, tp)]['al']:.2f}",
                         f"{sum(pnls) / n:.3f}", f"{sum(pnls):.2f}",
                         reasons["stop_loss"], reasons["target"], reasons["time"]])
    assert viol == 0, f"{viol} time-exit invariant violations"
    print("all cell/signal invariants hold (0 violations)")

    with open(OUT / "phase1_grid.csv", "w", newline="") as fh:
        fh.write("stop_pct;target_pct;n;wins;winrate_pct;avg_win;avg_loss;"
                 "expectancy;total_pnl;exits_stop;exits_target;exits_time\r\n")
        for r in rows:
            fh.write(";".join(str(x) for x in r) + "\r\n")

    print()
    print("=" * 100)
    print("Step 3 Phase 1 — EXPLORATORY reconstruction (fixed entries, no occupancy), NOT a validated backtest")
    print("=" * 100)
    print(f"{'stop%':>6} {'target%':>8} {'n':>5} {'win%':>7} {'avgW':>7} {'avgL':>8}"
          f" {'EXPT':>8} {'total':>10}  stop/target/time")
    for sp in STOPS:
        for tp in TARGETS:
            c = grid[(sp, tp)]
            print(f"{sp:>6.2f} {tp:>8.2f} {c['n']:>5} {c['wr']:>7.2f} {c['aw']:>7.2f}"
                  f" {c['al']:>8.2f} {c['expt']:>8.3f} {c['total']:>10.2f}"
                  f"  {c['r']['stop_loss']:>5}/{c['r']['target']:>5}/{c['r']['time']:>4}")

    # neighborhood analysis (shortlist input per GATE.md)
    print()
    print("neighborhood check (cell > 0 AND all orthogonal neighbors > 0):")
    passing = []
    for sp in STOPS:
        for tp in TARGETS:
            e = grid[(sp, tp)]["expt"]
            if e <= 0:
                continue
            nbrs = []
            if sp != STOPS[0]:
                nbrs.append(grid[(STOPS[STOPS.index(sp) - 1], tp)]["expt"])
            if sp != STOPS[-1]:
                nbrs.append(grid[(STOPS[STOPS.index(sp) + 1], tp)]["expt"])
            if tp != TARGETS[0]:
                nbrs.append(grid[(sp, TARGETS[TARGETS.index(tp) - 1])]["expt"])
            if tp != TARGETS[-1]:
                nbrs.append(grid[(sp, TARGETS[TARGETS.index(tp) + 1])]["expt"])
            ok = all(x > 0 for x in nbrs)
            if ok:
                passing.append((sp, tp, e))
            print(f"  ({sp:.2f},{tp:.2f}) expt {e:+.3f}  neighbors "
                  f"{'/'.join(f'{x:+.3f}' for x in nbrs)}  -> "
                  f"{'PASS' if ok else 'fail'}")
    print()
    print(f"strict-neighborhood shortlist: {passing if passing else 'NONE — fallback to structural pattern per GATE.md'}")
    print(f"wrote {OUT / 'phase1_grid.csv'}")


if __name__ == "__main__":
    main()
