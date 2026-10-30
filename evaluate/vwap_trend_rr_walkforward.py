# -*- coding: utf-8 -*-
"""
vwap_trend_rr_walkforward.py — Step 3, Phase 2 (the ONLY phase allowed to
be called a result): occupancy-modeled full re-simulation of the VWAP
trend-pullback strategy for the shortlisted (stop%, target%) candidates,
with walk-forward selection per GATE.md.

Occupancy model: one position at a time; signals while holding are skipped
(not queued). Exits follow the validated reference-engine conventions:
time-exit priority at the 15:55 bar open, then bar-open gap checks, then
same-bar both-level ambiguity via the close>=open path heuristic; early
close => carry to next session's first bar open. SL/TP rounded to the
0.01 trade grid. Entry = signal entry bar's open, only if flat at that
bar's start. Daily PnL zero-filled over session days.

Walk-forward: burn-in 2021; folds 2022H1..2026H1 (9 half-years); select by
trailing expectancy (>= 20 trailing trades, else default candidate).
"""

import csv
import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path

import vwap_trend_common as tc

OUT = Path(r"C:\Users\Milad\Desktop\RR\data\results\vwap_trend_pullback_rr")

# shortlist rationale (all 16 grid cells passed strict neighborhood; three
# diagonal cells spanning the excursion scale, incl. the a-priori default):
CANDIDATES = {
    "S_010_020": (0.10, 0.20),
    "D_020_030": (0.20, 0.30),   # a-priori default
    "L_030_045": (0.30, 0.45),
}
DEFAULT_KEY = "D_020_030"
MIN_TRAILING = 20

FOLD_BOUNDS = [
    (2022, 1), (2022, 7), (2023, 1), (2023, 7), (2024, 1), (2024, 7),
    (2025, 1), (2025, 7), (2026, 1), (2026, 5),
]


def fold_of(ny_date):
    y, m = ny_date // 10000, (ny_date // 100) % 100
    for k in range(len(FOLD_BOUNDS) - 1):
        (fy, fm), (ty, tm) = FOLD_BOUNDS[k], FOLD_BOUNDS[k + 1]
        if (y, m) >= (fy, fm) and (y, m) < (ty, tm):
            return k
    return None


def simulate(days, stop_pct, target_pct):
    """Occupancy-modeled re-simulation. Returns trades list."""
    trades = []
    pos = None
    for di, day in enumerate(days):
        bars = day["bars"]
        sig_at = {s["entry_idx"]: s for s in day["signals"]}

        # carried position from a prior early-close day: exit at this day's
        # first bar open (engine convention)
        if pos is not None:
            px = bars[0]["open"]
            sign = 1 if pos["dir"] == "up" else -1
            trades.append({"ny_date": pos["ny_date"], "dir": pos["dir"],
                           "open_ts": pos["open_ts"], "close_ts": bars[0]["time"],
                           "entry": pos["entry"], "exit": px,
                           "reason": "time_exit",
                           "pnl": round(sign * (px - pos["entry"]), 2)})
            pos = None

        for i, b in enumerate(bars):
            # ---- exits during bar i (engine priority: time > gap > levels)
            if pos is not None:
                sign = 1 if pos["dir"] == "up" else -1
                sl, tp = pos["sl"], pos["tp"]
                exit_px = reason = None
                if b["wall"] >= tc.FORCE_EXIT_M:
                    exit_px, reason = b["open"], "time_exit"
                elif pos["dir"] == "up":
                    if b["open"] <= sl:
                        exit_px, reason = b["open"], "stop_loss"
                    elif b["open"] >= tp:
                        exit_px, reason = b["open"], "target"
                    elif b["low"] <= sl and b["high"] >= tp:
                        if b["close"] >= b["open"]:
                            exit_px, reason = sl, "stop_loss"
                        else:
                            exit_px, reason = tp, "target"
                    elif b["low"] <= sl:
                        exit_px, reason = sl, "stop_loss"
                    elif b["high"] >= tp:
                        exit_px, reason = tp, "target"
                else:
                    if b["open"] >= sl:
                        exit_px, reason = b["open"], "stop_loss"
                    elif b["open"] <= tp:
                        exit_px, reason = b["open"], "target"
                    elif b["high"] >= sl and b["low"] <= tp:
                        if b["close"] >= b["open"]:
                            exit_px, reason = tp, "target"
                        else:
                            exit_px, reason = sl, "stop_loss"
                    elif b["high"] >= sl:
                        exit_px, reason = sl, "stop_loss"
                    elif b["low"] <= tp:
                        exit_px, reason = tp, "target"
                if exit_px is not None:
                    trades.append({"ny_date": pos["ny_date"], "dir": pos["dir"],
                                   "open_ts": pos["open_ts"],
                                   "close_ts": b["time"], "entry": pos["entry"],
                                   "exit": exit_px, "reason": reason,
                                   "pnl": round(sign * (exit_px - pos["entry"]), 2)})
                    pos = None

            # ---- entry at bar i open if flat at bar start and signal here
            if pos is None and i in sig_at:
                s = sig_at[i]
                sign = 1 if s["dir"] == "up" else -1
                entry = b["open"]
                pos = {"ny_date": day["ny_date"], "dir": s["dir"],
                       "open_ts": b["time"], "entry": entry,
                       "sl": tc.norm_price(entry * (1.0 - sign * stop_pct / 100.0)),
                       "tp": tc.norm_price(entry * (1.0 + sign * target_pct / 100.0))}

        # position open past the day's last bar (early close) -> carry
    # data end with position open: close at the very last bar's close
    if pos is not None:
        sign = 1 if pos["dir"] == "up" else -1
        lb = days[-1]["bars"][-1]
        trades.append({"ny_date": pos["ny_date"], "dir": pos["dir"],
                       "open_ts": pos["open_ts"], "close_ts": lb["time"],
                       "entry": pos["entry"], "exit": lb["close"],
                       "reason": "time_exit",
                       "pnl": round(sign * (lb["close"] - pos["entry"]), 2)})
    return trades


def metrics(trades, session_days, day_from=None, day_to=None):
    """Zero-filled daily pnl metrics. Days filtered to (day_from, day_to)
    ny_date ints if given."""
    days = [d for d in session_days
            if (day_from is None or d >= day_from)
            and (day_to is None or d < day_to)]
    per_day = defaultdict(float)
    for t in trades:
        per_day[t["ny_date"]] += t["pnl"]
    vals = [per_day.get(d, 0.0) for d in days]
    mean = sum(vals) / len(vals) if vals else 0.0
    var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1) if len(vals) > 1 else 0.0
    std = math.sqrt(var)
    sharpe = mean / std * math.sqrt(252) if std > 0 else None
    cum = peak = maxdd = 0.0
    for x in vals:
        cum += x
        peak = max(peak, cum)
        maxdd = max(maxdd, peak - cum)
    total = sum(t["pnl"] for t in trades)
    return {"trades": len(trades), "total": round(total, 2),
            "expt": round(total / len(trades), 3) if trades else None,
            "sharpe": None if sharpe is None else round(sharpe, 3),
            "maxdd": round(maxdd, 2)}


def fts(t):
    return dt.datetime.fromtimestamp(t, tz=dt.timezone.utc).strftime("%Y.%m.%d %H:%M")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    days = tc.prepare_days()
    session_days = [d["ny_date"] for d in days]
    print(f"days: {len(days)}  signals: {sum(len(d['signals']) for d in days)}")

    runs = {}
    for name, (sp, tp) in CANDIDATES.items():
        trades = simulate(days, sp, tp)
        runs[name] = trades
        m = metrics(trades, session_days)
        taken = len(trades)
        print(f"{name}: stop={sp:.2f}% target={tp:.2f}%  trades={taken}  "
              f"total={m['total']:+.2f}  expt={m['expt']:+.3f}  "
              f"sharpe={m['sharpe']}  maxdd={m['maxdd']}")

    # ---- in-sample full range -------------------------------------------
    wf_start = FOLD_BOUNDS[0][0] * 10000 + FOLD_BOUNDS[0][1] * 100
    wf_end = FOLD_BOUNDS[-1][0] * 10000 + FOLD_BOUNDS[-1][1] * 100
    print("\n== in-sample full range (2021-01..2026-04) ==")
    insample = {}
    for name, trades in runs.items():
        m = metrics(trades, session_days)
        wins = sum(1 for t in trades if t["pnl"] > 0)
        m["winrate"] = round(100.0 * wins / len(trades), 2) if trades else None
        m["expt_net04"] = round(m["expt"] - 0.4, 3) if m["expt"] is not None else None
        m["expt_net08"] = round(m["expt"] - 0.8, 3) if m["expt"] is not None else None
        insample[name] = m
        print(f"  {name}: {m}")

    # ---- walk-forward -----------------------------------------------------
    print("\n== walk-forward (folds 2022H1..2026H1, trailing expectancy) ==")
    wf_rows = []
    wf_trades = []
    for k in range(len(FOLD_BOUNDS) - 1):
        (fy, fm), (ty, tm) = FOLD_BOUNDS[k], FOLD_BOUNDS[k + 1]
        f_start, f_end = fy * 10000 + fm * 100, ty * 10000 + tm * 100
        best, best_exp = None, None
        for name, trades in runs.items():
            trail = [t for t in trades if t["ny_date"] < f_start]
            if len(trail) < MIN_TRAILING:
                continue
            exp = sum(t["pnl"] for t in trail) / len(trail)
            if best_exp is None or exp > best_exp:
                best, best_exp = name, exp
        sel = best if best is not None else DEFAULT_KEY
        fold_trades = [dict(t, cfg=sel) for t in runs[sel]
                       if f_start <= t["ny_date"] < f_end]
        wf_trades.extend(fold_trades)
        fp = sum(t["pnl"] for t in fold_trades)
        wf_rows.append({"fold": f"{fy}-{fm:02d}..{ty}-{tm:02d}", "selected": sel,
                        "trades": len(fold_trades), "pnl": round(fp, 2),
                        "expectancy": round(fp / len(fold_trades), 3)
                        if fold_trades else None})
        print(f"  {fy}-{fm:02d}..{ty}-{tm:02d}: sel={sel:>10}  n={len(fold_trades):>4}"
              f"  pnl={fp:+9.2f}")

    m = metrics(wf_trades, session_days, wf_start, wf_end)
    per_year = defaultdict(lambda: [0, 0.0])
    for t in wf_trades:
        y = t["ny_date"] // 10000
        per_year[y][0] += 1
        per_year[y][1] += t["pnl"]

    print("\n== WALK-FORWARD RESULT ==")
    print(f"  trades={m['trades']}  total={m['total']:+.2f}  expt={m['expt']:+.3f}"
          f"  sharpe={m['sharpe']}  maxdd={m['maxdd']}")
    print(f"  informational: expt - 0.4pt spread = "
          f"{None if m['expt'] is None else round(m['expt'] - 0.4, 3)}; "
          f"- 0.8pt = {None if m['expt'] is None else round(m['expt'] - 0.8, 3)}")
    print("  per-year:")
    for y in sorted(per_year):
        print(f"    {y}: n={per_year[y][0]:>4}  pnl={per_year[y][1]:+9.2f}")

    years_pos = sum(1 for y in per_year if per_year[y][1] > 0)
    wf_total = m["total"]
    best_year_share = (max(v[1] for v in per_year.values()) / wf_total * 100.0) \
        if wf_total > 0 else None
    g1 = m["expt"] is not None and m["expt"] > 0 and m["sharpe"] is not None and m["sharpe"] > 0
    g2 = m["trades"] >= 150
    g3 = (years_pos >= 4 and best_year_share is not None and best_year_share < 60.0)

    print("\n== GATES (pre-declared in GATE.md) ==")
    print(f"  G1 expt>0 & sharpe>0 : {m['expt']} / {m['sharpe']} -> {'PASS' if g1 else 'FAIL'}")
    print(f"  G2 trades>=150      : {m['trades']} -> {'PASS' if g2 else 'FAIL'}")
    print(f"  G3 >=4/5y pos & best<60%: {years_pos}/5, "
          f"best-share={'n/a' if best_year_share is None else f'{best_year_share:.1f}%'} "
          f"-> {'PASS' if g3 else 'FAIL'}")
    verdict = "CONFIRMED — proceed to Step 4 (EA)" if (g1 and g2 and g3) else \
              "NOT CONFIRMED — stop (no EA)"
    print(f"  VERDICT: {verdict}")

    # ---- artifacts ---------------------------------------------------------
    with open(OUT / "wf_folds.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["fold", "selected", "trades", "pnl",
                                           "expectancy"])
        w.writeheader()
        w.writerows(wf_rows)
    with open(OUT / "wf_trades.csv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["ticket", "open_time", "close_time", "direction", "entry",
                    "exit_price", "exit_reason", "pnl", "cfg"])
        for i, t in enumerate(wf_trades, 1):
            w.writerow([i, fts(t["open_ts"]), fts(t["close_ts"]),
                        "BUY" if t["dir"] == "up" else "SELL",
                        f"{t['entry']:.2f}", f"{t['exit']:.2f}", t["reason"],
                        f"{t['pnl']:.2f}", t["cfg"]])
    summary = {
        "candidates": {k: {"stop_pct": v[0], "target_pct": v[1]}
                       for k, v in CANDIDATES.items()},
        "in_sample_full_range": insample,
        "walk_forward": {
            "folds": wf_rows, **m,
            "per_year": {str(y): {"trades": v[0], "pnl": round(v[1], 2)}
                         for y, v in sorted(per_year.items())},
            "expt_net_0_4pt": None if m["expt"] is None else round(m["expt"] - 0.4, 3),
            "expt_net_0_8pt": None if m["expt"] is None else round(m["expt"] - 0.8, 3),
        },
        "gates": {"G1": g1, "G2": g2, "G3": g3, "verdict": verdict},
    }
    with open(OUT / "wf_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {OUT / 'wf_folds.csv'}, wf_trades.csv, wf_summary.json")


if __name__ == "__main__":
    main()
