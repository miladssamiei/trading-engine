# -*- coding: utf-8 -*-
"""
fvg_continuation_step3.py - Step 3 walk-forward for the FVG
retracement-continuation setup (hypothesis #10), per
data/results/fvg_continuation_signal_quality/GATE.md (S1-S4 pre-declared).

Fixed rules only (no parameter selection): swing SL, fixed 1:1 TP,
2 trades/day cap, session window. Occupancy-modeled re-simulation on the
verified M5 tape (one position at a time; overnight carries until
SL/TP). Costs 0.8 pt round trip. WF folds 2022H1..2026H1 (2021 burn-in)
are a stability check (nothing is trained).

Outputs (data/results/fvg_continuation_rr/):
  wf_trades.csv, wf_yearly.csv, wf_summary.json
"""

import csv
import json
import math
import time as timemod
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\fvg_continuation_rr")
M5 = Path(r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present\us500tick_m5.csv")

NY_LO, NY_HI = 570, 660
RISK_LO, RISK_HI = 0.14, 0.56
UNTOUCHED_EXPIRY_S = 30 * 86400
ARMED_EXPIRY_BARS = 576
COST_PT = 0.8
CAP_PER_DAY = 2
SQ252 = math.sqrt(252)

FOLD_BOUNDS = [(2022, 1), (2022, 7), (2023, 1), (2023, 7), (2024, 1),
               (2024, 7), (2025, 1), (2025, 7), (2026, 1), (2026, 5)]


def extract_raw_signals(m5):
    """Identical machinery to fvg_continuation_phase1.py; returns raw
    in-window trigger records (k, dir, swing, risk band applied at fill)."""
    t = m5["time"].to_numpy(np.int64)
    h = m5["high"].to_numpy(float)
    l = m5["low"].to_numpy(float)
    c = m5["close"].to_numpy(float)
    n = len(m5)
    m5b = m5.copy()
    m5b["bk"] = m5b["time"] // 900
    g5 = m5b.groupby("bk")
    cnt5 = g5.size()
    m15 = pd.DataFrame({"time": cnt5.index * 900,
                        "high": g5["high"].max(), "low": g5["low"].min()})
    t15 = m15["time"].to_numpy(np.int64)
    h15 = m15["high"].to_numpy(float)
    l15 = m15["low"].to_numpy(float)
    n15 = len(m15)
    span_hi = np.full(n15, -1e18)
    span_lo = np.full(n15, 1e18)
    j0 = np.searchsorted(t, t15)
    j1 = np.searchsorted(t, t15 + 900)
    for i in range(n15):
        if j1[i] > j0[i]:
            span_hi[i] = h[j0[i]:j1[i]].max()
            span_lo[i] = l[j0[i]:j1[i]].min()
    fvg_events = []
    for i in range(2, n15):
        if h15[i - 2] < l15[i]:
            fvg_events.append((t15[i] + 900, i, 1, h15[i - 2], l15[i]))
        if l15[i - 2] > h15[i]:
            fvg_events.append((t15[i] + 900, i, -1, h15[i], l15[i - 2]))
    fvg_events.sort(key=lambda x: x[0])
    idx = pd.DatetimeIndex(pd.to_datetime(t, unit="s", utc=True))
    ny = idx.tz_convert("America/New_York")
    ny_min = (ny.hour * 60 + ny.minute).to_numpy()
    raw = []
    waiting, armed = [], []
    ei = 0
    for k in range(n):
        tk = t[k]
        while ei < len(fvg_events) and fvg_events[ei][0] <= tk:
            ct, i, d, zb, zt = fvg_events[ei]
            ei += 1
            if d == 1:
                waiting.append({"d": d, "zbot": zb, "ztop": zt, "ct": ct,
                                "rmax": span_hi[i]})
            else:
                waiting.append({"d": d, "zbot": zb, "ztop": zt, "ct": ct,
                                "rmin": span_lo[i]})
        hk, lk, ck = h[k], l[k], c[k]
        still_w = []
        for f in waiting:
            if tk - f["ct"] > UNTOUCHED_EXPIRY_S:
                continue
            if f["d"] == 1:
                if lk <= f["zbot"]:
                    continue
                if lk <= f["ztop"]:
                    f["arm_k"], f["swing"], f["touch_t"] = k, lk, tk
                    armed.append(f)
                    continue
                f["rmax"] = max(f["rmax"], hk)
            else:
                if hk >= f["ztop"]:
                    continue
                if hk >= f["zbot"]:
                    f["arm_k"], f["swing"], f["touch_t"] = k, hk, tk
                    armed.append(f)
                    continue
                f["rmin"] = min(f["rmin"], lk)
            still_w.append(f)
        waiting = still_w
        still_a = []
        for f in armed:
            if k - f["arm_k"] > ARMED_EXPIRY_BARS:
                continue
            if f["d"] == 1:
                if lk <= f["zbot"]:
                    continue
                if ck > f["rmax"]:
                    if NY_LO <= ny_min[k] < NY_HI:
                        raw.append({"k": k, "d": 1,
                                    "swing": min(f["swing"], lk),
                                    "touch_t": f["touch_t"]})
                    continue
                f["rmax"] = max(f["rmax"], hk)
                f["swing"] = min(f["swing"], lk)
            else:
                if hk >= f["ztop"]:
                    continue
                if ck < f["rmin"]:
                    if NY_LO <= ny_min[k] < NY_HI:
                        raw.append({"k": k, "d": -1,
                                    "swing": max(f["swing"], hk),
                                    "touch_t": f["touch_t"]})
                    continue
                f["rmin"] = min(f["rmin"], lk)
                f["swing"] = max(f["swing"], hk)
            still_a.append(f)
        armed = still_a
    return raw


def main():
    t0 = timemod.time()
    m5 = pd.read_csv(M5)
    t = m5["time"].to_numpy(np.int64)
    o = m5["open"].to_numpy(float)
    h = m5["high"].to_numpy(float)
    l = m5["low"].to_numpy(float)
    c = m5["close"].to_numpy(float)
    n = len(m5)
    idx = pd.DatetimeIndex(pd.to_datetime(t, unit="s", utc=True))
    ny = idx.tz_convert("America/New_York")
    ny_date = (ny.year * 10000 + ny.month * 100 + ny.day).to_numpy()
    ny_year = ny.year.to_numpy()

    raw = extract_raw_signals(m5)
    print(f"raw in-window triggers: {len(raw)} ({timemod.time() - t0:.0f}s)")
    by_k = defaultdict(list)
    for r in raw:
        by_k[r["k"]].append(r)

    # ---------------- occupancy sim ----------------
    trades = []
    tallies = defaultdict(int)
    pos = None
    pending = None          # signal triggered at bar k, fills at k+1 open
    day_entries = defaultdict(int)
    cur_day = None
    for k in range(n):
        # fills at this bar's open
        if pending is not None and pending["k"] + 1 == k:
            if pos is None:
                entry = round(o[k], 2)
                d = pending["d"]
                swing = pending["swing"]
                risk = ((entry - swing) if d == 1 else (swing - entry))
                risk_pct = risk / entry * 100.0
                if RISK_LO <= risk_pct <= RISK_HI and day_entries[ny_date[k]] < CAP_PER_DAY:
                    sl = round(swing, 2)
                    tp = round(entry + d * risk, 2)
                    pos = {"d": d, "entry": entry, "sl": sl, "tp": tp,
                           "k0": k, "nd": int(ny_date[k])}
                    day_entries[ny_date[k]] += 1
                    tallies["entered"] += 1
                else:
                    tallies["risk_out_of_band_or_cap"] += 1
            else:
                tallies["skipped_in_position"] += 1
            pending = None
        # exits during bar k
        if pos is not None:
            d = pos["d"]
            exit_px, reason = None, None
            entry_bar = (k == pos["k0"])
            if d == 1:
                if o[k] <= pos["sl"]:
                    exit_px, reason = o[k], "sl_gap"
                elif o[k] >= pos["tp"]:
                    exit_px, reason = o[k], "tp_gap"
                elif l[k] <= pos["sl"]:
                    exit_px, reason = pos["sl"], "sl"
                elif entry_bar:
                    # entry bar: TP only on close beyond (conservative)
                    if c[k] >= pos["tp"]:
                        exit_px, reason = pos["tp"], "tp"
                elif h[k] >= pos["tp"]:
                    exit_px, reason = pos["tp"], "tp"
            else:
                if o[k] >= pos["sl"]:
                    exit_px, reason = o[k], "sl_gap"
                elif o[k] <= pos["tp"]:
                    exit_px, reason = o[k], "tp_gap"
                elif h[k] >= pos["sl"]:
                    exit_px, reason = pos["sl"], "sl"
                elif entry_bar:
                    if c[k] <= pos["tp"]:
                        exit_px, reason = pos["tp"], "tp"
                elif l[k] <= pos["tp"]:
                    exit_px, reason = pos["tp"], "tp"
            if exit_px is not None:
                gross = ((exit_px - pos["entry"]) if d == 1
                         else (pos["entry"] - exit_px))
                trades.append({"nd": pos["nd"], "year": int(str(pos["nd"])[:4]),
                               "dir": d, "entry": pos["entry"],
                               "exit": round(exit_px, 2), "reason": reason,
                               "gross": round(gross, 2),
                               "net": round(gross - COST_PT, 2),
                               "entry_t": int(t[pos["k0"]]),
                               "exit_t": int(t[k]),
                               "bars_held": k - pos["k0"]})
                if t[k] - t[pos["k0"]] > 3600 * 4:
                    tallies["overnight_or_long_hold"] += 1
                pos = None
        # new signal at bar k
        if k in by_k and pos is None and pending is None:
            for r in by_k[k]:
                pending = r
                break
    print(f"sim done: {len(trades)} trades; tallies {dict(tallies)}")

    # ---------------- stats + gates ----------------
    days_rth = sorted(set(ny_date[(ny.hour * 60 + ny.minute >= 570)
                                  & (ny.hour * 60 + ny.minute < 960)]))
    per_day = defaultdict(float)
    for tr in trades:
        per_day[tr["nd"]] += tr["net"]
    daily = np.array([per_day.get(d, 0.0) for d in days_rth])
    tot_net = float(daily.sum())
    expt = tot_net / len(trades) if trades else None
    sharpe = (daily.mean() / daily.std(ddof=1) * SQ252
              if daily.std(ddof=1) > 0 else None)
    per_year = defaultdict(float)
    yr_n = defaultdict(int)
    for tr in trades:
        per_year[tr["year"]] += tr["net"]
        yr_n[tr["year"]] += 1
    years_full = [y for y in sorted(per_year) if y < 2026]
    n_pos_years = sum(1 for y in years_full if per_year[y] > 0)
    best_share = (max(per_year[y] for y in years_full) / tot_net * 100
                  if tot_net > 0 and years_full else None)

    wf_folds = []
    for fi in range(len(FOLD_BOUNDS) - 1):
        (fy, fm), (ty, tm) = FOLD_BOUNDS[fi], FOLD_BOUNDS[fi + 1]
        lo, hi = fy * 10000 + fm * 100, ty * 10000 + tm * 100
        fold_tr = [tr for tr in trades if lo <= tr["nd"] < hi]
        pnl = sum(tr["net"] for tr in fold_tr)
        fd = [d for d in days_rth if lo <= d < hi]
        fday = np.array([per_day.get(d, 0.0) for d in fd])
        fsh = (fday.mean() / fday.std(ddof=1) * SQ252
               if len(fd) > 1 and fday.std(ddof=1) > 0 else None)
        wf_folds.append({"fold": f"{fy}H{1 if fm == 1 else 2}",
                         "trades": len(fold_tr),
                         "net": round(pnl, 2),
                         "expt": round(pnl / len(fold_tr), 3) if fold_tr else None,
                         "sharpe": round(fsh, 3) if fsh is not None else None})

    s1 = expt is not None and expt > 0 and sharpe is not None and sharpe > 0
    s2 = len(trades) >= 150
    s3 = (n_pos_years >= 4 and best_share is not None and best_share <= 60)
    verdict = ("PASS -> Step 4 (EA build) + robustness arms"
               if (s1 and s2 and s3) else "FAIL -> STOP (null result)")
    print(f"\nnet total {tot_net:+.2f} pts, expt {expt:+.3f}, "
          f"Sharpe {sharpe:.3f}, trades {len(trades)}")
    print(f"years net: {dict(per_year)}; positive full years "
          f"{n_pos_years}/{len(years_full)}; best-year share {best_share}%")
    print(f"S1={s1} S2={s2} S3={s3} -> {verdict}")

    pd.DataFrame(trades).to_csv(BASE / "wf_trades.csv", index=False)
    with open(BASE / "wf_yearly.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["year", "trades", "net"])
        for y in sorted(per_year):
            w.writerow([y, yr_n[y], round(per_year[y], 2)])
    with open(BASE / "wf_folds.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(wf_folds[0].keys()))
        w.writeheader()
        w.writerows(wf_folds)
    summary = {"n_trades": len(trades), "net_total": round(tot_net, 2),
               "expt_net": round(expt, 3) if expt else None,
               "sharpe_net": round(sharpe, 3) if sharpe else None,
               "per_year": {str(k): round(v, 2) for k, v in sorted(per_year.items())},
               "n_pos_full_years": f"{n_pos_years}/{len(years_full)}",
               "best_year_share_pct": round(best_share, 1) if best_share else None,
               "gates": {"S1": bool(s1), "S2": bool(s2), "S3": bool(s3)},
               "verdict": verdict, "folds": wf_folds,
               "tallies": dict(tallies),
               "runtime_s": round(timemod.time() - t0, 1)}
    with open(BASE / "wf_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote wf artifacts ({summary['runtime_s']}s)")


if __name__ == "__main__":
    main()
