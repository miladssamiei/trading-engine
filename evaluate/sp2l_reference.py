# -*- coding: utf-8 -*-
"""
sp2l_reference.py — standalone Python reference for the SP2L spike/gap
strategy, WALK-FORWARD-CONFIRMED configuration NY_E (NY session 09:30-15:30
entry window, EMA60 filter, TP_R = 2.0 — selected by trailing expectancy in
ALL 9 walk-forward folds), E-NEXT entry convention (signal on completed M1
bar u -> market fill at open of bar u+1; SL/TP from the actual entry
reference price), 2021-01..2026-04 on the verified US500-tick M1 export.

Standalone by design (no imports from sp2l_phase1/sp2l_step3): it is the
independent reimplementation the MT5 EA is reconciled against, and doubles
as a cross-check of the Step-3 occupancy engine for this configuration.

Rule (SPEC.md data/results/sp2l_ablation/SPEC.md; params from
step3/wf_folds.csv): spike/gap/continuation setup on closed M1 bars,
pending -> entry on first subsequent break bar passing risk-cap + EMA
filters inside the NY window; SL = pre-spike extreme (normalized to the
0.01 grid); TP = entry +/- 2.0 x risk; forced exit at the 15:55 NY bar
open; one position at a time (no setup detection while holding; pending
expires at NY date end). Exits bar-path engine, SL-first on same-bar
both-level ambiguity (conservative; the MT5 tester resolves by tick order
— expected, documented reconciliation class).

Outputs: data/results/sp2l_2021_2026/py_trades.csv, py_days.csv
"""

import csv
import datetime as dt
import math
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\sp2l_2021_2026")
M1_CSV = Path(r"C:\Users\Milad\Desktop\RR\data\results\vwap_2021_2026\us500tick_m1.csv")

NY = ZoneInfo("America/New_York")
UTC = dt.timezone.utc

SPIKE_RATIO = 1.5
PGAP_PCT = 0.04
MAX_SL_PCT = 0.40
EMA_PERIOD = 60
TP_R = 2.0

SESSION_FIRST_M = 570          # 09:30 NY — entry window open
SESSION_LAST_ENTRY_M = 930     # 15:30 NY — last eligible SIGNAL bar
FORCE_EXIT_M = 955             # 15:55 NY — forced exit at this bar's open


def norm(p):
    return round(round(p / 0.01) * 0.01, 2)


def fts(t):
    return dt.datetime.fromtimestamp(int(t), tz=UTC).strftime("%Y.%m.%d %H:%M")


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(M1_CSV)
    t = pd.to_datetime(df["time"], unit="s", utc=True)
    ny = t.dt.tz_convert(NY)
    wall = (ny.dt.hour * 60 + ny.dt.minute).to_numpy(np.int32)
    nyd = (ny.dt.year * 10000 + ny.dt.month * 100 + ny.dt.day).to_numpy(np.int64)
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    ts = df["time"].to_numpy(np.int64)
    ema = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean().to_numpy(float)
    N = len(df)
    print(f"bars: {N}  range {fts(ts[0])} .. {fts(ts[-1])} UTC")

    # ---- setup detection (vectorized, SPEC §3) ------------------------
    body = c - o
    idx = np.arange(3, N)
    t1, t2, t3 = idx - 1, idx - 2, idx - 3
    g = PGAP_PCT / 100.0
    buy = (
        (l[idx] < l[t1])
        & (c[t1] > c[t2]) & (o[t1] > o[t2])
        & (c[t2] > c[t3]) & (o[t2] > o[t3])
        & (c[t1] > o[t1]) & (c[t2] > o[t2]) & (c[t3] > o[t3])
        & (l[t1] > h[t3] * (1.0 + g))
        & (body[t2] > SPIKE_RATIO * body[t1])
        & (body[t2] > SPIKE_RATIO * body[t3])
        & (body[t2] > SPIKE_RATIO * body[idx])
    )
    sell = (
        (h[idx] > h[t1])
        & (c[t1] < c[t2]) & (o[t1] < o[t2])
        & (c[t2] < c[t3]) & (o[t2] < o[t3])
        & (c[t1] < o[t1]) & (c[t2] < o[t2]) & (c[t3] < o[t3])
        & (h[t1] < l[t3] * (1.0 - g))
        & (-body[t2] > SPIKE_RATIO * -body[t1])
        & (-body[t2] > SPIKE_RATIO * -body[t3])
        & (-body[t2] > SPIKE_RATIO * -body[idx])
    )
    both = buy & sell
    buy &= ~both
    sell &= ~both
    setups = ([(int(i), 1, float(l[i - 3])) for i in idx[buy]]
              + [(int(i), -1, float(h[i - 3])) for i in idx[sell]])
    setups.sort()
    print(f"setups: {len(setups)} (buy {int(buy.sum())} / sell {int(sell.sum())},"
          f" both-dropped {int(both.sum())})")

    # ---- NY-date blocks ------------------------------------------------
    starts = np.flatnonzero(np.r_[True, nyd[1:] != nyd[:-1]])
    ends = np.r_[starts[1:], N]
    blk_of = np.repeat(np.arange(len(starts), dtype=np.int32), ends - starts)
    blk_end = {b: int(ends[b]) for b in range(len(starts))}
    blk_day = {b: int(nyd[int(starts[b])]) for b in range(len(starts))}
    bar_blk = blk_of

    # per-block: last bar opening before 15:55, and the 15:55 bar index
    walk_end = {}
    idx955 = {}
    for b in range(len(starts)):
        s, e = int(starts[b]), int(ends[b])
        w = wall[s:e]
        pre = np.nonzero(w < FORCE_EXIT_M)[0]
        walk_end[b] = s + int(pre[-1]) if pre.size else e - 1
        j = np.nonzero(w == FORCE_EXIT_M)[0]
        idx955[b] = s + int(j[0]) if j.size else None

    # ---- pending machine + occupancy + exits ---------------------------
    trades = []
    days = defaultdict(lambda: {"setups": 0, "entered": 0, "killed": 0,
                                "expired": 0, "exits_tp": 0, "exits_sl": 0,
                                "exits_time": 0, "pnl": 0.0})
    ns = len(setups)
    si = 0
    cur = -1
    while True:
        while si < ns and setups[si][0] <= cur:
            si += 1
        if si >= ns:
            break
        tidx, d, sl = setups[si]
        si += 1
        b = int(bar_blk[tidx])
        e0 = blk_end[b]
        day = blk_day[b]
        days[day]["setups"] += 1
        u = tidx + 1
        outcome = None
        while u < e0:
            w = int(wall[u])
            if w > SESSION_LAST_ENTRY_M:      # past window: linger to date end
                outcome = ("expired", e0 - 1)
                break
            if d == 1:
                if l[u] >= l[u - 1]:
                    u += 1
                    continue
                risk = l[u] - sl
                if risk <= 0:
                    u += 1
                    continue
                if risk > MAX_SL_PCT / 100.0 * l[u]:
                    outcome = ("killed", u)
                    break
            else:
                if h[u] <= h[u - 1]:
                    u += 1
                    continue
                risk = sl - h[u]
                if risk <= 0:
                    u += 1
                    continue
                if risk > MAX_SL_PCT / 100.0 * h[u]:
                    outcome = ("killed", u)
                    break
            # EMA filter (NY_E)
            if d == 1 and not (c[u] > ema[u]):
                u += 1
                continue
            if d == -1 and not (c[u] < ema[u]):
                u += 1
                continue
            if w < SESSION_FIRST_M:
                u += 1
                continue
            outcome = ("entered", u)
            break
        if outcome is None:
            outcome = ("expired", e0 - 1)
        kind, obar = outcome
        days[day][kind] += 1
        cur = obar
        if kind != "entered":
            continue

        # E-NEXT entry: market fill at open of the bar after the signal bar
        if u + 1 >= e0:
            cur = e0 - 1
            continue
        E = norm(float(o[u + 1]))
        sl_n = norm(sl)
        risk = (E - sl_n) if d == 1 else (sl_n - E)
        if risk <= 0:                         # gap through the stop
            continue
        tp = norm(E + d * TP_R * risk)
        fill = u + 1
        we_ = walk_end[b]
        i955 = idx955[b]

        # exit engine: fill bar first (fill at OPEN), then bar-by-bar
        exit_px = reason = None
        eb = fill
        if d == 1:
            if o[fill] <= sl_n:
                exit_px, reason = float(o[fill]), "stop_loss"
            elif o[fill] >= tp:
                exit_px, reason = float(o[fill]), "target"
            elif l[fill] <= sl_n and h[fill] >= tp:
                exit_px, reason = sl_n, "stop_loss"
            elif l[fill] <= sl_n:
                exit_px, reason = sl_n, "stop_loss"
            elif h[fill] >= tp:
                exit_px, reason = tp, "target"
        else:
            if o[fill] >= sl_n:
                exit_px, reason = float(o[fill]), "stop_loss"
            elif o[fill] <= tp:
                exit_px, reason = float(o[fill]), "target"
            elif h[fill] >= sl_n and l[fill] <= tp:
                exit_px, reason = sl_n, "stop_loss"
            elif h[fill] >= sl_n:
                exit_px, reason = sl_n, "stop_loss"
            elif l[fill] <= tp:
                exit_px, reason = tp, "target"
        if exit_px is None:
            eb = fill + 1
            while eb < e0:
                w = int(wall[eb])
                if w >= FORCE_EXIT_M:
                    exit_px, reason = float(o[eb]), "time_exit"
                    break
                if d == 1:
                    if o[eb] <= sl_n:
                        exit_px, reason = float(o[eb]), "stop_loss"
                    elif o[eb] >= tp:
                        exit_px, reason = float(o[eb]), "target"
                    elif l[eb] <= sl_n and h[eb] >= tp:
                        exit_px, reason = sl_n, "stop_loss"
                    elif l[eb] <= sl_n:
                        exit_px, reason = sl_n, "stop_loss"
                    elif h[eb] >= tp:
                        exit_px, reason = tp, "target"
                else:
                    if o[eb] >= sl_n:
                        exit_px, reason = float(o[eb]), "stop_loss"
                    elif o[eb] <= tp:
                        exit_px, reason = float(o[eb]), "target"
                    elif h[eb] >= sl_n and l[eb] <= tp:
                        exit_px, reason = sl_n, "stop_loss"
                    elif h[eb] >= sl_n:
                        exit_px, reason = sl_n, "stop_loss"
                    elif l[eb] <= tp:
                        exit_px, reason = tp, "target"
                if exit_px is not None:
                    break
                eb += 1
            if exit_px is None:               # early-close day: last bar close
                eb = e0 - 1
                exit_px, reason = float(c[eb]), "time_exit"
        pnl = round(d * (exit_px - E), 2)
        trades.append({"nd": day, "dir": d, "setup_ts": int(ts[tidx]),
                       "signal_ts": int(ts[u]), "open_ts": int(ts[fill]),
                       "close_ts": int(ts[eb]), "entry": E, "sl": sl_n,
                       "tp": tp, "exit": exit_px, "reason": reason,
                       "pnl": pnl})
        days[day]["entries"] = days[day].get("entries", 0) + 1
        days[day]["exits_tp" if reason == "target" else
                 ("exits_sl" if reason == "stop_loss" else "exits_time")] += 1
        days[day]["pnl"] += pnl
        cur = eb

    # ---- outputs --------------------------------------------------------
    with open(BASE / "py_trades.csv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["ticket", "open_time", "close_time", "type", "open_price",
                    "stop_loss", "take_profit", "close_price", "exit_reason",
                    "profit", "setup_time", "signal_time"])
        for i, tr in enumerate(trades, 1):
            w.writerow([i, fts(tr["open_ts"]), fts(tr["close_ts"]),
                        "BUY" if tr["dir"] == 1 else "SELL",
                        f"{tr['entry']:.2f}", f"{tr['sl']:.2f}",
                        f"{tr['tp']:.2f}", f"{tr['exit']:.2f}", tr["reason"],
                        f"{tr['pnl']:.2f}", fts(tr["setup_ts"]),
                        fts(tr["signal_ts"])])
    with open(BASE / "py_days.csv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["ny_date", "setups", "entered", "killed", "expired",
                    "entries", "exits_tp", "exits_sl", "exits_time",
                    "total_pnl"])
        for day in sorted(days):
            v = days[day]
            w.writerow([day, v["setups"], v["entered"], v["killed"],
                        v["expired"], v.get("entries", 0), v["exits_tp"],
                        v["exits_sl"], v["exits_time"], f"{v['pnl']:.2f}"])

    # ---- summary --------------------------------------------------------
    n = len(trades)
    wins = sum(1 for tr in trades if tr["pnl"] > 0)
    total = sum(tr["pnl"] for tr in trades)
    per_year = defaultdict(lambda: [0, 0.0])
    for tr in trades:
        per_year[tr["nd"] // 10000][0] += 1
        per_year[tr["nd"] // 10000][1] += tr["pnl"]
    reasons = defaultdict(int)
    for tr in trades:
        reasons[tr["reason"]] += 1
    print(f"\ntrades: {n}  winrate: {100.0 * wins / n:.2f}%  "
          f"total: {total:+.2f} pts  expt: {total / n:+.3f}")
    print("exit reasons:", dict(reasons))
    print("per-year:")
    for y in sorted(per_year):
        print(f"  {y}: n={per_year[y][0]:>4}  pnl={per_year[y][1]:+9.2f}")
    print(f"\nwrote {BASE / 'py_trades.csv'} and py_days.csv")


if __name__ == "__main__":
    main()
