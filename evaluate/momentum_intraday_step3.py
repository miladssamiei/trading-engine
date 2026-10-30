# -*- coding: utf-8 -*-
"""
momentum_intraday_step3.py — Step 3 for the intraday multi-horizon trend
alignment hypothesis (project-original; NOT a test of Lempérière et al. or
the video rubric).

Protocol pre-declared in data/results/momentum_intraday_rr/GATE-STEP3.md:
score4 (30m/1h/2h/4h) state-change entries on M5, NY session, occupancy sim
(1 lot, gross), grid SL x TP_R, 9 half-year walk-forward folds with
trailing-expectancy selection, gates G1-G5 incl. the day-block bootstrap.

Outputs (data/results/momentum_intraday_rr/):
  grid_insample.csv, wf_trades.csv, wf_summary.json
"""

import json
import time as timemod
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\momentum_intraday_rr")
M5 = Path(r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present\us500tick_m5.csv")

HORIZONS = [1800, 3600, 7200, 14400]                 # 30m 1h 2h 4h
SL_GRID = (0.15, 0.25, 0.40, 0.60)                   # % of entry
TP_GRID = (0.5, 0.75, 1.0, 1.5, 2.0)
DEFAULT_SL, DEFAULT_TP = 0.25, 1.0
MIN_TRAILING = 20
ENTRY_LO, ENTRY_HI = 570, 930                        # 09:30..15:30 NY open-time
FLAT_MIN = 955                                       # 15:55 NY
SESSION_GAP = 3600
FOLD_BOUNDS = [(2022, 1), (2022, 7), (2023, 1), (2023, 7), (2024, 1),
               (2024, 7), (2025, 1), (2025, 7), (2026, 1), (2026, 5)]
SPREAD_RT = 0.8                                      # pts round trip (informational)


def fold_of(y, m):
    for k in range(len(FOLD_BOUNDS) - 1):
        (fy, fm), (ty, tm) = FOLD_BOUNDS[k], FOLD_BOUNDS[k + 1]
        if (y, m) >= (fy, fm) and (y, m) < (ty, tm):
            return k
    return None


def simulate(ctx, sl_pct, tp_r, conv, th_abs):
    """Occupancy state machine. Returns trades (gross, pnl in pts and %)."""
    t, o, h, l, c = ctx["t"], ctx["o"], ctx["h"], ctx["l"], ctx["c"]
    score4, ny_min, ny_yr, ny_date, sess_end = (ctx["score4"], ctx["ny_min"],
                                                ctx["ny_year"], ctx["ny_date"],
                                                ctx["sess_end"])
    n = len(c)
    trades = []
    pos = 0
    d = 0
    entry = stop = tgt = 0.0
    e_idx = -1
    prev_mag = False                                   # |score4| >= th at prev bar
    for i in range(n):
        s4 = int(score4[i])
        mag = abs(s4) >= th_abs
        if pos == 0:
            wm = ny_min[i]
            if (not prev_mag) and mag and wm is not None \
                    and ENTRY_LO <= wm <= ENTRY_HI and i + 1 <= sess_end[i]:
                d = 1 if s4 > 0 else -1
                entry = float(o[i + 1]) if conv == "next" else float(c[i])
                stop = entry - d * entry * sl_pct / 100.0
                tgt = entry + d * entry * sl_pct * tp_r / 100.0
                e_idx = i + 1
                pos = d
                prev_mag = mag
                continue
            prev_mag = mag
            continue

        # ---- in position at bar i ----
        wm = ny_min[i]
        if i == sess_end[i]:                            # tape session end (half days)
            px = float(c[i])
            trades.append(mk(i, d, entry, px, "session_end", ny_yr, ny_date, t))
            pos = 0
            prev_mag = mag
            continue
        if wm is not None and wm >= FLAT_MIN:           # forced flat at 15:55 open
            px = float(o[i])
            trades.append(mk(i, d, entry, px, "flat", ny_yr, ny_date, t))
            pos = 0
            prev_mag = mag
            continue
        if i >= e_idx:                                  # entry bar (next) or later
            hi, lo = float(h[i]), float(l[i])
            stop_hit = (lo <= stop) if d == 1 else (hi >= stop)
            tgt_hit = (hi >= tgt) if d == 1 else (lo <= tgt)
            if stop_hit:                                # pessimistic priority
                trades.append(mk(i, d, entry, stop, "sl", ny_yr, ny_date, t))
                pos = 0
                prev_mag = mag
                continue
            if tgt_hit:
                trades.append(mk(i, d, entry, tgt, "tp", ny_yr, ny_date, t))
                pos = 0
                prev_mag = mag
                continue
            if not mag:                                 # signal exit
                if i + 1 <= sess_end[i] and conv == "next":
                    px = float(o[i + 1])
                    trades.append(mk(i + 1, d, entry, px, "sig", ny_yr, ny_date, t))
                else:
                    px = float(c[i])
                    trades.append(mk(i, d, entry, px, "sig", ny_yr, ny_date, t))
                pos = 0
                prev_mag = mag
                continue
        prev_mag = mag
    return trades


def mk(i, d, entry, px, reason, ny_yr, ny_date, t):
    return {"dir": d, "entry": entry, "exit": px, "reason": reason,
            "pnl_pts": (px - entry) * d,
            "pnl_pct": (px - entry) / entry * 100.0 * d,
            "nd": int(ny_date[i]), "year": int(ny_yr[i]), "time": int(t[i]),
            "bars": i}


def eval_trades(trs, day_from=None, day_to=None):
    sel = [x for x in trs if (day_from is None or x["nd"] >= day_from)
           and (day_to is None or x["nd"] < day_to)]
    if not sel:
        return {"n": 0}
    pts = [x["pnl_pts"] for x in sel]
    per_day = defaultdict(float)
    for x in sel:
        per_day[x["nd"]] += x["pnl_pts"]
    days = sorted(per_day)
    vals = np.array([per_day[d] for d in days])
    sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    sharpe = float(vals.mean() / sd * np.sqrt(252)) if sd > 0 else None
    years = defaultdict(float)
    for x in sel:
        years[x["year"]] += x["pnl_pts"]
    total = float(sum(pts))
    return {"n": len(sel), "total_pts": total,
            "expt": total / len(sel), "sharpe": sharpe,
            "per_year": dict(sorted(years.items())),
            "n_days": len(days),
            "per_day_pnl": {d: per_day[d] for d in days}}


def day_bootstrap(per_day_pnl, n_boot=10000, seed=7):
    vals = np.array(list(per_day_pnl.values()))
    rng = np.random.default_rng(seed)
    tot = rng.choice(vals, size=(n_boot, len(vals)), replace=True).sum(axis=1)
    return {"p05": float(np.percentile(tot, 5)),
            "mean": float(tot.mean()),
            "p_le0": float((tot <= 0).mean())}


def main():
    t0 = timemod.time()
    df = pd.read_csv(M5)
    t = df["time"].to_numpy(dtype=np.int64)
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)

    sess = np.zeros(n, dtype=np.int64)
    sess[1:] = np.cumsum(np.diff(t) > SESSION_GAP)
    sess_end = np.zeros(n, dtype=np.int64)
    last = n - 1
    for i in range(n - 1, -1, -1):
        if i == n - 1 or sess[i] != sess[i + 1]:
            last = i
        sess_end[i] = last

    idx = pd.DatetimeIndex(pd.to_datetime(t, unit="s", utc=True))
    ny = idx.tz_convert("America/New_York")
    ny_min_arr = (ny.hour * 60 + ny.minute).to_numpy()
    ny_year = ny.year.to_numpy()
    ny_date = (ny.year * 10000 + ny.month * 100 + ny.day).to_numpy()

    score4 = np.zeros(n, dtype=np.int8)
    for hs in HORIZONS:
        j = np.searchsorted(t, t - hs, side="right") - 1
        ok = j >= 0
        d = np.zeros(n, dtype=np.int8)
        d[ok] = np.sign(c[ok] - c[j[ok]])
        score4 += d

    ctx = {"t": t, "o": o, "h": h, "l": l, "c": c, "score4": score4,
           "ny_min": ny_min_arr, "ny_year": ny_year, "ny_date": ny_date,
           "sess_end": sess_end}
    print(f"ctx ready ({timemod.time() - t0:.0f}s); simulating grid...")

    wf_start = FOLD_BOUNDS[0][0] * 10000 + FOLD_BOUNDS[0][1] * 100
    wf_end = FOLD_BOUNDS[-1][0] * 10000 + FOLD_BOUNDS[-1][1] * 100
    grid_rows = []
    runs = {}
    for th in (4, 2):
        for conv in ("next", "cls"):
            for sl in SL_GRID:
                for tp_r in TP_GRID:
                    trs = simulate(ctx, sl, tp_r, conv, th)
                    runs[(th, conv, sl, tp_r)] = trs
                    m = eval_trades(trs)
                    grid_rows.append({
                        "th": th, "conv": conv, "sl_pct": sl, "tp_r": tp_r,
                        "n": m.get("n", 0), "total_pts": round(m.get("total_pts", 0), 1),
                        "expt_pts": round(m.get("expt", 0), 4),
                        "sharpe": None if m.get("sharpe") is None else round(m["sharpe"], 3),
                        **{"y" + str(k): round(v, 1) for k, v in m.get("per_year", {}).items()}})
    pd.DataFrame(grid_rows).to_csv(BASE / "grid_insample.csv", index=False)
    print(f"grid done ({timemod.time() - t0:.0f}s); walk-forward...")

    summary = {}
    for th in (4, 2):
        for conv in ("next", "cls"):
            wf_trades = []
            fold_log = []
            for k in range(len(FOLD_BOUNDS) - 1):
                (fy, fm), (ty, tm) = FOLD_BOUNDS[k], FOLD_BOUNDS[k + 1]
                f_start, f_end = fy * 10000 + fm * 100, ty * 10000 + tm * 100
                best, best_exp, best_n = (DEFAULT_SL, DEFAULT_TP), None, 0
                for sl in SL_GRID:
                    for tp_r in TP_GRID:
                        trail = [x for x in runs[(th, conv, sl, tp_r)]
                                 if x["nd"] < f_start]
                        if len(trail) < MIN_TRAILING:
                            continue
                        exp = sum(x["pnl_pts"] for x in trail) / len(trail)
                        if best_exp is None or exp > best_exp:
                            best, best_exp, best_n = (sl, tp_r), exp, len(trail)
                sl, tp_r = best
                fold_tr = [x for x in runs[(th, conv, sl, tp_r)]
                           if f_start <= x["nd"] < f_end]
                for x in fold_tr:
                    x = dict(x)
                    x["fold"] = k
                    x["sel_sl"], x["sel_tp"] = sl, tp_r
                    wf_trades.append(x)
                fold_log.append({"fold": k, "sl": sl, "tp_r": tp_r,
                                 "trail_expt": None if best_exp is None else round(best_exp, 4),
                                 "trail_n": best_n, "n": len(fold_tr),
                                 "pnl": round(sum(x["pnl_pts"] for x in fold_tr), 1)})
            m = eval_trades(wf_trades)
            boot = day_bootstrap(m["per_day_pnl"]) if m.get("n") else {}
            years = m.get("per_year", {})
            wf_years = {y: v for y, v in years.items() if y >= 2022}
            pos_years = sum(1 for v in wf_years.values() if v > 0)
            share = (max(wf_years.values()) / sum(wf_years.values())
                     if wf_years and sum(wf_years.values()) > 0 else None)
            expt_net = (m.get("expt", 0) - SPREAD_RT) if m.get("n") else None
            summ = {
                "trades": m.get("n", 0), "total_pts": round(m.get("total_pts", 0), 1),
                "expt_pts": round(m.get("expt", 0), 4),
                "expt_net_of_spread": round(expt_net, 4) if expt_net is not None else None,
                "sharpe": None if m.get("sharpe") is None else round(m["sharpe"], 3),
                "per_year": {str(k): round(v, 1) for k, v in wf_years.items()},
                "g1": bool(m.get("n") and m["expt"] > 0 and m["sharpe"] and m["sharpe"] > 0),
                "g2": bool(m.get("n", 0) >= 150),
                "g3": bool(pos_years >= 4 and (share is None or share <= 0.60)
                           and sum(wf_years.values()) > 0),
                "g4_boot": {kk: round(v, 2) for kk, v in boot.items()} if boot else {},
                "g4": bool(boot and boot["p05"] > 0),
                "folds": fold_log,
            }
            summ["pass_all"] = summ["g1"] and summ["g2"] and summ["g3"] and summ["g4"]
            summary[f"th{th}_{conv}"] = summ
            print(f"th{th}/{conv}: n={summ['trades']} total={summ['total_pts']} "
                  f"expt={summ['expt_pts']} (net {summ['expt_net_of_spread']}) "
                  f"Sh={summ['sharpe']} per_year={summ['per_year']} "
                  f"G1={summ['g1']} G2={summ['g2']} G3={summ['g3']} "
                  f"G4={summ['g4']} ({summ['g4_boot']}) PASS={summ['pass_all']}")

    # primary detail: TH4 next WF trades
    wf_trades = []
    for k in range(len(FOLD_BOUNDS) - 1):
        (fy, fm), (ty, tm) = FOLD_BOUNDS[k], FOLD_BOUNDS[k + 1]
        f_start, f_end = fy * 10000 + fm * 100, ty * 10000 + tm * 100
        best, best_exp = (DEFAULT_SL, DEFAULT_TP), None
        for sl in SL_GRID:
            for tp_r in TP_GRID:
                trail = [x for x in runs[(4, "next", sl, tp_r)] if x["nd"] < f_start]
                if len(trail) < MIN_TRAILING:
                    continue
                exp = sum(x["pnl_pts"] for x in trail) / len(trail)
                if best_exp is None or exp > best_exp:
                    best, best_exp = (sl, tp_r), exp
        sl, tp_r = best
        for x in runs[(4, "next", sl, tp_r)]:
            if f_start <= x["nd"] < f_end:
                wf_trades.append({**x, "fold": k, "sel_sl": sl, "sel_tp": tp_r})
    pd.DataFrame(wf_trades).to_csv(BASE / "wf_trades.csv", index=False)

    verdict = ("PASS -> Step 4 (EA)" if summary["th4_next"]["pass_all"]
               else "FAIL -> STOP (null result, report)")
    with open(BASE / "wf_summary.json", "w") as fh:
        json.dump({"summary": summary, "verdict": verdict,
                   "params": {"SL_GRID": SL_GRID, "TP_GRID": TP_GRID,
                              "FOLD_BOUNDS": FOLD_BOUNDS, "MIN_TRAILING": MIN_TRAILING,
                              "DEFAULTS": [DEFAULT_SL, DEFAULT_TP],
                              "SPREAD_RT_PTS": SPREAD_RT}},
                  fh, indent=2)
    print("\nVERDICT:", verdict)
    print(f"wrote grid_insample.csv, wf_trades.csv, wf_summary.json "
          f"({timemod.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
