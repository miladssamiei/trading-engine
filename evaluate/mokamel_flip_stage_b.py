# -*- coding: utf-8 -*-
"""
mokamel_flip_stage_b.py — Hypothesis #16 MOKAMEL_FLIP Stage B:
literal 40/80-point exit arm + non-binding unit ring + the SPEC sec 5
money-management day machine; GATE B1.

Implements SPEC_MOKAMEL_FLIP.txt sections 5/7 / GATE_MOKAMEL_FLIP.txt
exactly (frozen 2026-09-03 BEFORE Stage A ran; this script written
2026-09-03 after the Stage-A PASS and the task owner's explicit
go-ahead):
- BINDING arm: stop 40.00 pt / target 80.00 pt, full window
  2021-01..2026-04, exits on the M1 tape with the SP2L/Bollinger/
  scalping-screen conventions (first-crossing, SL-first same-bar
  ambiguity, gap fills at open, 15:55-bar-open forced exit, no
  overnight) — sim_m0 copied VERBATIM in its logic from
  evaluate/scalping_screen_stage_ab.py.
- Money machine: first trade AGGRESSIVE; any trade with realized
  points < 0 -> CONSERVATIVE (close-confirmed entries, I-6);
  day cumulative R >= 0 -> back to AGGRESSIVE; day closes at 3 total
  TPs or 3 total SLs (Q3); one position at a time (skipped signals
  are consumed); filter exits and forced exits are neutral (I-10);
  "realized net points" read literally as the trade's gross realized
  points (cost is a reporting layer, not part of the price path).
- Sizing: binding 2.5 lots ($100 risk / 40 pt); ring lots = 100/stop.
- Ring {20/40, 30/60, 50/100, 60/120} pt at constant 1:2 shape is
  NON-BINDING, non-gating, promotion forbidden (Q2/GATE).

Outputs (data/results/mokamel_flip/): matrix_master.csv,
trades_all_arms.csv, yearly_breakdown.csv, stage_b_stats.json.
Does NOT touch SPEC/GATE. Stage C is NOT run here.
"""

import json
import time as timemod
from pathlib import Path

import numpy as np
import pandas as pd

from mokamel_flip_state_machine import Tape, atr14, run_all, day_bounds, M5, M1

ROOT = Path(r"C:\Users\Milad\Desktop\RR")
DAYS = ROOT / "data/results/orb_2021_present/mt5_days.csv"
BASE = ROOT / "data/results/mokamel_flip"

INDECISION = 0.10
WARMUP = 200
COST_PRIMARY, COST_LEGACY = 0.51, 0.80
FORCE_WALL = 955
MIN_N = 150
WF_YEARS = (2021, 2022, 2023, 2024, 2025, 2026)

ARMS = [
    ("binding_40_80", 40.0, 80.0, 2.5),
    ("ring_0.50x_20_40", 20.0, 40.0, 100.0 / 20.0),
    ("ring_0.75x_30_60", 30.0, 60.0, 100.0 / 30.0),
    ("ring_1.25x_50_100", 50.0, 100.0, 100.0 / 50.0),
    ("ring_1.50x_60_120", 60.0, 120.0, 100.0 / 60.0),
]


def sim_m0(o, h, l, i0, fidx, fpx, entry, d, sl, tp):
    """M0 fixed SL/TP walk on the M1 tape. Exit-walk conventions VERBATIM
    from scalping_screen_stage_ab.py (first-crossing; same-bar both-level
    SL-first; gap-through fills at open; 15:55-bar-open forced exit).
    Returns (pnl_pts, exit_idx, reason, mfe, mae)."""
    mfe = mae = 0.0
    for j in range(i0, fidx):
        hj, lj, oj = h[j], l[j], o[j]
        if d == 1:
            mfe = max(mfe, hj - entry)
            mae = max(mae, entry - lj)
            if oj <= sl:
                return oj - entry, j, "stop_gap", mfe, mae
            if oj >= tp:
                return oj - entry, j, "target_gap", mfe, mae
            if lj <= sl and hj >= tp:
                return sl - entry, j, "stop_loss", mfe, mae
            if lj <= sl:
                return sl - entry, j, "stop_loss", mfe, mae
            if hj >= tp:
                return tp - entry, j, "target", mfe, mae
        else:
            mfe = max(mfe, entry - lj)
            mae = max(mae, hj - entry)
            if oj >= sl:
                return entry - oj, j, "stop_gap", mfe, mae
            if oj <= tp:
                return entry - oj, j, "target_gap", mfe, mae
            if hj >= sl and lj <= tp:
                return entry - sl, j, "stop_loss", mfe, mae
            if hj >= sl:
                return entry - sl, j, "stop_loss", mfe, mae
            if lj <= tp:
                return entry - tp, j, "target", mfe, mae
    px = fpx
    if d == 1:
        mfe = max(mfe, px - entry)
        mae = max(mae, entry - px)
    else:
        mfe = max(mfe, entry - px)
        mae = max(mae, px - entry)
    return (d == 1) * (px - entry) + (d == -1) * (entry - px), fidx, \
        "time_exit", mfe, mae


def simulate(sigs_by_day, m1, force, stop, target, lots):
    """SPEC sec 5 money machine for one arm. Returns (trades, tallies)."""
    trades = []
    tal = {"entries_aggressive": 0, "entries_conservative": 0,
           "cons_skipped_no_confirm": 0, "cons_skipped_filtered": 0,
           "filter_exits": 0, "occupancy_skips": 0, "day_closed_blocks": 0,
           "days_closed_by_wins": 0, "days_closed_by_losses": 0}
    o1, h1, l1 = m1.o, m1.h, m1.l

    for nd, sigs in sorted(sigs_by_day.items()):
        mode = "AGG"
        wins = losses = 0
        cumR = 0.0
        closed = False
        pos = None
        fidx, fpx = force[nd]

        def apply_closure(t):
            nonlocal mode, wins, losses, cumR, closed
            if t["reason"].startswith("target"):
                wins += 1
            elif t["reason"].startswith("stop"):
                losses += 1
            r = t["pnl_pts"] / stop
            cumR += r
            if wins >= 3 or losses >= 3:
                if not closed:
                    if wins >= 3:
                        tal["days_closed_by_wins"] += 1
                    else:
                        tal["days_closed_by_losses"] += 1
                closed = True
            if t["pnl_pts"] < 0:
                mode = "CON"
            elif cumR >= 0:
                mode = "AGG"

        for s in sigs:
            if pos is not None and s["entry_ts"] >= pos["exit_ts"]:
                apply_closure(pos)
                pos = None
            if pos is not None:
                tal["occupancy_skips"] += 1
                continue
            if closed:
                tal["day_closed_blocks"] += 1
                continue
            d = int(s["dir"])
            if mode == "AGG":
                entry_px, entry_tsx, ex0 = float(s["fill"]), int(s["entry_ts"]), int(s["ex_idx"])
                if s["filtered"]:
                    exit_px, exit_tsx = float(s["trig_close"]), int(s["trigger_ts"]) + 300
                    pnl = d * (exit_px - entry_px)
                    tal["filter_exits"] += 1
                    tal["entries_aggressive"] += 1
                    pos = dict(arm=None, ny_date=nd, mode=mode, dir=d,
                               entry_ts=entry_tsx, exit_ts=exit_tsx,
                               entry_px=entry_px, exit_px=exit_px,
                               pnl_pts=pnl, reason="filter_exit")
                    trades.append(pos)
                    continue
                sl = entry_px - d * stop
                tp = entry_px + d * target
                pnl, j, reason, mfe, mae = sim_m0(o1, h1, l1, ex0, fidx, fpx,
                                                  entry_px, d, sl, tp)
                if reason in ("stop_loss", "stop_gap"):
                    assert mae >= stop - 1e-6
                elif reason in ("target", "target_gap"):
                    assert mfe >= target - 1e-6
                else:
                    assert mae < stop + 1e-6 and mfe < target + 1e-6
                exit_tsx = int(m1.t[j])
                exit_px = entry_px + d * pnl
                tal["entries_aggressive"] += 1
                pos = dict(arm=None, ny_date=nd, mode=mode, dir=d,
                           entry_ts=entry_tsx, exit_ts=exit_tsx,
                           entry_px=entry_px, exit_px=exit_px,
                           pnl_pts=pnl, reason=reason)
                trades.append(pos)
            else:
                if not s["cons_ok"]:
                    tal["cons_skipped_no_confirm"] += 1
                    continue
                if s["filtered"]:
                    tal["cons_skipped_filtered"] += 1
                    continue
                entry_px = float(s["trig_close"])
                entry_tsx = int(s["trigger_ts"]) + 300
                ex0 = int(np.searchsorted(m1.t, entry_tsx, side="left"))
                assert ex0 <= fidx
                sl = entry_px - d * stop
                tp = entry_px + d * target
                pnl, j, reason, mfe, mae = sim_m0(o1, h1, l1, ex0, fidx, fpx,
                                                  entry_px, d, sl, tp)
                if reason in ("stop_loss", "stop_gap"):
                    assert mae >= stop - 1e-6
                elif reason in ("target", "target_gap"):
                    assert mfe >= target - 1e-6
                else:
                    assert mae < stop + 1e-6 and mfe < target + 1e-6
                tal["entries_conservative"] += 1
                pos = dict(arm=None, ny_date=nd, mode=mode, dir=d,
                           entry_ts=entry_tsx, exit_ts=int(m1.t[j]),
                           entry_px=entry_px, exit_px=entry_px + d * pnl,
                           pnl_pts=pnl, reason=reason)
                trades.append(pos)
        if pos is not None:
            apply_closure(pos)
            pos = None
    return trades, tal


def metrics(trades, universe_days, lots):
    pnls = np.array([t["pnl_pts"] for t in trades], float)
    nds = np.array([t["ny_date"] for t in trades], np.int64)
    yrs = np.array([t["ny_date"] // 10000 for t in trades], np.int32)
    n = len(pnls)
    net = pnls - COST_PRIMARY
    daily = {}
    for p, d0 in zip(net, nds):
        daily[int(d0)] = daily.get(int(d0), 0.0) + p * lots
    dd = np.array([daily.get(int(d), 0.0) for d in universe_days])
    sd = dd.std(ddof=1) if len(dd) > 1 else np.nan
    sharpe = float(dd.mean() / sd * np.sqrt(252)) if sd and sd > 0 else 0.0
    cum = np.cumsum(dd)
    peak = np.maximum.accumulate(np.r_[0.0, cum])[:-1]
    maxdd = float((peak - cum).max() / 10000.0 * 100.0) if n else np.nan
    yearly = {}
    for p, y0 in zip(net, yrs):
        yearly[int(y0)] = yearly.get(int(y0), 0.0) + p * lots
    total = float(net.sum()) * lots
    wf = {y: v for y, v in yearly.items() if y >= 2022}
    years_pos = int(sum(1 for v in wf.values() if v > 0))
    best_share = (max(wf.values()) / sum(wf.values()) * 100.0) \
        if sum(wf.values()) > 0 else np.nan
    reasons = {}
    for t in trades:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    return {"n_trades": n,
            "win_rate": round(float((net > 0).mean()), 4) if n else np.nan,
            "expectancy_pts": round(float(pnls.mean()), 4) if n else np.nan,
            "expectancy_after_cost": round(float(pnls.mean() - COST_PRIMARY), 4) if n else np.nan,
            "expectancy_after_cost_legacy08": round(float(pnls.mean() - COST_LEGACY), 4) if n else np.nan,
            "Sharpe": round(sharpe, 4), "max_DD_pct": round(maxdd, 4),
            "total_net_dollars": round(total, 2),
            "best_year_share_pct": round(best_share, 2) if best_share == best_share else "",
            "years_positive_of_5": years_pos,
            "yearly_net_dollars": {int(k): round(v, 2) for k, v in sorted(yearly.items())},
            "exit_reasons": reasons}


def main():
    t0 = timemod.time()
    m5 = Tape(M5, 300)
    m1 = Tape(M1, 60)
    rows, sigs, tal_machine = run_all(m5, m1)
    print(f"machine: {tal_machine} ({timemod.time()-t0:.0f}s)", flush=True)

    force = {}
    for nd, (s, e) in day_bounds(m1).items():
        w = m1.wall[s:e]
        m = w == FORCE_WALL
        if m.any():
            fi = s + int(np.argmax(m))
            force[nd] = (int(fi), float(m1.o[fi]))
        else:
            force[nd] = (e - 1, float(m1.c[e - 1]))

    days_df = pd.read_csv(DAYS, sep=";")
    universe = days_df.loc[(days_df.session_open > 0) |
                           (days_df.prev_close > 0), "ny_date"] \
        .astype(np.int64).to_numpy()
    universe.sort()
    universe_days = [int(d) for d in universe]

    sigs_by_day = {}
    universe_set = set(universe_days)
    n_inelig = 0
    for s in sigs.to_dict("records"):
        k = int(s["m5_idx_trigger"])
        if k < WARMUP or int(s["ny_date"]) not in universe_set:
            n_inelig += 1
            continue
        body = abs(m5.c[k] - m5.o[k])
        rng_ = m5.h[k] - m5.l[k]
        s["body_ratio"] = body / rng_ if rng_ > 0 else np.inf
        s["filtered"] = bool(s["body_ratio"] < INDECISION)
        s["trig_close"] = float(m5.c[k])
        if s["dir"] == -1:
            s["cons_ok"] = bool(m5.c[k] < s["r_low"])
        else:
            s["cons_ok"] = bool(m5.c[k] > s["r_high"])
        ex = int(np.searchsorted(m1.t, s["entry_ts"], side="left"))
        assert m1.t[ex] == s["entry_ts"]
        s["ex_idx"] = ex
        sigs_by_day.setdefault(int(s["ny_date"]), []).append(s)
    print(f"signals: {sum(len(v) for v in sigs_by_day.values())} on "
          f"{len(sigs_by_day)} days ({n_inelig} ineligible: warmup/universe) "
          f"({timemod.time()-t0:.0f}s)", flush=True)

    all_rows, all_trades, stats = [], [], {"arms": {}, "gate_B1": {}}
    for name, stop, target, lots in ARMS:
        trades, tal = simulate(sigs_by_day, m1, force, stop, target, lots)
        for t in trades:
            t["arm"] = name
        all_trades.extend(trades)
        met = metrics(trades, universe_days, lots)
        met.update(arm=name, stop_pts=stop, target_pts=target, lots=round(lots, 4),
                   source_stage="B")
        all_rows.append(met)
        stats["arms"][name] = {**met, "money_machine_tallies": tal}
        print(f"{name}: n={met['n_trades']} exp_net={met['expectancy_after_cost']} "
              f"Sharpe={met['Sharpe']} maxDD={met['max_DD_pct']} "
              f"({timemod.time()-t0:.0f}s)", flush=True)

    b = stats["arms"]["binding_40_80"]
    c1 = b["expectancy_after_cost"] > 0
    c2 = "VACUOUS BY Q2 RESOLUTION (single exogenous cell; unit ring reported non-binding)"
    c3 = b["n_trades"] >= MIN_N
    stats["gate_B1"] = {
        "clause_i_expectancy_after_cost_gt_0": bool(c1),
        "clause_ii_neighborhood": c2,
        "clause_iii_n_trades_ge_150": bool(c3),
        "VERDICT": "PASS" if (c1 and c3) else "FAIL"}

    md = pd.DataFrame(all_rows).sort_values("expectancy_after_cost", ascending=False)
    md.to_csv(BASE / "matrix_master.csv", index=False)
    td = pd.DataFrame(all_trades,
                      columns=["arm", "ny_date", "mode", "dir", "entry_ts",
                               "exit_ts", "entry_px", "exit_px", "pnl_pts",
                               "reason"])
    td.to_csv(BASE / "trades_all_arms.csv", index=False)
    yb = []
    for name, stop, target, lots in ARMS:
        sub = td[td.arm == name]
        for y, g in sub.groupby(sub.ny_date // 10000):
            net = (g.pnl_pts - COST_PRIMARY) * lots
            yb.append(dict(arm=name, year=int(y), n_trades=len(g),
                           gross_pts=round(float(g.pnl_pts.sum()), 2),
                           net_dollars=round(float(net.sum()), 2),
                           n_tp=int(g.reason.str.startswith("target").sum()),
                           n_sl=int(g.reason.str.startswith("stop").sum()),
                           n_neutral=int((~g.reason.str.startswith("target") &
                                          ~g.reason.str.startswith("stop")).sum())))
    pd.DataFrame(yb).to_csv(BASE / "yearly_breakdown.csv", index=False)
    with open(BASE / "stage_b_stats.json", "w") as fh:
        json.dump(stats, fh, indent=1, default=str)
    print(json.dumps(stats["gate_B1"], indent=1))
    print(f"DONE {timemod.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
