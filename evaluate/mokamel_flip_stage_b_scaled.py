# -*- coding: utf-8 -*-
"""
mokamel_flip_stage_b_scaled.py — Hypothesis #16 Amendment 1: the two
scaled-exit arms under GATE B1-AMENDED.

Implements SPEC_MOKAMEL_FLIP_AMENDMENT_SCALED_EXIT.txt /
GATE_MOKAMEL_FLIP_AMENDMENT_SCALED_EXIT.txt exactly (frozen
2026-09-03 BEFORE this script was written):
- derives both stop/target pairs from the frozen anchors (dow/
  us500_ratio_derivation.csv is the audit trail);
- re-runs the ORIGINAL Stage-B machinery unchanged (simulate/metrics
  imported from evaluate/mokamel_flip_stage_b.py) with only the
  stop/target distances changed;
- evaluates B1-AMENDED clauses (i)/(ii)/(iii) per arm, separately.

Outputs: dow_us500_ratio_derivation.csv, matrix_scaled_arms.csv,
stage_b_scaled_stats.json. Does NOT touch any prior artifact.
"""

import json
import time as timemod
from pathlib import Path

import numpy as np
import pandas as pd

from mokamel_flip_state_machine import Tape, day_bounds, M1
from mokamel_flip_stage_b import simulate, metrics, COST_PRIMARY, FORCE_WALL

ROOT = Path(r"C:\Users\Milad\Desktop\RR")
DAYS = ROOT / "data/results/orb_2021_present/mt5_days.csv"
BASE = ROOT / "data/results/mokamel_flip"

R_PRICE = 8.0          # brief's 7-9x Dow/US500 band midpoint
ATR_MULT = 1.0
RAW_STOP_VIDEO = 40.0
RAW_TARGET_VIDEO = 80.0


def round_half(x):
    return np.floor(x * 2.0 + 0.5) / 2.0


def main():
    t0 = timemod.time()
    m1 = Tape(M1, 60)

    df = pd.read_csv(BASE / "signals.csv")
    b = df[df.eligible & ~df.exclude_indecision]
    atr_med = float(b.atr14_trigger.median())

    stop_m1 = round_half(RAW_STOP_VIDEO / R_PRICE)
    target_m1 = 2.0 * stop_m1
    stop_m2 = round_half(ATR_MULT * atr_med)
    target_m2 = 2.0 * stop_m2
    assert stop_m1 == 5.0 and stop_m2 == 5.0, (stop_m1, stop_m2)

    q = b.atr14_trigger.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(4)
    band = [(7.0, round_half(RAW_STOP_VIDEO / 7.0)), (9.0, round_half(RAW_STOP_VIDEO / 9.0))]
    drv = pd.DataFrame([
        dict(item="statement", value="NO Dow Jones / US30 / DJI series exists in this project"),
        dict(item="terminal_bases", value="EURUSD GBPUSD US500.c USDCHF USDJPY XAUUSD + Custom US500-tick USA500_Tick"),
        dict(item="sourcing_decision", value="US500-native adoption per task owner 2026-09-03"),
        dict(item="M1_anchor_R_price", value=R_PRICE),
        dict(item="M1_anchor_source", value="midpoint of the 7-9x band stated in the commissioning brief (public knowledge, LOWER-RIGOR, flagged)"),
        dict(item="M1_raw_stop", value=RAW_STOP_VIDEO / R_PRICE),
        dict(item="M1_stop_rounded", value=stop_m1),
        dict(item="M1_target", value=target_m1),
        dict(item="M2_atr_multiple", value=ATR_MULT),
        dict(item="M2_median_atr14_trigger", value=round(atr_med, 4)),
        dict(item="M2_median_source", value="signals.csv, Stage-A binding population (eligible & ~exclude_indecision)"),
        dict(item="M2_stop_rounded", value=stop_m2),
        dict(item="M2_target", value=target_m2),
        dict(item="converged", value=bool(stop_m1 == stop_m2 and target_m1 == target_m2)),
        dict(item="band_7x_context_untested", value=band[0][1]),
        dict(item="band_9x_context_untested", value=band[1][1]),
        dict(item="atr_q05", value=q.loc[0.05]), dict(item="atr_q25", value=q.loc[0.25]),
        dict(item="atr_q50", value=q.loc[0.5]), dict(item="atr_q75", value=q.loc[0.75]),
        dict(item="atr_q95", value=q.loc[0.95]),
        dict(item="rounding_rule", value="stop to nearest 0.5 pt; target = 2 x stop (1:2 shape preserved)"),
    ])
    drv.to_csv(BASE / "dow_us500_ratio_derivation.csv", index=False)
    print(f"derivation: median ATR {atr_med:.4f}; M1 {stop_m1}/{target_m1}; "
          f"M2 {stop_m2}/{target_m2}; converged={stop_m1 == stop_m2} "
          f"({timemod.time()-t0:.0f}s)", flush=True)

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
    universe_days = [int(x) for x in universe]

    sigs_by_day = {}
    for s in df[df.eligible].to_dict("records"):
        d = int(s["dir"])
        s["filtered"] = bool(s["exclude_indecision"])
        s["trig_close"] = float(s["trigger_close"])
        if d == -1:
            s["cons_ok"] = bool(s["trigger_close"] < s["r_low"])
        else:
            s["cons_ok"] = bool(s["trigger_close"] > s["r_high"])
        ex = int(np.searchsorted(m1.t, s["entry_ts"], side="left"))
        assert m1.t[ex] == s["entry_ts"]
        s["ex_idx"] = ex
        sigs_by_day.setdefault(int(s["ny_date"]), []).append(s)

    arms = [("scaled_M1_price_5_10", stop_m1, target_m1, 100.0 / stop_m1),
            ("scaled_M2_vol_5_10", stop_m2, target_m2, 100.0 / stop_m2)]
    rows, stats = [], {"arms": {}, "gate_B1_amended": {}}
    all_trades = []
    for name, stop, target, lots in arms:
        trades, tal = simulate(sigs_by_day, m1, force, stop, target, lots)
        for t in trades:
            t["arm"] = name
        all_trades.extend(trades)
        met = metrics(trades, universe_days, lots)
        met.pop("exit_reasons", None)
        reasons = {}
        for t in trades:
            reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
        te_share = reasons.get("time_exit", 0) / max(1, len(trades))
        met.update(arm=name, stop_pts=stop, target_pts=target,
                   lots=round(lots, 4), source_stage="B_SCALED",
                   exit_reasons=reasons,
                   time_exit_share=round(te_share, 4))
        rows.append(met)
        c1 = met["expectancy_after_cost"] > 0
        c2 = met["n_trades"] >= 150
        c3 = te_share < 0.60
        stats["arms"][name] = met
        stats["arms"][name]["money_machine_tallies"] = tal
        stats["gate_B1_amended"][name] = {
            "clause_i_expectancy_after_cost_gt_0": bool(c1),
            "clause_ii_n_trades_ge_150": bool(c2),
            "clause_iii_time_exit_share_lt_0.60": bool(c3),
            "VERDICT": "PASS" if (c1 and c2 and c3) else "FAIL"}
        print(f"{name}: n={met['n_trades']} exp_net={met['expectancy_after_cost']} "
              f"te_share={te_share:.3f} Sharpe={met['Sharpe']} "
              f"({timemod.time()-t0:.0f}s)", flush=True)

    md = pd.DataFrame(rows).sort_values("expectancy_after_cost", ascending=False)
    md.to_csv(BASE / "matrix_scaled_arms.csv", index=False)
    pd.DataFrame(all_trades).to_csv(
        BASE / "trades_scaled_arms.csv", index=False)
    with open(BASE / "stage_b_scaled_stats.json", "w") as fh:
        json.dump(stats, fh, indent=1, default=str)
    print(json.dumps(stats["gate_B1_amended"], indent=1))
    print(f"DONE {timemod.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
