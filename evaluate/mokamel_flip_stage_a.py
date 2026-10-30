# -*- coding: utf-8 -*-
"""
mokamel_flip_stage_a.py — Hypothesis #16 MOKAMEL_FLIP Stage A:
exit-free signal quality + GATE A1.

Implements SPEC_MOKAMEL_FLIP.txt section 6 / GATE_MOKAMEL_FLIP.txt
exactly (frozen 2026-09-03 BEFORE this code was written):
- population: every AGGRESSIVE-fill flip entry, eligible window, cash
  day in the ORB universe, M5 dataset warmup 200 bars;
- direction-native MFE/MAE over the next 60 minutes of M1 bars with
  the SP2L/scalping-screen E-XTM first-bar convention (functions
  copied VERBATIM from evaluate/scalping_screen_stage_ab.py);
- binding thresholds t in {0.50, 0.75, 1.00} x ATR14(trigger M5 bar,
  continuous tape); e = (MFE-MAE)/ATR14; day-block bootstrap
  (10,000 draws, seed 20260903);
- GATE A1 evaluated on the post-indecision-filter population
  (body/range < 0.10, Q4); unfiltered population reported alongside,
  non-gating.

Outputs (data/results/mokamel_flip/): base_candle_state_log.csv,
signals.csv, stage_a_stats.json. Does NOT touch SPEC/GATE.
"""

import json
import time as timemod
from pathlib import Path

import numpy as np
import pandas as pd

from mokamel_flip_state_machine import (Tape, atr14, run_all, M5, M1)

ROOT = Path(r"C:\Users\Milad\Desktop\RR")
DAYS = ROOT / "data/results/orb_2021_present/mt5_days.csv"
BASE = ROOT / "data/results/mokamel_flip"

WARMUP = 200
HORIZON_S = 3600
THR_ATR = (0.50, 0.75, 1.00)
THR_PCT = (0.10, 0.15, 0.20)
N_BOOT, SEED = 10000, 20260903
INDECISION = 0.10
MIN_N = 150
BOOT_ALPHA = 0.20


def excursions(m1, entry, d, i0):
    """Direction-native MFE/MAE in POINTS over the 60-min M1 walk from i0.
    E-XTM first-bar convention. VERBATIM from scalping_screen_stage_ab.py."""
    i1 = int(np.searchsorted(m1.t, m1.t[i0] + HORIZON_S, side="left")) - 1
    if i1 < i0:
        return None
    if d == 1:
        fav = max(m1.c[i0] - entry, 0.0)
        adv = max(entry - m1.l[i0], 0.0)
        if i1 > i0:
            fav = max(fav, float(m1.h[i0 + 1:i1 + 1].max()) - entry)
            adv = max(adv, entry - float(m1.l[i0 + 1:i1 + 1].min()))
    else:
        fav = max(entry - m1.c[i0], 0.0)
        adv = max(m1.h[i0] - entry, 0.0)
        if i1 > i0:
            fav = max(fav, entry - float(m1.l[i0 + 1:i1 + 1].min()))
            adv = max(adv, float(m1.h[i0 + 1:i1 + 1].max()) - entry)
    return max(fav, 0.0), max(adv, 0.0)


def boot_p_le0(e, nd):
    """VERBATIM from scalping_screen_stage_ab.py (day-block bootstrap)."""
    per_day = {}
    for x, d0 in zip(e, nd):
        per_day.setdefault(d0, []).append(x)
    days = np.array([np.mean(v) for v in per_day.values()])
    rng = np.random.default_rng(SEED)
    means = rng.choice(days, size=(N_BOOT, len(days)), replace=True).mean(axis=1)
    return float((means <= 0).mean()), len(days)


def main():
    t0 = timemod.time()
    BASE.mkdir(parents=True, exist_ok=True)

    m5 = Tape(M5, 300)
    m5.atr = atr14(m5.h, m5.l, m5.c)
    m1 = Tape(M1, 60)
    print(f"tapes: M5 {m5.n}, M1 {m1.n} ({timemod.time()-t0:.0f}s)", flush=True)

    rows, sigs, tal = run_all(m5, m1)
    print(f"machine: {tal['cash_days_run']} cash days, {len(sigs)} flip entries, "
          f"tallies {tal} ({timemod.time()-t0:.0f}s)", flush=True)

    ny = pd.DatetimeIndex(pd.to_datetime(rows["time"].to_numpy(), unit="s", utc=True))
    rows.insert(1, "ny_time", ny.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M"))
    rows.to_csv(BASE / "base_candle_state_log.csv", index=False)

    days_df = pd.read_csv(DAYS, sep=";")
    universe = set(days_df.loc[(days_df.session_open > 0) |
                               (days_df.prev_close > 0), "ny_date"]
                    .astype(np.int64).tolist())

    out = []
    for s in sigs.to_dict("records"):
        k = int(s["m5_idx_trigger"])
        body = abs(m5.c[k] - m5.o[k])
        rng_ = m5.h[k] - m5.l[k]
        ratio = body / rng_ if rng_ > 0 else np.inf
        ex = int(np.searchsorted(m1.t, s["entry_ts"], side="left"))
        assert m1.t[ex] == s["entry_ts"]
        assert 570 <= s["entry_wall"] <= 870
        r = excursions(m1, float(s["fill"]), int(s["dir"]), ex)
        atr = float(m5.atr[k])
        out.append(dict(
            ny_date=s["ny_date"], entry_ts=s["entry_ts"], entry_wall=s["entry_wall"],
            dir=s["dir"], fill=s["fill"], r_id=s["r_id"], r_high=s["r_high"],
            r_low=s["r_low"], base_id=s["base_id"], trigger_ts=s["trigger_ts"],
            trigger_open=m5.o[k], trigger_high=m5.h[k], trigger_low=m5.l[k],
            trigger_close=m5.c[k], body_ratio=round(ratio, 4),
            exclude_indecision=bool(ratio < INDECISION),
            atr14_trigger=atr,
            eligible=bool(k >= WARMUP and s["ny_date"] in universe),
            mfe_pts=(r[0] if r else np.nan), mae_pts=(r[1] if r else np.nan)))
    df = pd.DataFrame(out)
    df["mfe_atr"] = df.mfe_pts / df.atr14_trigger
    df["mae_atr"] = df.mae_pts / df.atr14_trigger
    df["e_atr"] = (df.mfe_pts - df.mae_pts) / df.atr14_trigger
    df.to_csv(BASE / "signals.csv", index=False)

    # degeneracy assertions (scalping-screen lesson)
    el = df[df.eligible]
    assert len(el) > 0, "no eligible signals"
    assert (el.dir.abs() == 1).all()
    assert np.isfinite(el.fill).all() and (el.fill > 0).all()
    assert np.isfinite(el.atr14_trigger).all() and (el.atr14_trigger > 0).all()
    assert el.ny_date.nunique() > 100

    stats = {"tallies": {k: int(v) for k, v in tal.items()},
             "raw_flip_entries": int(len(df)),
             "eligible": int(len(el)),
             "i9_same_bar_fraction_of_candidates":
                 round(tal["P2_same_bar_guard"] /
                       max(1, tal["P2_same_bar_guard"] + len(df)), 4)}

    for pop, name in ((el[~el.exclude_indecision], "binding_filtered"),
                      (el, "unfiltered_context")):
        n = len(pop)
        e = pop.e_atr.to_numpy()
        nd = pop.ny_date.to_numpy()
        thr_rows = {}
        for t in THR_ATR:
            thr_rows[str(t)] = {
                "n_mfe_ge": int((pop.mfe_atr >= t).sum()),
                "n_mae_ge": int((pop.mae_atr >= t).sum()),
                "mfe_share": round(float((pop.mfe_atr >= t).mean()), 4) if n else None}
        pct_rows = {str(t): {
                "n_mfe_ge": int((pop.mfe_pts >= t / 100 * pop.fill).sum()),
                "n_mae_ge": int((pop.mae_pts >= t / 100 * pop.fill).sum())}
            for t in THR_PCT}
        p_boot, n_days = boot_p_le0(e, nd)
        cnt_ok = all(v["n_mfe_ge"] > v["n_mae_ge"] for v in thr_rows.values())
        stats[name] = {
            "n": n, "n_days": n_days,
            "mean_e_atr": round(float(e.mean()), 4),
            "median_e_atr": round(float(np.median(e)), 4),
            "thresholds": thr_rows, "pct_thresholds_descriptive": pct_rows,
            "counts_mfe_gt_mae_all": bool(cnt_ok),
            "p_boot": round(p_boot, 4),
            "dir_split": {str(d): {"n": int((pop.dir == d).sum()),
                                   "mean_e_atr": round(float(pop.e_atr[pop.dir == d].mean()), 4)}
                          for d in (-1, 1)},
            "yearly_n": {int(y): int(c) for y, c in
                         pop.ny_date.astype(str).str[:4].value_counts().sort_index().items()}}

    b = stats["binding_filtered"]
    c1 = b["n"] >= MIN_N
    c2 = b["counts_mfe_gt_mae_all"]
    c3 = b["p_boot"] <= BOOT_ALPHA
    stats["gate_A1"] = {"clause_i_n_ge_150": bool(c1),
                        "clause_ii_mfe_gt_mae_every_threshold": bool(c2),
                        "clause_iii_p_boot_le_0.20": bool(c3),
                        "VERDICT": "PASS" if (c1 and c2 and c3) else "FAIL"}
    with open(BASE / "stage_a_stats.json", "w") as fh:
        json.dump(stats, fh, indent=1)
    print(json.dumps(stats["gate_A1"], indent=1))
    print(f"binding n={b['n']} mean_e={b['mean_e_atr']} p_boot={b['p_boot']}")
    print(f"DONE {timemod.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
