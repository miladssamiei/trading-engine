# -*- coding: utf-8 -*-
"""
albrooks_phase1.py — Phase 1 signal quality for the Al Brooks H1/H2
pullback pattern on US500-tick M5 (project hypothesis #7).

Implements SPEC.md exactly (state machine, EMA60+12-bar-slope trend filter,
stop-entry at trigger-bar extreme, 6-bar timeout, 90-min uncensored
direction-native excursions with SP2L E-XTM first-bar convention), evaluates
the pre-declared GATE.md gates, and reports the H1-vs-H2 comparison
(Brooks' own claim) explicitly regardless of outcome.

Outputs (data/results/al_brooks_h1h2_signal_quality/):
  phase1_buckets.csv, phase1_yearly.csv, phase1_signals.csv,
  phase1_stats.json
"""

import csv
import json
import time as timemod
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\al_brooks_h1h2_signal_quality")
M5 = Path(r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present\us500tick_m5.csv")

EMA_SPAN = 60
SLOPE_LB = 12
TIMEOUT = 6
FWD_BARS = 18
MIN_FWD = 6
THRESHOLDS = (0.10, 0.15, 0.20)
NY_LO, NY_HI = 570, 930          # trigger-bar window (primary arm)
SESSION_GAP = 3600


def main():
    t0 = timemod.time()
    df = pd.read_csv(M5)
    t = df["time"].to_numpy(dtype=np.int64)
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)
    ema = df["close"].ewm(span=EMA_SPAN, adjust=False).mean().to_numpy(float)
    slope = np.r_[np.zeros(SLOPE_LB), ema[SLOPE_LB:] - ema[:-SLOPE_LB]]
    up = (c > ema) & (slope > 0)
    dn = (c < ema) & (slope < 0)

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
    ny_min = (ny.hour * 60 + ny.minute).to_numpy()
    ny_year = ny.year.to_numpy()
    ny_date = (ny.year * 10000 + ny.month * 100 + ny.day).to_numpy()

    # ---------------- state machines (bull + bear) ----------------
    signals = []          # sampled (filled, eligible) records
    tallies = defaultdict(int)
    for d in (+1, -1):
        engaged = False
        trend = False     # False = pullback active
        leg = 0
        swing = 0.0
        pb_ext = 0.0      # pullback_low (bull) / pullback_high (bear)
        lh_seen = False   # lower-high (bull) / higher-low (bear) since last trigger
        filt = up if d == 1 else dn
        for i in range(1, n):
            if not filt[i]:
                if engaged:
                    engaged = False
                continue
            if not engaged:
                engaged, trend, leg, lh_seen = True, True, 0, False
                swing = h[i] if d == 1 else l[i]
                continue
            hi, lo = h[i], l[i]
            prior = h[i - 1] if d == 1 else l[i - 1]
            cur = hi if d == 1 else lo
            if trend:
                if (d == 1 and hi <= prior) or (d == -1 and lo >= prior):
                    trend = False
                    leg = 0
                    lh_seen = True
                    pb_ext = lo if d == 1 else hi
                else:
                    swing = max(swing, hi) if d == 1 else min(swing, lo)
                continue
            # ---- pullback state ----
            if (d == 1 and hi > swing) or (d == -1 and lo < swing):
                trend = True
                leg = 0
                swing = cur
                continue
            if (d == 1 and hi > prior) or (d == -1 and lo < prior):
                if lh_seen:
                    leg += 1
                    lh_seen = False
                    tallies["triggers"] += 1
                    tallies[f"leg{min(leg, 4)}"] += 1
                    if not (NY_LO <= ny_min[i] <= NY_HI):
                        tallies["out_of_window"] += 1
                        continue
                    tallies["in_window"] += 1
                # consecutive higher-high: no retrigger
            else:
                lh_seen = True
            pb_ext = min(pb_ext, lo) if d == 1 else max(pb_ext, hi)
    print(f"state machines done ({timemod.time() - t0:.0f}s); tallies: "
          f"{dict(tallies)}")

    # ---------------- fills + excursions ----------------
    # machines rerun storing pullback-extreme context at each trigger
    signals2 = run_machines_with_context(t, o, h, l, up, dn, ny_min)
    print(f"triggers with context: {len(signals2)} "
          f"(check == in-window above: {tallies['in_window']})")

    out_rows = []
    for (i, d, leg, stop, pb_ext) in signals2:
        fill = None
        hi_i = min(i + TIMEOUT, sess_end[i])
        for f in range(i + 1, hi_i + 1):
            if (d == 1 and h[f] >= stop) or (d == -1 and l[f] <= stop):
                fill = f
                break
        if fill is None:
            tallies["abandoned_timeout"] += 1
            continue
        entry = max(stop, o[fill]) if d == 1 else min(stop, o[fill])
        wend = min(fill + FWD_BARS, sess_end[fill])
        if wend - fill < MIN_FWD:
            tallies["ineligible_fwd"] += 1
            continue
        tallies["filled"] += 1
        # excursions, direction-native
        if d == 1:
            mfe = max(c[fill] - entry, 0.0)
            mae = max(entry - l[fill], 0.0)
            wmax = -1e18
            wmin = 1e18
            for k in range(fill + 1, wend + 1):
                wmax = max(wmax, h[k])
                wmin = min(wmin, l[k])
            mfe = max(mfe, wmax - entry)
            mae = max(mae, entry - wmin)
            risk = (entry - pb_ext) / entry * 100.0
        else:
            mfe = max(entry - c[fill], 0.0)
            mae = max(h[fill] - entry, 0.0)
            wmax = -1e18
            wmin = 1e18
            for k in range(fill + 1, wend + 1):
                wmax = max(wmax, h[k])
                wmin = min(wmin, l[k])
            mfe = max(mfe, entry - wmin)
            mae = max(mae, wmax - entry)
            risk = (pb_ext - entry) / entry * 100.0
        mfe_p = mfe / entry * 100.0
        mae_p = mae / entry * 100.0
        # first-touch 0.15 informational
        ft = 0
        for k in range(fill, wend + 1):
            fav = (h[k] - entry) if d == 1 else (entry - l[k])
            adv = (entry - l[k]) if d == 1 else (h[k] - entry)
            favp = fav / entry * 100.0
            advp = adv / entry * 100.0
            fu = favp >= 0.15
            fd = advp >= 0.15
            if fu or fd:
                ft = 1 if (fu and not fd) else 0
                break
        out_rows.append({
            "time": int(t[fill]), "nd": int(ny_date[fill]),
            "year": int(ny_year[fill]), "dir": d, "leg": leg,
            "entry": round(entry, 3), "fwd_len": int(wend - fill),
            "mfe": round(mfe_p, 4), "mae": round(mae_p, 4),
            "risk_brooks_pct": round(risk, 4), "ft_win_015": ft,
            "trigger_time": int(t[i]),
        })

    pd.DataFrame(out_rows).to_csv(BASE / "phase1_signals.csv", index=False)

    # ---------------- aggregation ----------------
    def bucket(rows):
        nn = len(rows)
        if not nn:
            return None
        mfs = [r["mfe"] for r in rows]
        mas = [r["mae"] for r in rows]
        out = {"n": nn,
               "mean_mfe": round(sum(mfs) / nn, 4),
               "mean_mae": round(sum(mas) / nn, 4),
               "mean_diff": round((sum(mfs) - sum(mas)) / nn, 4)}
        for T in THRESHOLDS:
            out[f"mfe_ge_{int(T * 100):03d}"] = sum(1 for v in mfs if v >= T)
            out[f"mae_ge_{int(T * 100):03d}"] = sum(1 for v in mas if v >= T)
        out["ft_win_015"] = round(sum(r["ft_win_015"] for r in rows) / nn, 4)
        out["mean_risk_pct"] = round(sum(r["risk_brooks_pct"] for r in rows) / nn, 4)
        return out

    configs = {"H1": (1,), "H2": (2,), "H1+H2": (1, 2), "H3+": (3, 4, 5, 6, 7, 8, 9, 10)}
    buckets = []
    per_cfg = {}
    for cfg, legs in configs.items():
        for dname, dd in (("pooled", None), ("bull", 1), ("bear", -1)):
            rows = [r for r in out_rows if r["leg"] in legs
                    and (dd is None or r["dir"] == dd)]
            b = bucket(rows)
            if b is None:
                continue
            b.update({"config": cfg, "dir": dname})
            buckets.append(b)
            if dname == "pooled" and cfg in ("H1", "H2", "H1+H2"):
                per_cfg[cfg] = (b, rows)
    with open(BASE / "phase1_buckets.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(buckets[0].keys()))
        w.writeheader()
        w.writerows(buckets)

    yearly = []
    for cfg, legs in configs.items():
        per_year = defaultdict(list)
        for r in out_rows:
            if r["leg"] in legs:
                per_year[r["year"]].append(r["mfe"] - r["mae"])
        for y in sorted(per_year):
            vals = per_year[y]
            yearly.append({"config": cfg, "year": y, "n": len(vals),
                           "mean_diff": round(sum(vals) / len(vals), 4)})
    with open(BASE / "phase1_yearly.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["config", "year", "n", "mean_diff"])
        w.writeheader()
        w.writerows(yearly)

    # ---------------- gates ----------------
    def boot(rows, n_boot=10000, seed=11):
        per_day = defaultdict(list)
        for r in rows:
            per_day[r["nd"]].append(r["mfe"] - r["mae"])
        days = np.array([sum(v) / len(v) for v in per_day.values()])
        rng = np.random.default_rng(seed)
        means = rng.choice(days, size=(n_boot, len(days)), replace=True).mean(axis=1)
        return {"n_days": len(days), "p05": round(float(np.percentile(means, 5)), 5),
                "p_le0": round(float((means <= 0).mean()), 4)}

    gates = {}
    for cfg, (b, rows) in per_cfg.items():
        dom = all(b[f"mfe_ge_{int(T * 100):03d}"] > b[f"mae_ge_{int(T * 100):03d}"]
                  for T in THRESHOLDS)
        g1 = b["n"] >= 150
        g2 = dom and b["mean_diff"] > 0
        bt = boot(rows)
        g4 = bt["p05"] > 0
        gates[cfg] = {"n": b["n"], "g1": bool(g1), "dominance": bool(dom),
                      "mean_diff": b["mean_diff"], "g2": bool(g2),
                      "boot": bt, "g4": bool(g4)}
        print(f"{cfg}: n={b['n']} mfe {b['mean_mfe']} mae {b['mean_mae']} "
              f"diff {b['mean_diff']:+.4f} 015 {b['mfe_ge_015']} vs {b['mae_ge_015']} "
              f"risk {b['mean_risk_pct']}%  G1={g1} G2={g2} G4={g4} boot={bt}")

    any_pass = any(g["g1"] and g["g2"] and g["g4"] for g in gates.values())
    verdict = ("PASS -> Step 3 eligible configs: "
               + ",".join(k for k, g in gates.items() if g["g1"] and g["g2"] and g["g4"])
               if any_pass else "FAIL -> STOP (null result)")

    # H1 vs H2 explicit comparison + diff bootstrap (informational, always reported)
    h1, h2 = per_cfg.get("H1", (None, [])), per_cfg.get("H2", (None, []))
    cmp_boot = None
    if h1[0] and h2[0]:
        per_day = {d: defaultdict(list) for d in (1, 2)}
        for r in out_rows:
            if r["leg"] in (1, 2):
                per_day[r["leg"]][r["nd"]].append(r["mfe"] - r["mae"])
        days_all = sorted(set(per_day[1]) | set(per_day[2]))
        m1 = {d: (sum(v) / len(v) if v else None) for d, v in per_day[1].items()}
        m2 = {d: (sum(v) / len(v) if v else None) for d, v in per_day[2].items()}
        arr1 = np.array([m1.get(d) if m1.get(d) is not None else np.nan
                         for d in days_all])
        arr2 = np.array([m2.get(d) if m2.get(d) is not None else np.nan
                         for d in days_all])
        diffs = arr2 - arr1
        diffs = diffs[~np.isnan(diffs)]
        rng = np.random.default_rng(13)
        bs = rng.choice(diffs, size=(10000, len(diffs)), replace=True).mean(axis=1)
        cmp_boot = {"n_days_both": len(diffs),
                    "h2_minus_h1_daily": round(float(diffs.mean()), 5),
                    "p05": round(float(np.percentile(bs, 5)), 5),
                    "p95": round(float(np.percentile(bs, 95)), 5),
                    "p_h2_le_h1": round(float((bs <= 0).mean()), 4)}

    stats = {"gates": gates, "verdict": verdict,
             "h1_vs_h2": {"H1": per_cfg.get("H1", (None,))[0],
                          "H2": per_cfg.get("H2", (None,))[0],
                          "day_diff_bootstrap": cmp_boot},
             "tallies": dict(tallies),
             "params": {"EMA_SPAN": EMA_SPAN, "SLOPE_LB": SLOPE_LB,
                        "TIMEOUT": TIMEOUT, "FWD_BARS": FWD_BARS,
                        "MIN_FWD": MIN_FWD, "THRESHOLDS": THRESHOLDS,
                        "NY_WINDOW": [NY_LO, NY_HI]},
             "runtime_s": round(timemod.time() - t0, 1)}
    with open(BASE / "phase1_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    print("\nH1 vs H2 (Brooks claim):")
    if h1[0] and h2[0]:
        print(f"  H1: n={h1[0]['n']} diff {h1[0]['mean_diff']:+.4f} "
              f"ft {h1[0]['ft_win_015']}")
        print(f"  H2: n={h2[0]['n']} diff {h2[0]['mean_diff']:+.4f} "
              f"ft {h2[0]['ft_win_015']}")
        print(f"  day-level H2-H1 bootstrap: {cmp_boot}")
    print("\nVERDICT:", verdict)
    print(f"wrote phase1_buckets.csv, phase1_yearly.csv, phase1_signals.csv, "
          f"phase1_stats.json ({stats['runtime_s']}s)")


def run_machines_with_context(t, o, h, l, up, dn, ny_min):
    """Same machines as main pass but store (i, dir, leg, stop, pb_ext) for
    in-window triggers. Must mirror the main machine exactly."""
    n = len(t)
    out = []
    for d in (+1, -1):
        engaged = False
        trend = False
        leg = 0
        swing = 0.0
        pb_ext = 0.0
        lh_seen = False
        filt = up if d == 1 else dn
        for i in range(1, n):
            if not filt[i]:
                if engaged:
                    engaged = False
                continue
            if not engaged:
                engaged, trend, leg, lh_seen = True, True, 0, False
                swing = h[i] if d == 1 else l[i]
                continue
            hi, lo = h[i], l[i]
            prior = h[i - 1] if d == 1 else l[i - 1]
            if trend:
                if (d == 1 and hi <= prior) or (d == -1 and lo >= prior):
                    trend = False
                    leg = 0
                    lh_seen = True
                    pb_ext = lo if d == 1 else hi
                else:
                    swing = max(swing, hi) if d == 1 else min(swing, lo)
                continue
            if (d == 1 and hi > swing) or (d == -1 and lo < swing):
                trend = True
                leg = 0
                swing = hi if d == 1 else lo
                continue
            if (d == 1 and hi > prior) or (d == -1 and lo < prior):
                if lh_seen:
                    leg += 1
                    lh_seen = False
                    if NY_LO <= ny_min[i] <= NY_HI:
                        out.append((i, d, leg,
                                    h[i] if d == 1 else l[i], pb_ext))
            else:
                lh_seen = True
            pb_ext = min(pb_ext, lo) if d == 1 else max(pb_ext, hi)
    return out


if __name__ == "__main__":
    main()
