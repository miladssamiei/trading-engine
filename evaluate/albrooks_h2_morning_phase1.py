# -*- coding: utf-8 -*-
"""
albrooks_h2_morning_phase1.py - Phase 1 signal quality for the Al Brooks
H2 + IBS>65 + NY-morning variant on US500-tick M5 (project hypothesis #8).

Implements data/results/al_brooks_h2_ibs_morning/SPEC.md exactly:
- leg-counting machine reused from study #7, gated by EMA(20) side only
- leg==2 triggers only (H2 bull + L2 bear mirror)
- filter funnel: IBS(>65/<35) -> doji (body<=10% range) -> spike
  (range > 1.7*ATR14[i-1]) -> morning window [09:30,11:30) -> prior-day
  context (close beyond prior-session extreme)
- stop-entry at signal-bar extreme, 6-bar timeout, 90-min uncensored
  direction-native excursions (E-XTM first-bar convention)
- evaluates the pre-declared GATE.md gates (combined H2+L2 unit)

Outputs (data/results/al_brooks_h2_ibs_morning/):
  phase1_funnel.csv, phase1_buckets.csv, phase1_yearly.csv,
  phase1_signals.csv, phase1_stats.json
"""

import csv
import json
import time as timemod
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\al_brooks_h2_ibs_morning")
M5 = Path(r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present\us500tick_m5.csv")

EMA_SPAN = 20
ATR_PERIOD = 14
ATR_MULT = 1.7
IBS_HI = 65.0
IBS_LO = 35.0
DOJI_BODY_FRAC = 0.10
TIMEOUT = 6
FWD_BARS = 18
MIN_FWD = 6
THRESHOLDS = (0.10, 0.15, 0.20)
NY_LO, NY_HI = 570, 690          # signal-bar window [09:30, 11:30)
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
    up = c > ema
    dn = c < ema

    # Wilder ATR(14) over the continuous tape
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.r_[c[0], c[:-1]]),
                                      np.abs(l - np.r_[c[0], c[:-1]])))
    atr = pd.Series(tr).ewm(alpha=1.0 / ATR_PERIOD, adjust=False).mean().to_numpy(float)

    # session segmentation (gap > 60 min) + per-session high/low
    sess = np.zeros(n, dtype=np.int64)
    sess[1:] = np.cumsum(np.diff(t) > SESSION_GAP)
    n_sess = sess[-1] + 1
    sess_hi = np.full(n_sess, -1e18)
    sess_lo = np.full(n_sess, 1e18)
    for i in range(n):
        s = sess[i]
        if h[i] > sess_hi[s]:
            sess_hi[s] = h[i]
        if l[i] < sess_lo[s]:
            sess_lo[s] = l[i]
    # prior-session arrays (index by session id)
    prior_hi = np.r_[np.nan, sess_hi[:-1]]
    prior_lo = np.r_[np.nan, sess_lo[:-1]]

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

    # ---------------- state machines (bull + bear), leg-2 triggers ----------
    triggers = []        # dicts, all leg-2 triggers with context
    tallies = defaultdict(int)
    for d in (+1, -1):
        engaged = False
        trend = False     # False = pullback active
        leg = 0
        swing = 0.0
        pb_ext = 0.0      # pullback_low (bull) / pullback_high (bear)
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
                    tallies["triggers_all_legs"] += 1
                    tallies[f"leg{min(leg, 4)}"] += 1
                    if leg == 2:
                        rng = hi - lo
                        ibs = (c[i] - lo) / rng * 100.0 if rng > 0 else 50.0
                        triggers.append({
                            "i": i, "d": d, "stop": hi if d == 1 else lo,
                            "pb_ext": pb_ext, "ibs": ibs,
                            "body_pct": abs(c[i] - o[i]) / rng * 100.0 if rng > 0 else 0.0,
                            "atr_ratio": rng / atr[i - 1] if atr[i - 1] > 0 else 9.9,
                            "in_window": bool(NY_LO <= ny_min[i] < NY_HI),
                            "pd_hi": prior_hi[sess[i]], "pd_lo": prior_lo[sess[i]],
                        })
            else:
                lh_seen = True
            pb_ext = min(pb_ext, lo) if d == 1 else max(pb_ext, hi)
    tallies["leg2_raw"] = len(triggers)
    print(f"machines done ({timemod.time() - t0:.0f}s); tallies: {dict(tallies)}")

    # ---------------- filter funnel (cumulative, per SPEC order) ----------
    def stage(rows, key):
        return [r for r in rows if key(r)]

    f1 = stage(triggers, lambda r: r["ibs"] > IBS_HI if r["d"] == 1 else r["ibs"] < IBS_LO)
    f2 = stage(f1, lambda r: r["body_pct"] > DOJI_BODY_FRAC * 100.0)
    f3 = stage(f2, lambda r: r["atr_ratio"] <= ATR_MULT)
    f4 = stage(f3, lambda r: r["in_window"])
    f5 = stage(f4, lambda r: (r["d"] == 1 and c[r["i"]] > r["pd_hi"]) or
                             (r["d"] == -1 and c[r["i"]] < r["pd_lo"]))
    stages = [("F0_raw_leg2", triggers), ("F1_after_ibs", f1),
              ("F2_after_doji", f2), ("F3_after_atr_spike", f3),
              ("F4_after_morning_window", f4), ("F5_after_prior_day", f5)]
    with open(BASE / "phase1_funnel.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stage", "bull", "bear", "total"])
        for name, rows in stages:
            b = sum(1 for r in rows if r["d"] == 1)
            be = len(rows) - b
            w.writerow([name, b, be, len(rows)])
            print(f"funnel {name}: bull {b} bear {be} total {len(rows)}")

    # ---------------- fills + excursions (study #7 conventions) ----------
    out_rows = []
    for r in f5:
        i, d = r["i"], r["d"]
        stop = r["stop"]
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
        if d == 1:
            mfe = max(c[fill] - entry, 0.0)
            mae = max(entry - l[fill], 0.0)
            wmax = h[fill + 1:wend + 1].max() if wend > fill else entry
            wmin = l[fill + 1:wend + 1].min() if wend > fill else entry
            mfe = max(mfe, wmax - entry)
            mae = max(mae, entry - wmin)
            risk = (entry - r["pb_ext"]) / entry * 100.0
        else:
            mfe = max(entry - c[fill], 0.0)
            mae = max(h[fill] - entry, 0.0)
            wmax = h[fill + 1:wend + 1].max() if wend > fill else entry
            wmin = l[fill + 1:wend + 1].min() if wend > fill else entry
            mfe = max(mfe, entry - wmin)
            mae = max(mae, wmax - entry)
            risk = (r["pb_ext"] - entry) / entry * 100.0
        mfe_p = mfe / entry * 100.0
        mae_p = mae / entry * 100.0
        # first-touch 0.15 informational (study #7 form)
        ft = 0
        for k in range(fill, wend + 1):
            fav = (h[k] - entry) if d == 1 else (entry - l[k])
            adv = (entry - l[k]) if d == 1 else (h[k] - entry)
            fu = fav / entry * 100.0 >= 0.15
            fd = adv / entry * 100.0 >= 0.15
            if fu or fd:
                ft = 1 if (fu and not fd) else 0
                break
        out_rows.append({
            "time": int(t[fill]), "nd": int(ny_date[fill]), "year": int(ny_year[fill]),
            "dir": d, "entry": round(entry, 3), "fwd_len": int(wend - fill),
            "mfe": round(mfe_p, 4), "mae": round(mae_p, 4),
            "risk_pct": round(risk, 4), "ft_win_015": ft,
            "trigger_time": int(t[i]), "ibs": round(r["ibs"], 1),
            "body_pct": round(r["body_pct"], 1), "atr_ratio": round(r["atr_ratio"], 2),
        })

    pd.DataFrame(out_rows).to_csv(BASE / "phase1_signals.csv", index=False)

    # ---------------- aggregation ----------------
    def bucket(rows):
        nn = len(rows)
        if not nn:
            return None
        mfs = [x["mfe"] for x in rows]
        mas = [x["mae"] for x in rows]
        rsk = [x["risk_pct"] for x in rows]
        out = {"n": nn,
               "mean_mfe": round(sum(mfs) / nn, 4),
               "mean_mae": round(sum(mas) / nn, 4),
               "mean_diff": round((sum(mfs) - sum(mas)) / nn, 4)}
        for T in THRESHOLDS:
            out[f"mfe_ge_{int(T * 100):03d}"] = sum(1 for v in mfs if v >= T)
            out[f"mae_ge_{int(T * 100):03d}"] = sum(1 for v in mas if v >= T)
        out["ft_win_015"] = round(sum(x["ft_win_015"] for x in rows) / nn, 4)
        out["mean_risk_pct"] = round(sum(rsk) / nn, 4)
        # 1R descriptive (fixed 1:1 rule; MFE/MAE window-censored)
        out["mfe_ge_1r"] = sum(1 for k in range(nn) if mfs[k] >= rsk[k])
        out["mae_ge_1r"] = sum(1 for k in range(nn) if mas[k] >= rsk[k])
        return out

    buckets = []
    per_cfg = {}
    for dname, dd in (("combined", None), ("bull_H2", 1), ("bear_L2", -1)):
        rows = [x for x in out_rows if dd is None or x["dir"] == dd]
        b = bucket(rows)
        if b is None:
            continue
        b.update({"config": "H2L2morning", "dir": dname})
        buckets.append(b)
        per_cfg[dname] = (b, rows)
    with open(BASE / "phase1_buckets.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(buckets[0].keys()))
        w.writeheader()
        w.writerows(buckets)

    yearly = []
    per_year = defaultdict(list)
    for x in out_rows:
        per_year[x["year"]].append(x["mfe"] - x["mae"])
    for y in sorted(per_year):
        vals = per_year[y]
        yearly.append({"year": y, "n": len(vals),
                       "mean_diff": round(sum(vals) / len(vals), 4)})
    with open(BASE / "phase1_yearly.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["year", "n", "mean_diff"])
        w.writeheader()
        w.writerows(yearly)

    # signals/day distribution (cap impact, informational)
    per_nd = defaultdict(int)
    for x in out_rows:
        per_nd[x["nd"]] += 1
    nd_counts = np.array(sorted(per_nd.values()))
    sigday = {"n_signal_days": len(per_nd),
              "max_signals_day": int(nd_counts.max()),
              "days_gt3": int((nd_counts > 3).sum()),
              "signals_lost_if_capped": int(max(0, nd_counts[nd_counts > 3].sum()
                                                - 3 * (nd_counts > 3).sum()))}

    # ---------------- gates (combined unit) ----------------
    def boot(rows, n_boot=10000, seed=11):
        per_day = defaultdict(list)
        for x in rows:
            per_day[x["nd"]].append(x["mfe"] - x["mae"])
        days = np.array([sum(v) / len(v) for v in per_day.values()])
        rng = np.random.default_rng(seed)
        means = rng.choice(days, size=(n_boot, len(days)), replace=True).mean(axis=1)
        return {"n_days": len(days), "p05": round(float(np.percentile(means, 5)), 5),
                "p95": round(float(np.percentile(means, 95)), 5),
                "p_le0": round(float((means <= 0).mean()), 4)}

    gates = {}
    b, rows = per_cfg.get("combined", (None, []))
    if b:
        dom = all(b[f"mfe_ge_{int(T * 100):03d}"] > b[f"mae_ge_{int(T * 100):03d}"]
                  for T in THRESHOLDS)
        g1 = b["n"] >= 150 and sigday["n_signal_days"] >= 80
        g2 = dom and b["mean_diff"] > 0
        bt = boot(rows)
        g3 = bt["p_le0"] <= 0.40
        gates["combined"] = {"n": b["n"], "g1": bool(g1), "dominance": bool(dom),
                             "mean_diff": b["mean_diff"], "g2": bool(g2),
                             "boot": bt, "g3": bool(g3)}
        verdict = ("PASS -> Step 3" if (g1 and g2 and g3)
                   else "FAIL -> STOP (null result)")
        print(f"\ncombined: n={b['n']} days={sigday['n_signal_days']} "
              f"mfe {b['mean_mfe']} mae {b['mean_mae']} diff {b['mean_diff']:+.4f} "
              f"010 {b['mfe_ge_010']} vs {b['mae_ge_010']} "
              f"015 {b['mfe_ge_015']} vs {b['mae_ge_015']} "
              f"020 {b['mfe_ge_020']} vs {b['mae_ge_020']}")
        print(f"1R descriptive: mfe>=1R {b['mfe_ge_1r']} vs mae>=1R {b['mae_ge_1r']} "
              f"(of {b['n']}); mean risk {b['mean_risk_pct']}%")
        print(f"G1={g1} G2={g2} G3={g3} boot={bt}")
        print(f"VERDICT: {verdict}")
    else:
        verdict = "FAIL -> STOP (zero signals)"
        print("VERDICT:", verdict)

    for dname in ("bull_H2", "bear_L2"):
        if dname in per_cfg:
            bb = per_cfg[dname][0]
            print(f"{dname}: n={bb['n']} diff {bb['mean_diff']:+.4f} "
                  f"mfe {bb['mean_mfe']} mae {bb['mean_mae']} risk {bb['mean_risk_pct']}%")

    stats = {"gates": gates, "verdict": verdict, "tallies": dict(tallies),
             "signal_days": sigday,
             "params": {"EMA_SPAN": EMA_SPAN, "ATR_PERIOD": ATR_PERIOD,
                        "ATR_MULT": ATR_MULT, "IBS_HI": IBS_HI, "IBS_LO": IBS_LO,
                        "DOJI_BODY_FRAC": DOJI_BODY_FRAC, "TIMEOUT": TIMEOUT,
                        "FWD_BARS": FWD_BARS, "MIN_FWD": MIN_FWD,
                        "THRESHOLDS": THRESHOLDS, "NY_WINDOW": [NY_LO, NY_HI]},
             "runtime_s": round(timemod.time() - t0, 1)}
    with open(BASE / "phase1_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"wrote funnel/buckets/yearly/signals/stats ({stats['runtime_s']}s)")


if __name__ == "__main__":
    main()
