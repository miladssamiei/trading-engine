# -*- coding: utf-8 -*-
"""
fvg_continuation_phase1.py - Phase 1 signal quality for the Fair Value
Gap retracement-continuation setup (hypothesis #10, continuation family
attempt #5) on US500-tick, per
data/results/fvg_continuation_signal_quality/SPEC.md.

M15 anchor FVGs (3-candle, strict), mitigation/staleness state tracked
on the M5 execution tape, running-extreme trigger (close beyond the
max/min since the impulse bar), swing stop context, fixed session
window [09:30,11:00), risk band [0.14,0.56]%. Exit-free direction-native
MFE/MAE (E-XTM, 18-bar/90-min window capped at session end, MIN_FWD 6).
Day-block bootstrap gate.

Outputs (data/results/fvg_continuation_signal_quality/):
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

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\fvg_continuation_signal_quality")
M5 = Path(r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present\us500tick_m5.csv")
M1 = Path(r"C:\Users\Milad\Desktop\RR\data\results\vwap_2021_2026\us500tick_m1.csv")

NY_LO, NY_HI = 570, 660          # trigger-bar window [09:30, 11:00)
RISK_LO, RISK_HI = 0.14, 0.56    # converted 10-40 pt band, % of entry
UNTOUCHED_EXPIRY_S = 30 * 86400
ARMED_EXPIRY_BARS = 576
TIMEOUT = 0                      # entry is market at next bar open
FWD_BARS = 18
MIN_FWD = 6
SESSION_GAP = 3600
THRESHOLDS = (0.10, 0.15, 0.20)


def main():
    t0 = timemod.time()

    # ---------------- data: verified M5 + M15 aggregation ----------------
    m5 = pd.read_csv(M5)
    t = m5["time"].to_numpy(np.int64)
    o = m5["open"].to_numpy(float)
    h = m5["high"].to_numpy(float)
    l = m5["low"].to_numpy(float)
    c = m5["close"].to_numpy(float)
    n = len(m5)

    m5["bk"] = m5["time"] // 900
    g5 = m5.groupby("bk")
    cnt5 = g5.size()
    partial5 = int((cnt5 != 3).sum())
    m15 = pd.DataFrame({"time": cnt5.index * 900,
                        "open": g5["open"].first(), "high": g5["high"].max(),
                        "low": g5["low"].min(), "close": g5["close"].last()})
    t15 = m15["time"].to_numpy(np.int64)
    h15 = m15["high"].to_numpy(float)
    l15 = m15["low"].to_numpy(float)
    n15 = len(m15)
    # cross-check vs M1-derived on complete-in-both buckets
    m1 = pd.read_csv(M1)
    m1["bk"] = m1["time"] // 900
    g1 = m1.groupby("bk")
    cnt1 = g1.size()
    m15b = pd.DataFrame({"time": cnt1.index * 900,
                         "open": g1["open"].first(), "high": g1["high"].max(),
                         "low": g1["low"].min(), "close": g1["close"].last()})
    both = cnt5[cnt5 == 3].index.intersection(cnt1[cnt1 == 15].index)
    a = m15[m15["time"].isin(both)].sort_values("time").reset_index(drop=True)
    bq = m15b[m15b["time"].isin(both)].sort_values("time").reset_index(drop=True)
    mism = int(((a["open"] != bq["open"]) | (a["high"] != bq["high"]) |
                (a["low"] != bq["low"]) | (a["close"] != bq["close"])).sum())
    print(f"M15 agg: {n15} buckets, partial {partial5}; M1 cross-check "
          f"mismatches on {len(both)} complete-in-both buckets: {mism}")
    if mism:
        print("AGGREGATION VERIFICATION FAILED - abort")
        return

    # per-M15-bar span extremes on M5 (for running-extreme init)
    span_hi = np.full(n15, -1e18)
    span_lo = np.full(n15, 1e18)
    j0 = np.searchsorted(t, t15)
    j1 = np.searchsorted(t, t15 + 900)
    for i in range(n15):
        if j1[i] > j0[i]:
            span_hi[i] = h[j0[i]:j1[i]].max()
            span_lo[i] = l[j0[i]:j1[i]].min()

    # FVG detection (M15, strict)
    fvg_events = []   # (create_time, sort key) -> fields
    for i in range(2, n15):
        if h15[i - 2] < l15[i]:
            fvg_events.append((t15[i] + 900, i, 1, h15[i - 2], l15[i]))
        if l15[i - 2] > h15[i]:
            fvg_events.append((t15[i] + 900, i, -1, h15[i], l15[i - 2]))
    fvg_events.sort(key=lambda x: x[0])
    print(f"FVGs detected on M15: {len(fvg_events)} "
          f"(bull {sum(1 for e in fvg_events if e[2] == 1)}, "
          f"bear {sum(1 for e in fvg_events if e[2] == -1)})")

    idx = pd.DatetimeIndex(pd.to_datetime(t, unit="s", utc=True))
    ny = idx.tz_convert("America/New_York")
    ny_min = (ny.hour * 60 + ny.minute).to_numpy()
    ny_year = ny.year.to_numpy()
    ny_date = (ny.year * 10000 + ny.month * 100 + ny.day).to_numpy()

    sess_end = np.zeros(n, dtype=np.int64)
    last = n - 1
    for i in range(n - 1, -1, -1):
        if i == n - 1 or t[i] - t[i + 1] > SESSION_GAP or t[i + 1] - t[i] > SESSION_GAP:
            last = i
        sess_end[i] = last

    # ---------------- chronological M5 pass ----------------
    funnel = defaultdict(lambda: [0, 0])
    tallies = defaultdict(int)
    raw = []           # trigger records pre-fill
    waiting = []       # dicts: untouched
    armed = []         # dicts: touched, watching trigger
    ei = 0             # fvg_events cursor
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
            funnel["F0_raw_fvgs"][0 if d == 1 else 1] += 0  # counted later
        hk, lk, ck = h[k], l[k], c[k]

        still_w = []
        for f in waiting:
            si = 0 if f["d"] == 1 else 1
            if tk - f["ct"] > UNTOUCHED_EXPIRY_S:
                tallies["untouched_expired"] += 1
                continue
            if f["d"] == 1:
                if lk <= f["zbot"]:
                    tallies["inverted_on_contact"] += 1
                    continue
                if lk <= f["ztop"]:
                    f["arm_k"] = k
                    f["swing"] = lk
                    f["touch_t"] = tk
                    armed.append(f)
                    funnel["F1_touched"][si] += 1
                    continue
                f["rmax"] = max(f["rmax"], hk)
            else:
                if hk >= f["ztop"]:
                    tallies["inverted_on_contact"] += 1
                    continue
                if hk >= f["zbot"]:
                    f["arm_k"] = k
                    f["swing"] = hk
                    f["touch_t"] = tk
                    armed.append(f)
                    funnel["F1_touched"][si] += 1
                    continue
                f["rmin"] = min(f["rmin"], lk)
            still_w.append(f)
        waiting = still_w

        still_a = []
        for f in armed:
            si = 0 if f["d"] == 1 else 1
            if k - f["arm_k"] > ARMED_EXPIRY_BARS:
                tallies["armed_expired"] += 1
                continue
            if f["d"] == 1:
                if lk <= f["zbot"]:
                    tallies["inverted_while_armed"] += 1
                    continue
                if ck > f["rmax"]:
                    funnel["F2_trigger_fired"][si] += 1
                    if not (NY_LO <= ny_min[k] < NY_HI):
                        tallies["trigger_out_of_window"] += 1
                        continue
                    funnel["F3_in_window"][si] += 1
                    raw.append({"k": k, "d": 1,
                                "swing": min(f["swing"], lk),
                                "zone_pct": (f["ztop"] - f["zbot"]) / ck * 100.0,
                                "touch_t": f["touch_t"]})
                    continue
                f["rmax"] = max(f["rmax"], hk)
                f["swing"] = min(f["swing"], lk)
            else:
                if hk >= f["ztop"]:
                    tallies["inverted_while_armed"] += 1
                    continue
                if ck < f["rmin"]:
                    funnel["F2_trigger_fired"][si] += 1
                    if not (NY_LO <= ny_min[k] < NY_HI):
                        tallies["trigger_out_of_window"] += 1
                        continue
                    funnel["F3_in_window"][si] += 1
                    raw.append({"k": k, "d": -1,
                                "swing": max(f["swing"], hk),
                                "zone_pct": (f["ztop"] - f["zbot"]) / ck * 100.0,
                                "touch_t": f["touch_t"]})
                    continue
                f["rmin"] = min(f["rmin"], lk)
                f["swing"] = max(f["swing"], hk)
            still_a.append(f)
        armed = still_a
    # F0 counts (per direction) from the detection pass
    funnel["F0_raw_fvgs"][0] = sum(1 for e in fvg_events if e[2] == 1)
    funnel["F0_raw_fvgs"][1] = sum(1 for e in fvg_events if e[2] == -1)
    print(f"M5 pass done ({timemod.time() - t0:.0f}s); tallies {dict(tallies)}")

    # ---------------- fills + risk band + excursions ----------------
    out_rows = []
    for r in raw:
        k = r["k"]
        if k + 1 >= n:
            tallies["no_next_bar"] += 1
            continue
        entry = o[k + 1]
        d = r["d"]
        risk = ((entry - r["swing"]) if d == 1 else (r["swing"] - entry)) / entry * 100.0
        if not (RISK_LO <= risk <= RISK_HI):
            tallies["risk_out_of_band"] += 1
            continue
        funnel["F4_risk_band_final"][0 if d == 1 else 1] += 1
        fill = k + 1
        wend = min(fill + FWD_BARS, sess_end[fill])
        if wend - fill < MIN_FWD:
            tallies["ineligible_fwd"] += 1
            continue
        tallies["filled_eligible"] += 1
        if d == 1:
            mfe = max(c[fill] - entry, 0.0)
            mae = max(entry - l[fill], 0.0)
            for kk in range(fill + 1, wend + 1):
                mfe = max(mfe, h[kk] - entry)
                mae = max(mae, entry - l[kk])
        else:
            mfe = max(entry - c[fill], 0.0)
            mae = max(h[fill] - entry, 0.0)
            for kk in range(fill + 1, wend + 1):
                mfe = max(mfe, entry - l[kk])
                mae = max(mae, h[kk] - entry)
        ft = 0
        for kk in range(fill, wend + 1):
            fav = (h[kk] - entry) if d == 1 else (entry - l[kk])
            adv = (entry - l[kk]) if d == 1 else (h[kk] - entry)
            fu = fav / entry * 100.0 >= 0.15
            fd = adv / entry * 100.0 >= 0.15
            if fu or fd:
                ft = 1 if (fu and not fd) else 0
                break
        out_rows.append({
            "time": int(t[fill]), "nd": int(ny_date[fill]),
            "year": int(ny_year[fill]), "dir": d,
            "entry": round(entry, 3), "fwd_len": int(wend - fill),
            "mfe": round(mfe / entry * 100.0, 4),
            "mae": round(mae / entry * 100.0, 4),
            "risk_pct": round(risk, 4), "zone_pct": round(r["zone_pct"], 4),
            "ret_bars": int((t[k] - r["touch_t"]) // 300),
            "ft_win_015": ft, "trigger_time": int(t[k])})

    pd.DataFrame(out_rows).to_csv(BASE / "phase1_signals.csv", index=False)
    with open(BASE / "phase1_funnel.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stage", "bull", "bear", "total"])
        for st in ("F0_raw_fvgs", "F1_touched", "F2_trigger_fired",
                   "F3_in_window", "F4_risk_band_final"):
            a, b = funnel[st]
            w.writerow([st, a, b, a + b])
            print(f"funnel {st}: bull {a} bear {b} total {a + b}")

    # ---------------- aggregation ----------------
    def bucket(rows):
        if not rows:
            return None
        nn = len(rows)
        mfs = [x["mfe"] for x in rows]
        mas = [x["mae"] for x in rows]
        out = {"n": nn,
               "mean_mfe": round(sum(mfs) / nn, 4),
               "mean_mae": round(sum(mas) / nn, 4),
               "mean_diff": round((sum(mfs) - sum(mas)) / nn, 4)}
        for T in THRESHOLDS:
            out[f"mfe_ge_{int(T * 100):03d}"] = sum(1 for v in mfs if v >= T)
            out[f"mae_ge_{int(T * 100):03d}"] = sum(1 for v in mas if v >= T)
        out["ft_win_015"] = round(sum(x["ft_win_015"] for x in rows) / nn, 4)
        out["mean_risk_pct"] = round(sum(x["risk_pct"] for x in rows) / nn, 4)
        out["mean_zone_pct"] = round(sum(x["zone_pct"] for x in rows) / nn, 4)
        return out

    buckets = []
    per_cfg = {}
    for name, dd in (("pooled", None), ("long", 1), ("short", -1)):
        rows = [x for x in out_rows if dd is None or x["dir"] == dd]
        b = bucket(rows)
        if b is None:
            continue
        b.update({"config": "fvg_cont", "dir": name})
        buckets.append(b)
        per_cfg[name] = (b, rows)
    with open(BASE / "phase1_buckets.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(buckets[0].keys()))
        w.writeheader()
        w.writerows(buckets)

    yearly = []
    py = defaultdict(list)
    for x in out_rows:
        py[x["year"]].append(x["mfe"] - x["mae"])
    for y in sorted(py):
        v = py[y]
        yearly.append({"year": y, "n": len(v),
                       "mean_diff": round(sum(v) / len(v), 4)})
    with open(BASE / "phase1_yearly.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["year", "n", "mean_diff"])
        w.writeheader()
        w.writerows(yearly)

    per_nd = defaultdict(int)
    for x in out_rows:
        per_nd[x["nd"]] += 1
    ndc = np.array(sorted(per_nd.values())) if per_nd else np.array([0])
    sigday = {"n_signal_days": len(per_nd),
              "max_signals_day": int(ndc.max()),
              "days_gt2": int((ndc > 2).sum()),
              "signals_lost_if_capped": int(max(0, ndc[ndc > 2].sum()
                                                - 2 * (ndc > 2).sum()))}

    # ---------------- gates ----------------
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

    b, rows = per_cfg.get("pooled", (None, []))
    gates = {}
    if b:
        dom = all(b[f"mfe_ge_{int(T * 100):03d}"] > b[f"mae_ge_{int(T * 100):03d}"]
                  for T in THRESHOLDS)
        bt = boot(rows)
        g1 = b["n"] >= 150
        g2 = dom and b["mean_diff"] > 0
        g3 = bt["p_le0"] <= 0.40
        gates["pooled"] = {"n": b["n"], "g1": bool(g1), "dominance": bool(dom),
                           "mean_diff": b["mean_diff"], "g2": bool(g2),
                           "boot": bt, "g3": bool(g3)}
        verdict = ("PASS -> Step 3" if (g1 and g2 and g3)
                   else "FAIL -> STOP (null result)")
        print(f"\npooled: n={b['n']} mfe {b['mean_mfe']} mae {b['mean_mae']} "
              f"diff {b['mean_diff']:+.4f} 010 {b['mfe_ge_010']} vs {b['mae_ge_010']} "
              f"015 {b['mfe_ge_015']} vs {b['mae_ge_015']} "
              f"020 {b['mfe_ge_020']} vs {b['mae_ge_020']}")
        print(f"G1={g1} G2={g2} G3={g3} boot={bt}")
        print(f"signal_days={sigday}")
        print(f"VERDICT: {verdict}")
        for nm in ("long", "short"):
            if nm in per_cfg:
                bb = per_cfg[nm][0]
                print(f"{nm}: n={bb['n']} diff {bb['mean_diff']:+.4f} "
                      f"mfe {bb['mean_mfe']} mae {bb['mean_mae']} "
                      f"risk {bb['mean_risk_pct']}%")
    else:
        verdict = "FAIL -> STOP (zero signals)"
        print("VERDICT:", verdict)

    stats = {"gates": gates, "verdict": verdict, "tallies": dict(tallies),
             "signal_days": sigday,
             "params": {"NY_WINDOW": [NY_LO, NY_HI],
                        "RISK_BAND": [RISK_LO, RISK_HI],
                        "UNTOUCHED_EXPIRY_S": UNTOUCHED_EXPIRY_S,
                        "ARMED_EXPIRY_BARS": ARMED_EXPIRY_BARS,
                        "FWD_BARS": FWD_BARS, "MIN_FWD": MIN_FWD,
                        "THRESHOLDS": THRESHOLDS},
             "runtime_s": round(timemod.time() - t0, 1)}
    with open(BASE / "phase1_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"wrote artifacts ({stats['runtime_s']}s)")


if __name__ == "__main__":
    main()
