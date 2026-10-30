# -*- coding: utf-8 -*-
"""
bollinger_mr_phase1.py - Phase 1 signal quality for Bollinger Band mean
reversion (Strategy 6) on US500-tick, per
data/results/bollinger_mr_signal_quality/SPEC.md (hypothesis #9).

M15 aggregated from the verified M1 export (verified: 15 bars/bucket,
cross-checked vs M5->M15). BB(20, 2, ddof=0) on the continuous tape.
Signals = M15 closes at/beyond bands under VWAP-MR range-day + max-move
conditions (rolling, reused verbatim). Exit-free direction-native
MFE/MAE on the M5 path to the 15:55 forced exit, E-XTM first-bar
convention. Day-block bootstrap gate per the intraday-momentum lesson.

Outputs (data/results/bollinger_mr_signal_quality/):
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

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\bollinger_mr_signal_quality")
M1 = Path(r"C:\Users\Milad\Desktop\RR\data\results\vwap_2021_2026\us500tick_m1.csv")
M5 = Path(r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present\us500tick_m5.csv")

BB_N, BB_K = 20, 2.0
RANGE_NET_PCT = 0.6
RANGE_HL_PCT = 1.0
MAXMOVE_PCT = 0.42
SESSION_OPEN_M, SESSION_LAST_M = 570, 959
LAST_ENTRY_M = 930          # entry-bar open cutoff -> signal wall <= 915
FORCE_EXIT_M = 955
THRESHOLDS = (0.10, 0.15, 0.20)


def main():
    t0 = timemod.time()

    # ---------------- M15 aggregation + verification ----------------
    # Primary: M5 -> M15 (verified tape, 3 bars/bucket, partials tallied).
    # Cross-check: M1 -> M15 must match OHLC on buckets complete in BOTH.
    m5 = pd.read_csv(M5)
    m5["bk"] = m5["time"] // 900
    g5 = m5.groupby("bk")
    cnt5 = g5.size()
    partial5 = cnt5[cnt5 != 3]
    m15 = pd.DataFrame({"time": cnt5.index * 900,
                        "open": g5["open"].first(), "high": g5["high"].max(),
                        "low": g5["low"].min(), "close": g5["close"].last()})
    m1 = pd.read_csv(M1)
    m1["bk"] = m1["time"] // 900
    g = m1.groupby("bk")
    cnt = g.size()
    m15b = pd.DataFrame({"time": cnt.index * 900,
                         "open": g["open"].first(), "high": g["high"].max(),
                         "low": g["low"].min(), "close": g["close"].last()})
    ok15 = cnt[cnt == 15].index
    ok3 = cnt5[cnt5 == 3].index
    both = ok15.intersection(ok3)
    a = m15[m15["time"].isin(both)].sort_values("time").reset_index(drop=True)
    bq = m15b[m15b["time"].isin(both)].sort_values("time").reset_index(drop=True)
    mism = int(((a["open"] != bq["open"]) | (a["high"] != bq["high"]) |
                (a["low"] != bq["low"]) | (a["close"] != bq["close"])).sum())
    print(f"M15 agg (M5 primary): {len(m15)} buckets; partial M5 buckets: "
          f"{len(partial5)}; M1 partial buckets: {int((cnt != 15).sum())} "
          f"(M1 skips no-tick minutes); OHLC mismatches on complete-in-both "
          f"buckets ({len(both)}): {mism}")
    if mism:
        print("AGGREGATION VERIFICATION FAILED - abort")
        return

    # ---------------- Bollinger Bands on continuous tape ----------------
    close = m15["close"]
    mid = close.rolling(BB_N).mean()
    sd = close.rolling(BB_N).std(ddof=0)
    up = mid + BB_K * sd
    lo = mid - BB_K * sd
    bw = up - lo
    t15 = m15["time"].to_numpy(np.int64)
    o15 = m15["open"].to_numpy(float)
    h15 = m15["high"].to_numpy(float)
    l15 = m15["low"].to_numpy(float)
    c15 = close.to_numpy(float)
    mid_a, up_a, lo_a, bw_a = (mid.to_numpy(float), up.to_numpy(float),
                               lo.to_numpy(float), bw.to_numpy(float))

    idx = pd.DatetimeIndex(pd.to_datetime(t15, unit="s", utc=True))
    ny = idx.tz_convert("America/New_York")
    wall = (ny.hour * 60 + ny.minute).to_numpy()
    nd = (ny.year * 10000 + ny.month * 100 + ny.day).to_numpy()
    year = ny.year.to_numpy()

    # M5 path arrays per date
    idx5 = pd.DatetimeIndex(pd.to_datetime(m5["time"].to_numpy(np.int64), unit="s", utc=True))
    ny5 = idx5.tz_convert("America/New_York")
    nd5 = (ny5.year * 10000 + ny5.month * 100 + ny5.day).to_numpy()
    wall5 = (ny5.hour * 60 + ny5.minute).to_numpy()
    m5_by_date = defaultdict(list)
    for k in range(len(m5)):
        if wall5[k] < FORCE_EXIT_M:
            m5_by_date[nd5[k]].append(k)
    for k in m5_by_date:
        m5_by_date[k].sort()
    h5 = m5["high"].to_numpy(float)
    l5 = m5["low"].to_numpy(float)
    c5 = m5["close"].to_numpy(float)
    o5 = m5["open"].to_numpy(float)
    t5 = m5["time"].to_numpy(np.int64)

    # ---------------- per-day session walk ----------------
    days = sorted(set(nd))
    sess_by_day = defaultdict(list)
    for i in range(len(nd)):
        if SESSION_OPEN_M <= wall[i] <= SESSION_LAST_M:
            sess_by_day[nd[i]].append(i)

    signals = []
    funnel = defaultdict(lambda: [0, 0])   # stage -> [long, short]
    tallies = defaultdict(int)
    for d in days:
        bars = sess_by_day[d]
        if not bars or wall[bars[0]] != SESSION_OPEN_M:
            tallies["no_session_open"] += 1
            continue
        sopen = o15[bars[0]]
        sh, sl_ = -1e18, 1e18
        for bi, i in enumerate(bars):
            sh = max(sh, h15[i])
            sl_ = min(sl_, l15[i])
            if bi < BB_N or not np.isfinite(bw_a[i]) or bw_a[i] <= 0:
                continue
            dlong = c15[i] <= lo_a[i]
            dshrt = c15[i] >= up_a[i]
            if not (dlong or dshrt):
                continue
            s = 0 if dlong else 1
            funnel["F0_raw_band_closes"][s] += 1
            net_pct = abs(c15[i] / sopen - 1.0) * 100.0
            hl_pct = (sh - sl_) / sopen * 100.0
            if not (net_pct <= RANGE_NET_PCT and hl_pct <= RANGE_HL_PCT):
                continue
            funnel["F1_after_range_day"][s] += 1
            if abs(c15[i] - sopen) > sopen * MAXMOVE_PCT / 100.0:
                continue
            funnel["F2_after_max_move"][s] += 1
            if wall[i] > LAST_ENTRY_M - 15:
                continue
            funnel["F3_after_entry_window"][s] += 1
            path = m5_by_date.get(d, [])
            entry_ix = None
            for e, k in enumerate(path):
                if t5[k] >= t15[i] + 900:
                    entry_ix = e
                    break
            if entry_ix is None:
                tallies["no_next_bar"] += 1
                continue
            funnel["F4_final_signals"][s] += 1
            prev = bars[bi - 1] if bi > 0 else None
            ep_first = (prev is None or
                        (dlong and c15[prev] > lo_a[prev]) or
                        (dshrt and c15[prev] < up_a[prev]))
            signals.append({"i": i, "d": d, "dir": 1 if dlong else -1,
                            "nd": d, "year": year[i], "entry_ix": entry_ix,
                            "path": path, "ep_first": bool(ep_first),
                            "bw_pct": bw_a[i] / c15[i] * 100.0,
                            "beyond_pct": (lo_a[i] - c15[i]) / c15[i] * 100.0
                            if dlong else (c15[i] - up_a[i]) / c15[i] * 100.0})
    print(f"session walk done ({timemod.time() - t0:.0f}s); "
          f"signals {len(signals)}; tallies {dict(tallies)}")

    # ---------------- excursions (E-XTM on M5 path) ----------------
    out_rows = []
    for s in signals:
        path = s["path"]
        k0 = path[s["entry_ix"]]
        entry = o5[k0]
        d = s["dir"]
        if d == 1:
            mfe = max(c5[k0] - entry, 0.0)
            mae = max(entry - l5[k0], 0.0)
            for k in path[s["entry_ix"] + 1:]:
                mfe = max(mfe, h5[k] - entry)
                mae = max(mae, entry - l5[k])
        else:
            mfe = max(entry - c5[k0], 0.0)
            mae = max(h5[k0] - entry, 0.0)
            for k in path[s["entry_ix"] + 1:]:
                mfe = max(mfe, entry - l5[k])
                mae = max(mae, h5[k] - entry)
        # first-touch 0.15 informational
        ft = 0
        for k in path[s["entry_ix"]:]:
            fav = (h5[k] - entry) if d == 1 else (entry - l5[k])
            adv = (entry - l5[k]) if d == 1 else (h5[k] - entry)
            fu = fav / entry * 100.0 >= 0.15
            fd = adv / entry * 100.0 >= 0.15
            if fu or fd:
                ft = 1 if (fu and not fd) else 0
                break
        out_rows.append({
            "time": int(t5[k0]), "nd": s["nd"], "year": s["year"],
            "dir": s["dir"], "entry": round(entry, 3),
            "mfe": round(mfe / entry * 100.0, 4),
            "mae": round(mae / entry * 100.0, 4),
            "bw_pct": round(s["bw_pct"], 4),
            "beyond_pct": round(s["beyond_pct"], 4),
            "ft_win_015": ft, "ep_first": int(s["ep_first"]),
            "fwd_len": len(path) - s["entry_ix"]})

    pd.DataFrame(out_rows).to_csv(BASE / "phase1_signals.csv", index=False)

    with open(BASE / "phase1_funnel.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stage", "long", "short", "total"])
        for st in ("F0_raw_band_closes", "F1_after_range_day",
                   "F2_after_max_move", "F3_after_entry_window",
                   "F4_final_signals"):
            a, b = funnel[st]
            w.writerow([st, a, b, a + b])
            print(f"funnel {st}: long {a} short {b} total {a + b}")

    # ---------------- aggregation ----------------
    def bucket(rows):
        if not rows:
            return None
        nn = len(rows)
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
        out["mean_bw_pct"] = round(sum(r["bw_pct"] for r in rows) / nn, 4)
        out["mean_beyond_pct"] = round(
            sum(r["beyond_pct"] for r in rows) / nn, 4)
        return out

    buckets = []
    per_cfg = {}
    for name, dd in (("pooled", None), ("long", 1), ("short", -1)):
        rows = [r for r in out_rows if dd is None or r["dir"] == dd]
        b = bucket(rows)
        if b is None:
            continue
        b.update({"config": "bb_mr", "dir": name})
        buckets.append(b)
        per_cfg[name] = (b, rows)
    ep_rows = [r for r in out_rows if r["ep_first"] == 1]
    be = bucket(ep_rows)
    if be:
        be.update({"config": "bb_mr", "dir": "episode_first"})
        buckets.append(be)
    with open(BASE / "phase1_buckets.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(buckets[0].keys()))
        w.writeheader()
        w.writerows(buckets)

    yearly = []
    py = defaultdict(list)
    for r in out_rows:
        py[r["year"]].append(r["mfe"] - r["mae"])
    for y in sorted(py):
        v = py[y]
        yearly.append({"year": y, "n": len(v),
                       "mean_diff": round(sum(v) / len(v), 4)})
    with open(BASE / "phase1_yearly.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["year", "n", "mean_diff"])
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
        print(f"episode_first: n={be['n'] if be else 0} "
              f"diff {be['mean_diff'] if be else None}")
        print(f"G1={g1} G2={g2} G3={g3} boot={bt}")
        print(f"VERDICT: {verdict}")
        for nm in ("long", "short"):
            if nm in per_cfg:
                bb = per_cfg[nm][0]
                print(f"{nm}: n={bb['n']} diff {bb['mean_diff']:+.4f} "
                      f"mfe {bb['mean_mfe']} mae {bb['mean_mae']}")
    else:
        verdict = "FAIL -> STOP (zero signals)"
        print("VERDICT:", verdict)

    stats = {"gates": gates, "verdict": verdict, "tallies": dict(tallies),
             "params": {"BB_N": BB_N, "BB_K": BB_K,
                        "RANGE_NET_PCT": RANGE_NET_PCT,
                        "RANGE_HL_PCT": RANGE_HL_PCT,
                        "MAXMOVE_PCT": MAXMOVE_PCT,
                        "LAST_ENTRY_M": LAST_ENTRY_M,
                        "FORCE_EXIT_M": FORCE_EXIT_M,
                        "THRESHOLDS": THRESHOLDS},
             "runtime_s": round(timemod.time() - t0, 1)}
    with open(BASE / "phase1_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"wrote artifacts ({stats['runtime_s']}s)")


if __name__ == "__main__":
    main()
