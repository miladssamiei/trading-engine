# -*- coding: utf-8 -*-
"""
session_transfer_phase1.py - Phase 1 signal quality for the session
transfer mechanism (hypothesis #11), per
data/results/session_transfer/SPEC_SESSION_TRANSFER.txt sections 2-6
and GATE_SESSION_TRANSFER.txt (G1-G5 pre-declared).

Two arms: CARRY (|r_pre| and vol_pre >= causal trailing-252d P80 ->
direction sign(r_pre) through the cash block) and FADE (|r_post| >= P80
AND sign(r_post) = -sign(r_cash) -> fade the overnight 20:00 -> next
09:30). Direction-native E-XTM MFE/MAE, window-scaled thresholds, iid
day bootstrap + 10-day moving-block bootstrap, descriptive splits
(hl_cash, vol_cash, D1 regime).

Outputs (data/results/session_transfer/):
  day_blocks.csv, phase1_carry.csv, phase1_fade.csv,
  signal_funnel.csv, yearly_breakdown.csv, phase1_stats.json
"""

import csv
import json
import time as timemod
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\session_transfer")
M5 = Path(r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present\us500tick_m5.csv")
D1 = Path(r"C:\Users\Milad\Desktop\RR\data\results\momentum_multihorizon\us500c_d1.csv")

PCT_WIN = 252
PCT_Q = 80
CARRY_T = (0.25, 0.50, 0.75)
FADE_T = (0.10, 0.20, 0.30)
MEDIAN_GAP_MIN = 0.05
N_BOOT = 10000
BLOCK_LEN = 10
COST_PT = 0.8


def excursion(bars, entry, d, o, h, l, c):
    """Direction-native E-XTM excursion: first bar close/extremes then
    running extremes. d=+1 long, -1 short. Returns (mfe_pct, mae_pct,
    fwd_bars)."""
    if len(bars) == 0:
        return None
    k0 = bars[0]
    if d == 1:
        mfe = max(c[k0] - entry, 0.0)
        mae = max(entry - l[k0], 0.0)
        for k in bars[1:]:
            mfe = max(mfe, h[k] - entry)
            mae = max(mae, entry - l[k])
    else:
        mfe = max(entry - c[k0], 0.0)
        mae = max(h[k0] - entry, 0.0)
        for k in bars[1:]:
            mfe = max(mfe, entry - l[k])
            mae = max(mae, h[k] - entry)
    return mfe / entry * 100.0, mae / entry * 100.0, len(bars)


def boot_iid(vals, seed=11):
    rng = np.random.default_rng(seed)
    v = np.array(vals)
    means = rng.choice(v, size=(N_BOOT, len(v)), replace=True).mean(axis=1)
    return round(float((means <= 0).mean()), 4)


def boot_moving_block(vals, seed=13):
    rng = np.random.default_rng(seed)
    v = np.array(vals)
    n = len(v)
    if n < BLOCK_LEN:
        return boot_iid(vals, seed)
    nblk = int(np.ceil(n / BLOCK_LEN))
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        starts = rng.integers(0, n - BLOCK_LEN + 1, size=nblk)
        idx = (starts[:, None] + np.arange(BLOCK_LEN)[None, :]).ravel()[:n]
        out[b] = v[idx].mean()
    return round(float((out <= 0).mean()), 4)


def arm_stats(rows, thresholds):
    if not rows:
        return None
    mfe = np.array([r["mfe"] for r in rows])
    mae = np.array([r["mae"] for r in rows])
    diff = mfe - mae
    s = {"n": len(rows),
         "mean_mfe": round(float(mfe.mean()), 4),
         "mean_mae": round(float(mae.mean()), 4),
         "mean_diff": round(float(diff.mean()), 4),
         "median_mfe": round(float(np.median(mfe)), 4),
         "median_mae": round(float(np.median(mae)), 4),
         "median_diff": round(float(np.median(diff)), 4)}
    for t in thresholds:
        s[f"mfe_ge_{int(t * 100):03d}"] = int((mfe >= t).sum())
        s[f"mae_ge_{int(t * 100):03d}"] = int((mae >= t).sum())
    s["boot_iid_p_le0"] = boot_iid(diff.tolist())
    s["boot_mb10_p_le0"] = boot_moving_block(diff.tolist())
    # after-cost descriptive: median(MFE - cost%) per SPEC section 6
    cost = np.array([0.8 / r["entry"] * 100.0 for r in rows])
    s["median_mfe_net_of_cost"] = round(float(np.median(mfe - cost)), 4)
    return s


def evaluate_gates(rows, thresholds, all_days, eligible_days, label):
    s = arm_stats(rows, thresholds)
    if s is None:
        return None, {"G1": False, "G2": False, "G3": False, "G4": False,
                      "G5": False}, "FAIL (no signals)"
    years = defaultdict(list)
    for r in rows:
        years[r["year"]].append(r["mfe"] - r["mae"])
    full_years = {y: v for y, v in years.items() if y < 2026}
    n_pos = sum(1 for v in full_years.values() if np.mean(v) > 0)
    max_share = max((len(v) for y, v in years.items())) / s["n"] * 100
    dom = all(s[f"mfe_ge_{int(t * 100):03d}"] > s[f"mae_ge_{int(t * 100):03d}"]
              for t in thresholds)
    g1 = s["n"] >= 150 and s["n"] >= 0.05 * eligible_days
    g2 = dom and s["mean_diff"] > 0
    g3 = s["median_diff"] >= MEDIAN_GAP_MIN
    g4 = s["boot_iid_p_le0"] <= 0.40 and s["boot_mb10_p_le0"] <= 0.45
    g5 = max_share <= 40 and n_pos >= 4
    gates = {"G1": bool(g1), "G2": bool(g2), "G3": bool(g3),
             "G4": bool(g4), "G5": bool(g5)}
    passed = all(gates.values())
    verdict = "PASS -> Phase 2" if passed else "FAIL -> STOP"
    print(f"\n{label}: n={s['n']} ({s['n'] / eligible_days * 100:.1f}% of "
          f"{eligible_days} eligible days)")
    print(f"  mean mfe/mae {s['mean_mfe']}/{s['mean_mae']} "
          f"diff {s['mean_diff']:+.4f}; median mfe/mae "
          f"{s['median_mfe']}/{s['median_mae']} diff {s['median_diff']:+.4f}")
    for t in thresholds:
        k = int(t * 100)
        print(f"  t={t:.2f}%: mfe {s[f'mfe_ge_{k:03d}']} vs mae {s[f'mae_ge_{k:03d}']}")
    print(f"  boot iid P<=0 {s['boot_iid_p_le0']}, moving-block(10) "
          f"{s['boot_mb10_p_le0']}; max-year share {max_share:.1f}%; "
          f"positive full years {n_pos}/{len(full_years)}")
    print(f"  gates {gates} -> {verdict}")
    return s, gates, verdict


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
    wall = (ny.hour * 60 + ny.minute).to_numpy()
    nd = (ny.year * 10000 + ny.month * 100 + ny.day).to_numpy()
    nyear = ny.year.to_numpy()

    # ---------------- per-day block aggregation ----------------
    days = []
    seen = set()
    bar_day = defaultdict(list)
    for k in range(n):
        if nd[k] not in seen:
            seen.add(nd[k])
            days.append(nd[k])
        bar_day[nd[k]].append(k)

    day_rows = []
    day_index = {}
    for d in days:
        ks = bar_day[d]
        walls = wall[ks]
        pre = [k for k in ks if 480 <= wall[k] < 570]
        cash = [k for k in ks if 570 <= wall[k] < 960]
        post = [k for k in ks if 960 <= wall[k] < 1200]
        pre_ok = (len(pre) > 0 and wall[pre[0]] == 480 and 565 in walls)
        cash_ok = (len(cash) > 0 and wall[cash[0]] == 570)
        post_ok = (len(post) > 0 and wall[post[0]] == 960 and 1195 in walls)
        row = {"nd": d, "year": d // 10000,
               "pre_ok": pre_ok, "cash_ok": cash_ok, "post_ok": post_ok}
        if pre_ok:
            o0, cN = o[pre[0]], c[pre[-1]]
            row["r_pre"] = float(np.log(cN / o0))
            row["vol_pre"] = float(sum(abs(np.log(c[k] / o[k])) for k in pre))
            row["pre_close_t"] = int(t[pre[-1]])
        if cash_ok:
            o0, cN = o[cash[0]], c[cash[-1]]
            row["r_cash"] = float(np.log(cN / o0))
            row["vol_cash"] = float(sum(abs(np.log(c[k] / o[k])) for k in cash))
            row["hl_cash"] = float((max(h[k] for k in cash)
                                    - min(l[k] for k in cash)) / o0)
            row["cash_open"] = float(o0)
            row["cash_open_t"] = int(t[cash[0]])
            row["cash_close_t"] = int(t[cash[-1]])
        if post_ok:
            o0, cN = o[post[0]], c[post[-1]]
            row["r_post"] = float(np.log(cN / o0))
            row["post_close"] = float(cN)
            row["post_close_t"] = int(t[post[-1]])
        day_rows.append(row)
        day_index[d] = row

    cash_days = [r["nd"] for r in day_rows if r["cash_ok"]]
    pre_days = [r["nd"] for r in day_rows if r["pre_ok"]]
    post_days = [r["nd"] for r in day_rows if r["post_ok"] and r["cash_ok"]]
    print(f"days total {len(day_rows)}; cash {len(cash_days)}, pre-valid "
          f"{len(pre_days)}, post+cash-valid {len(post_days)} "
          f"({timemod.time() - t0:.0f}s)")
    hl_med = float(np.median([day_index[d]["hl_cash"] for d in cash_days]))
    vc_med = float(np.median([day_index[d]["vol_cash"] for d in cash_days]))

    # D1 regime (causal: last COMPLETED D1 bar strictly before day start)
    d1 = pd.read_csv(D1)
    d1t = d1["time"].to_numpy(np.int64)
    d1c = d1["close"].to_numpy(float)
    ema200 = pd.Series(d1c).ewm(span=200, adjust=False).mean().to_numpy(float)
    day_epoch0 = {d: None for d in cash_days}
    # NY midnight epoch for day d: use the first bar's date; approximate via
    # the cash-open bar time minus 9.5h is fragile across DST; instead key
    # by matching completed D1 bars whose close date < the cash date.
    d1_nydate = []
    didx = pd.DatetimeIndex(pd.to_datetime(d1t, unit="s", utc=True)
                            ).tz_convert("America/New_York")
    d1_nydate = (didx.year * 10000 + didx.month * 100 + didx.day).to_numpy()
    regime = {}
    for d in cash_days:
        prior = np.where(d1_nydate < d)[0]
        if len(prior) == 0:
            regime[d] = "na"
            continue
        i = prior[-1]
        rel = (d1c[i] / ema200[i] - 1.0) * 100.0
        regime[d] = ("up" if rel > 0.5 else "down" if rel < -0.5 else "range")

    # ---------------- causal trailing percentiles ----------------
    def trailing_pct(base_days, key):
        vals, out = [], {}
        hist = []
        for d in base_days:
            if len(hist) >= PCT_WIN:
                out[d] = float(np.percentile(hist, PCT_Q))
            hist.append(day_index[d][key])
        return out

    pre_abs = {d: abs(day_index[d]["r_pre"]) for d in pre_days if "r_pre" in day_index[d]}
    # store |r_pre| on rows for percentile history
    for d in pre_days:
        day_index[d]["abs_r_pre"] = abs(day_index[d]["r_pre"])
    for d in post_days:
        day_index[d]["abs_r_post"] = abs(day_index[d]["r_post"])
    p_abs_pre = trailing_pct(pre_days, "abs_r_pre")
    p_vol_pre = trailing_pct(pre_days, "vol_pre")
    p_abs_post = trailing_pct(post_days, "abs_r_post")

    # ---------------- signal days ----------------
    carry_rows, fade_rows = [], []
    funnel = defaultdict(lambda: [0, 0])   # stage -> [carry, fade]
    funnel["F0_cash_days"] = [len(cash_days), len(cash_days)]
    funnel["F1_block_valid"] = [len(pre_days), len(post_days)]
    for d in pre_days:
        r = day_index[d]
        if d in p_abs_pre and d in p_vol_pre:
            if r["abs_r_pre"] >= p_abs_pre[d] and r["vol_pre"] >= p_vol_pre[d]:
                funnel["F2_pct_conditions"][0] += 1
                cash = [k for k in bar_day[d] if 570 <= wall[k] < 960]
                if not cash or wall[cash[0]] != 570:
                    continue
                entry = o[cash[0]]
                ddir = 1 if r["r_pre"] > 0 else -1
                mfe, mae, fwd = excursion(cash, entry, ddir, None, h, l, c)
                funnel["F3_signal_window_ok"][0] += 1
                carry_rows.append({
                    "nd": d, "year": r["year"], "dir": ddir,
                    "entry": round(entry, 3), "mfe": round(mfe, 4),
                    "mae": round(mae, 4), "fwd_bars": fwd,
                    "r_pre_pct": round(r["r_pre"] * 100, 4),
                    "vol_pre": round(r["vol_pre"] * 100, 4),
                    "hl_split": "wide" if r.get("hl_cash", 0) > hl_med else "normal",
                    "vc_split": "hi" if r.get("vol_cash", 0) > vc_med else "lo",
                    "regime": regime.get(d, "na")})
    for d in post_days:
        r = day_index[d]
        if d in p_abs_post and "r_post" in r and "r_cash" in r:
            if (r["abs_r_post"] >= p_abs_post[d]
                    and np.sign(r["r_post"]) == -np.sign(r["r_cash"])
                    and r["r_post"] != 0):
                funnel["F2_pct_conditions"][1] += 1
                # window: bars (post_close_t, next cash open t)
                nxt = None
                ci = cash_days.index(d) if d in cash_days else None
                if ci is not None and ci + 1 < len(cash_days):
                    nxt = cash_days[ci + 1]
                if nxt is None:
                    continue
                t0w = r["post_close_t"]
                t1w = day_index[nxt]["cash_open_t"]
                lo = np.searchsorted(t, t0w, side="right")
                hi = np.searchsorted(t, t1w, side="left")
                window = list(range(lo, hi))
                if not window:
                    continue
                funnel["F3_signal_window_ok"][1] += 1
                entry = r["post_close"]
                ddir = -1 if r["r_post"] > 0 else 1
                exc = excursion(window, entry, ddir, None, h, l, c)
                mfe, mae, fwd = exc
                # secondary: full next cash block (informational)
                ncash = [k for k in bar_day[nxt] if 570 <= wall[k] < 960]
                exc2 = excursion(ncash, day_index[nxt]["cash_open"], ddir,
                                 None, h, l, c) if ncash else None
                fade_rows.append({
                    "nd": d, "year": r["year"], "dir": ddir,
                    "entry": round(entry, 3), "mfe": round(mfe, 4),
                    "mae": round(mae, 4), "fwd_bars": fwd,
                    "r_post_pct": round(r["r_post"] * 100, 4),
                    "r_cash_pct": round(r["r_cash"] * 100, 4),
                    "next_nd": nxt, "weekend": int(nxt - d > 1),
                    "mfe2_nextcash": round(exc2[0], 4) if exc2 else None,
                    "mae2_nextcash": round(exc2[1], 4) if exc2 else None,
                    "hl_split": "wide" if r.get("hl_cash", 0) > hl_med else "normal",
                    "vc_split": "hi" if r.get("vol_cash", 0) > vc_med else "lo",
                    "regime": regime.get(d, "na")})

    pd.DataFrame([r for r in day_rows]).to_csv(BASE / "day_blocks.csv", index=False)
    pd.DataFrame(carry_rows).to_csv(BASE / "phase1_carry.csv", index=False)
    pd.DataFrame(fade_rows).to_csv(BASE / "phase1_fade.csv", index=False)
    with open(BASE / "signal_funnel.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stage", "carry", "fade"])
        for st in ("F0_cash_days", "F1_block_valid", "F2_pct_conditions",
                   "F3_signal_window_ok"):
            a, b = funnel[st]
            w.writerow([st, a, b])
            print(f"funnel {st}: carry {a} fade {b}")

    # ---------------- gates ----------------
    res = {}
    cs, cg, cv = evaluate_gates(carry_rows, CARRY_T, None, len(pre_days),
                                "CARRY")
    fs, fg, fv = evaluate_gates(fade_rows, FADE_T, None, len(post_days),
                                "FADE")
    res["carry"] = {"stats": cs, "gates": cg, "verdict": cv}
    res["fade"] = {"stats": fs, "gates": fg, "verdict": fv}

    # yearly breakdown
    yearly = []
    for arm, rows in (("carry", carry_rows), ("fade", fade_rows)):
        py = defaultdict(list)
        for r in rows:
            py[r["year"]].append(r["mfe"] - r["mae"])
        for y in sorted(py):
            yearly.append({"arm": arm, "year": y, "n": len(py[y]),
                           "mean_diff": round(float(np.mean(py[y])), 4)})
    with open(BASE / "yearly_breakdown.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "year", "n", "mean_diff"])
        w.writeheader()
        w.writerows(yearly)

    # descriptive splits (NOT gates, per SPEC section 5)
    splits = {}
    for arm, rows, ths in (("carry", carry_rows, CARRY_T),
                           ("fade", fade_rows, FADE_T)):
        for key in ("hl_split", "vc_split", "regime"):
            for val in sorted(set(str(r[key]) for r in rows)):
                sub = [r for r in rows if str(r[key]) == val]
                splits[f"{arm}.{key}={val}"] = {
                    "n": len(sub),
                    "mean_diff": round(float(np.mean([r["mfe"] - r["mae"]
                                                      for r in sub])), 4)
                    if sub else None}

    stats = {"carry": res["carry"], "fade": res["fade"],
             "eligible_days": {"carry_pre_valid": len(pre_days),
                               "fade_post_valid": len(post_days),
                               "cash_days": len(cash_days)},
             "medians": {"hl_cash": round(hl_med, 5),
                         "vol_cash": round(vc_med, 5)},
             "splits_descriptive": splits,
             "params": {"PCT_WIN": PCT_WIN, "PCT_Q": PCT_Q,
                        "CARRY_T": CARRY_T, "FADE_T": FADE_T,
                        "MEDIAN_GAP_MIN": MEDIAN_GAP_MIN,
                        "N_BOOT": N_BOOT, "BLOCK_LEN": BLOCK_LEN,
                        "COST_PT": COST_PT},
             "runtime_s": round(timemod.time() - t0, 1)}
    with open(BASE / "phase1_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2, default=str)
    print(f"\nwrote artifacts ({stats['runtime_s']}s)")


if __name__ == "__main__":
    main()
