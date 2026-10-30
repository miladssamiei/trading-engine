# -*- coding: utf-8 -*-
"""
calendar_effects_phase1.py - Phase 1 for hypothesis #14 (calendar effects):
HYPOTHESIS OVERNIGHT (unconditional overnight premium) and HYPOTHESIS DOW
(classic Monday effect), per
data/results/calendar_effects/SPEC_CALENDAR_EFFECTS.txt sections 2-6 and
GATE_CALENDAR_EFFECTS.txt (PG1-PG3, O1-O4, D1-D3 pre-declared).

Unconditional static rules; no filters, no thresholds, no lookback.
Both financing conventions reported (A documented -20/+2 $/day/lot
Wednesday x5; B strict MT5 points -0.20/+0.02). Bootstrap p-values raw
and Bonferroni-adjusted (2 tests).

Outputs (data/results/calendar_effects/):
  overnight_returns.csv, dow_returns.csv, yearly_breakdown_calendar.csv,
  phase1_calendar_stats.json
"""

import csv
import datetime as dt
import json
import time as timemod
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\calendar_effects")
M5 = Path(r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present\us500tick_m5.csv")
D1T = Path(r"C:\Users\Milad\Desktop\RR\data\results\momentum_multihorizon\us500tick_d1.csv")

N_BOOT = 10000
BLOCK_LEN = 10
COST_PT = 0.8
SWAP_A = -20.0
SWAP_B = -0.20
WED_EXTRA = 4          # Wednesday x5 = 1 + 4
BEST_SHARPE = 0.337    # ORB, current best validated single-instrument
BARS = 252
BASE_USD = 10000.0
FULL_YEARS = (2021, 2022, 2023, 2024, 2025)


def wday_of(nd):
    return dt.date(nd // 10000, nd // 100 % 100, nd % 100).weekday()


def sharpe(series):
    s = np.asarray(series, dtype=float)
    sd = float(s.std(ddof=1))
    return float(s.mean() / sd * np.sqrt(BARS)) if sd > 0 else None


def boot_iid_le0(vals, seed=11):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    m = rng.choice(v, size=(N_BOOT, len(v)), replace=True).mean(axis=1)
    return round(float((m <= 0).mean()), 4)


def boot_iid_ge0(vals, seed=11):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    m = rng.choice(v, size=(N_BOOT, len(v)), replace=True).mean(axis=1)
    return round(float((m >= 0).mean()), 4)


def boot_mb_le0(vals, seed=13):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    n = len(v)
    if n < BLOCK_LEN:
        return boot_iid_le0(vals, seed)
    nblk = int(np.ceil(n / BLOCK_LEN))
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        starts = rng.integers(0, n - BLOCK_LEN + 1, size=nblk)
        idx = (starts[:, None] + np.arange(BLOCK_LEN)[None, :]).ravel()[:n]
        out[b] = v[idx].mean()
    return round(float((out <= 0).mean()), 4)


def boot_delta(vals, mon_mask, seed=17):
    """Day-block bootstrap of delta = mean(Mon) - pooled mean(Tue..Fri),
    weekday labels preserved. Returns (P(delta>=0), P(delta<=0))."""
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    mon = np.asarray(mon_mask, dtype=bool)
    n = len(v)
    p_ge = p_le = 0
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        m = mon[idx]
        if m.any() and (~m).any():
            d = float(v[idx][m].mean() - v[idx][~m].mean())
        elif m.any():
            d = float(v[idx][m].mean())
        else:
            d = -float(v[idx][~m].mean())
        p_ge += d >= 0
        p_le += d <= 0
    return round(p_ge / N_BOOT, 4), round(p_le / N_BOOT, 4)


def summary(vals):
    v = np.asarray(vals, dtype=float)
    return {"n": int(len(v)),
            "mean": round(float(v.mean()), 5),
            "median": round(float(np.median(v)), 5),
            "std": round(float(v.std(ddof=1)), 5),
            "sharpe": round(sharpe(v), 4) if sharpe(v) is not None else None,
            "win_rate": round(float((v > 0).mean()), 4),
            "skew": round(float(pd.Series(v).skew()), 4)}


def main():
    t0 = timemod.time()
    res = {"hypothesis": "calendar_effects_phase1", "gates": {}}

    # ---------------- tape loading and cash-day construction ----------------
    m5 = pd.read_csv(M5)
    t = m5["time"].to_numpy(np.int64)
    o = m5["open"].to_numpy(float)
    c = m5["close"].to_numpy(float)
    n = len(m5)
    idx = pd.DatetimeIndex(pd.to_datetime(t, unit="s", utc=True))
    ny = idx.tz_convert("America/New_York")
    wall = (ny.hour * 60 + ny.minute).to_numpy()
    nd = (ny.year * 10000 + ny.month * 100 + ny.day).to_numpy()

    dates = sorted(set(nd.tolist()))
    cash = []
    for d in dates:
        m = nd == d
        if m.any() and (wall[m] == 570).any():
            cash.append(d)
    n_dates = len(dates)
    weds_dates = sum(1 for d in dates if wday_of(d) == 2)
    swap_anchor = SWAP_A * (n_dates + WED_EXTRA * weds_dates)

    cash_rows = {}
    for d in cash:
        m = nd == d
        o0 = float(o[m][wall[m] == 570][0])
        cN = float(c[m][(wall[m] >= 570) & (wall[m] < 960)][-1])
        cash_rows[d] = {"open": o0, "close": cN, "wday": wday_of(d),
                        "year": d // 10000}

    # ---------------- PG1-PG3 pre-gates ----------------
    d1t = pd.read_csv(D1T)
    d1t_close = d1t["close"].to_numpy(float)
    d1t_open0 = float(d1t["open"].iloc[0])
    d1t_close0 = float(d1t["close"].iloc[0])
    d1t_last = float(d1t["close"].iloc[-1])
    pg3_close2close = (d1t_last - d1t_close0) / 100.0
    pg3_anchor = (d1t_last - d1t_open0) / 100.0
    cl0 = cash_rows[cash[0]]["close"]
    clN = cash_rows[cash[-1]]["close"]
    pg3_m5_cash = (clN - cl0) / 100.0
    pg3_ok = 34.66 <= pg3_close2close <= 34.69
    pgs = {"PG1_cash_days": {"value": len(cash), "expect": 1370,
                             "ok": len(cash) == 1370},
           "PG2_swap_anchor": {"n_dates": n_dates, "wednesdays": weds_dates,
                               "swap_usd": swap_anchor,
                               "expect_swap_usd": -55360.0,
                               "expect_n_dates": 1656, "expect_weds": 278,
                               "ok": (n_dates == 1656 and weds_dates == 278
                                      and abs(swap_anchor - (-55360.0)) < 0.01)},
           "PG3_bh_reproduction": {
               "band": [34.66, 34.69],
               "d1_close2close_pct": round(pg3_close2close, 4),
               "d1_first_open_anchor_pct": round(pg3_anchor, 4),
               "m5_cash_close2close_pct": round(pg3_m5_cash, 4),
               "classified_residual": ("M5 first-cash-close reading drops the "
                                       "pre-first-cash-day D1 bar (weekend "
                                       "Globex) - documented first-bar "
                                       "open-vs-close convention residual"),
               "ok": bool(pg3_ok)}}
    res["pgs"] = pgs
    print("PRE-GATES:")
    for k, v in pgs.items():
        print(f"  {k}: ok={v['ok']}  {v}")
    if not all(v["ok"] for v in pgs.values()):
        print("\nPG FAILURE - abort per GATE_CALENDAR_EFFECTS (data problem, "
              "no hypothesis result reported).")
        with open(BASE / "phase1_calendar_stats.json", "w") as fh:
            json.dump(res, fh, indent=2, default=str)
        return

    # ---------------- HYPOTHESIS OVERNIGHT ----------------
    spans = []
    for i in range(len(cash) - 1):
        d, d_next = cash[i], cash[i + 1]
        r = cash_rows[d]
        open_next = cash_rows[d_next]["open"]
        pts = open_next - r["close"]
        r_ovn = float(np.log(open_next / r["close"]))
        wed = 1 if r["wday"] == 2 else 0
        fin_a = SWAP_A * (1 + WED_EXTRA * wed)
        fin_b = SWAP_B * (1 + WED_EXTRA * wed)
        spans.append({"nd": d, "nd_next": d_next, "year": r["year"],
                      "wday": r["wday"], "wed": wed,
                      "r_ovn": round(r_ovn, 8),
                      "pts": round(pts, 3),
                      "r_intra": round(float(np.log(r["close"] / r["open"])), 8),
                      "pts_intra": round(r["close"] - r["open"], 3),
                      "fin_a": round(fin_a, 2), "fin_b": round(fin_b, 3),
                      "exfin_pts": round(pts - COST_PT, 3),
                      "net_a": round(pts - COST_PT + fin_a, 3),
                      "net_b": round(pts - COST_PT + fin_b, 3)})

    r_ovn = np.array([s["r_ovn"] for s in spans])
    pts = np.array([s["pts"] for s in spans])
    r_intra = np.array([s["r_intra"] for s in spans])
    pts_intra = np.array([s["pts_intra"] for s in spans])
    net_a = np.array([s["net_a"] for s in spans])
    net_b = np.array([s["net_b"] for s in spans])

    ovn = {}
    ovn["n_spans"] = len(spans)
    ovn["overnight_gross"] = summary(pts)
    ovn["overnight_r_ovn"] = summary(r_ovn)
    ovn["intraday_benchmark"] = summary(pts_intra)
    ovn["intraday_r_intra"] = summary(r_intra)
    ovn["exfin_net"] = summary(np.array([s["exfin_pts"] for s in spans]))
    ovn["net_convA"] = summary(net_a)
    ovn["net_convB"] = summary(net_b)
    ovn["fin_total_A"] = round(float(np.array([s["fin_a"] for s in spans]).sum()), 2)
    ovn["fin_total_B"] = round(float(np.array([s["fin_b"] for s in spans]).sum()), 2)

    # yearly
    yearly_rows = []
    for label, series, key in (("overnight_gross", pts, "gross"),
                               ("overnight_netA", net_a, "netA"),
                               ("overnight_netB", net_b, "netB")):
        py = defaultdict(list)
        for s, v in zip(spans, series):
            py[s["year"]].append(v)
        for y in sorted(py):
            arr = np.array(py[y])
            yearly_rows.append({"hypothesis": "overnight", "convention": key,
                                "year": y, "n": len(arr),
                                "mean_return": round(float(arr.mean()), 4),
                                "net": round(float(arr.sum()), 2),
                                "sharpe": round(sharpe(arr), 4)
                                if sharpe(arr) is not None else None,
                                "share_pct": None})
    ovn["yearly"] = {k: v for k, v in
                     [(f"{r['year']}_{r['convention']}",
                       {"n": r["n"], "mean_return": r["mean_return"],
                        "net": r["net"], "sharpe": r["sharpe"]})
                      for r in yearly_rows]}

    # O gates
    p_o1 = boot_iid_le0(r_ovn)
    p_o1_mb = boot_mb_le0(r_ovn)
    o1 = float(r_ovn.mean()) > 0 and p_o1 <= 0.10
    o2 = (sharpe(net_a) is not None and sharpe(net_a) > 0
          and sharpe(net_b) is not None and sharpe(net_b) > 0)
    o3, o4 = {}, {}
    for conv, series in (("A", net_a), ("B", net_b)):
        ys = defaultdict(float)
        for s, v in zip(spans, series):
            if s["year"] in FULL_YEARS:
                ys[s["year"]] += v
        tot = sum(ys.values())
        n_pos = sum(1 for y in FULL_YEARS if ys[y] > 0)
        shares = {y: (ys[y] / tot * 100 if tot else None) for y in ys}
        max_share = max((abs(v) for v in shares.values() if v is not None),
                        default=None)
        o3[conv] = {"positive_years": n_pos, "need": 4,
                    "max_year_share_pct": round(max_share, 2)
                    if max_share is not None else None,
                    "ok": n_pos >= 4 and (max_share is None or max_share <= 60.0)}
        sh = sharpe(series)
        o4[conv] = {"sharpe": round(sh, 4) if sh is not None else None,
                    "need_gt": BEST_SHARPE,
                    "ok": sh is not None and sh > BEST_SHARPE}
    o3_ok = all(o3[k]["ok"] for k in ("A", "B"))
    o4_ok = all(o4[k]["ok"] for k in ("A", "B"))
    adj_o1 = round(min(1.0, 2 * p_o1), 4)
    ovn_gates = {"O1": {"mean_r_ovn": round(float(r_ovn.mean()), 5),
                        "p_boot_le0_raw": p_o1,
                        "p_boot_le0_bonferroni": adj_o1,
                        "p_mb10_le0": p_o1_mb,
                        "ok": bool(o1)},
                 "O2": {"sharpe_A": ovn["net_convA"]["sharpe"],
                        "sharpe_B": ovn["net_convB"]["sharpe"],
                        "ok": bool(o2)},
                 "O3": {"A": o3["A"], "B": o3["B"], "ok": bool(o3_ok)},
                 "O4": {"A": o4["A"], "B": o4["B"], "ok": bool(o4_ok)}}
    ovn["gates"] = ovn_gates
    ovn_pass = bool(o1 and o2 and o3_ok and o4_ok)
    ovn["verdict"] = "PASS -> Phase 2" if ovn_pass else "FAIL -> null result"
    ovn["fragile_multi_testing"] = bool(o1) and p_o1 <= 0.10 and adj_o1 > 0.10

    # ---------------- HYPOTHESIS DOW ----------------
    dow_rows = []
    for i in range(1, len(cash)):
        d, d_prev = cash[i], cash[i - 1]
        r = cash_rows[d]
        rp = cash_rows[d_prev]
        pts_cc = r["close"] - rp["close"]
        fin_a = SWAP_A * (1 + WED_EXTRA * (rp["wday"] == 2))
        fin_b = SWAP_B * (1 + WED_EXTRA * (rp["wday"] == 2))
        dow_rows.append({"nd": d, "year": r["year"], "wday": r["wday"],
                         "wday_prev": rp["wday"],
                         "r_cc": round(float(np.log(r["close"] / rp["close"])), 8),
                         "pts_cc": round(pts_cc, 3),
                         "fin_a": round(fin_a, 2), "fin_b": round(fin_b, 3)})

    r_cc = np.array([x["r_cc"] for x in dow_rows])
    wday_arr = np.array([x["wday"] for x in dow_rows])
    mon_mask = wday_arr == 0

    dow = {"n_days": len(dow_rows)}
    per_wd = {}
    for w in range(5):
        m = wday_arr == w
        v = r_cc[m]
        per_wd[w] = {"n": int(m.sum()),
                     "mean_pct": round(float(v.mean()) * 100, 5),
                     "median_pct": round(float(np.median(v)) * 100, 5),
                     "std_pct": round(float(v.std(ddof=1)) * 100, 5),
                     "sharpe": round(sharpe(v * 100), 4) if sharpe(v) is not None else None,
                     "win_rate": round(float((v > 0).mean()), 4),
                     "p_le0": boot_iid_le0(v, seed=20 + w),
                     "p_ge0": boot_iid_ge0(v, seed=30 + w)}
    wd_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    ranked = sorted(range(5), key=lambda w: per_wd[w]["mean_pct"], reverse=True)
    dow["per_weekday"] = {wd_names[w]: per_wd[w] for w in range(5)}
    dow["ranked_by_mean"] = [wd_names[w] for w in ranked]
    dow["best_weekday"] = wd_names[ranked[0]]
    dow["worst_weekday"] = wd_names[ranked[-1]]

    mon_mean = float(r_cc[mon_mask].mean())
    pooled_mean = float(r_cc[~mon_mask].mean())
    delta = mon_mean - pooled_mean
    p_d1_ge, p_d1_le = boot_delta(r_cc, mon_mask)
    d1 = delta < 0 and p_d1_ge <= 0.10
    classic = ("HOLDS" if (delta < 0 and p_d1_ge <= 0.10)
               else "REVERSED" if (delta > 0 and p_d1_le <= 0.10)
               else "ABSENT")
    adj_d1 = round(min(1.0, 2 * p_d1_ge), 4)
    dow["delta"] = {"mon_mean_pct": round(mon_mean * 100, 5),
                    "pooled_tue_fri_mean_pct": round(pooled_mean * 100, 5),
                    "delta_pct": round(delta * 100, 5),
                    "p_delta_ge0_raw": p_d1_ge,
                    "p_delta_ge0_bonferroni": adj_d1,
                    "p_delta_le0": p_d1_le,
                    "classic_verdict": classic}

    dow_gates = {"D1": {"delta_pct": round(delta * 100, 5),
                        "p_no_effect_raw": p_d1_ge,
                        "p_no_effect_bonferroni": adj_d1,
                        "ok": bool(d1)},
                 "D2": {"run": False, "ok": False, "note":
                        "NOT RUN per GATE (D1 failed -> STOP)"},
                 "D3": {"run": False, "ok": False, "note":
                        "NOT RUN per GATE (D1 failed -> STOP)"}}
    dow["gates"] = dow_gates
    dow["fragile_multi_testing"] = bool(d1) and p_d1_ge <= 0.10 and adj_d1 > 0.10

    if d1:
        # tradeable version: flat over Monday-ending spans, long the rest
        fin_a = np.array([x["fin_a"] for x in dow_rows])
        fin_b = np.array([x["fin_b"] for x in dow_rows])
        pts_cc = np.array([x["pts_cc"] for x in dow_rows])
        # per SPEC: hold night s->d when weekday(d) in {Tue,Wed,Thu,Fri}
        # -> held[d] = weekday(d) != Monday, i.e. wday != 0
        held = wday_arr != 0
        block_start = np.zeros(len(dow_rows), dtype=bool)
        for i in range(len(dow_rows)):
            if held[i] and (i == 0 or not held[i - 1]):
                block_start[i] = True
        for conv, fin in (("A", fin_a), ("B", fin_b)):
            strat = np.where(held, pts_cc + fin - COST_PT * block_start, 0.0)
            base = pts_cc + fin
            sh_s = sharpe(strat)
            sh_b = sharpe(base)
            ys = defaultdict(float)
            for i, v in enumerate(strat):
                if dow_rows[i]["year"] in FULL_YEARS:
                    ys[dow_rows[i]["year"]] += v
            tot = sum(ys.values())
            n_pos = sum(1 for y in FULL_YEARS if ys[y] > 0)
            shares = {y: (ys[y] / tot * 100 if tot else None) for y in ys}
            max_share = max((abs(v) for v in shares.values() if v is not None),
                            default=None)
            dow_gates["D2"][conv] = {"sharpe_strategy": round(sh_s, 4)
                                     if sh_s is not None else None,
                                     "sharpe_baseline": round(sh_b, 4)
                                     if sh_b is not None else None,
                                     "ok": (sh_s is not None and sh_b is not None
                                            and sh_s > sh_b)}
            dow_gates["D3"][conv] = {"positive_years": n_pos, "need": 4,
                                     "max_year_share_pct":
                                     round(max_share, 2)
                                     if max_share is not None else None,
                                     "ok": n_pos >= 4 and (max_share is None
                                                           or max_share <= 60.0)}
        delta_by_year = {}
        for y in FULL_YEARS:
            m = np.array([x["year"] == y for x in dow_rows])
            mm = m & mon_mask
            pm = m & ~mon_mask
            if mm.any() and pm.any():
                delta_by_year[y] = round(float(r_cc[mm].mean()
                                               - r_cc[pm].mean()) * 100, 5)
        d2_ok = all(dow_gates["D2"][k]["ok"] for k in ("A", "B"))
        d3_ok = (all(dow_gates["D3"][k]["ok"] for k in ("A", "B"))
                 and sum(1 for y in delta_by_year
                         if delta_by_year[y] < 0) >= 4)
        dow_gates["D2"]["run"] = True
        dow_gates["D2"]["ok"] = bool(d2_ok)
        dow_gates["D3"]["run"] = True
        dow_gates["D3"]["ok"] = bool(d3_ok)
        dow_gates["D3"]["delta_by_year_pct"] = delta_by_year
        dow["delta_by_year_pct"] = delta_by_year
    else:
        dow_gates["D2"]["note"] = "NOT RUN per GATE (D1 failed -> STOP)"
        dow_gates["D3"]["note"] = "NOT RUN per GATE (D1 failed -> STOP)"

    dow_pass = bool(d1) and dow_gates["D2"]["ok"] and dow_gates["D3"]["ok"]
    dow["verdict"] = ("PASS (Phase 2 requires separate SPEC amendment)"
                      if dow_pass else "FAIL -> null result")

    # ---------------- artifacts ----------------
    pd.DataFrame(spans).to_csv(BASE / "overnight_returns.csv", index=False)
    pd.DataFrame(dow_rows).to_csv(BASE / "dow_returns.csv", index=False)
    yearly_rows.sort(key=lambda r: (r["hypothesis"], r["year"]))
    for r in yearly_rows:
        r["hypothesis"] = r["hypothesis"].replace("overnight_", "OVERNIGHT/")
    pd.DataFrame(yearly_rows).to_csv(BASE / "yearly_breakdown_calendar.csv",
                                     index=False)

    res["overnight"] = ovn
    res["dow"] = dow
    res["universes"] = {"m5_dates": n_dates, "cash_dates": len(cash),
                        "overnight_spans": len(spans),
                        "dow_days": len(dow_rows),
                        "wednesdays_all_dates": weds_dates}
    res["params"] = {"N_BOOT": N_BOOT, "BLOCK_LEN": BLOCK_LEN,
                     "COST_PT": COST_PT, "SWAP_A": SWAP_A, "SWAP_B": SWAP_B,
                     "WED_EXTRA": WED_EXTRA, "BEST_SHARPE": BEST_SHARPE,
                     "BARS": BARS, "BASE_USD": BASE_USD,
                     "FULL_YEARS": list(FULL_YEARS)}
    res["runtime_s"] = round(timemod.time() - t0, 1)
    with open(BASE / "phase1_calendar_stats.json", "w") as fh:
        json.dump(res, fh, indent=2, default=str)

    # ---------------- console summary ----------------
    print(f"\nOVERNIGHT: n={ovn['n_spans']} spans")
    for k in ("overnight_gross", "overnight_r_ovn", "intraday_benchmark",
              "exfin_net", "net_convA", "net_convB"):
        s = ovn[k]
        print(f"  {k}: mean {s['mean']} med {s['median']} std {s['std']} "
              f"sh {s['sharpe']} win {s['win_rate']} skew {s['skew']}")
    print(f"  financing totals: A {ovn['fin_total_A']}  B {ovn['fin_total_B']}")
    print(f"  O1 p(raw)={p_o1} adj={adj_o1} mb10={p_o1_mb} ok={o1}")
    print(f"  O2 sharpe A/B {ovn['net_convA']['sharpe']}/"
          f"{ovn['net_convB']['sharpe']} ok={o2}")
    print(f"  O3 ok={o3_ok}  {o3}")
    print(f"  O4 ok={o4_ok}  {o4}")
    print(f"  OVERNIGHT verdict: {ovn['verdict']}")
    print(f"\nDOW: n={dow['n_days']} days; per-weekday mean_pct: "
          + ", ".join(f"{wd_names[w]}={per_wd[w]['mean_pct']}" for w in range(5)))
    print(f"  ranked: {dow['ranked_by_mean']}")
    print(f"  delta_pct {dow['delta']['delta_pct']} p_ge0(raw)={p_d1_ge} "
          f"adj={adj_d1} p_le0={p_d1_le} -> classic {classic}")
    print(f"  D1 ok={d1}; D2 {dow_gates['D2']}")
    print(f"  D3 {dow_gates['D3']}")
    print(f"  DOW verdict: {dow['verdict']}")
    print(f"\nwrote artifacts ({res['runtime_s']}s)")


if __name__ == "__main__":
    main()
