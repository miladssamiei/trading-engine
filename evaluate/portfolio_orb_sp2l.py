# -*- coding: utf-8 -*-
"""
portfolio_orb_sp2l.py - combined-portfolio exercise over the two validated
MT5 net-of-spread runs (ORB + SP2L), per
data/results/portfolio_orb_sp2l/SPEC.md.

Step 0 reproduction gate -> pre-declared allocation schemes S1/S2/S3
(SPEC Step 1) -> combined metrics, diversification/tail analysis,
year-by-year stability. No new strategy logic; reads existing artifacts
only. Aborts if standalone stats do not reproduce documented numbers.

Outputs (data/results/portfolio_orb_sp2l/):
  combined_daily_pnl.csv, combined_equity.csv, yearly_breakdown.csv,
  portfolio_summary.json
"""

import csv
import json
import math
import time as timemod
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\portfolio_orb_sp2l")
ORB = Path(r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present")
SP2L = Path(r"C:\Users\Milad\Desktop\RR\data\results\sp2l_2021_2026")
M5 = ORB / "us500tick_m5.csv"

ACCOUNT = 10000.0
SQ252 = math.sqrt(252)

# documented standalone anchors (Step 0 reproduction gate)
ORB_DOC = {"trades": 447, "pnl": 470.67, "sharpe": 0.34, "dd_pct": 5.13}
SP2L_DOC = {"trades": 1343, "pnl": 281.84, "sharpe": 0.25, "dd_pts": 416.71}
BH_DOC = {"ret": 34.66, "cagr": 5.75, "sharpe": 0.82, "dd": 32.24, "rdd": 1.07}


def load_trades(path):
    df = pd.read_csv(path, sep=";")
    df["open_time"] = pd.to_datetime(df["open_time"], format="%Y.%m.%d %H:%M", utc=True)
    df["d"] = df["open_time"].dt.strftime("%Y%m%d")
    df["profit"] = df["profit"].astype(float)
    return df


def metrics(daily):
    s = pd.Series(daily)
    total = float(s.sum())
    cum = s.cumsum()
    mdd = float((cum.cummax() - cum).max())
    mean, std = float(s.mean()), float(s.std(ddof=1))
    return {"total": total, "ret_pct": total / ACCOUNT * 100.0,
            "cagr_pct": None,  # filled by caller with years
            "sharpe": mean / std * SQ252 if std > 0 else None,
            "mdd": mdd, "mdd_pct": mdd / ACCOUNT * 100.0,
            "mean": mean, "std": std}


def main():
    t0 = timemod.time()
    orb_tr = load_trades(ORB / "mt5_trades.csv")
    sp2l_tr = load_trades(SP2L / "mt5_trades.csv")

    # ---- day universe: ORB session days ----
    od = pd.read_csv(ORB / "mt5_days.csv", sep=";", dtype=str)
    od["session_open"] = pd.to_numeric(od["session_open"], errors="coerce")
    od["prev_close"] = pd.to_numeric(od["prev_close"], errors="coerce")
    universe = sorted(od[(od["session_open"] > 0) | (od["prev_close"] > 0)]["ny_date"])
    ndays = len(universe)
    print(f"day universe (ORB session days): {ndays} "
          f"({universe[0]}..{universe[-1]})")

    orb_day = orb_tr.groupby("d")["profit"].sum().to_dict()
    sp2l_day = sp2l_tr.groupby("d")["profit"].sum().to_dict()
    # cross-check SP2L days CSV plumbing
    sd = pd.read_csv(SP2L / "mt5_days.csv", sep=";")
    sp2l_days_map = dict(zip(sd["ny_date"].astype(str),
                             sd["total_pnl"].astype(float)))
    mism = sum(1 for k, v in sp2l_day.items()
               if abs(sp2l_days_map.get(k, 0.0) - v) > 0.01)
    print(f"SP2L trades-groupby vs mt5_days.total_pnl mismatches: {mism} "
          f"(trade days: {len(sp2l_day)})")

    orb_v = [orb_day.get(k, 0.0) for k in universe]
    sp2l_v = [sp2l_day.get(k, 0.0) for k in universe]

    # ---- Step 0: reproduction gate ----
    m_orb = metrics(orb_v)
    m_sp2l = metrics(sp2l_v)
    rep = []
    ok = True
    for name, m, doc, tr in (("ORB", m_orb, ORB_DOC, orb_tr),
                             ("SP2L", m_sp2l, SP2L_DOC, sp2l_tr)):
        r = {"leg": name, "trades": len(tr), "trades_doc": doc["trades"],
             "pnl": round(m["total"], 2), "pnl_doc": doc["pnl"],
             "sharpe": round(m["sharpe"], 3), "sharpe_doc": doc["sharpe"],
             "mdd_pct": round(m["mdd_pct"], 2),
             "mdd_doc": doc.get("dd_pct", doc.get("dd_pts"))}
        if "dd_pts" in doc:
            r["mdd_pts"] = round(m["mdd"], 2)
            r.pop("mdd_doc")
        ok_leg = (r["trades"] == r["trades_doc"]
                  and abs(r["pnl"] - r["pnl_doc"]) <= 0.05
                  and abs(r["mdd_pct"] - doc.get("dd_pct", 4.17)) <= 0.03
                  and abs(m["sharpe"] - doc["sharpe"]) <= 0.03)
        if name == "SP2L" and abs(m["sharpe"] - doc["sharpe"]) > 0.01:
            # proven universe-convention delta (SPEC Step 0 note):
            # documented 0.25 used sp2l_stats' 2,605-day union universe;
            # trades/pnl/DD reproduce exactly on the 1,369-day common one
            print(f"SP2L Sharpe {m['sharpe']:.3f} vs documented 0.25: "
                  f"day-universe convention delta (SPEC Step 0 note), "
                  f"tolerance 0.03")
        r["reproduced"] = bool(ok_leg)
        ok = ok and ok_leg
        rep.append(r)
        print(f"repro {name}: {r}")
    if not ok:
        print("REPRODUCTION GATE FAILED - abort (see SPEC Step 0)")
        return

    # ---- years / CAGR exponent (orb_stats convention) ----
    tm = pd.read_csv(M5, usecols=["time"])["time"]
    years = (tm.iloc[-1] - tm.iloc[0]) / 86400.0 / 365.25
    print(f"years = {years:.4f}")

    def with_cagr(m):
        m = dict(m)
        m["cagr_pct"] = ((1 + m["ret_pct"] / 100) ** (1 / years) - 1) * 100
        m["rdd"] = m["ret_pct"] / m["mdd_pct"] if m["mdd_pct"] > 0 else None
        return m

    m_orb, m_sp2l = with_cagr(m_orb), with_cagr(m_sp2l)

    # ---- Step 1: pre-declared allocation schemes ----
    sig_o, sig_s = m_orb["std"], m_sp2l["std"]
    w_s2_o = (1 / sig_o) / (1 / sig_o + 1 / sig_s) * 2.0
    w_s3_o = (0.75 / sig_o) / (0.75 / sig_o + 0.25 / sig_s) * 2.0
    schemes = {
        "S1_equal_notional": (1.0, 1.0),
        "S2_equal_risk": (w_s2_o, 2.0 - w_s2_o),
        "S3_orb75_risk25": (w_s3_o, 2.0 - w_s3_o),
    }
    print(f"sigma: ORB {sig_o:.3f}  SP2L {sig_s:.3f}")
    for k, (wo, ws) in schemes.items():
        rc_o = wo * sig_o / (wo * sig_o + ws * sig_s) * 100
        print(f"{k}: ORB {wo:.3f} lots ({rc_o:.0f}% risk) "
              f"+ SP2L {ws:.3f} lots ({100 - rc_o:.0f}% risk)")

    # ---- Step 2: combined metrics ----
    combo = {}
    for k, (wo, ws) in schemes.items():
        daily = [wo * a + ws * b for a, b in zip(orb_v, sp2l_v)]
        m = with_cagr(metrics(daily))
        combo[k] = {"w_orb": round(wo, 4), "w_sp2l": round(ws, 4), **m,
                    "daily": daily}
    # correlation (mt5-mt5, zero-filled universe)
    so = pd.Series(orb_v)
    ss = pd.Series(sp2l_v)
    rho = float(so.corr(ss))

    # ---- Step 3c: joint tails ----
    both_neg = [(k, a, b) for k, a, b in zip(universe, orb_v, sp2l_v)
                if a < 0 and b < 0]
    worst_joint = sorted(both_neg, key=lambda x: x[1] + x[2])[:5]
    tails = {"rho_mt5mt5": round(rho, 4), "rho_documented_py_based": 0.054,
             "both_neg_days": len(both_neg), "both_neg_days_pct":
                 round(100.0 * len(both_neg) / ndays, 2),
             "worst_joint_days": [{"date": k, "orb": round(a, 2),
                                   "sp2l": round(b, 2),
                                   "sum": round(a + b, 2)}
                                  for k, a, b in worst_joint]}
    # worst combined DD windows (peak->trough dates)
    def dd_window(daily):
        cum = pd.Series(daily).cumsum()
        peak = cum.cummax()
        dd = peak - cum
        tr = int(dd.idxmax())
        pk = int(cum.iloc[:tr + 1].idxmax())
        return {"depth": round(float(dd.iloc[tr]), 2),
                "peak_date": universe[pk], "trough_date": universe[tr],
                "peak_eq": round(float(cum.iloc[pk]), 2),
                "trough_eq": round(float(cum.iloc[tr]), 2)}
    ddwin = {k: dd_window(c["daily"]) for k, c in combo.items()}
    ddwin["ORB_alone"] = dd_window(orb_v)
    ddwin["SP2L_alone"] = dd_window(sp2l_v)

    # ---- Step 4: yearly for S3 (+S1 context) ----
    yearly = []
    per_year = defaultdict(lambda: [0.0, 0.0, 0.0])  # orb, sp2l, s3
    for k, a, b, s3 in zip(universe, orb_v, sp2l_v,
                           combo["S3_orb75_risk25"]["daily"]):
        y = int(k[:4])
        per_year[y][0] += a
        per_year[y][1] += b
        per_year[y][2] += s3
    tot_s3 = sum(v[2] for v in per_year.values())
    for y in sorted(per_year):
        a, b, s3 = per_year[y]
        s1 = a + b
        yearly.append({"year": y, "orb": round(a, 2), "sp2l": round(b, 2),
                       "s1_equal": round(s1, 2), "s3_primary": round(s3, 2),
                       "s3_share_pct": round(100.0 * s3 / tot_s3, 1)
                       if tot_s3 > 0 else None})
    best_share = max(r["s3_share_pct"] for r in yearly) if tot_s3 > 0 else None
    print(f"S3 best-year share of total profit: {best_share}% "
          f"(flag >60%: {best_share is not None and best_share > 60})")

    # ---- outputs ----
    pd.DataFrame({"date": universe, "orb": orb_v, "sp2l": sp2l_v,
                  **{k: c["daily"] for k, c in combo.items()}}
                 ).to_csv(BASE / "combined_daily_pnl.csv", index=False)
    eq = {"date": universe, "orb_cum": so.cumsum(), "sp2l_cum": ss.cumsum()}
    for k, c in combo.items():
        eq[f"{k}_cum"] = pd.Series(c["daily"]).cumsum()
    pd.DataFrame(eq).to_csv(BASE / "combined_equity.csv", index=False)
    with open(BASE / "yearly_breakdown.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(yearly[0].keys()))
        w.writeheader()
        w.writerows(yearly)

    def row(m):
        return {"total_ret_pct": round(m["ret_pct"], 2),
                "cagr_pct": round(m["cagr_pct"], 2),
                "sharpe": round(m["sharpe"], 3),
                "max_dd_pct": round(m["mdd_pct"], 2),
                "ret_dd": round(m["rdd"], 2) if m["rdd"] else None}

    summary = {
        "reproduction": rep, "years": round(years, 4),
        "sigma": {"orb": round(sig_o, 3), "sp2l": round(sig_s, 3)},
        "legs": {"ORB_alone": row(m_orb), "SP2L_alone": row(m_sp2l),
                 "BH_documented": BH_DOC},
        "schemes": {k: {"weights": {"orb": c["w_orb"], "sp2l": c["w_sp2l"]},
                        **row(c)} for k, c in combo.items()},
        "tails": tails, "dd_windows": ddwin, "yearly": yearly,
        "s3_best_year_share_pct": best_share,
        "runtime_s": round(timemod.time() - t0, 1)}
    with open(BASE / "portfolio_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- console summary table ----
    print("\n=== SUMMARY (base $10k, no compounding, MT5 net legs) ===")
    hdr = f"{'portfolio':<22}{'ret%':>8}{'cagr%':>8}{'sharpe':>8}" \
          f"{'maxdd%':>8}{'ret/dd':>8}"
    print(hdr)
    print(f"{'ORB alone':<22}{m_orb['ret_pct']:>8.2f}{m_orb['cagr_pct']:>8.2f}"
          f"{m_orb['sharpe']:>8.3f}{m_orb['mdd_pct']:>8.2f}"
          f"{m_orb['rdd']:>8.2f}")
    print(f"{'SP2L alone':<22}{m_sp2l['ret_pct']:>8.2f}"
          f"{m_sp2l['cagr_pct']:>8.2f}{m_sp2l['sharpe']:>8.3f}"
          f"{m_sp2l['mdd_pct']:>8.2f}{m_sp2l['rdd']:>8.2f}")
    for k, c in combo.items():
        print(f"{k:<22}{c['ret_pct']:>8.2f}{c['cagr_pct']:>8.2f}"
              f"{c['sharpe']:>8.3f}{c['mdd_pct']:>8.2f}{c['rdd']:>8.2f}")
    print(f"{'B&H ex-fin (doc)':<22}{BH_DOC['ret']:>8.2f}"
          f"{BH_DOC['cagr']:>8.2f}{BH_DOC['sharpe']:>8.3f}"
          f"{BH_DOC['dd']:>8.2f}{BH_DOC['rdd']:>8.2f}")
    print(f"\nrho(mt5,mt5)={rho:+.4f}  both-neg days={len(both_neg)}/{ndays}")
    print("worst joint days:", tails["worst_joint_days"][:3])
    for k in ("S1_equal_notional", "S3_orb75_risk25"):
        print(f"DD window {k}: {ddwin[k]}")
    print(f"\nyearly S3: {[(r['year'], r['s3_primary']) for r in yearly]}")
    print(f"runtime {summary['runtime_s']}s")


if __name__ == "__main__":
    main()
