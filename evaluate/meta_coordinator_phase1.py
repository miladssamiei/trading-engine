# -*- coding: utf-8 -*-
"""
meta_coordinator_phase1.py - Phase 1 offline ensemble for the
meta-coordinator (hypothesis #12), per
data/results/meta_coordinator/SPEC_META_COORDINATOR.txt sections 3-4
and GATE_META_COORDINATOR.txt (E1-E4 pre-declared).

Children: ORB + SP2L, MT5 net-of-spread streams, zero-filled over the
common ORB session-day universe. Meta-rules: R1 activation gating
(causal trailing 126d Sharpe > 0 AND trailing DD < 8% of $10k; first
126 days ungated burn-in), R2 conflict veto (opposite net direction on
the same day blocks both), R3 caps (<=1 lot/child). Arms: A
equal-notional, B causal trailing-vol equal-risk (cap 1.00, not
renormalized after cap). Gates E1-E4 per arm.

Outputs (data/results/meta_coordinator/):
  child_streams.csv, ensemble_daily_pnl.csv, yearly_breakdown.csv,
  contributions.csv, phase1_stats.json
"""

import csv
import json
import math
import time as timemod
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\meta_coordinator")
ORB = Path(r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present")
SP2L = Path(r"C:\Users\Milad\Desktop\RR\data\results\sp2l_2021_2026")
PORT = Path(r"C:\Users\Milad\Desktop\RR\data\results\portfolio_orb_sp2l")

T_WIN = 126
DD_CAP_PCT = 8.0
SQ252 = math.sqrt(252)
ACCOUNT = 10000.0
CHILD_DOC = {"ORB": {"trades": 447, "pnl": 470.67, "sharpe": 0.337,
                     "dd_pct": 5.13},
             "SP2L": {"trades": 1343, "pnl": 281.84, "sharpe": 0.275,
                      "dd_pct": 4.17}}


def load_child(trades_path):
    df = pd.read_csv(trades_path, sep=";")
    df["d"] = df["open_time"].str[:10].str.replace(".", "").astype(int)
    df["profit"] = df["profit"].astype(float)
    df["lot"] = df["lot"].astype(float)
    return df


def sharpe(vals):
    v = np.asarray(vals, dtype=float)
    sd = v.std(ddof=1)
    return float(v.mean() / sd * SQ252) if sd > 0 else None


def max_dd(vals):
    cum = np.cumsum(np.asarray(vals, dtype=float))
    peak = np.maximum.accumulate(cum)
    return float((peak - cum).max())


def trailing_stats(pnl_arr, j):
    """Causal stats over the T_WIN days strictly before index j."""
    w = pnl_arr[max(0, j - T_WIN):j]
    if len(w) < 2:
        return None
    sh = sharpe(w)
    cum = np.cumsum(w)
    peak = np.maximum.accumulate(cum)
    dd = float((peak - cum).max()) / ACCOUNT * 100.0
    sd = float(w.std(ddof=1))
    return {"sharpe": sh, "dd_pct": dd, "sigma": sd}


def main():
    t0 = timemod.time()
    orb_tr = load_child(ORB / "mt5_trades.csv")
    sp2l_tr = load_child(SP2L / "mt5_trades.csv")

    od = pd.read_csv(ORB / "mt5_days.csv", sep=";", dtype=str)
    od["session_open"] = pd.to_numeric(od["session_open"], errors="coerce")
    od["prev_close"] = pd.to_numeric(od["prev_close"], errors="coerce")
    universe = sorted(od[(od["session_open"] > 0) | (od["prev_close"] > 0)]
                      ["ny_date"].astype(int))
    nu = len(universe)
    uidx = {d: j for j, d in enumerate(universe)}

    def stream(df):
        pnl = np.zeros(nu)
        sig = np.zeros(nu, dtype=bool)
        dirn = np.zeros(nu, dtype=int)
        for d, g in df.groupby("d"):
            j = uidx.get(d)
            if j is None:
                continue
            pnl[j] = g["profit"].sum()
            sig[j] = len(g) > 0
            buy = g.loc[g["type"] == "BUY", "lot"].sum()
            sell = g.loc[g["type"] == "SELL", "lot"].sum()
            dirn[j] = int(np.sign(buy - sell))
        return pnl, sig, dirn

    orb_pnl, orb_sig, orb_dir = stream(orb_tr)
    sp2l_pnl, sp2l_sig, sp2l_dir = stream(sp2l_tr)

    # reproduction gate (portfolio-study tolerances)
    rep = {}
    ok = True
    for name, tr, pnl in (("ORB", orb_tr, orb_pnl), ("SP2L", sp2l_tr, sp2l_pnl)):
        r = {"trades": len(tr), "trades_doc": CHILD_DOC[name]["trades"],
             "pnl": round(float(pnl.sum()), 2),
             "pnl_doc": CHILD_DOC[name]["pnl"],
             "sharpe": round(sharpe(pnl), 3),
             "sharpe_doc": CHILD_DOC[name]["sharpe"],
             "dd_pct": round(max_dd(pnl) / ACCOUNT * 100, 2),
             "dd_doc": CHILD_DOC[name]["dd_pct"]}
        good = (r["trades"] == r["trades_doc"]
                and abs(r["pnl"] - r["pnl_doc"]) <= 0.05
                and abs(r["sharpe"] - r["sharpe_doc"]) <= 0.03
                and abs(r["dd_pct"] - r["dd_doc"]) <= 0.03)
        r["reproduced"] = bool(good)
        ok = ok and good
        rep[name] = r
        print(f"repro {name}: {r}")
    if not ok:
        print("REPRODUCTION GATE FAILED - abort")
        return

    # static comparators from the portfolio study (sanity + E1 anchors)
    port = pd.read_csv(PORT / "combined_daily_pnl.csv")
    port["date"] = port["date"].astype(int)
    port = port.set_index("date").reindex(universe).fillna(0.0)
    s1_sharpe = sharpe(port["S1_equal_notional"].to_numpy())
    s2_sharpe = sharpe(port["S2_equal_risk"].to_numpy())
    print(f"static anchors: S1 Sharpe {s1_sharpe:.3f}, S2 {s2_sharpe:.3f} "
          f"(documented 0.420 / 0.418)")

    # ---------------- day-by-day coordination ----------------
    arms = {"A_equal_notional": np.zeros(nu), "B_equal_risk": np.zeros(nu)}
    decision = []
    contrib = {"ORB": 0.0, "SP2L": 0.0}
    blocked = defaultdict(int)
    child_stats = {"ORB": [], "SP2L": []}
    for j in range(nu):
        st_o = trailing_stats(orb_pnl, j) if j >= T_WIN else None
        st_s = trailing_stats(sp2l_pnl, j) if j >= T_WIN else None
        if st_o:
            child_stats["ORB"].append((universe[j], st_o["sharpe"],
                                       st_o["dd_pct"]))
        if st_s:
            child_stats["SP2L"].append((universe[j], st_s["sharpe"],
                                        st_s["dd_pct"]))
        # R1
        allow = {}
        for name, st in (("ORB", st_o), ("SP2L", st_s)):
            if st is None:
                allow[name] = (True, "burn_in")
            elif st["sharpe"] <= 0:
                allow[name] = (False, "r1_sharpe")
            elif st["dd_pct"] >= DD_CAP_PCT:
                allow[name] = (False, "r1_dd")
            else:
                allow[name] = (True, "ok")
        # R2 conflict veto
        a_o, a_s = allow["ORB"][0], allow["SP2L"][0]
        conflict = (a_o and a_s and orb_sig[j] and sp2l_sig[j]
                    and orb_dir[j] * sp2l_dir[j] == -1)
        if conflict:
            a_o = a_s = False
            blocked["r2_conflict"] += 1
        for name, a in (("ORB", a_o), ("SP2L", a_s)):
            if not a and allow[name][1] != "burn_in":
                blocked[allow[name][1]] += 1
        # R3 + weights
        n_allowed = int(a_o) + int(a_s)
        w = {"ORB": 0.0, "SP2L": 0.0}
        if n_allowed:
            if a_o:
                w["ORB"] = 1.0
            if a_s:
                w["SP2L"] = 1.0
            if n_allowed == 2:  # arm B renormalizes to sum 2, cap 1.00
                raw = 1.0 / (st_o["sigma"] if st_o else 1.0)
                raw2 = 1.0 / (st_s["sigma"] if st_s else 1.0)
                if st_o is None or st_s is None:
                    pass  # burn-in: equal weights already set
                else:
                    w["ORB"] = min(1.0, 2.0 * raw / (raw + raw2))
                    w["SP2L"] = min(1.0, 2.0 * raw2 / (raw + raw2))
        arms["A_equal_notional"][j] = ((1.0 if a_o else 0.0) * orb_pnl[j]
                                       + (1.0 if a_s else 0.0) * sp2l_pnl[j])
        arms["B_equal_risk"][j] = w["ORB"] * orb_pnl[j] + w["SP2L"] * sp2l_pnl[j]
        contrib["ORB"] += w["ORB"] * orb_pnl[j]
        contrib["SP2L"] += w["SP2L"] * sp2l_pnl[j]
        decision.append({"date": universe[j],
                         "orb_allowed": int(a_o), "sp2l_allowed": int(a_s),
                         "orb_sig": int(orb_sig[j]),
                         "sp2l_sig": int(sp2l_sig[j]),
                         "orb_reason": allow["ORB"][1],
                         "sp2l_reason": allow["SP2L"][1],
                         "conflict": int(conflict),
                         "w_orb_B": round(w["ORB"], 3),
                         "w_sp2l_B": round(w["SP2L"], 3),
                         "ens_A": round(arms["A_equal_notional"][j], 2),
                         "ens_B": round(arms["B_equal_risk"][j], 2)})
    print(f"coordination done ({timemod.time() - t0:.0f}s); "
          f"blocked tallies {dict(blocked)}")

    # ---------------- outputs ----------------
    pd.DataFrame({"date": universe, "orb": orb_pnl, "sp2l": sp2l_pnl}
                 ).to_csv(BASE / "child_streams.csv", index=False)
    dd = pd.DataFrame(decision)
    dd.to_csv(BASE / "ensemble_daily_pnl.csv", index=False)
    with open(BASE / "contributions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["child", "net_contribution", "blocked_days"])
        w.writerow(["ORB", round(contrib["ORB"], 2),
                    sum(1 for r in decision if not r["orb_allowed"]
                        and r["orb_reason"] != "burn_in")])
        w.writerow(["SP2L", round(contrib["SP2L"], 2),
                    sum(1 for r in decision if not r["sp2l_allowed"]
                        and r["sp2l_reason"] != "burn_in")])

    # ---------------- metrics + gates ----------------
    results = {}
    yearly_rows = []
    for arm, arr in arms.items():
        sh = sharpe(arr)
        dd_pct = max_dd(arr) / ACCOUNT * 100.0
        traded_days = int(sum(1 for r in decision
                              if (r["orb_allowed"] and r["orb_sig"])
                              or (r["sp2l_allowed"] and r["sp2l_sig"])))
        per_year = defaultdict(float)
        for j, d in enumerate(universe):
            per_year[d // 10000] += arr[j]
        tot = float(arr.sum())
        full = {y: v for y, v in per_year.items() if y < 2026}
        n_pos = sum(1 for v in full.values() if v > 0)
        best_share = (max(full.values()) / tot * 100
                      if tot > 0 and full else None)
        static = s1_sharpe if arm.startswith("A") else s2_sharpe
        e1 = (sh is not None and sh > 0 and sh > max(CHILD_DOC["ORB"]["sharpe"],
                                                     CHILD_DOC["SP2L"]["sharpe"])
              and sh > static)
        e2 = dd_pct <= 11.29
        e3 = traded_days >= 150
        e4 = n_pos >= 4 and best_share is not None and best_share <= 60
        results[arm] = {
            "sharpe": round(sh, 3) if sh else None,
            "static_anchor": round(static, 3),
            "total_net": round(tot, 2),
            "max_dd_pct": round(dd_pct, 2),
            "traded_days": traded_days, "win_days":
                round(100.0 * float((arr > 0).mean()), 1),
            "per_year": {str(k): round(v, 2) for k, v in sorted(per_year.items())},
            "pos_full_years": f"{n_pos}/{len(full)}",
            "best_year_share_pct": round(best_share, 1) if best_share else None,
            "gates": {"E1": bool(e1), "E2": bool(e2), "E3": bool(e3),
                      "E4": bool(e4)}}
        verdict = "PASS" if all(results[arm]["gates"].values()) else "FAIL"
        results[arm]["verdict"] = verdict
        print(f"\n{arm}: Sharpe {results[arm]['sharpe']} vs static "
              f"{static:.3f}; net {tot:+.2f}; DD {dd_pct:.2f}%; "
              f"traded days {traded_days}; years "
              f"{results[arm]['per_year']}; best-year "
              f"{results[arm]['best_year_share_pct']}%; "
              f"pos years {results[arm]['pos_full_years']}")
        print(f"  gates {results[arm]['gates']} -> {verdict}")
        for y, v in sorted(per_year.items()):
            yearly_rows.append({"arm": arm, "year": y,
                                "net": round(v, 2),
                                "share_pct": round(100 * v / tot, 1)
                                if tot > 0 else None})
    overall = ("PASS -> Phase 2" if any(r["verdict"] == "PASS"
                                        for r in results.values())
               else "FAIL -> STOP (null result)")
    print(f"\nOVERALL: {overall}")

    with open(BASE / "yearly_breakdown.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "year", "net", "share_pct"])
        w.writeheader()
        w.writerows(yearly_rows)

    stats = {"reproduction": rep,
             "static_anchors": {"S1": round(s1_sharpe, 3),
                                "S2": round(s2_sharpe, 3)},
             "arms": results, "blocked_tallies": dict(blocked),
             "contributions": {k: round(v, 2) for k, v in contrib.items()},
             "params": {"T_WIN": T_WIN, "DD_CAP_PCT": DD_CAP_PCT,
                        "universe_days": nu, "burn_in_days": T_WIN},
             "overall_verdict": overall,
             "runtime_s": round(timemod.time() - t0, 1)}
    with open(BASE / "phase1_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"wrote artifacts ({stats['runtime_s']}s)")


if __name__ == "__main__":
    main()
