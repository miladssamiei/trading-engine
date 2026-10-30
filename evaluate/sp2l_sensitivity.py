# -*- coding: utf-8 -*-
"""
sp2l_sensitivity.py — transparency/robustness audit of the confirmed SP2L
NY_E validation. NOT a re-optimization; nothing here modifies the confirmed
result, the EA, or its parameters. Post-hoc sensitivity runs are labeled as
such (the only GATED confirmation remains NY_E @ PGAP 0.04% / MAX_SL 0.40%).

Part 1 — Stage A single-filter + session-variant table, from the ALREADY-RUN
  artifacts (phase1_ablation.csv, step3/wf_all.csv): these individual
  configurations were among the 35 Phase-1 passers, so their Step 3
  walk-forwards were computed in the original run — pulled, not rebuilt.

Part 2 — broker-digit-convention bracket for the unit conversion: PGAP
  {0.004, 0.02, 0.04, 0.2, 0.4}% x MAX_SL scaled 10:1
  {0.04, 0.2, 0.4, 2.0, 4.0}%, for NY_E and the ALL24 baseline:
  Phase 1 gate (both conventions) at each pair; for pairs that pass, the
  same Step 3 protocol (TP grid {0.5,0.75,1.0,1.5,2.0}, occupancy
  re-simulation, 9 half-year WF folds, trailing-expectancy selection,
  gates G1-G3) under all three conventions (xtm/cls/next).
  The (0.04, 0.40) pair doubles as an exact regression anchor against the
  original run (NY_E n=1479, WF-next 1165 trades +811.18 sharpe 0.834;
  24_base n=2759).

Outputs: data/results/sp2l_ablation/SENSITIVITY_ADDENDUM.md,
  sensitivity_phase1.csv, sensitivity_wf.csv + console.
"""

import csv
import json
import sys
import time as timemod
from array import array
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp2l_phase1 as p1
import sp2l_step3 as s3

BASE = p1.BASE
PAIRS = [(0.004, 0.04), (0.02, 0.2), (0.04, 0.4), (0.2, 2.0), (0.4, 4.0)]
AUDIT_CFGS = [("24_base", "24", (False, False, False, False)),
              ("NY_E", "NY", (True, False, False, False))]
ANCHOR = {("24_base", 0.04): 2759, ("NY_E", 0.04): 1479}
ANCHOR_WF = {("NY_E", "next"): (1165, 811.18, 0.834)}

STAGE_A = ["24_base", "24_E", "24_T", "24_A", "24_S",
           "NY_base", "LON_base"]


# ------------------------------------------------------------- part 1 ----

def part1():
    ab = {}
    with open(BASE / "phase1_ablation.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            ab[r["cfg"]] = r
    wf = defaultdict(dict)
    with open(BASE / "step3" / "wf_all.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            wf[r["cfg"]][r["conv"]] = r

    lines = []
    lines.append("### Part 1 — Stage A single-filter / session-variant breakdown"
                 " (from the original run artifacts)\n")
    lines.append("Phase 1 (counts = #signals with excursion >= t; margins = "
                 "MFE−MAE; verdict = pre-declared gate under BOTH "
                 "conventions + G2):\n")
    lines.append("| config | n_first (+2nd) | up/dn | days | xtm m@.10/.15/.20 "
                 "| cls m@.10/.15/.20 | P1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for cfg in STAGE_A:
        r = ab[cfg]
        def m(c, t):
            return int(r[f"{c}_{t:03d}_mfe"]) - int(r[f"{c}_{t:03d}_mae"])
        lines.append(
            f"| {cfg} | {r['n_first']} (+{r['n_second']}) | "
            f"{r['n_up']}/{r['n_down']} | {r['days']} | "
            f"{m('xtm', 10):+d} / {m('xtm', 15):+d} / {m('xtm', 20):+d} | "
            f"{m('cls', 10):+d} / {m('cls', 15):+d} / {m('cls', 20):+d} | "
            f"{'PASS' if r['pass'] == 'True' else 'fail'} |")
    lines.append("")
    lines.append("Step 3 walk-forward for the same configurations (ALL were "
                 "among the 35 Phase-1 passers, so their WF was computed in "
                 "the original run — this is a pull, not a rebuild; no gap):\n")
    lines.append("| config | conv | trades | total | expt | sharpe | y2022..2026 "
                 "| y+ | best% | G1G2G3 | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for cfg in STAGE_A + ["NY_E"]:
        for conv in ("xtm", "cls", "next"):
            r = wf[cfg].get(conv)
            if not r:
                continue
            ys = "/".join(f"{float(r[f'y{y}']):+.0f}" for y in range(2022, 2027))
            lines.append(
                f"| {cfg} | {conv} | {r['trades']} | {float(r['total']):+.2f} | "
                f"{float(r['expt']):+.3f} | {r['sharpe']} | {ys} | "
                f"{r['years_pos']} | {r['best_share']} | "
                f"{r['G1']}{r['G2']}{r['G3']} | "
                f"{'PASS' if r['verdict'] == 'True' else 'FAIL'} |")
    return "\n".join(lines)


# ------------------------------------------------------------- part 2 ----

def wf_gates(runs, all_days, wf_start, wf_end):
    """Same WF protocol as s3.main: fold selection + gates per convention."""
    out = {}
    for conv, tp_runs in runs.items():
        wf_trades = []
        for k in range(len(s3.FOLD_BOUNDS) - 1):
            (fy, fm), (ty, tm) = s3.FOLD_BOUNDS[k], s3.FOLD_BOUNDS[k + 1]
            f_start = fy * 10000 + fm * 100
            f_end = ty * 10000 + tm * 100
            best, best_exp = None, None
            for tp_r in s3.TP_GRID:
                trail = [t for t in tp_runs[tp_r] if t["nd"] < f_start]
                if len(trail) < s3.MIN_TRAILING:
                    continue
                exp = sum(t["pnl"] for t in trail) / len(trail)
                if best_exp is None or exp > best_exp:
                    best, best_exp = tp_r, exp
            sel = best if best is not None else s3.DEFAULT_TP
            wf_trades.extend(t for t in tp_runs[sel]
                             if f_start <= t["nd"] < f_end)
        m = s3.metrics(wf_trades, all_days, wf_start, wf_end)
        per_year = defaultdict(float)
        for t in wf_trades:
            per_year[t["nd"] // 10000] += t["pnl"]
        py = {y: round(v, 2) for y, v in sorted(per_year.items())}
        years_pos = sum(1 for v in py.values() if v > 0)
        best_share = (max(py.values()) / m["total"] * 100.0
                      if m["total"] > 0 and py else None)
        g1 = m["expt"] is not None and m["expt"] > 0 \
            and m["sharpe"] is not None and m["sharpe"] > 0
        g2 = m["trades"] >= 150
        g3 = years_pos >= 4 and best_share is not None and best_share < 60.0
        out[conv] = {**m, "per_year": py, "years_pos": years_pos,
                     "best_share": None if best_share is None
                     else round(best_share, 1),
                     "G1": g1, "G2": g2, "G3": g3,
                     "verdict": g1 and g2 and g3}
    return out


def part2():
    t0 = timemod.time()
    ctx = p1.load_ctx()
    ctx["A_h_o"] = array("d", ctx["o"])
    ctx["_times_list"] = ctx["time"].tolist()
    sessions = {s: p1.build_session(s, ctx) for s in p1.SESSIONS}
    import numpy as np
    all_days = sorted(int(x) for x in np.unique(ctx["ny_date"]))
    wf_start = s3.FOLD_BOUNDS[0][0] * 10000 + s3.FOLD_BOUNDS[0][1] * 100
    wf_end = s3.FOLD_BOUNDS[-1][0] * 10000 + s3.FOLD_BOUNDS[-1][1] * 100

    p1_rows, wf_rows = [], []
    wf_summary = {}
    anchors_ok = True
    for gap, maxsl in PAIRS:
        setups, nb, nsell, nboth = p1.detect_setups(ctx, gap)
        p1.MAX_SL_PCT = maxsl          # read at runtime by run_config/simulate
        for name, sess, flt in AUDIT_CFGS:
            sigs, _ = p1.run_config(sess, flt[0], flt[1], flt[2], flt[3],
                                    ctx, sessions[sess], setups)
            sigs = p1.add_excursions(sigs, ctx, sessions[sess])
            row = p1.summarize(name, sess, flt, sigs)
            gate = row["pass"]
            p1_rows.append({"cfg": name, "pgap_pct": gap, "maxsl_pct": maxsl,
                            "setups": len(setups), "n_first": row["n_first"],
                            "n_second": row["n_second"],
                            "g1_xtm": row["g1_xtm"], "g1_cls": row["g1_cls"],
                            "g2": row["g2"], "pass_both": gate})
            if (name, gap) in ANCHOR and row["n_first"] != ANCHOR[(name, gap)]:
                anchors_ok = False
                print(f"!! ANCHOR MISMATCH {name} @gap {gap}: "
                      f"n_first {row['n_first']} != {ANCHOR[(name, gap)]}")
            print(f"[{timemod.time() - t0:6.0f}s] gap={gap}% sl={maxsl}% "
                  f"{name:<8} setups={len(setups):>6} n={row['n_first']:>5} "
                  f"g1xtm={int(row['g1_xtm'])} g1cls={int(row['g1_cls'])} "
                  f"g2={int(row['g2'])} -> "
                  f"{'P1-PASS' if gate else 'p1-fail'}")
            if not gate:
                continue
            # Step 3 protocol under all three conventions
            runs = {}
            for conv in ("xtm", "cls", "next"):
                runs[conv] = {
                    tp_r: s3.simulate_config(name, sess, flt, tp_r, conv,
                                             ctx, sessions, setups)
                    for tp_r in s3.TP_GRID}
            res = wf_gates(runs, all_days, wf_start, wf_end)
            wf_summary[(name, gap)] = res
            a = ANCHOR_WF.get(("NY_E", "next"))
            if a and name == "NY_E" and gap == 0.04:
                m = res["next"]
                if (m["trades"], m["total"], m["sharpe"]) != a:
                    anchors_ok = False
                    print(f"!! WF ANCHOR MISMATCH: {m['trades']}, "
                          f"{m['total']}, {m['sharpe']} != {a}")
                else:
                    print(f"   WF anchor reproduced exactly: {a}")
            for conv, m in res.items():
                wf_rows.append({"cfg": name, "pgap_pct": gap,
                                "maxsl_pct": maxsl, "conv": conv, **{
                                    k: v for k, v in m.items()
                                    if k != "per_year"},
                                **{f"y{y}": m["per_year"].get(y)
                                   for y in range(2022, 2027)}})
            clears_bar = all(res[c]["verdict"] for c in ("xtm", "cls", "next"))
            print(f"   WF: " + "  ".join(
                f"{c}: n={res[c]['trades']} tot={res[c]['total']:+.1f} "
                f"sh={res[c]['sharpe']} y+={res[c]['years_pos']} "
                f"{'PASS' if res[c]['verdict'] else 'FAIL'}"
                for c in ("xtm", "cls", "next")) +
                f"  -> clears-NY_E-bar={clears_bar}")
    p1.MAX_SL_PCT = 0.40               # restore
    return p1_rows, wf_rows, wf_summary, anchors_ok


def main():
    part1_md = part1()
    p1_rows, wf_rows, wf_summary, anchors_ok = part2()

    lines = ["# SENSITIVITY_ADDENDUM — SP2L NY_E robustness audit (2026-08-27)",
             "",
             "Post-hoc transparency audit. Nothing here was pre-declared as "
             "a gating run except the original PGAP 0.04% / MAX_SL 0.40% "
             "configuration; all other rows are informational robustness "
             "checks. No parameter, EA, or confirmed result was modified.",
             ""]
    lines.append(part1_md)
    lines.append("")
    lines.append("### Part 2 — broker-digit-convention bracket "
                 "(PGAP / MAX_SL scaled 10:1)\n")
    lines.append("Motivation: the original conversion assumed a 2-decimal "
                 "XAUUSD broker (point = 0.01 → PGAP $1.00 → 0.04%). A "
                 "3-decimal broker (point = 0.001) would make the source's "
                 "100 points = $0.10 → 0.004%; a coarse 1-decimal feed "
                 "would push the other way. The 10:1 MAX_SL:PGAP ratio of "
                 "the source is held fixed.\n")
    lines.append("| config | PGAP / MAX_SL | setups | n_first | G1-xtm | "
                 "G1-cls | G2 | P1 gate (both conv) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in p1_rows:
        lines.append(f"| {r['cfg']} | {r['pgap_pct']}% / {r['maxsl_pct']}% | "
                     f"{r['setups']} | {r['n_first']} | "
                     f"{int(r['g1_xtm'])} | {int(r['g1_cls'])} | "
                     f"{int(r['g2'])} | {'PASS' if r['pass_both'] else 'fail'} |")
    lines.append("")
    if wf_rows:
        lines.append("Step 3 walk-forward (same protocol as the confirmed "
                     "run; post-hoc, informational) for Phase-1-passing "
                     "pairs:\n")
        lines.append("| config | PGAP / MAX_SL | conv | trades | total | expt "
                     "| sharpe | y+ | best% | G1G2G3 | verdict | clears NY_E "
                     "bar (xtm+cls+next) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in wf_rows:
            key = (r["cfg"], r["pgap_pct"])
            bar = all(wf_summary[key][c]["verdict"]
                      for c in ("xtm", "cls", "next")) if key in wf_summary \
                else None
            lines.append(
                f"| {r['cfg']} | {r['pgap_pct']}% / {r['maxsl_pct']}% | "
                f"{r['conv']} | {r['trades']} | {r['total']:+.2f} | "
                f"{r['expt']} | {r['sharpe']} | {r['years_pos']} | "
                f"{r['best_share']} | {int(r['G1'])}{int(r['G2'])}"
                f"{int(r['G3'])} | {'PASS' if r['verdict'] else 'FAIL'} | "
                f"{bar if bar is not None else ''} |")
    lines.append("")
    anchor_txt = "REPRODUCED EXACTLY" if anchors_ok else "MISMATCH — investigate"
    lines.append(f"Regression anchor: {anchor_txt} "
                 "(NY_E n=1479 & WF-next 1165/+811.18/0.834; 24_base 2759 "
                 "at PGAP 0.04%).")
    md = "\n".join(lines)
    (BASE / "SENSITIVITY_ADDENDUM.md").write_text(md, encoding="utf-8")

    with open(BASE / "sensitivity_phase1.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(p1_rows[0].keys()))
        w.writeheader()
        w.writerows(p1_rows)
    if wf_rows:
        fields = ["cfg", "pgap_pct", "maxsl_pct", "conv", "trades", "total",
                  "expt", "sharpe", "maxdd", "years_pos", "best_share",
                  "G1", "G2", "G3", "verdict"] + [f"y{y}" for y in range(2022, 2027)]
        with open(BASE / "sensitivity_wf.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(wf_rows)
    print(md.encode("ascii", "replace").decode("ascii"))
    print(f"\nwrote {BASE / 'SENSITIVITY_ADDENDUM.md'}, "
          f"sensitivity_phase1.csv, sensitivity_wf.csv")


if __name__ == "__main__":
    main()
