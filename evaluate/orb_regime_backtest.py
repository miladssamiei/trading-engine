#!/usr/bin/env python3
"""Step 2: backtest the HMM-regime-filtered ORB rule against unfiltered ORB.

PRE-DECLARED GATE (fixed before any result was computed; evaluated
mechanically at the end of this script -- if the gate FAILS we stop and
report honestly; no re-tuning of windows/states/features afterwards):
  G1  PRIMARY (w252_k2, trade only HIGHVOL days) annualized Sharpe >=
      baseline Sharpe + 0.15 AND total return > baseline total return.
  G2  INVERSE (w252_k2, trade only LOWVOL days) strictly worse than PRIMARY
      on BOTH Sharpe and total return (sanity: labels capture something real;
      if the inverse does as well or better, they do not).
  G3  Robustness to the walk-forward design: at least 2 of the 3 sensitivity
      HIGHVOL variants (w126_k2, w504_k2, w252_k3) beat the baseline on BOTH
      Sharpe and total return (i.e. the improvement is not a single-window
      artifact).
  OVERALL = G1 AND G2 AND G3.  Only if OVERALL passes do we proceed to an
  MT5 implementation (Step 4/5).

METHOD: the validated ORB baseline is strictly per-day independent (one
trade/day max; entry, SL/TP and session-end exit are all computed from day
D's own bars; the only cross-day input, the gap filter, is already encoded in
py_days.csv skipped/skip_reason). A regime filter that arms or disarms the
orders on day D therefore yields EXACTLY the subset of baseline trades whose
session-day label matches -- no re-simulation is required, and this mirrors
1:1 what the MT5 EA's regime-file gate will do. Metrics replicate
evaluate/orb_stats.py verbatim (daily PnL zero-filled over session days,
Sharpe = mean/std(ddof=1)*sqrt(252), maxDD on cumulative PnL, CAGR over
ACT/365.25 years, return/DD), computed over the COMMON window
2022-01-03 .. 2026-04-30 where the PRIMARY variant has labels. Days without
a label (w504 warmup) are untraded (0 PnL) -- the honest exposure treatment.
"""

import argparse
import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
ACCOUNT_BASE = 10000.0
WINDOW_FIRST_NY, WINDOW_LAST_NY = 20220103, 20260430


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-dir", default=os.path.join("data", "results", "orb_2021_present"))
    ap.add_argument("--regime-dir", default=os.path.join("data", "results", "orb_regime"))
    ap.add_argument("--out-dir", default=os.path.join("data", "results", "orb_regime"))
    return ap.parse_args()


def load_inputs(base_dir, regime_dir):
    trades = pd.read_csv(os.path.join(base_dir, "py_trades.csv"), sep=";")
    trades["open_dt"] = pd.to_datetime(trades["open_time"], format="%Y.%m.%d %H:%M", utc=True)
    trades["ny_date"] = [int(d.astimezone(NY).strftime("%Y%m%d")) for d in trades["open_dt"]]
    trades["profit"] = trades["profit"].astype(float)
    days = pd.read_csv(os.path.join(base_dir, "py_days.csv"), sep=";", dtype=str)
    days["ny_date_i"] = days["ny_date"].astype(int)
    days["date"] = pd.to_datetime(days["ny_date"], format="%Y%m%d", utc=True).dt.date
    days["session_open"] = pd.to_numeric(days["session_open"], errors="coerce")
    days["prev_close"] = pd.to_numeric(days["prev_close"], errors="coerce")
    days["is_session"] = (days["session_open"] > 0) | (days["prev_close"] > 0)
    labels = pd.read_csv(os.path.join(regime_dir, "regime_labels.csv"))
    labels["ny_date_i"] = labels["ny_date"].astype(int)
    m5 = pd.read_csv(os.path.join(base_dir, "us500tick_m5.csv"))
    m5["dt"] = pd.to_datetime(m5["time"], unit="s", utc=True)
    return trades, days, labels, m5


def win_years(m5):
    w = m5[(m5["dt"] >= pd.Timestamp("2022-01-03 00:00", tz=NY)) &
           (m5["dt"] <= pd.Timestamp("2026-04-30 23:59", tz=NY))]
    first_ts, last_ts = w["dt"].iloc[0], w["dt"].iloc[-1]
    return (last_ts - first_ts).total_seconds() / 86400.0 / 365.25, w


def variant_stats(name, trades_var, sess_nys, years):
    """orb_stats.aggregated conventions, zero-filled over session days."""
    per_day = trades_var.groupby("ny_date")["profit"].sum().to_dict()
    vals = np.array([per_day.get(int(ny), 0.0) for ny in sess_nys], dtype=float)
    total_pnl = float(vals.sum())
    total_return_pct = total_pnl / ACCOUNT_BASE * 100
    cum = pd.Series(vals).cumsum()
    maxdd = float((cum.cummax() - cum).max())
    maxdd_pct = maxdd / ACCOUNT_BASE * 100
    s = pd.Series(vals)
    std = float(s.std(ddof=1))
    sharpe = (float(s.mean()) / std * math.sqrt(252)) if std > 0 else None
    cagr = ((1 + total_return_pct / 100) ** (1 / years) - 1) * 100
    rdd = (total_return_pct / maxdd_pct) if maxdd_pct > 0 else None
    wins = int((trades_var["profit"] > 0).sum())
    return {
        "variant": name, "trades": int(len(trades_var)),
        "winrate_pct": (wins / len(trades_var) * 100) if len(trades_var) else 0.0,
        "traded_days": int(len(per_day)), "session_days": len(sess_nys),
        "exposure_pct": (len(per_day) / len(sess_nys) * 100) if len(sess_nys) else 0.0,
        "total_pnl": round(total_pnl, 2), "total_return_pct": round(total_return_pct, 4),
        "cagr_pct": round(cagr, 4), "max_dd_pct": round(maxdd_pct, 4),
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
        "return_dd": round(rdd, 4) if rdd is not None else None,
        "_per_day": per_day,
    }


def bh_window(w):
    first_price, last_price = float(w["open"].iloc[0]), float(w["close"].iloc[-1])
    first_ts, last_ts = w["dt"].iloc[0], w["dt"].iloc[-1]
    years = (last_ts - first_ts).total_seconds() / 86400.0 / 365.25
    price_ret_usd = (last_price - first_price) * 1.0
    bh_ret_price = (last_price / first_price - 1) * 100
    bh_ret_base = price_ret_usd / ACCOUNT_BASE * 100
    daily = w.groupby(w["dt"].dt.date)["close"].last()
    counts = w.groupby(w["dt"].dt.date).size()
    daily_full = daily[counts >= 50]
    eq = (daily_full / first_price - 1.0) * ACCOUNT_BASE
    maxdd_pct = float((eq.cummax() - eq).max()) / ACCOUNT_BASE * 100
    rets = daily_full.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std(ddof=1) * math.sqrt(252)) if len(rets) > 1 else None
    return {
        "price_ret_pct": round(bh_ret_price, 2), "base_ret_pct_ex_fin": round(bh_ret_base, 2),
        "cagr_price_pct": round(((1 + bh_ret_price / 100) ** (1 / years) - 1) * 100, 2),
        "cagr_base_pct_ex_fin": round(((1 + bh_ret_base / 100) ** (1 / years) - 1) * 100, 2),
        "max_dd_pct": round(maxdd_pct, 2),
        "sharpe_price": round(sharpe, 3) if sharpe is not None else None,
        "return_dd_ex_fin": round(bh_ret_base / maxdd_pct, 3) if maxdd_pct > 0 else None,
        "_daily": daily_full,
    }


def main():
    global win_years_cache
    args = parse_args()
    trades, days, labels, m5 = load_inputs(args.base_dir, args.regime_dir)
    years, w = win_years(m5)
    win_years_cache = (years, w)
    sess = days[(days["is_session"]) & (days["ny_date_i"] >= WINDOW_FIRST_NY) &
                (days["ny_date_i"] <= WINDOW_LAST_NY)].sort_values("date")
    sess_ny = [int(n) for n in sess["ny_date_i"]]
    sess_ny_full = [int(n) for n in days[days["is_session"]].sort_values("date")["ny_date_i"]]
    tw = trades[(trades["ny_date"] >= WINDOW_FIRST_NY) & (trades["ny_date"] <= WINDOW_LAST_NY)]
    lab = {(r.variant, r.ny_date_i): r.label for r in labels.itertuples()}

    def filtered(variant, wanted):
        ok = [d for d in tw["ny_date"] if lab.get((variant, d)) in wanted]
        return tw[tw["ny_date"].isin(ok)]

    years_full = (m5["dt"].iloc[-1] - m5["dt"].iloc[0]).total_seconds() / 86400.0 / 365.25
    results = {}
    results["baseline_fullrange_2021"] = variant_stats("baseline_fullrange_2021", trades, sess_ny_full, years_full)
    results["baseline_window"] = variant_stats("baseline_window", tw, sess_ny, years)
    results["primary_w252_k2_HIGHVOL"] = variant_stats("primary_w252_k2_HIGHVOL", filtered("w252_k2", {"HIGHVOL"}), sess_ny, years)
    results["inverse_w252_k2_LOWVOL"] = variant_stats("inverse_w252_k2_LOWVOL", filtered("w252_k2", {"LOWVOL"}), sess_ny, years)
    results["sens_w126_k2_HIGHVOL"] = variant_stats("sens_w126_k2_HIGHVOL", filtered("w126_k2", {"HIGHVOL"}), sess_ny, years)
    results["sens_w504_k2_HIGHVOL"] = variant_stats("sens_w504_k2_HIGHVOL", filtered("w504_k2", {"HIGHVOL"}), sess_ny, years)
    results["sens_w252_k3_HIGHVOL"] = variant_stats("sens_w252_k3_HIGHVOL", filtered("w252_k3", {"HIGHVOL"}), sess_ny, years)
    bh = bh_window(w)

    base, prim, inv = results["baseline_window"], results["primary_w252_k2_HIGHVOL"], results["inverse_w252_k2_LOWVOL"]
    sens = [results["sens_w126_k2_HIGHVOL"], results["sens_w504_k2_HIGHVOL"], results["sens_w252_k3_HIGHVOL"]]
    g1 = prim["sharpe"] is not None and prim["sharpe"] >= base["sharpe"] + 0.15 and prim["total_return_pct"] > base["total_return_pct"]
    g2 = (inv["sharpe"] < prim["sharpe"]) and (inv["total_return_pct"] < prim["total_return_pct"])
    g3 = sum(1 for s in sens if s["sharpe"] > base["sharpe"] and s["total_return_pct"] > base["total_return_pct"]) >= 2
    gate = {"G1_primary_beats_baseline_sharpe015_ret": bool(g1),
            "G2_inverse_strictly_worse_than_primary": bool(g2),
            "G3_2of3_sensitivity_variants_beat_baseline": bool(g3),
            "OVERALL_PROCEED_TO_MT5": bool(g1 and g2 and g3)}

    # monthly matrix
    sess_month = ["%04d-%02d" % (d.year, d.month) for d in sess["date"]]
    months = sorted(set(sess_month))
    rows = []
    cum_bh = 0.0
    for m in months:
        nys_m = [ny for ny, mm in zip(sess_ny, sess_month) if mm == m]

        def m_pnl(res):
            return round(float(sum(res["_per_day"].get(int(ny), 0.0) for ny in nys_m)), 2)
        dm = w[w["dt"].dt.strftime("%Y-%m") == m]
        if len(dm):
            month_close = dm["close"].iloc[-1]
            if not rows:
                prev_close = dm["open"].iloc[0]
            bh_m = (month_close / prev_close - 1) * 100
            prev_close = month_close
            cum_bh = (1 + cum_bh / 100) * (1 + bh_m / 100) * 100 - 100
        rows.append({"month": m, "base_pnl": m_pnl(base), "primary_pnl": m_pnl(prim),
                     "inverse_pnl": m_pnl(inv), "bh_month_ret_pct": round(bh_m, 2), "bh_cum_ret_pct": round(cum_bh, 2)})

    months_prim_better = sum(1 for r in rows if r["primary_pnl"] > r["base_pnl"])
    months_prim_worse = sum(1 for r in rows if r["primary_pnl"] < r["base_pnl"])

    out = {"window": {"first_ny": WINDOW_FIRST_NY, "last_ny": WINDOW_LAST_NY,
                      "years": round(years, 4), "session_days": len(sess_ny)},
           "gate_predeclared": gate,
           "variants": {k: {kk: vv for kk, vv in v.items() if kk != "_per_day"} for k, v in results.items()},
           "buy_hold_window_ex_financing": {k: v for k, v in bh.items() if k != "_daily"},
           "monthly_matrix": rows,
           "monthly_stability_primary_vs_baseline": {"months_better": months_prim_better,
                                                     "months_worse": months_prim_worse,
                                                     "months_total": len(rows)}}
    with open(os.path.join(args.out_dir, "regime_backtest.json"), "w") as f:
        json.dump(out, f, indent=1)

    lines = ["# Regime-filtered ORB vs unfiltered ORB (Python, common window 2022-01-03..2026-04-30)", ""]
    lines.append("## Pre-declared gate")
    for k, v in gate.items():
        lines.append("- %s: %s" % (k, "PASS" if v else "FAIL"))
    lines += ["", "## Variants (metrics = orb_stats.py conventions, zero-filled session days)", "",
              "| variant | trades | winrate% | exposure% | PnL | ret% | CAGR% | maxDD% | Sharpe | ret/DD |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for k in ["baseline_fullrange_2021", "baseline_window", "primary_w252_k2_HIGHVOL",
              "inverse_w252_k2_LOWVOL", "sens_w126_k2_HIGHVOL", "sens_w504_k2_HIGHVOL",
              "sens_w252_k3_HIGHVOL"]:
        v = out["variants"][k]
        lines.append("| %s | %d | %.1f | %.1f | %.2f | %.2f | %.2f | %.2f | %s | %s |" % (
            v["variant"], v["trades"], v["winrate_pct"], v["exposure_pct"], v["total_pnl"],
            v["total_return_pct"], v["cagr_pct"], v["max_dd_pct"],
            "%.3f" % v["sharpe"] if v["sharpe"] is not None else "-", 
            "%.2f" % v["return_dd"] if v["return_dd"] is not None else "-"))
    b = out["buy_hold_window_ex_financing"]
    lines += ["", "Buy&hold same window (ex-financing, $10k/1-lot): ret %.2f%% (price %.2f%%), CAGR %.2f%%, maxDD %.2f%%, Sharpe %.3f, ret/DD %.2f" % (
        b["base_ret_pct_ex_fin"], b["price_ret_pct"], b["cagr_base_pct_ex_fin"], b["max_dd_pct"],
        b["sharpe_price"], b["return_dd_ex_fin"]),
        "", "## Monthly matrix", "",
        "| month | base PnL | primary PnL | inverse PnL | B&H m% | B&H cum% |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %.2f | %.2f | %.2f | %.2f | %.2f |" % (
            r["month"], r["base_pnl"], r["primary_pnl"], r["inverse_pnl"],
            r["bh_month_ret_pct"], r["bh_cum_ret_pct"]))
    lines += ["", "Monthly stability (primary vs baseline): better %d / worse %d / total %d" % (
        months_prim_better, months_prim_worse, len(rows)), ""]
    with open(os.path.join(args.out_dir, "regime_backtest_summary.md"), "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(gate, indent=1))
    for k, v in out["variants"].items():
        print("%-32s trades=%3d pnl=%8.2f ret=%6.2f%% sharpe=%s" % (
            k, v["trades"], v["total_pnl"], v["total_return_pct"],
            "%.3f" % v["sharpe"] if v["sharpe"] is not None else "-"))


if __name__ == "__main__":
    main()
