"""Frozen-plan H95 close-only two-stage feasibility diagnostic; no orders."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/vix_weekday_h95"
TRADES = OUT / "trades.csv"
EXPECTED = "dcaf647df4fec656c97f3e93fdb6f189a57a17ab370ebba891f955131297bcc3"
END = pd.Timestamp("2026-09-23")


def stage(rows: pd.DataFrame, first: bool, euros_per_r: float):
    balance = 10000.0
    target = 11000 if first else 10500
    for count, row in enumerate(rows.itertuples(), 1):
        pnl = euros_per_r * row.stress_r
        balance += pnl
        if balance < 9000 or pnl < -500:
            return "BREACH", row.monday, count
        if balance >= target and count >= 4:
            return "PASS", row.monday, count
    return "TIMEOUT", pd.NaT, len(rows)


def main():
    assert hashlib.sha256(TRADES.read_bytes()).hexdigest() == EXPECTED
    assert (OUT / "PACE_PREFLIGHT_PLAN_2026-09-25.txt").exists()
    x = pd.read_csv(TRADES, parse_dates=["monday", "exit_ny"])
    x = x.loc[x.year >= 2024].sort_values("monday").reset_index(drop=True)
    assert x.monday.is_unique and x.monday.is_monotonic_increasing
    assert (x.exit_ny.dt.normalize() <= END).all()
    x["stress_r"] = x.stress_net / (2 * x.prior_atr20)
    assert np.isfinite(x.stress_r).all()
    starts = x.loc[x.monday + pd.Timedelta(days=180) <= END, "monday"]
    results = []
    for risk, euros in ((0.5, 50), (1.0, 100), (2.0, 200)):
        for start in starts:
            window = x.loc[(x.monday >= start) &
                           (x.monday < start + pd.Timedelta(days=180))].copy()
            s1, hit1, n1 = stage(window, True, euros)
            if s1 == "PASS":
                s2, hit2, _ = stage(window.iloc[n1:], False, euros)
                outcome = s2
                hit = hit2 if s2 == "PASS" else pd.NaT
            else:
                outcome = s1
                hit = pd.NaT
            results.append({"risk_pct": risk, "start": start, "year": start.year,
                            "stage1_hit": s1 == "PASS", "outcome": outcome,
                            "days": int((hit-start).days) if pd.notna(hit) else None})
    result = pd.DataFrame(results)
    result.to_csv(OUT / "pace_preflight_outcomes.csv", index=False)
    summary = []
    cohorts = []
    for risk, group in result.groupby("risk_pct"):
        counts = group.outcome.value_counts()
        summary.append({"risk_pct": risk, "starts": len(group),
                        "stage1_hits_before_breach": int(group.stage1_hit.sum()),
                        "pass": int(counts.get("PASS", 0)),
                        "breach": int(counts.get("BREACH", 0)),
                        "timeout": int(counts.get("TIMEOUT", 0)),
                        "pass_rate": float(group.outcome.eq("PASS").mean()),
                        "breach_rate": float(group.outcome.eq("BREACH").mean()),
                        "median_pass_days": float(group.days.median())
                        if group.days.notna().any() else None})
        for year, yg in group.groupby("year"):
            cohorts.append({"risk_pct": risk, "year": int(year),
                            "starts": len(yg), "pass": int(yg.outcome.eq("PASS").sum()),
                            "breach": int(yg.outcome.eq("BREACH").sum()),
                            "timeout": int(yg.outcome.eq("TIMEOUT").sum())})
    h94 = pd.read_csv(ROOT / "data/results/us100_weekday_hold_h94/trades.csv",
                      parse_dates=["monday"])
    h94 = h94.loc[h94.year >= 2024, ["monday", "stress_net", "prior_atr20"]].copy()
    h94["h94_r"] = h94.stress_net / (2*h94.prior_atr20)
    joined = x[["monday", "stress_r"]].merge(h94[["monday", "h94_r"]],
                                              on="monday", validate="one_to_one")
    assert len(joined) == len(x)
    comparison = {"matched_weeks": len(joined),
                  "h95_mean_r": float(joined.stress_r.mean()),
                  "h94_mean_r": float(joined.h94_r.mean()),
                  "changed_week_count": int((~np.isclose(joined.stress_r,
                                                          joined.h94_r)).sum())}
    report = OUT / "PACE_PREFLIGHT_REPORT_2026-09-25.txt"
    assert not report.exists()
    report.write_text("H95 US100 VIX-CURVE CLOSE-ONLY 180-DAY TWO-STAGE PREFLIGHT\n"
                      + "Frozen plan: PACE_PREFLIGHT_PLAN_2026-09-25.txt\n"
                      + "H95 trades SHA256: " + EXPECTED + "\n"
                      + "Risk rows: " + json.dumps(summary) + "\n"
                      + "Start-year cohorts: " + json.dumps(cohorts) + "\n"
                      + "H94 matched comparison: " + json.dumps(comparison) + "\n"
                      + "No native Bid/Ask, floating equity, exact FX, lot rounding or actual swap. No orders.\n",
                      encoding="utf-8")
    print(json.dumps({"risk_rows": summary, "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
