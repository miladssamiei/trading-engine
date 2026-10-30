"""Predeclared read-only H92 gold benchmark + H94 US100 feasibility check."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from us100_vix_weekday_h95_pace import stage


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/project_status"
GOLD = ROOT / "data/results/xauusd_weekly_trend_h92/trades.csv"
US100 = ROOT / "data/results/us100_weekday_hold_h94/trades.csv"
HASHES = {GOLD: "56a88b4608b6846df1df4afad2815eae750d1fda128e75b0ed2a01f4ad07af6a",
          US100: "5cf3b14bab33e0f6bfff79fbff2a5793f33822981273bb35376c1cb2a75e51e1"}
END = pd.Timestamp("2026-09-01")


def main():
    assert (OUT / "H92_GOLD_BENCH_H94_US100_PLAN_2026-09-25.txt").exists()
    for path, expected in HASHES.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    gold = pd.read_csv(GOLD, parse_dates=["monday"])
    index = pd.read_csv(US100, parse_dates=["monday"])
    gold = gold.loc[(gold.year >= 2024) & gold.benchmark_stress.notna(),
                    ["monday", "benchmark_stress", "prior_atr20"]].copy()
    index = index.loc[index.year >= 2024,
                      ["monday", "stress_net", "prior_atr20"]].copy()
    assert gold.monday.is_unique and index.monday.is_unique
    gold["gold_r"] = gold.benchmark_stress / (2 * gold.prior_atr20)
    index["index_r"] = index.stress_net / (2 * index.prior_atr20)
    missing_gold = sorted(set(index.monday.dt.strftime("%Y-%m-%d")) -
                          set(gold.monday.dt.strftime("%Y-%m-%d")))
    missing_index = sorted(set(gold.monday.dt.strftime("%Y-%m-%d")) -
                           set(index.monday.dt.strftime("%Y-%m-%d")))
    joined = gold[["monday", "gold_r"]].merge(index[["monday", "index_r"]],
                                             on="monday", validate="one_to_one")
    joined = joined.sort_values("monday").reset_index(drop=True)
    joined["stress_r"] = joined.gold_r + joined.index_r
    assert np.isfinite(joined[["gold_r", "index_r", "stress_r"]].to_numpy()).all()
    matched = {"weeks": len(joined), "gold_only": missing_index,
               "index_only": missing_gold,
               "gold_mean_r": float(joined.gold_r.mean()),
               "index_mean_r": float(joined.index_r.mean()),
               "correlation": float(joined.gold_r.corr(joined.index_r)),
               "both_negative_share": float(((joined.gold_r < 0) &
                                             (joined.index_r < 0)).mean()),
               "worst_joint_r": float(joined.stress_r.min()),
               "annual_joint_r": {str(y): float(joined.loc[joined.monday.dt.year == y,
                                                          "stress_r"].sum())
                                  for y in (2024, 2025, 2026)}}
    starts = joined.loc[joined.monday + pd.Timedelta(days=180) <= END, "monday"]
    results = []
    for start in starts:
        window = joined.loc[(joined.monday >= start) &
                            (joined.monday < start + pd.Timedelta(days=180))].copy()
        s1, hit1, n1 = stage(window, True, 200.0)
        if s1 == "PASS":
            s2, hit2, _ = stage(window.iloc[n1:], False, 200.0)
            outcome = s2
            hit = hit2 if s2 == "PASS" else pd.NaT
        else:
            outcome = s1
            hit = pd.NaT
        results.append({"start": start, "year": start.year,
                        "stage1_hit": s1 == "PASS", "outcome": outcome,
                        "days": int((hit-start).days) if pd.notna(hit) else None})
    result = pd.DataFrame(results)
    result.to_csv(OUT / "H92_GOLD_BENCH_H94_US100_OUTCOMES_2026-09-25.csv",
                  index=False)
    counts = result.outcome.value_counts()
    summary = {"starts": len(result),
               "stage1_hits_before_breach": int(result.stage1_hit.sum()),
               "pass": int(counts.get("PASS", 0)),
               "breach": int(counts.get("BREACH", 0)),
               "timeout": int(counts.get("TIMEOUT", 0)),
               "pass_rate": float(result.outcome.eq("PASS").mean()),
               "breach_rate": float(result.outcome.eq("BREACH").mean()),
               "median_pass_days": float(result.days.median())
               if result.days.notna().any() else None}
    cohorts = {str(y): {"starts": len(g), "pass": int(g.outcome.eq("PASS").sum()),
                        "breach": int(g.outcome.eq("BREACH").sum()),
                        "timeout": int(g.outcome.eq("TIMEOUT").sum())}
               for y, g in result.groupby("year")}
    report = OUT / "H92_GOLD_BENCH_H94_US100_REPORT_2026-09-25.txt"
    assert not report.exists()
    report.write_text(
        "H92 GOLD BENCHMARK + H94 US100 2%+2% 180-DAY BOOK FEASIBILITY\n"
        + "Predeclared plan: H92_GOLD_BENCH_H94_US100_PLAN_2026-09-25.txt\n"
        + "Input SHA256: " + json.dumps({str(k.relative_to(ROOT)): v
                                        for k, v in HASHES.items()}) + "\n"
        + "Matched/source coverage: " + json.dumps(matched) + "\n"
        + "Close-only two-stage: " + json.dumps(summary) + "\n"
        + "Start-year cohorts: " + json.dumps(cohorts) + "\n"
        + "Descriptive exposed ledgers only; no exact account execution or orders.\n",
        encoding="utf-8")
    print(json.dumps({"matched": matched, "two_stage": summary,
                      "cohorts": cohorts}, indent=2))


if __name__ == "__main__":
    main()
