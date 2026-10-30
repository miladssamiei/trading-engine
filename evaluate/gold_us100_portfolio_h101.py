"""Frozen H101 exposed-ledger portfolio pace preflight; read only."""
from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/gold_us100_portfolio_h101"
GOLD = ROOT / "data/results/xauusd_saleh_intraday_real_quotes_close1559_v1/native_quote_trades.csv"
US100 = ROOT / "data/results/us100_weekday_hold_h94/trades.csv"
COVERAGE = ROOT / "data/results/us100_weekday_hold_h94/coverage.csv"
EXPECTED = {
    GOLD: "17d0f1fc14c716c9cc428c6c4283bc13528ce28c4e1853ad4715de00df34107c",
    US100: "5cf3b14bab33e0f6bfff79fbff2a5793f33822981273bb35376c1cb2a75e51e1",
    COVERAGE: "28251f94667e33fe498c01ef4803f438bbe7dc962fd1192a20fc1feb8190891d",
}
GAPS = (date(2024, 2, 14), date(2025, 12, 22), date(2025, 12, 29))
END = date(2026, 9, 1)
GOLD_ROWS = (100, 150, 200)
INDEX_RISK = 200


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load() -> pd.DataFrame:
    g = pd.read_csv(GOLD)
    g = g.loc[(g.year >= 2024) & (~g.unresolved)].copy()
    gbuy = g.side.eq("BUY")
    entry_px = np.where(gbuy, g.entry_ask, g.entry_bid)
    distance = np.abs(entry_px - g.initial_sl.to_numpy(float))
    if not (distance > 0).all():
        raise ValueError("Gold stop distance not positive")
    ge = pd.to_datetime(g.entry_ms, unit="ms", utc=True).dt.tz_convert("America/New_York").dt.date
    gx = pd.to_datetime(g.exit_ms, unit="ms", utc=True).dt.tz_convert("America/New_York").dt.date
    gold = pd.DataFrame({"asset": "GOLD", "entry_date": ge, "exit_date": gx,
                         "stressed_r": g.stress_units.to_numpy(float) / distance})
    u = pd.read_csv(US100)
    u = u.loc[u.year >= 2024].copy()
    ud = (u.entry - u.initial_stop).to_numpy(float)
    if not (ud > 0).all():
        raise ValueError("US100 stop distance not positive")
    idx = pd.DataFrame({"asset": "US100", "entry_date": pd.to_datetime(u.entry_ny).dt.date,
                        "exit_date": pd.to_datetime(u.exit_ny).dt.date,
                        "stressed_r": u.stress_net.to_numpy(float) / ud})
    ledger = pd.concat([gold, idx], ignore_index=True).sort_values(
        ["exit_date", "asset", "entry_date"]).reset_index(drop=True)
    ledger = ledger.loc[ledger.exit_date <= END].copy()
    return ledger


def replay(ledger: pd.DataFrame, start: date, gold_risk: int) -> dict:
    end = start + timedelta(days=180)
    window = ledger.loc[(ledger.entry_date >= start) & (ledger.exit_date < end)].copy()
    window["eur"] = window.stressed_r * np.where(window.asset.eq("GOLD"), gold_risk, INDEX_RISK)
    stage = 1
    bal = 10000.0
    stage_start = start
    unique_entries: set[date] = set()
    stage1_date = None
    for exit_day, group in window.groupby("exit_date", sort=True):
        current = group.loc[group.entry_date >= stage_start]
        if len(current) == 0:
            continue
        unique_entries.update(current.entry_date)
        bal += float(current.eur.sum())
        if bal <= 9000:
            return dict(outcome="BREACH", stage=stage, day=exit_day,
                        stage1_date=stage1_date, balance=bal)
        target = 11000 if stage == 1 else 10500
        if bal >= target and len(unique_entries) >= 4:
            if stage == 2:
                return dict(outcome="PASS", stage=2, day=exit_day,
                            stage1_date=stage1_date, balance=bal)
            stage1_date = exit_day
            stage = 2
            stage_start = exit_day + timedelta(days=1)
            bal = 10000.0
            unique_entries = set()
    return dict(outcome="TIMEOUT", stage=stage, day=None,
                stage1_date=stage1_date, balance=bal)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = OUT / "REPORT_GOLD_US100_PORTFOLIO_H101.txt"
    if report.exists():
        raise FileExistsError(report)
    spec = OUT / "SPEC_GOLD_US100_PORTFOLIO_H101.txt"
    gate = OUT / "GATE_GOLD_US100_PORTFOLIO_H101.txt"
    lines = [x for x in spec.read_text(encoding="utf-8").splitlines() if x.startswith("G-PF")]
    if lines != gate.read_text(encoding="utf-8").splitlines():
        raise ValueError("SPEC/GATE not verbatim")
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in EXPECTED}
    if not all(sha(p) == expected for p, expected in EXPECTED.items()):
        raise ValueError("Source ledger changed")
    ledger = load()
    starts = [d.date() for d in pd.date_range("2024-01-01", "2026-03-05", freq="W-MON")
              if d.date() + timedelta(days=180) <= END]
    clean = [s for s in starts if not any(s <= gap < s + timedelta(days=180) for gap in GAPS)]
    rows = []
    for risk in GOLD_ROWS:
        for start in clean:
            out = replay(ledger, start, risk)
            rows.append({"gold_risk_eur": risk, "us100_risk_eur": INDEX_RISK,
                         "start": str(start), "outcome": out["outcome"],
                         "stage_at_end": out["stage"],
                         "end_date": str(out["day"]) if out["day"] else "",
                         "stage1_date": str(out["stage1_date"]) if out["stage1_date"] else "",
                         "days_to_pass": (out["day"] - start).days if out["outcome"] == "PASS" else np.nan,
                         "balance_at_end": out["balance"]})
    outcomes = pd.DataFrame(rows)
    outcomes.to_csv(OUT / "rolling_outcomes.csv", index=False)
    ledger.to_csv(OUT / "normalized_ledger.csv", index=False)
    annual = (ledger.assign(year=ledger.entry_date.map(lambda x: x.year))
              .groupby(["year", "asset"]).stressed_r.agg(["sum", "size"]).to_dict("index"))
    annual_json = {f"{y}-{asset}": {"sum_r": float(v["sum"]), "n": int(v["size"])}
                   for (y, asset), v in annual.items()}
    summaries = []
    for risk, group in outcomes.groupby("gold_risk_eur"):
        counts = group.outcome.value_counts()
        passes = group.loc[group.outcome.eq("PASS"), "days_to_pass"]
        summaries.append({"gold_risk_eur": int(risk), "starts": len(group),
                          "pass": int(counts.get("PASS", 0)),
                          "breach": int(counts.get("BREACH", 0)),
                          "timeout": int(counts.get("TIMEOUT", 0)),
                          "pass_rate": float(group.outcome.eq("PASS").mean()),
                          "breach_rate": float(group.outcome.eq("BREACH").mean()),
                          "median_pass_days": float(passes.median()) if len(passes) else None})
    by_asset = ledger.groupby("asset").stressed_r.sum().to_dict()
    years = sorted(set(ledger.entry_date.map(lambda x: x.year)))
    by_year = {str(y): float(sum(annual.get((y, a), {}).get("sum", 0) for a in ("GOLD", "US100")))
               for y in years}
    gates = {"G-PF1": bool(len(clean) >= 50),
             "G-PF2": bool(all(by_year.get(str(y), 0) > 0 for y in (2024, 2025, 2026))
                           and all(by_asset.get(a, 0) > 0 for a in ("GOLD", "US100"))),
             "G-PF3": bool(any(s["pass_rate"] >= .70 and s["breach_rate"] <= .10
                                for s in summaries)),
             "G-PF4": True}
    result = {"source_sha256": hashes, "source_gaps": [str(x) for x in GAPS],
              "candidate_starts": len(starts), "clean_starts": len(clean),
              "excluded_starts": len(starts) - len(clean), "ledger_rows": len(ledger),
              "annual": annual_json, "combined_year_r": by_year,
              "child_pooled_r": by_asset, "summary": summaries, "gates": gates,
              "verdict": "PASS_PREFLIGHT_ONLY" if all(gates.values()) else "FINAL_FAIL"}
    (OUT / "run_meta.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report.write_text("H101 EXPOSED GOLD H70 + US100 H94 CLOSE-ONLY PORTFOLIO PREFLIGHT\n"
                      + json.dumps(result, indent=2) + "\n"
                      + "This excludes all windows with known native quote gaps; full floating equity, concurrent position handling, daily-loss resets, native US100 ticks, EUR conversion and forward validation remain unproven. No account order.\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
