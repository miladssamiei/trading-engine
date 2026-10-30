"""Frozen Hypothesis 70 account-speed diagnostic; never promotes failed execution."""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/xauusd_saleh_prop_pace_audit_v2"
SPEC = OUT / "SPEC_PROP_PACE.txt"
GATE = OUT / "GATE_PROP_PACE.txt"
SOURCE = ROOT / "data/results/xauusd_saleh_intraday_real_quotes_close1559_v1/native_quote_trades.csv"
PRIOR_META = ROOT / "data/results/xauusd_saleh_intraday_real_quotes_close1559_v1/run_meta.json"
RISKS = (0.001, 0.0025, 0.005, 0.01, 0.015, 0.02)
HORIZONS = (90, 180)
INITIAL, TARGET, STATIC_FLOOR, DAILY_ALLOWANCE = 10_000.0, 11_000.0, 9_000.0, 500.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trades() -> pd.DataFrame:
    trades = pd.read_csv(SOURCE)
    trades = trades.loc[trades.year >= 2024].copy().sort_values(["entry_ms", "exit_ms"])
    trades["entry_ny_date"] = pd.to_datetime(trades.entry_ms, unit="ms", utc=True).dt.tz_convert("America/New_York").dt.date
    trades["entry_prague_date"] = pd.to_datetime(trades.entry_ms, unit="ms", utc=True).dt.tz_convert("Europe/Prague").dt.date
    trades["exit_prague_date"] = pd.to_datetime(trades.exit_ms, unit="ms", utc=True).dt.tz_convert("Europe/Prague").dt.date
    assert (trades.entry_prague_date == trades.exit_prague_date).all()
    assert (trades.entry_ny_date.astype(str).str.replace("-", "") == trades.ny_date.astype(str)).all()
    trades["risk_units"] = np.where(trades.side.eq("BUY"),
                                     trades.entry_ask - trades.initial_sl,
                                     trades.initial_sl - trades.entry_bid)
    assert np.isfinite(trades.risk_units).all() and (trades.risk_units > 0).all()
    trades["r"] = trades.quote_net_units / trades.risk_units
    assert np.isfinite(trades.r).all()
    return trades


def replay_window(rows: pd.DataFrame, risk_fraction: float, start, horizon: int) -> dict:
    end = start + timedelta(days=horizon)
    balance = INITIAL
    day_open = INITIAL
    last_day = None
    trading_days = set()
    trade_count = 0
    for trade in rows.itertuples():
        if trade.entry_ny_date < start or trade.entry_ny_date >= end:
            continue
        day = trade.entry_prague_date
        if day != last_day:
            day_open = balance
            last_day = day
        trading_days.add(day)
        trade_count += 1
        balance += INITIAL * risk_fraction * trade.r
        if balance < STATIC_FLOOR or balance < day_open - DAILY_ALLOWANCE:
            return {"outcome": "BREACH", "days": (day - start).days,
                    "trades": trade_count, "balance": balance, "trading_days": len(trading_days)}
        if balance >= TARGET and len(trading_days) >= 4:
            return {"outcome": "PASS", "days": (day - start).days,
                    "trades": trade_count, "balance": balance, "trading_days": len(trading_days)}
    return {"outcome": "TIMEOUT", "days": horizon,
            "trades": trade_count, "balance": balance, "trading_days": len(trading_days)}


def main() -> None:
    assert not (OUT / "REPORT_PROP_PACE.txt").exists(), "Frozen report exists"
    spec = SPEC.read_text(encoding="utf-8")
    gate_lines = GATE.read_text(encoding="utf-8").splitlines()
    assert len(gate_lines) == 4 and all(line in spec for line in gate_lines)
    prior = json.loads(PRIOR_META.read_text(encoding="utf-8"))
    trades = load_trades()
    last_date = trades.entry_ny_date.max()
    starts = sorted(set(trades.entry_ny_date))
    outcomes = []
    for horizon in HORIZONS:
        eligible_starts = [start for start in starts if start + timedelta(days=horizon) <= last_date + timedelta(days=1)]
        for fraction in RISKS:
            for start in eligible_starts:
                result = replay_window(trades, fraction, start, horizon)
                outcomes.append({"horizon_days": horizon, "planned_risk_pct": fraction * 100,
                                 "start": str(start), "start_year": start.year, **result})
    out = pd.DataFrame(outcomes)
    out.to_csv(OUT / "rolling_outcomes.csv", index=False, float_format="%.8f")

    summaries = []
    for (horizon, risk), group in out.groupby(["horizon_days", "planned_risk_pct"], sort=True):
        counts = group.outcome.value_counts()
        passes = group[group.outcome == "PASS"]
        summaries.append({"horizon_days": int(horizon), "planned_risk_pct": float(risk),
                          "starts": len(group), "pass": int(counts.get("PASS", 0)),
                          "breach": int(counts.get("BREACH", 0)),
                          "timeout": int(counts.get("TIMEOUT", 0)),
                          "pass_rate": float((group.outcome == "PASS").mean()),
                          "breach_rate": float((group.outcome == "BREACH").mean()),
                          "median_days_to_pass": None if passes.empty else float(passes.days.median())})
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.8f")
    yearly_starts = []
    for (horizon, risk, year), group in out.groupby(["horizon_days", "planned_risk_pct", "start_year"], sort=True):
        yearly_starts.append({"horizon_days": int(horizon), "planned_risk_pct": float(risk),
                              "start_year": int(year), "starts": len(group),
                              "pass": int((group.outcome == "PASS").sum()),
                              "breach": int((group.outcome == "BREACH").sum()),
                              "timeout": int((group.outcome == "TIMEOUT").sum())})
    pd.DataFrame(yearly_starts).to_csv(OUT / "start_year_summary.csv", index=False)

    eligible = summary[(summary.horizon_days == 180) & (summary.planned_risk_pct <= 1.0)]
    pace_ok = eligible[(eligible.pass_rate >= 0.70) & (eligible.breach_rate <= 0.10)]
    pass_rows = out[out.outcome == "PASS"]
    gates = {"G-PACE1": bool((eligible.pass_rate >= 0.70).any()),
             "G-PACE2": bool(not pace_ok.empty),
             "G-PACE3": bool(((pass_rows.trading_days >= 4) & (pass_rows.balance >= TARGET)).all()),
             "G-PACE4": bool(prior.get("verdict") == "PASS" and prior.get("execution_funnel", {}).get("unresolved_exits") == 0)}
    monthly = trades.groupby(trades.entry_ny_date.astype(str).str[:7]).size()
    year_r = trades.groupby("year").r.agg(["count", "mean", "sum"]).reset_index()
    report = ["XAUUSD SALEH SAME-DAY PROP CHALLENGE PACE AUDIT V2",
              "Frozen #70 quote replay remains FAIL; this is a diagnostic, not an executable recommendation.",
              "The earlier V1 SPEC/GATE draft had non-verbatim gates and was abandoned before code/run.",
              "Rules: FTMO 10k 2-Step first Challenge, target 11000, daily close floor prior balance-500, static floor 9000, four Prague entry days.",
              "All fills and PnL come from prior native Bid/Ask replay, including one unresolved exit marked at last quote.",
              "Close-only floor checks, fractional sizes, no FTMO feed or floating drawdown: pass rates are optimistic upper bounds.",
              f"Input quote trades SHA256: {sha256(SOURCE)}",
              f"Prior metadata SHA256: {sha256(PRIOR_META)}",
              f"SPEC SHA256: {sha256(SPEC)}; GATE SHA256: {sha256(GATE)}",
              f"Gated trade count: {len(trades)}, unresolved: {int(trades.unresolved.sum())}, last NY date: {last_date}",
              f"Monthly trade count mean: {monthly.mean():.3f}; median: {monthly.median():.3f}; min/max: {monthly.min()}/{monthly.max()}",
              f"Risk-unit median: {trades.risk_units.median():.3f}; mean R/trade: {trades.r.mean():.5f}",
              "R by entry year:", year_r.to_string(index=False, float_format=lambda x: f"{x:.5f}"),
              "Rolling starts, pass/breach/timeout and pass-time median (days from start):",
              summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
              "Start-year summary:", pd.DataFrame(yearly_starts).to_string(index=False),
              f"GATES: {json.dumps(gates, sort_keys=True)}",
              "PACE SCREEN: " + ("PASS" if pace_ok.shape[0] else "FAIL"),
              "OVERALL EXECUTABLE STRATEGY: FAIL (prior native-quote execution gate remains failed).",
              "FTMO actual symbol economics, volume grid, margin and floating equity path are not known here."]
    (OUT / "REPORT_PROP_PACE.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"gates": gates, "pace_screen": "PASS" if len(pace_ok) else "FAIL",
                      "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()
