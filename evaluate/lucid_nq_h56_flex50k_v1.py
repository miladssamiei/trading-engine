"""Frozen NQ H56 proxy to Lucid 50K Flex account diagnostic; see SPEC/GATE."""
from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/results/futures_prop_validation_intraday_remeasurement_v2/trades.csv"
OUT = ROOT / "data/results/lucid_nq_h56_flex50k_v1"
SPEC = OUT / "SPEC_LUCID_NQ_H56_FLEX50K_V1.txt"
GATE = OUT / "GATE_LUCID_NQ_H56_FLEX50K_V1.txt"
REPORT = OUT / "REPORT_LUCID_NQ_H56_FLEX50K_V1.txt"
START_BALANCE = 50000.0
TARGET = 53000.0
LOCKED_MLL = 50100.0
TRAIL_TRIGGER = 52100.0
HORIZON = timedelta(days=180)
CASES = (
    ("2_NQ_1tick", 2, 20, 0.425, 0.0),
    ("4_NQ_1tick", 4, 20, 0.425, 0.0),
    ("2_NQ_2tick_plus_0p5", 2, 20, 0.675, 0.5),
    ("4_NQ_2tick_plus_0p5", 4, 20, 0.675, 0.5),
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load() -> pd.DataFrame:
    if REPORT.exists():
        raise FileExistsError(f"Immutable report already exists: {REPORT}")
    if not SPEC.exists() or not GATE.exists():
        raise FileNotFoundError("SPEC and GATE must predate evaluator output")
    data = pd.read_csv(SOURCE)
    data = data.loc[(data.asset == "NQ") & (data.engine == "C_H56_M1")].copy()
    assert len(data) == 447
    data = data.sort_values(["entry_time", "exit_time"]).reset_index(drop=True)
    assert data.entry_time.is_monotonic_increasing
    data["entry_ny"] = pd.to_datetime(data.entry_time, unit="s", utc=True).dt.tz_convert("America/New_York")
    data["exit_ny"] = pd.to_datetime(data.exit_time, unit="s", utc=True).dt.tz_convert("America/New_York")
    data["day"] = data.entry_ny.dt.date
    return data


def replay(window: pd.DataFrame, usd_per_point: int, count: int, cost: float, extra: float) -> dict:
    balance = START_BALANCE
    mll = START_BALANCE - 2000.0
    peak_close = START_BALANCE
    day = None
    day_pnl = 0.0
    largest_day = 0.0
    n = 0

    def close_day() -> None:
        nonlocal peak_close, mll
        peak_close = max(peak_close, balance)
        mll = LOCKED_MLL if peak_close > TRAIL_TRIGGER else peak_close - 2000.0

    for trade in window.itertuples(index=False):
        if day is not None and trade.day != day:
            close_day()
            day_pnl = 0.0
        day = trade.day
        n += 1
        # The short initial risk is measured to the structural ask stop,
        # while a long adds round-trip entry cost to its bid structural stop.
        stop_units = trade.initial_risk + extra + (cost if trade.side == 1 else cost - trade.cost)
        if balance - stop_units * usd_per_point * count <= mll:
            return {"outcome": "breach_risk_reserve", "trades": n, "ending_balance": balance,
                    "mll": mll, "largest_day": largest_day}
        pnl = (trade.gross - cost - extra) * usd_per_point * count
        balance += pnl
        day_pnl += pnl
        largest_day = max(largest_day, day_pnl)
        if balance <= mll:
            return {"outcome": "breach_balance", "trades": n, "ending_balance": balance,
                    "mll": mll, "largest_day": largest_day}
        if balance >= TARGET and largest_day / (balance - START_BALANCE) <= 0.5:
            return {"outcome": "pass", "trades": n, "ending_balance": balance,
                    "mll": mll, "largest_day": largest_day}
    if day is not None:
        close_day()
    return {"outcome": "timeout", "trades": n, "ending_balance": balance,
            "mll": mll, "largest_day": largest_day}


def main() -> None:
    data = load()
    same_day = data.entry_ny.dt.date.eq(data.exit_ny.dt.date)
    weekday = data.entry_ny.dt.weekday.lt(5)
    cutoff = data.exit_ny.dt.hour * 60 + data.exit_ny.dt.minute
    g1 = bool((same_day & weekday & cutoff.lt(16 * 60 + 45)).all())
    last_day = data.day.max()
    starts = [d for d in data.day.unique() if d + HORIZON <= last_day]
    g2 = len(starts) >= 50
    reports = []
    rolling_rows = []
    for label, count, usd_per_point, cost, extra in CASES:
        net = (data.gross - cost - extra) * usd_per_point * count
        yearly = {str(y): round(float(net[data.ny_year == y].sum()), 2)
                  for y in (2024, 2025, 2026)}
        outcomes = []
        for start in starts:
            end = start + HORIZON
            window = data.loc[(data.day >= start) & (data.day < end)]
            result = replay(window, usd_per_point, count, cost, extra)
            result.update(case=label, start=start.isoformat(), end=end.isoformat())
            rolling_rows.append(result)
            outcomes.append(result["outcome"])
        passed = outcomes.count("pass")
        breached = sum(o.startswith("breach") for o in outcomes)
        timed_out = outcomes.count("timeout")
        g3 = (passed / len(starts) >= 0.5 and breached / len(starts) <= 0.1) if starts else False
        g4 = all(v > 0 for v in yearly.values())
        reports.append({"case": label, "contracts": count, "usd_per_point_each": usd_per_point,
                        "cost_points": cost, "extra_points": extra,
                        "all_trade_mean_usd": round(float(net.mean()), 2),
                        "all_trade_sum_usd": round(float(net.sum()), 2),
                        "yearly_usd": yearly, "starts": len(starts), "pass": passed,
                        "breach": breached, "timeout": timed_out, "G3": g3,
                        "G4": g4, "diagnostic_pass": bool(g1 and g2 and g3 and g4)})
    pd.DataFrame(rolling_rows).to_csv(OUT / "rolling_starts.csv", index=False)
    metadata = {"source_sha256": digest(SOURCE), "spec_sha256": digest(SPEC),
                "gate_sha256": digest(GATE), "script_sha256": digest(Path(__file__)),
                "source_rows": len(data), "first_entry_ny": str(data.entry_ny.iloc[0]),
                "last_entry_ny": str(data.entry_ny.iloc[-1]), "G1": g1,
                "G2": g2, "G5": False, "results": reports}
    (OUT / "run_meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    lines = ["LUCID NQ H56 50K FLEX - ACCOUNT TRANSFER REPORT", "OVERALL VERDICT: NO GO",
             "2026-09-26. Frozen USTECH bid-M1 proxy; not native CME NQ or Lucid fills.",
             f"Source rows {len(data)}, complete 180-day rolling entry-day starts {len(starts)}.",
             f"G1 same-day weekday before 16:45 NY: {'PASS' if g1 else 'FAIL'}.",
             f"G2 >=50 complete starts: {'PASS' if g2 else 'FAIL'}.",
             "G5 native CME NQ price/quote and Lucid forward fills: FAIL (unavailable).",
             "Assumed one-tick spread, listed Lucid commissions and zero slippage are floors.",
             "Pretrade structural risk-reserve check is conservative but not a floating tick replay.",
             "The signal's M1-extreme entry is optimistic. No contract rolls or historical spread.",
             "Historical full-tape and rolling starts overlap; none is unseen forward validation.", ""]
    for r in reports:
        lines.extend([f"{r['case']}: {r['contracts']} x ${r['usd_per_point_each']}/point, "
                      f"cost {r['cost_points']} + extra {r['extra_points']} index points",
                      f"  Full-ledger mean ${r['all_trade_mean_usd']:.2f}/trade; "
                      f"sum ${r['all_trade_sum_usd']:.2f}; annual $ {r['yearly_usd']}",
                      f"  180d starts: pass {r['pass']}/{r['starts']}, breach {r['breach']}/{r['starts']}, "
                      f"timeout {r['timeout']}/{r['starts']}",
                      f"  G3 {'PASS' if r['G3'] else 'FAIL'}; G4 {'PASS' if r['G4'] else 'FAIL'}; "
                      f"diagnostic {'PASS' if r['diagnostic_pass'] else 'FAIL'}.", ""])
    lines.extend(["References:", "https://support.lucidtrading.com/en/articles/12945790-lucidflex-evaluation-account",
                  "https://support.lucidtrading.com/en/articles/12945815-lucidflex-drawdown",
                  "https://support.lucidtrading.com/en/articles/12945805-lucidflex-consistency-percentage",
                  "https://support.lucidtrading.com/en/articles/11404729-allowed-trading-times",
                  "https://support.lucidtrading.com/en/articles/11508978-approved-products-and-commissions"])
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
