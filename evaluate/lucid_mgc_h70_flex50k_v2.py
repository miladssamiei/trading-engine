"""V2 H70 source-subset correction for MGC/Lucid 50K; see V2 SPEC/GATE."""
from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/results/xauusd_saleh_floating_pace_audit_v1/native_worst_trades.csv"
QUOTE_LEDGER = ROOT / "data/results/xauusd_saleh_intraday_real_quotes_close1559_v1/native_quote_trades.csv"
OUT = ROOT / "data/results/lucid_mgc_h70_flex50k_v2"
SPEC = OUT / "SPEC_LUCID_MGC_H70_FLEX50K_V2.txt"
GATE = OUT / "GATE_LUCID_MGC_H70_FLEX50K_V2.txt"
REPORT = OUT / "REPORT_LUCID_MGC_H70_FLEX50K_V2.txt"
HORIZON = timedelta(days=180)
CASES = (("1_MGC_mid_floor", 1, False), ("2_MGC_mid_floor", 2, False),
         ("1_MGC_double_spread_stress", 1, True),
         ("2_MGC_double_spread_stress", 2, True))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load() -> tuple[pd.DataFrame, bool]:
    if REPORT.exists():
        raise FileExistsError(f"Historical report cannot be overwritten: {REPORT}")
    if not SPEC.exists() or not GATE.exists():
        raise FileNotFoundError("SPEC/GATE must exist before evaluation")
    trades = pd.read_csv(SOURCE)
    quote = pd.read_csv(QUOTE_LEDGER)
    quote = quote.loc[quote.year >= 2024].reset_index(drop=True)
    keys = ["ny_date", "side", "entry_ms", "exit_ms", "exit_type", "unresolved"]
    exact_match = (len(trades) == len(quote) == 321 and
                   trades[keys].reset_index(drop=True).equals(quote[keys].reset_index(drop=True)))
    trades = trades.sort_values("entry_ms").reset_index(drop=True)
    trades["day"] = pd.to_datetime(trades.ny_date.astype(str), format="%Y%m%d").dt.date
    trades["entry_ny"] = pd.to_datetime(trades.entry_ms, unit="ms", utc=True).dt.tz_convert("America/New_York")
    trades["exit_ny"] = pd.to_datetime(trades.exit_ms, unit="ms", utc=True).dt.tz_convert("America/New_York")
    return trades, exact_match


def replay(window: pd.DataFrame, count: int, stressed: bool) -> dict:
    if window.unresolved.any():
        return {"outcome": "invalid_source", "trades": 0, "ending_balance": 50000.0,
                "mll": 48000.0, "largest_day": 0.0}
    balance, mll, peak_close = 50000.0, 48000.0, 50000.0
    day = None
    day_pnl, largest_day = 0.0, 0.0
    n = 0
    for row in window.itertuples(index=False):
        if day is not None and row.day != day:
            peak_close = max(peak_close, balance)
            mll = 50100.0 if peak_close > 52100.0 else peak_close - 2000.0
            day_pnl = 0.0
        day = row.day
        n += 1
        realized = (row.quote_net_units - 0.26 - 0.41 if stressed
                    else row.gross_mid_units - 0.26)
        worst = row.worst_float_net_units - 0.26 - (0.41 if stressed else 0.0)
        worst = min(worst, realized)
        if balance + worst * 10.0 * count <= mll:
            return {"outcome": "breach_float", "trades": n, "ending_balance": balance,
                    "mll": mll, "largest_day": largest_day}
        pnl = realized * 10.0 * count
        balance += pnl
        day_pnl += pnl
        largest_day = max(largest_day, day_pnl)
        if balance <= mll:
            return {"outcome": "breach_balance", "trades": n, "ending_balance": balance,
                    "mll": mll, "largest_day": largest_day}
        if balance >= 53000.0 and largest_day / (balance - 50000.0) <= 0.5:
            return {"outcome": "pass", "trades": n, "ending_balance": balance,
                    "mll": mll, "largest_day": largest_day}
    if day is not None:
        peak_close = max(peak_close, balance)
        mll = 50100.0 if peak_close > 52100.0 else peak_close - 2000.0
    return {"outcome": "timeout", "trades": n, "ending_balance": balance,
            "mll": mll, "largest_day": largest_day}


def main() -> None:
    trades, exact_match = load()
    same_day = trades.entry_ny.dt.date.eq(trades.exit_ny.dt.date)
    weekday = trades.entry_ny.dt.weekday.lt(5)
    exit_minute = trades.exit_ny.dt.hour * 60 + trades.exit_ny.dt.minute
    g1 = bool(exact_match and not trades.unresolved.any() and
              (same_day & weekday & exit_minute.lt(16 * 60 + 45)).all())
    source_end = trades.day.max()
    starts = [d for d in trades.day.unique() if d + HORIZON <= source_end]
    g2 = len(starts) >= 50
    rolling = []
    summary = []
    for label, count, stressed in CASES:
        valid = trades.loc[~trades.unresolved]
        net = (valid.quote_net_units - 0.26 - 0.41 if stressed
               else valid.gross_mid_units - 0.26) * 10.0 * count
        annual = {str(y): round(float(net[valid.year == y].sum()), 2)
                  for y in (2024, 2025, 2026)}
        outcomes = []
        for start in starts:
            end = start + HORIZON
            window = trades.loc[(trades.day >= start) & (trades.day < end)]
            result = replay(window, count, stressed)
            result.update(case=label, start=start.isoformat(), end=end.isoformat())
            rolling.append(result)
            outcomes.append(result["outcome"])
        passed = outcomes.count("pass")
        breached = sum(o.startswith("breach") for o in outcomes)
        invalid = outcomes.count("invalid_source")
        timeout = outcomes.count("timeout")
        g3 = (passed / len(starts) >= 0.5 and breached / len(starts) <= 0.1) if starts else False
        g4 = all(v > 0 for v in annual.values())
        summary.append({"case": label, "contracts": count, "stressed": stressed,
                        "source_valid_trades": len(valid), "annual_usd": annual,
                        "net_mean_usd": round(float(net.mean()), 2),
                        "net_total_usd": round(float(net.sum()), 2),
                        "starts": len(starts), "pass": passed, "breach": breached,
                        "timeout": timeout, "invalid": invalid, "G3": g3, "G4": g4,
                        "diagnostic_pass": bool(g1 and g2 and g3 and g4)})
    pd.DataFrame(rolling).to_csv(OUT / "rolling_starts.csv", index=False)
    metadata = {"source_sha256": digest(SOURCE), "quote_ledger_sha256": digest(QUOTE_LEDGER),
                "spec_sha256": digest(SPEC), "gate_sha256": digest(GATE),
                "script_sha256": digest(Path(__file__)), "source_rows": len(trades),
                "source_exact_match": exact_match, "unresolved": int(trades.unresolved.sum()),
                "G1": g1, "G2": g2, "G5": False, "results": summary}
    (OUT / "run_meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    lines = ["LUCID MGC H70 50K FLEX - V2 TRANSFER REPORT", "OVERALL VERDICT: NO GO",
             "V2 source check uses the 2024-26 subset of the full 2020-26 quote ledger.",
             f"H70 15:59 native CFD quote rows {len(trades)}; exact gated-ledger match {exact_match}.",
             f"Unresolved required exits {int(trades.unresolved.sum())}; G1 {'PASS' if g1 else 'FAIL'}.",
             f"Complete 180-day rolling starts {len(starts)}; G2 {'PASS' if g2 else 'FAIL'}.",
             "G5 FAIL: no native CME GC/MGC quotes, contract rolls or Lucid forward fills.",
             "Floor uses spot midpoint and one-tick MGC cost; stress double-charges spread.",
             "Worst floating path is a native CFD quote proxy, not observed MGC equity.",
             "Every historical row is exposed, so rolling starts are not forward confirmation.", ""]
    for r in summary:
        lines.extend([f"{r['case']}: {r['contracts']} MGC, {r['source_valid_trades']} resolved trades",
                      f"  Net mean ${r['net_mean_usd']:.2f}/trade, total ${r['net_total_usd']:.2f}; annual $ {r['annual_usd']}",
                      f"  180d: pass {r['pass']}, breach {r['breach']}, timeout {r['timeout']}, "
                      f"invalid source {r['invalid']} / {r['starts']} starts",
                      f"  G3 {'PASS' if r['G3'] else 'FAIL'}, G4 {'PASS' if r['G4'] else 'FAIL'}, "
                      f"diagnostic {'PASS' if r['diagnostic_pass'] else 'FAIL'}", ""])
    lines.extend(["No evaluation purchase or account order is justified by this report.",
                  "Sources:", "https://support.lucidtrading.com/en/articles/12945790-lucidflex-evaluation-account",
                  "https://support.lucidtrading.com/en/articles/12945815-lucidflex-drawdown",
                  "https://support.lucidtrading.com/en/articles/12945805-lucidflex-consistency-percentage",
                  "https://support.lucidtrading.com/en/articles/11404729-allowed-trading-times",
                  "https://support.lucidtrading.com/en/articles/11508978-approved-products-and-commissions"])
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
