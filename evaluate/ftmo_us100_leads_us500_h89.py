"""Frozen H89 cross-index quote replay. Read only; never sends an order."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date, timedelta
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from ftmo_us500_open_persistence_h88 import (
    broker_dt, broker_epoch, challenge, eur_pnl, fx_quote,
    percentile, price_round, price_up, sha,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/ftmo_us100_leads_us500_h89"
START = date(2025, 1, 2)
END = date(2026, 9, 24)
EXCLUDE = date(2025, 3, 21)
SEED = 20260930


def quotes(name: str, start: float, end: float):
    ticks = mt5.copy_ticks_range(name, broker_dt(start), broker_dt(end),
                                 mt5.COPY_TICKS_ALL)
    if ticks is None:
        return None, "API_ERROR"
    if len(ticks):
        ms = ticks["time_msc"].astype(np.int64)
        if np.any(ms < math.floor(start * 1000)) or np.any(ms > math.ceil(end * 1000)):
            return ticks, "OUT_OF_WINDOW"
        order = np.lexsort((np.arange(len(ticks)), ms))
        ticks = ticks[order]
        bid, ask = ticks["bid"], ticks["ask"]
        if np.any(~np.isfinite(bid)) or np.any(~np.isfinite(ask)) or np.any(bid <= 0) or np.any(ask < bid):
            return ticks, "BAD_QUOTES"
    return ticks, "OK"


def opening(ticks, start: float, end: float):
    if ticks is None or len(ticks) == 0:
        return None
    ts = ticks["time_msc"].astype(np.float64) / 1000
    idx = np.flatnonzero((ts >= start) & (ts < end))
    if not len(idx) or ts[idx[0]] >= start + 60:
        return None
    minutes = np.unique(np.floor((ts[idx] - start) / 60).astype(int))
    if len(minutes) != 15 or not np.array_equal(minutes, np.arange(15)):
        return None
    mid = (ticks["bid"][idx] + ticks["ask"][idx]) / 2
    return {"first": float(mid[0]), "last": float(mid[-1]),
            "range": float(mid.max() - mid.min()), "quotes": len(idx)}


def main() -> None:
    spec = OUT / "SPEC_FTMO_US100_LEADS_US500_H89.txt"
    gate = OUT / "GATE_FTMO_US100_LEADS_US500_H89.txt"
    report_path = OUT / "REPORT_FTMO_US100_LEADS_US500_H89.txt"
    assert not report_path.exists(), "H89 REPORT is immutable"
    lines = [s for s in spec.read_text(encoding="utf-8").splitlines()
             if s.startswith("G-XI")]
    assert lines == gate.read_text(encoding="utf-8").splitlines()
    assert mt5.initialize(), mt5.last_error()
    account = mt5.account_info()
    assert account and account.server == "FTMO-Demo" and account.currency == "EUR"
    assert mt5.positions_total() == 0 and mt5.orders_total() == 0
    for name in ("US500.cash", "US100.cash", "EURUSD"):
        assert mt5.symbol_select(name, True), (name, mt5.last_error())
    sym = mt5.symbol_info("US500.cash")
    assert sym and sym.trade_contract_size == 1 and sym.digits == 2

    source = hashlib.sha256()
    coverage, trades, unresolved = [], [], []
    spreads_in, spreads_out = [], []
    total500 = total100 = 0
    day = START
    try:
        while day <= END:
            if day.weekday() >= 5 or day == EXCLUDE:
                day += timedelta(days=1)
                continue
            t0 = broker_epoch(day, 9, 30)
            t1 = broker_epoch(day, 9, 45)
            t2 = broker_epoch(day, 12, 0)
            t3 = broker_epoch(day, 12, 5)
            a, status500 = quotes("US500.cash", t0, t3)
            b, status100 = quotes("US100.cash", t0, t1)
            if a is not None:
                source.update(day.isoformat().encode("ascii") + b"US500")
                source.update(a.tobytes())
                total500 += len(a)
            if b is not None:
                source.update(day.isoformat().encode("ascii") + b"US100")
                source.update(b.tobytes())
                total100 += len(b)
            if status500 != "OK" or status100 != "OK":
                coverage.append({"ny_date": day.isoformat(), "status": "SOURCE_ERROR",
                                 "us500_status": status500, "us100_status": status100})
                day += timedelta(days=1)
                continue
            op500, op100 = opening(a, t0, t1), opening(b, t0, t1)
            if op500 is None or op100 is None:
                coverage.append({"ny_date": day.isoformat(), "status": "INVALID_OPEN",
                                 "us500_quotes": len(a), "us100_quotes": len(b)})
                day += timedelta(days=1)
                continue
            ta = a["time_msc"].astype(np.float64) / 1000
            post = np.flatnonzero((ta >= t1) & (ta < t1 + 60))
            if not len(post):
                coverage.append({"ny_date": day.isoformat(), "status": "NO_ENTRY_QUOTE"})
                day += timedelta(days=1)
                continue
            r100 = op100["last"] / op100["first"] - 1
            r500 = op500["last"] / op500["first"] - 1
            residual = r100 - 1.4 * r500
            if residual == 0:
                coverage.append({"ny_date": day.isoformat(), "status": "ZERO_RESIDUAL"})
                day += timedelta(days=1)
                continue
            direction = 1 if residual > 0 else -1
            coverage.append({"ny_date": day.isoformat(), "status": "SIGNAL",
                             "r100": r100, "r500": r500, "residual": residual,
                             "us500_open_quotes": op500["quotes"],
                             "us100_open_quotes": op100["quotes"]})
            entry_i = int(post[0])
            entry = float(a["ask"][entry_i] if direction == 1 else a["bid"][entry_i])
            risk_points = price_up(max(op500["range"], 1.0))
            stop = price_round(entry - direction * risk_points)
            target = price_round(entry + direction * 2 * risk_points)
            mark = a["bid"] if direction == 1 else a["ask"]
            worst_points = float(direction * (mark[entry_i] - entry))
            exit_i, exit_reason = None, None
            for i in range(entry_i + 1, len(a)):
                if ta[i] >= t3:
                    break
                pts = float(direction * (mark[i] - entry))
                worst_points = min(worst_points, pts)
                if ta[i] >= t2:
                    exit_i, exit_reason = i, "NOON"
                    break
                if (direction == 1 and mark[i] <= stop) or (direction == -1 and mark[i] >= stop):
                    exit_i, exit_reason = i, "STOP"
                    break
                if (direction == 1 and mark[i] >= target) or (direction == -1 and mark[i] <= target):
                    exit_i, exit_reason = i, "TARGET"
                    break
            if exit_i is None:
                unresolved.append(day.isoformat() + ":EXIT")
                day += timedelta(days=1)
                continue
            fx_enter = fx_quote(float(ta[entry_i]))
            fx_exit = fx_quote(float(ta[exit_i]))
            if fx_enter is None or fx_exit is None:
                unresolved.append(day.isoformat() + ":FX")
                day += timedelta(days=1)
                continue
            base_points = float(direction * (mark[exit_i] - entry))
            spreads_in.append(float(a["ask"][entry_i] - a["bid"][entry_i]))
            spreads_out.append(float(a["ask"][exit_i] - a["bid"][exit_i]))
            row = {"ny_date": day.isoformat(), "direction": direction,
                   "r100": r100, "r500": r500, "residual": residual,
                   "entry_server_ms": int(a["time_msc"][entry_i]),
                   "exit_server_ms": int(a["time_msc"][exit_i]),
                   "entry_price": entry, "exit_price": float(mark[exit_i]),
                   "stop": stop, "target": target, "stop_distance": risk_points,
                   "exit_reason": exit_reason, "base_points": base_points,
                   "stress_points": base_points - 0.76,
                   "worst_points": worst_points,
                   "entry_spread": spreads_in[-1], "exit_spread": spreads_out[-1],
                   "fx_entry_bid": fx_enter[0], "fx_entry_ask": fx_enter[1],
                   "fx_exit_bid": fx_exit[0], "fx_exit_ask": fx_exit[1],
                   "fx_entry_delay_s": fx_enter[2],
                   "fx_exit_delay_s": fx_exit[2]}
            for risk in (0.5, 1.0):
                risk_eur = 10000 * risk / 100
                risk_lots = math.floor(risk_eur * fx_enter[0] / risk_points * 100 + 1e-8) / 100
                margin_lots = math.floor(10000 * 15 * fx_enter[0] / entry * 100 + 1e-8) / 100
                lot = max(0.0, min(risk_lots, margin_lots))
                row[f"lots_{risk:g}"] = lot
                row[f"base_eur_{risk:g}"] = eur_pnl(base_points * lot, fx_exit[0], fx_exit[1])
                row[f"stress_eur_{risk:g}"] = eur_pnl((base_points - 0.76) * lot,
                                                      fx_exit[0], fx_exit[1])
            trades.append(row)
            if len(trades) % 25 == 0:
                print(f"trades={len(trades)} through={day} source_quotes={total500 + total100}", flush=True)
            day += timedelta(days=1)
    finally:
        mt5.shutdown()

    pd.DataFrame(coverage).to_csv(OUT / "coverage.csv", index=False)
    pd.DataFrame(trades).to_csv(OUT / "trades.csv", index=False)
    boot, annual, pace, gates = {}, {}, {}, {}
    for risk in (0.5, 1.0):
        z = str(risk)
        key = f"stress_eur_{risk:g}"
        values = np.array([r[key] for r in trades], dtype=float)
        rng = np.random.default_rng(SEED)
        means = (rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
                 if len(values) else np.array([]))
        boot[z] = {"n": len(values),
                   "mean_stressed_eur": float(values.mean()) if len(values) else None,
                   "p_nonpositive": float(np.mean(means <= 0)) if len(values) else None,
                   "ci95": np.percentile(means, [2.5, 97.5]).tolist() if len(values) else None}
        annual[z] = {str(y): {"n": sum(r["ny_date"].startswith(str(y)) for r in trades),
                            "stressed_total_eur": sum(r[key] for r in trades
                                                      if r["ny_date"].startswith(str(y)))}
                     for y in (2025, 2026)}
        paid = challenge(trades, risk, 90, 11000, 4)
        pd.DataFrame(paid.pop("outcomes")).to_csv(OUT / f"paid_90d_{risk:g}.csv", index=False)
        pace[z] = {"paid_90d": paid}
    source_error = any(r["status"] == "SOURCE_ERROR" for r in coverage)
    for risk in (0.5, 1.0):
        z = str(risk)
        pp = pace[z]["paid_90d"]
        gates[z] = {
            "G-XI1": len(trades) >= 300 and len({r["ny_date"] for r in trades}) >= 300
                       and not unresolved and not source_error,
            "G-XI2": bool(trades) and boot[z]["mean_stressed_eur"] > 0,
            "G-XI3": bool(trades) and boot[z]["p_nonpositive"] <= 0.05,
            "G-XI4": all(annual[z][str(y)]["stressed_total_eur"] > 0
                         for y in (2025, 2026)),
            "G-XI5": pp["starts"] >= 150 and pp["pass_rate"] >= 0.70
                       and pp["breach_rate"] <= 0.10,
            "G-XI6": not unresolved and not source_error
                       and all(r["stop_distance"] > 0 and
                               r["exit_server_ms"] > r["entry_server_ms"]
                               for r in trades),
        }
    status = "PASS" if any(all(g.values()) for g in gates.values()) else "FAIL"
    summary = {"status": status, "spec_sha256": sha(spec), "gate_sha256": sha(gate),
               "script_sha256": sha(Path(__file__)),
               "source_raw_tick_sha256": source.hexdigest(),
               "source_days": len(coverage), "us500_quotes": total500,
               "us100_quotes": total100,
               "valid_signal_days": sum(r["status"] == "SIGNAL" for r in coverage),
               "trade_count": len(trades), "unresolved": unresolved,
               "source_errors": [r for r in coverage if r["status"] == "SOURCE_ERROR"],
               "entry_spread_p50": percentile(spreads_in, 50),
               "entry_spread_p90": percentile(spreads_in, 90),
               "exit_spread_p50": percentile(spreads_out, 50),
               "exit_spread_p90": percentile(spreads_out, 90),
               "base_mean_points": float(np.mean([r["base_points"] for r in trades])) if trades else None,
               "stress_mean_points": float(np.mean([r["stress_points"] for r in trades])) if trades else None,
               "boot": boot, "annual": annual, "pace": pace, "gates": gates}
    (OUT / "run_meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    text = [f"H89 US100 OPENING RELATIVE MOVE -> US500 - FINAL {status}",
            "Predeclared SPEC/GATE unchanged. FTMO-Demo native raw Bid/Ask.",
            "No account orders.",
            f"Source days/US500 quotes/US100 quotes: {len(coverage)}/{total500}/{total100}",
            f"Signals/executed/unresolved: {summary['valid_signal_days']}/{len(trades)}/{unresolved}",
            f"Entry/exit spread p50/p90: {summary['entry_spread_p50']}/{summary['entry_spread_p90']}/{summary['exit_spread_p50']}/{summary['exit_spread_p90']}",
            f"Base/stressed mean points: {summary['base_mean_points']}/{summary['stress_mean_points']}",
            "Bootstrap: " + json.dumps(boot),
            "Annual stressed EUR: " + json.dumps(annual),
            "Paid 90-day pace: " + json.dumps(pace),
            "Gates: " + json.dumps(gates),
            "Source errors: " + json.dumps(summary["source_errors"]),
            "Any historical PASS is selected exploratory evidence, not a verified trading system."]
    with report_path.open("x", encoding="utf-8") as f:
        f.write("\n".join(text) + "\n")
    print(json.dumps({k: summary[k] for k in
                      ("status", "trade_count", "base_mean_points", "stress_mean_points",
                       "boot", "annual", "pace", "gates")}, indent=2))


if __name__ == "__main__":
    main()
