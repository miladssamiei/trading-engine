"""H88 causal-FX remeasurement of frozen H87 rule; no account orders."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/ftmo_us500_open_persistence_h88"
NY = ZoneInfo("America/New_York")
PRAGUE = ZoneInfo("Europe/Prague")
UTC = timezone.utc
EXCLUDE = date(2025, 3, 21)
START = date(2025, 1, 2)
END = date(2026, 9, 24)
SEED = 20260929


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def broker_epoch(day: date, hour: int, minute: int) -> float:
    wall = datetime.combine(day, time(hour, minute), NY)
    utc = wall.astimezone(UTC)
    offset = utc.astimezone(PRAGUE).utcoffset().total_seconds() + 3600
    return utc.timestamp() + offset


def broker_dt(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, UTC)


def price_up(x: float) -> float:
    return math.ceil(x * 100 - 1e-10) / 100


def price_round(x: float) -> float:
    return round(x, 2)


def fx_quote(epoch: float) -> tuple[float, float, float] | None:
    epoch_ms = round(epoch * 1000)
    ticks = mt5.copy_ticks_range("EURUSD", broker_dt(epoch - 60),
                                 broker_dt(math.floor(epoch) + 1), mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        return None
    eligible = ticks[(ticks["time_msc"] <= epoch_ms) &
                     (ticks["time_msc"] >= epoch_ms - 60000)]
    if len(eligible) == 0:
        return None
    t = eligible[-1]
    delay = (epoch_ms - int(t["time_msc"])) / 1000
    bid, ask = float(t["bid"]), float(t["ask"])
    if not 0 <= delay <= 60 or bid <= 0 or ask < bid:
        return None
    return bid, ask, delay


def eur_pnl(usd: float, fx_bid: float, fx_ask: float) -> float:
    converted = usd / (fx_ask if usd >= 0 else fx_bid)
    return converted - 0.007 * abs(converted)


def percentile(v: list[float], pct: float) -> float | None:
    return round(float(np.percentile(v, pct)), 5) if v else None


def challenge(rows: list[dict], risk: float, horizon: int, target: float,
              min_days: int) -> dict:
    starts = [r for r in rows if date.fromisoformat(r["ny_date"]) + timedelta(days=horizon) <= END]
    counts = {"PASS": 0, "BREACH": 0, "TIMEOUT": 0}
    pass_days = []
    outcomes = []
    for start in starts:
        first = date.fromisoformat(start["ny_date"])
        cutoff = first + timedelta(days=horizon)
        balance = 10000.0
        trade_days = set()
        verdict = "TIMEOUT"
        for r in rows:
            d = date.fromisoformat(r["ny_date"])
            if d < first or d >= cutoff:
                continue
            lot = r[f"lots_{risk:g}"]
            if lot <= 0:
                continue
            # A conservative fixed conversion bound for open-trade losses.
            worst_usd = r["worst_points"] * lot
            worst_fx_bid = min(r["fx_entry_bid"], r["fx_exit_bid"])
            floating = eur_pnl(worst_usd, worst_fx_bid, max(r["fx_entry_ask"], r["fx_exit_ask"]))
            if balance + floating <= balance - 500 or balance + floating <= 9000:
                verdict = "BREACH"
                break
            balance += r[f"stress_eur_{risk:g}"]
            trade_days.add(d.isoformat())
            if balance <= 9000 or balance <= (balance - r[f"stress_eur_{risk:g}"]) - 500:
                verdict = "BREACH"
                break
            if balance >= target and len(trade_days) >= min_days:
                verdict = "PASS"
                pass_days.append((d - first).days)
                break
        counts[verdict] += 1
        outcomes.append({"start": first.isoformat(), "risk_pct": risk,
                         "horizon_days": horizon, "verdict": verdict,
                         "end_balance": round(balance, 2)})
    n = len(starts)
    return {"starts": n, "pass": counts["PASS"], "breach": counts["BREACH"],
            "timeout": counts["TIMEOUT"],
            "pass_rate": counts["PASS"] / n if n else None,
            "breach_rate": counts["BREACH"] / n if n else None,
            "median_pass_days": float(np.median(pass_days)) if pass_days else None,
            "outcomes": outcomes}


def main() -> None:
    spec = OUT / "SPEC_FTMO_US500_OPEN_PERSISTENCE_H88.txt"
    gate = OUT / "GATE_FTMO_US500_OPEN_PERSISTENCE_H88.txt"
    script = Path(__file__).resolve()
    lines = [x for x in spec.read_text(encoding="utf-8").splitlines() if x.startswith("G-OC")]
    assert lines == gate.read_text(encoding="utf-8").splitlines(), "SPEC/GATE mismatch"
    assert mt5.initialize(), mt5.last_error()
    acct = mt5.account_info()
    assert acct and acct.server == "FTMO-Demo" and acct.currency == "EUR"
    assert mt5.positions_total() == 0 and mt5.orders_total() == 0
    sym = mt5.symbol_info("US500.cash")
    assert sym and sym.trade_contract_size == 1 and sym.digits == 2
    assert mt5.symbol_select("US500.cash", True)
    assert mt5.symbol_select("EURUSD", True)

    source = hashlib.sha256()
    previous_ranges: list[float] = []
    coverage: list[dict] = []
    trades: list[dict] = []
    unresolved = []
    all_spreads_entry: list[float] = []
    all_spreads_exit: list[float] = []
    total_quotes = 0
    day = START
    k = 0
    try:
        while day <= END:
            if day.weekday() >= 5 or day == EXCLUDE:
                day += timedelta(days=1)
                continue
            t0 = broker_epoch(day, 9, 30)
            t1 = broker_epoch(day, 10, 0)
            t2 = broker_epoch(day, 12, 0)
            t3 = broker_epoch(day, 12, 5)
            ticks = mt5.copy_ticks_range("US500.cash", broker_dt(t0), broker_dt(t3), mt5.COPY_TICKS_ALL)
            if ticks is None:
                coverage.append({"ny_date": day.isoformat(), "status": "API_ERROR", "error": str(mt5.last_error())})
                day += timedelta(days=1)
                continue
            source.update(day.isoformat().encode("ascii"))
            source.update(ticks.tobytes())
            n = len(ticks)
            total_quotes += n
            if n == 0:
                coverage.append({"ny_date": day.isoformat(), "status": "NO_QUOTES", "quotes": 0})
                day += timedelta(days=1)
                continue
            order = np.lexsort((np.arange(n), ticks["time_msc"]))
            ticks = ticks[order]
            ts = ticks["time_msc"] / 1000.0
            bid, ask = ticks["bid"], ticks["ask"]
            bad = (bid <= 0) | (ask < bid) | ~np.isfinite(bid) | ~np.isfinite(ask)
            if bool(bad.any()):
                coverage.append({"ny_date": day.isoformat(), "status": "BAD_QUOTES", "quotes": n, "bad": int(bad.sum())})
                day += timedelta(days=1)
                continue
            opening = np.flatnonzero((ts >= t0) & (ts < t1))
            post = np.flatnonzero((ts >= t1) & (ts < t1 + 60))
            minutes = np.unique(np.floor((ts[opening] - t0) / 60).astype(int)) if len(opening) else []
            if not len(opening) or not len(post) or ts[opening[0]] >= t0 + 60 or len(minutes) != 30:
                coverage.append({"ny_date": day.isoformat(), "status": "INVALID_OPEN", "quotes": n,
                                 "opening_quotes": len(opening), "minutes": len(minutes)})
                day += timedelta(days=1)
                continue
            mid = (bid[opening] + ask[opening]) / 2
            q = float(mid.max() - mid.min())
            f, l = float(mid[0]), float(mid[-1])
            baseline = float(np.median(previous_ranges[-20:])) if len(previous_ranges) >= 20 else None
            signal = baseline is not None and q >= baseline and q > 0 and abs(l - f) / q >= 0.5
            coverage.append({"ny_date": day.isoformat(), "status": "VALID_OPEN", "quotes": n,
                             "opening_quotes": len(opening), "q": q, "baseline": baseline, "signal": signal})
            previous_ranges.append(q)
            if not signal:
                day += timedelta(days=1)
                continue
            direction = 1 if l > f else -1
            entry_i = int(post[0])
            entry = float(ask[entry_i] if direction == 1 else bid[entry_i])
            d = price_up(max(q / 2, 1.0))
            stop = price_round(entry - direction * d)
            target = price_round(entry + direction * 2 * d)
            entry_spread = float(ask[entry_i] - bid[entry_i])
            exit_i = None
            exit_reason = None
            mark = bid if direction == 1 else ask
            worst_pts = float(direction * (mark[entry_i] - entry))
            for i in range(entry_i + 1, n):
                if ts[i] >= t3:
                    break
                pts = float(direction * (mark[i] - entry))
                worst_pts = min(worst_pts, pts)
                if ts[i] >= t2:
                    exit_i, exit_reason = i, "NOON"
                    break
                if (direction == 1 and bid[i] <= stop) or (direction == -1 and ask[i] >= stop):
                    exit_i, exit_reason = i, "STOP"
                    break
                if (direction == 1 and bid[i] >= target) or (direction == -1 and ask[i] <= target):
                    exit_i, exit_reason = i, "TARGET"
                    break
            if exit_i is None:
                unresolved.append(day.isoformat())
                day += timedelta(days=1)
                continue
            exit_price = float(mark[exit_i])
            base_pt = float(direction * (exit_price - entry))
            exit_spread = float(ask[exit_i] - bid[exit_i])
            fx_enter = fx_quote(float(ts[entry_i]))
            fx_exit = fx_quote(float(ts[exit_i]))
            if fx_enter is None or fx_exit is None:
                unresolved.append(day.isoformat() + ":FX")
                day += timedelta(days=1)
                continue
            all_spreads_entry.append(entry_spread)
            all_spreads_exit.append(exit_spread)
            r = {"ny_date": day.isoformat(), "direction": direction,
                 "entry_server_ms": int(ticks["time_msc"][entry_i]),
                 "exit_server_ms": int(ticks["time_msc"][exit_i]),
                 "opening_range": q, "opening_efficiency": abs(l - f) / q,
                 "prior20_median_range": baseline, "stop_distance": d,
                 "entry_price": entry, "exit_price": exit_price, "stop": stop,
                 "target": target, "exit_reason": exit_reason,
                 "entry_spread": entry_spread, "exit_spread": exit_spread,
                 "base_points": base_pt, "stress_points": base_pt - 0.76,
                 "worst_points": worst_pts,
                 "fx_entry_bid": fx_enter[0], "fx_entry_ask": fx_enter[1],
                 "fx_exit_bid": fx_exit[0], "fx_exit_ask": fx_exit[1],
                 "fx_entry_delay_s": fx_enter[2], "fx_exit_delay_s": fx_exit[2]}
            for risk in (0.5, 1.0):
                risk_eur = 10000 * risk / 100
                risk_lots = math.floor((risk_eur * fx_enter[0] / d) * 100 + 1e-8) / 100
                margin_lots = math.floor((10000 * 15 * fx_enter[0] / entry) * 100 + 1e-8) / 100
                lots = max(0.0, min(risk_lots, margin_lots))
                r[f"lots_{risk:g}"] = lots
                r[f"base_eur_{risk:g}"] = eur_pnl(base_pt * lots, fx_exit[0], fx_exit[1])
                r[f"stress_eur_{risk:g}"] = eur_pnl((base_pt - 0.76) * lots, fx_exit[0], fx_exit[1])
            trades.append(r)
            day += timedelta(days=1)
            k += 1
            if k % 25 == 0:
                print(f"signals={k} through {day} quotes={total_quotes}", flush=True)
    finally:
        mt5.shutdown()

    pd.DataFrame(coverage).to_csv(OUT / "coverage.csv", index=False)
    pd.DataFrame(trades).to_csv(OUT / "trades.csv", index=False)
    boot = {}
    annual = {}
    pace = {}
    for risk in (0.5, 1.0):
        key = f"stress_eur_{risk:g}"
        values = np.array([r[key] for r in trades], dtype=float)
        rng = np.random.default_rng(SEED)
        means = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1) if len(values) else np.array([])
        boot[str(risk)] = {"n": len(values), "mean_stressed_eur": float(values.mean()) if len(values) else None,
                           "p_nonpositive": float(np.mean(means <= 0)) if len(values) else None,
                           "ci95": np.percentile(means, [2.5, 97.5]).tolist() if len(values) else None}
        annual[str(risk)] = {str(y): {"n": sum(r["ny_date"].startswith(str(y)) for r in trades),
                                 "stressed_total_eur": sum(r[key] for r in trades if r["ny_date"].startswith(str(y)))}
                             for y in (2025, 2026)}
        paid = challenge(trades, risk, 90, 11000, 4)
        free = challenge(trades, risk, 14, 10500, 2)
        pd.DataFrame(paid.pop("outcomes")).to_csv(OUT / f"paid_90d_{risk:g}.csv", index=False)
        pd.DataFrame(free.pop("outcomes")).to_csv(OUT / f"free_14d_{risk:g}.csv", index=False)
        pace[str(risk)] = {"paid_90d": paid, "free_14d": free}
    source_fail = any(x["status"] in ("API_ERROR", "BAD_QUOTES") for x in coverage)
    same_date = True  # NY 10:00..12:05 is the same Prague and NY date.
    gates = {}
    for risk in (0.5, 1.0):
        z = str(risk)
        gates[z] = {
            "G-OC1": len(trades) >= 150 and len({r["ny_date"] for r in trades}) >= 150 and not unresolved and not source_fail,
            "G-OC2": bool(trades) and boot[z]["mean_stressed_eur"] > 0,
            "G-OC3": bool(trades) and boot[z]["p_nonpositive"] <= 0.05,
            "G-OC4": all(annual[z][str(y)]["stressed_total_eur"] > 0 for y in (2025, 2026)),
            "G-OC5": pace[z]["paid_90d"]["starts"] >= 150 and
                       pace[z]["paid_90d"]["pass_rate"] >= 0.70 and
                       pace[z]["paid_90d"]["breach_rate"] <= 0.10,
            "G-OC6": same_date and not unresolved and all(r["stop_distance"] > 0 for r in trades),
        }
    summary = {"status": "PASS" if any(all(g.values()) for g in gates.values()) else "FAIL",
               "frozen_spec_sha256": sha(spec), "frozen_gate_sha256": sha(gate),
               "script_sha256": sha(script), "source_raw_tick_sha256": source.hexdigest(),
               "source_quotes": total_quotes, "source_days": len(coverage),
               "valid_open_days": sum(x["status"] == "VALID_OPEN" for x in coverage),
               "trade_count": len(trades), "unresolved": unresolved,
               "entry_spread_p50": percentile(all_spreads_entry, 50),
               "entry_spread_p90": percentile(all_spreads_entry, 90),
               "exit_spread_p50": percentile(all_spreads_exit, 50),
               "exit_spread_p90": percentile(all_spreads_exit, 90),
               "base_mean_points": float(np.mean([r["base_points"] for r in trades])) if trades else None,
               "stress_mean_points": float(np.mean([r["stress_points"] for r in trades])) if trades else None,
               "boot": boot, "annual": annual, "pace": pace, "gates": gates}
    (OUT / "run_meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = ["H88 US500 OPEN PERSISTENCE CAUSAL FX — FINAL " + summary["status"],
              "Predeclared SPEC/GATE unchanged. Native FTMO-Demo raw BID/ASK quotes.",
              "No account orders or tester strategy trades.",
              "Source days/quotes/valid opens: " + str((summary["source_days"], total_quotes, summary["valid_open_days"])),
              "Executed trades/unresolved: " + str((len(trades), unresolved)),
              "Entry/exit spread p50/p90: " + str((summary["entry_spread_p50"], summary["entry_spread_p90"],
                                                   summary["exit_spread_p50"], summary["exit_spread_p90"])),
              "Base/stressed mean points: " + str((summary["base_mean_points"], summary["stress_mean_points"])),
              "Bootstrap: " + json.dumps(boot), "Annual stressed EUR: " + json.dumps(annual),
              "Pace: " + json.dumps(pace), "Gates: " + json.dumps(gates),
              "Historical PASS, if any, is selected exploratory evidence requiring a new frozen native tester study and forward confirmation."]
    (OUT / "REPORT_FTMO_US500_OPEN_PERSISTENCE_H88.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("status", "trade_count", "base_mean_points", "stress_mean_points", "boot", "annual", "pace", "gates")}, indent=2))


if __name__ == "__main__":
    main()
