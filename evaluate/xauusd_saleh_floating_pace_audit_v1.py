"""Frozen H70 native-tick floating-risk audit; never promotes failed execution."""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/xauusd_saleh_floating_pace_audit_v1"
SOURCE = ROOT / "data/results/xauusd_saleh_intraday_real_quotes_close1559_v1/native_quote_trades.csv"
PRIOR_META = ROOT / "data/results/xauusd_saleh_intraday_real_quotes_close1559_v1/run_meta.json"
INVENTORY = ROOT / "data/results/xauusd_onboarding/onboarding_stats.json"
EXPECTED_TRADES_SHA = "17d0f1fc14c716c9cc428c6c4283bc13528ce28c4e1853ad4715de00df34107c"
RISKS = (0.001, 0.005, 0.01, 0.015, 0.02)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(4 << 20), b""):
            h.update(part)
    return h.hexdigest()


def load_frozen() -> tuple[pd.DataFrame, dict, list[dict]]:
    if sha256(SOURCE) != EXPECTED_TRADES_SHA:
        raise ValueError("Frozen H70 trade-list hash mismatch")
    trades = pd.read_csv(SOURCE)
    trades = trades.loc[trades.year >= 2024].copy().sort_values(["entry_ms", "exit_ms"])
    if len(trades) != 321 or int(trades.unresolved.sum()) != 1:
        raise ValueError("Frozen H70 gated trade count or unresolved flag changed")
    if not (trades.entry_ms < trades.exit_ms).all():
        raise ValueError("Invalid frozen trade interval")
    if (trades.entry_ms.to_numpy()[1:] <= trades.exit_ms.to_numpy()[:-1]).any():
        raise ValueError("Frozen trade intervals overlap")
    trades["risk_units"] = np.where(trades.side.eq("BUY"),
                                    trades.entry_ask - trades.initial_sl,
                                    trades.initial_sl - trades.entry_bid)
    if not (np.isfinite(trades.risk_units) & (trades.risk_units > 0)).all():
        raise ValueError("Invalid initial planned risk")
    prior = json.loads(PRIOR_META.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))["files"]
    return trades.reset_index(drop=True), prior, inventory


def native_worst(trades: pd.DataFrame, inventory: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    starts = trades.entry_ms.to_numpy(dtype=np.int64)
    ends = trades.exit_ms.to_numpy(dtype=np.int64)
    min_bid = np.full(len(trades), np.inf)
    max_ask = np.full(len(trades), -np.inf)
    counts = np.zeros(len(trades), dtype=np.int64)
    checked = []
    previous_last = -1
    for entry in inventory:
        source = Path(entry["file"])
        digest = sha256(source)
        if digest != entry["sha256"]:
            raise ValueError(f"Native source hash mismatch: {source.name}")
        checked.append({"file": str(source), "sha256": digest, "rows_inventory": entry["rows"]})
        if entry["last_ts"] < starts.min() or entry["first_ts"] > ends.max():
            continue
        rows_read = 0
        for chunk in pd.read_csv(source, usecols=["timestamp", "askPrice", "bidPrice"],
                                 dtype={"timestamp": "int64", "askPrice": "float64", "bidPrice": "float64"},
                                 chunksize=1_000_000):
            ts = chunk.timestamp.to_numpy(dtype=np.int64)
            bid = chunk.bidPrice.to_numpy(dtype=float)
            ask = chunk.askPrice.to_numpy(dtype=float)
            rows_read += len(ts)
            if len(ts) == 0:
                continue
            if ts[0] <= previous_last or (np.diff(ts) <= 0).any():
                raise ValueError("Native tick timestamps not strictly increasing")
            previous_last = int(ts[-1])
            valid = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask >= bid)
            if not valid.all():
                raise ValueError("Invalid Bid/Ask in source tick chunk")
            first_trade = int(np.searchsorted(ends, ts[0], side="left"))
            last_trade = int(np.searchsorted(starts, ts[-1], side="right"))
            for i in range(first_trade, min(last_trade, len(trades))):
                left = int(np.searchsorted(ts, starts[i], side="left"))
                right = int(np.searchsorted(ts, ends[i], side="right"))
                if right <= left:
                    continue
                min_bid[i] = min(min_bid[i], float(bid[left:right].min()))
                max_ask[i] = max(max_ask[i], float(ask[left:right].max()))
                counts[i] += right - left
        if rows_read != entry["rows"]:
            raise ValueError(f"Native source row count mismatch: {source.name}")
    result = trades.copy()
    result["path_tick_count"] = counts
    result["path_min_bid"] = np.where(counts > 0, min_bid, np.nan)
    result["path_max_ask"] = np.where(counts > 0, max_ask, np.nan)
    result["worst_float_net_units"] = np.where(result.side.eq("BUY"),
                                               result.path_min_bid - result.entry_ask,
                                               result.entry_bid - result.path_max_ask)
    result["worst_float_r"] = result.worst_float_net_units / result.risk_units
    result["realized_r"] = result.quote_net_units / result.risk_units
    return result, checked


def replay_window(trades: pd.DataFrame, risk: float, start, horizon: int = 180) -> dict:
    end = start + timedelta(days=horizon)
    balance = 10000.0
    day_open = 10000.0
    last_day = None
    trading_days = set()
    for trade in trades.itertuples(index=False):
        if trade.entry_ny_date < start or trade.entry_ny_date >= end:
            continue
        if trade.entry_prague_date != last_day:
            day_open = balance
            last_day = trade.entry_prague_date
        trading_days.add(trade.entry_prague_date)
        floating_equity = balance + 10000.0 * risk * trade.worst_float_r
        floor = max(9000.0, day_open - 500.0)
        if floating_equity < floor:
            return {"outcome": "BREACH", "days": (trade.entry_ny_date - start).days,
                    "balance": balance, "lowest_equity": floating_equity}
        balance += 10000.0 * risk * trade.realized_r
        if balance < floor:
            return {"outcome": "BREACH", "days": (trade.entry_ny_date - start).days,
                    "balance": balance, "lowest_equity": floating_equity}
        if balance >= 11000.0 and len(trading_days) >= 4:
            return {"outcome": "PASS", "days": (trade.entry_ny_date - start).days,
                    "balance": balance, "lowest_equity": floating_equity}
    return {"outcome": "TIMEOUT", "days": horizon, "balance": balance,
            "lowest_equity": None}


def challenge(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    entry_utc = pd.to_datetime(trades.entry_ms, unit="ms", utc=True)
    exit_utc = pd.to_datetime(trades.exit_ms, unit="ms", utc=True)
    trades = trades.copy()
    trades["entry_ny_date"] = entry_utc.dt.tz_convert("America/New_York").dt.date
    trades["entry_prague_date"] = entry_utc.dt.tz_convert("Europe/Prague").dt.date
    trades["exit_prague_date"] = exit_utc.dt.tz_convert("Europe/Prague").dt.date
    if not (trades.entry_prague_date == trades.exit_prague_date).all():
        raise ValueError("Cross-Prague-date frozen trade")
    last_date = trades.entry_ny_date.max()
    starts = sorted({day for day in trades.entry_ny_date
                     if day + timedelta(days=180) <= last_date + timedelta(days=1)})
    outcome_rows = []
    for risk in RISKS:
        for start in starts:
            outcome_rows.append({"risk_pct": risk * 100, "start_date": start.isoformat(),
                                 **replay_window(trades, risk, start)})
    outcomes = pd.DataFrame(outcome_rows)
    summaries = []
    for risk_pct, group in outcomes.groupby("risk_pct", sort=True):
        counts = group.outcome.value_counts()
        passes = group.loc[group.outcome.eq("PASS"), "days"]
        summaries.append({"risk_pct": risk_pct, "starts": len(group),
                          "pass": int(counts.get("PASS", 0)),
                          "breach": int(counts.get("BREACH", 0)),
                          "timeout": int(counts.get("TIMEOUT", 0)),
                          "pass_rate": float((group.outcome == "PASS").mean()),
                          "breach_rate": float((group.outcome == "BREACH").mean()),
                          "median_days_to_pass": float(passes.median()) if len(passes) else None})
    return outcomes, pd.DataFrame(summaries)


def main() -> None:
    report_path = OUT / "REPORT_FLOATING_PACE_AUDIT.txt"
    if report_path.exists():
        raise FileExistsError(f"Frozen report exists: {report_path}")
    spec = OUT / "SPEC_FLOATING_PACE_AUDIT.txt"
    gate = OUT / "GATE_FLOATING_PACE_AUDIT.txt"
    gate_lines = [line for line in spec.read_text(encoding="utf-8").splitlines()
                  if line.startswith("G-FP")]
    if gate_lines != gate.read_text(encoding="utf-8").splitlines():
        raise ValueError("SPEC/GATE lines differ")
    trades, prior, inventory = load_frozen()
    paths, checked = native_worst(trades, inventory)
    outcomes, summary = challenge(paths)
    path_complete = bool((paths.path_tick_count > 0).all())
    ordering = bool((paths.worst_float_net_units <= paths.quote_net_units + 1e-8).all()
                    and (paths.worst_float_net_units <= paths.risk_units + 1e-8).all())
    pace_ok = summary.loc[(summary.pass_rate >= 0.70) & (summary.breach_rate <= 0.10)]
    gates = {"G-FP1": path_complete and len(paths) == 321 and len(checked) == 4,
             "G-FP2": ordering,
             "G-FP3": bool(len(pace_ok)),
             "G-FP4": bool(prior.get("verdict") == "PASS" and
                           prior.get("execution_funnel", {}).get("unresolved_exits") == 0)}
    report = {"source_trades_sha256": sha256(SOURCE), "spec_sha256": sha256(spec),
              "gate_sha256": sha256(gate), "source_files": checked,
              "trades": len(paths), "unresolved": int(paths.unresolved.sum()),
              "intervals_without_ticks": int((paths.path_tick_count == 0).sum()),
              "minimum_path_ticks": int(paths.path_tick_count.min()),
              "floating_vs_realized_violations": int((paths.worst_float_net_units > paths.quote_net_units + 1e-8).sum()),
              "worst_float_r_min": float(paths.worst_float_r.min()),
              "worst_float_r_median": float(paths.worst_float_r.median()),
              "summary": summary.astype(object).where(pd.notna(summary), None).to_dict("records"),
              "gates": gates,
              "verdict": "PASS" if all(gates.values()) else "FAIL",
              "caveat": "Frozen H70 quote execution still FAIL; this is native-quote floating-risk diagnosis, not executable FTMO evidence."}
    paths.to_csv(OUT / "native_worst_trades.csv", index=False, float_format="%.8f")
    outcomes.to_csv(OUT / "rolling_outcomes.csv", index=False, float_format="%.8f")
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.8f")
    (OUT / "run_meta.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_path.write_text("XAUUSD H70 NATIVE FLOATING-EQUITY PACE AUDIT\n"
                           + json.dumps(report, indent=2, allow_nan=False)
                           + "\nOVERALL VERDICT: " + report["verdict"] + "\n", encoding="utf-8")
    print("VERDICT", report["verdict"], "GATES", gates)
    print(summary.to_string(index=False))
    print("WORST FLOAT R", report["worst_float_r_min"], report["worst_float_r_median"])


if __name__ == "__main__":
    main()
