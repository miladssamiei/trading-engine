"""Frozen H81: fade a large overnight US500 cash gap for 30 minutes."""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/us500_overnight_firsthalf_reversal_v1"
SOURCE = ROOT / "data/results/orb_2021_present/us500tick_m5.csv"
EXPECTED_SHA = "a8ee68f1a0010a1be9704c0d59fd5382aa798846cdae4d519b799ae21ec66c2d"
WINDOW = (570, 575, 580, 585, 590, 595)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_cash_days() -> tuple[pd.DataFrame, dict]:
    source_hash = sha256(SOURCE)
    if source_hash != EXPECTED_SHA:
        raise ValueError("Frozen US500 BID M5 source hash mismatch")
    bars = pd.read_csv(SOURCE, usecols=["time", "open", "high", "low", "close"])
    if bars.time.duplicated().any() or not bars.time.is_monotonic_increasing:
        raise ValueError("Duplicate or unsorted UTC M5 bar")
    ny = pd.to_datetime(bars.time, unit="s", utc=True).dt.tz_convert("America/New_York")
    minutes = ny.dt.hour * 60 + ny.dt.minute
    selected = minutes.isin([*WINDOW, 600, 955])
    bars = bars.loc[selected].copy()
    bars["ny_date"] = ny.loc[selected].dt.date
    bars["minute"] = minutes.loc[selected]
    if bars.duplicated(["ny_date", "minute"]).any():
        raise ValueError("Duplicate NY cash-clock M5 bar")
    values = bars[["open", "high", "low", "close"]].to_numpy(dtype=float)
    valid = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    valid &= (values[:, 2] <= np.minimum(values[:, 0], values[:, 3]))
    valid &= (values[:, 1] >= np.maximum(values[:, 0], values[:, 3]))
    bars["valid"] = valid
    rows = []
    for date, group in bars.groupby("ny_date", sort=True):
        by_minute = group.set_index("minute")
        if 570 not in by_minute.index or 955 not in by_minute.index:
            continue
        close_ok = bool(by_minute.loc[570, "valid"] and by_minute.loc[955, "valid"])
        path_ok = all(m in by_minute.index and bool(by_minute.loc[m, "valid"]) for m in [*WINDOW, 600])
        rec = {"ny_date": date, "cash_valid": close_ok,
               "path_valid": path_ok, "cash_open": float(by_minute.loc[570, "open"]),
               "cash_close": float(by_minute.loc[955, "close"])}
        if path_ok:
            path = by_minute.loc[list(WINDOW)]
            rec["range_firsthalf"] = float(path.high.max() - path.low.min())
            rec["exit_1000_bid"] = float(by_minute.loc[600, "open"])
            for m in WINDOW:
                for field in ("open", "high", "low"):
                    rec[f"m{m}_{field}"] = float(by_minute.loc[m, field])
        rows.append(rec)
    days = pd.DataFrame(rows).sort_values("ny_date").reset_index(drop=True)
    days = days.loc[days.cash_valid].reset_index(drop=True)
    if days.ny_date.duplicated().any():
        raise ValueError("Duplicate valid cash date")
    days["prev_cash_close"] = days.cash_close.shift(1)
    days["gap"] = days.cash_open - days.prev_cash_close
    days["prior_median_abs_gap"] = days.gap.abs().rolling(20, min_periods=20).median().shift(1)
    days["prior_median_firsthalf_range"] = days.range_firsthalf.rolling(20, min_periods=20).median().shift(1)
    info = {"sha256": source_hash, "selected_m5_rows": len(bars),
            "valid_cash_dates": len(days), "first_cash_date": str(days.ny_date.iloc[0]),
            "last_cash_date": str(days.ny_date.iloc[-1]),
            "valid_firsthalf_dates": int(days.path_valid.sum())}
    return days, info


def run_signals(days: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    signals = []
    trades = []
    for day in days.itertuples(index=False):
        date = day.ny_date
        record = {"ny_date": date.isoformat(), "year": date.year,
                  "cash_open_bid": day.cash_open, "prev_cash_close_bid": day.prev_cash_close,
                  "gap_points": day.gap, "prior_median_abs_gap": day.prior_median_abs_gap,
                  "prior_median_firsthalf_range": day.prior_median_firsthalf_range}
        if pd.isna(day.prior_median_abs_gap) or pd.isna(day.prior_median_firsthalf_range):
            record["skip_reason"] = "MISSING_20_PRIOR_COMPLETE_DAYS"
        elif not day.path_valid:
            record["skip_reason"] = "MISSING_OR_INVALID_CURRENT_REAL_M5_PATH"
        elif day.prior_median_firsthalf_range <= 0:
            record["skip_reason"] = "NONPOSITIVE_PRIOR_RANGE"
        elif not abs(day.gap) > day.prior_median_abs_gap:
            record["skip_reason"] = "GAP_NOT_ABOVE_PRIOR_MEDIAN"
        else:
            direction = -1 if day.gap > 0 else 1
            entry_bid = float(day.cash_open)
            distance = float(day.prior_median_firsthalf_range)
            stop_bid = entry_bid - distance if direction == 1 else entry_bid + distance - 0.51
            exit_bid = float(day.exit_1000_bid)
            exit_minute = 600
            exit_reason = "TIME_1000"
            adverse_bid = entry_bid
            for minute in WINDOW:
                lo = float(getattr(day, f"m{minute}_low"))
                hi = float(getattr(day, f"m{minute}_high"))
                bar_open = float(getattr(day, f"m{minute}_open"))
                if direction == 1:
                    adverse_bid = min(adverse_bid, lo)
                    hit = lo <= stop_bid
                    if hit:
                        exit_bid = min(stop_bid, bar_open)
                else:
                    adverse_bid = max(adverse_bid, hi)
                    hit = hi >= stop_bid
                    if hit:
                        exit_bid = max(stop_bid, bar_open)
                if hit:
                    exit_minute = minute
                    exit_reason = "INITIAL_STOP"
                    break
            gross = direction * (exit_bid - entry_bid)
            base_net = gross - 0.51
            worst_net = direction * (adverse_bid - entry_bid) - 0.51
            record.update(direction=direction, stop_distance_points=distance,
                          stop_bid=stop_bid, entry_bid=entry_bid, exit_bid=exit_bid,
                          exit_wall_minute=exit_minute, exit_reason=exit_reason,
                          worst_bid_during_hold=adverse_bid,
                          worst_floating_net_points=worst_net,
                          gross_points=gross, base_net_points=base_net,
                          stress_net_points=gross - 1.02, skip_reason="")
            trades.append(record.copy())
        signals.append(record)
    return pd.DataFrame(signals), pd.DataFrame(trades)


def stats(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"trades": 0, "base_total_points": 0.0}
    return {"trades": int(len(frame)), "dates": int(frame.ny_date.nunique()),
            "gross_mean_points": float(frame.gross_points.mean()),
            "base_mean_points": float(frame.base_net_points.mean()),
            "base_total_points": float(frame.base_net_points.sum()),
            "stress_mean_points": float(frame.stress_net_points.mean()),
            "base_win_rate": float((frame.base_net_points > 0).mean()),
            "stops": int(frame.exit_reason.eq("INITIAL_STOP").sum()),
            "longs": int(frame.direction.eq(1).sum()),
            "shorts": int(frame.direction.eq(-1).sum())}


def bootstrap(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"p_mean_nonpositive": 1.0, "ci95_points": None}
    values = frame.base_net_points.to_numpy(dtype=float)
    rng = np.random.default_rng(20261016)
    means = np.array([values[rng.integers(len(values), size=len(values))].mean()
                      for _ in range(10000)])
    return {"p_mean_nonpositive": float((means <= 0).mean()),
            "ci95_points": np.quantile(means, [0.025, 0.975]).tolist()}


def challenge_pace(trades: pd.DataFrame, last_source_date) -> dict:
    if trades.empty:
        return {"starts": 0, "pass": 0, "breach": 0, "timeout": 0,
                "pass_rate": None, "breach_rate": None}
    rows = trades.sort_values("ny_date").to_dict("records")
    starts = []
    for start in rows:
        start_date = pd.Timestamp(start["ny_date"]).date()
        end_exclusive = start_date + timedelta(days=180)
        if end_exclusive > last_source_date + timedelta(days=1):
            continue
        balance = 10000.0
        trade_dates = set()
        outcome = "timeout"
        pass_days = None
        for row in rows:
            day = pd.Timestamp(row["ny_date"]).date()
            if day < start_date or day >= end_exclusive:
                continue
            prior_midnight_balance = balance
            units = 100.0 / row["stop_distance_points"]
            low_equity = balance + units * row["worst_floating_net_points"]
            floor = max(9000.0, prior_midnight_balance - 500.0)
            if low_equity < floor:
                outcome = "breach"
                break
            balance += units * row["base_net_points"]
            if balance < floor:
                outcome = "breach"
                break
            entry_utc = pd.Timestamp(day).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
            prague_date = entry_utc.tz_convert("Europe/Prague").date()
            trade_dates.add(prague_date)
            if balance >= 11000.0 and len(trade_dates) >= 4:
                outcome = "pass"
                pass_days = (day - start_date).days
                break
        starts.append({"start_date": start_date.isoformat(), "outcome": outcome,
                       "days_to_pass": pass_days, "final_balance": balance})
    counts = pd.Series([row["outcome"] for row in starts]).value_counts()
    n = len(starts)
    return {"starts": n, "pass": int(counts.get("pass", 0)),
            "breach": int(counts.get("breach", 0)),
            "timeout": int(counts.get("timeout", 0)),
            "pass_rate": float(counts.get("pass", 0) / n) if n else None,
            "breach_rate": float(counts.get("breach", 0) / n) if n else None,
            "median_days_to_pass": float(np.median([r["days_to_pass"] for r in starts
                                                   if r["days_to_pass"] is not None]))
            if counts.get("pass", 0) else None}


def main() -> None:
    report = OUT / "REPORT_US500_OVERNIGHT_FIRSTHALF_REVERSAL.txt"
    if report.exists():
        raise FileExistsError(f"Frozen report exists: {report}")
    spec = OUT / "SPEC_US500_OVERNIGHT_FIRSTHALF_REVERSAL.txt"
    gate = OUT / "GATE_US500_OVERNIGHT_FIRSTHALF_REVERSAL.txt"
    extracted = [line for line in spec.read_text(encoding="utf-8").splitlines()
                 if line.startswith("G-OG")]
    if extracted != gate.read_text(encoding="utf-8").splitlines():
        raise ValueError("SPEC and GATE differ")
    days, source = read_cash_days()
    signals, trades = run_signals(days)
    gated = trades.loc[trades.ny_date.between("2024-01-01", "2026-04-30")].copy()
    calibration = trades.loc[trades.ny_date.lt("2024-01-01")].copy()
    boot = bootstrap(gated)
    annual = {str(year): stats(part) for year, part in trades.groupby("year")}
    pace = challenge_pace(gated, pd.Timestamp(source["last_cash_date"]).date())
    gates = {
        "G-OG1": bool(len(gated) >= 150 and gated.ny_date.nunique() == len(gated)),
        "G-OG2": bool(len(gated) and gated.stress_net_points.mean() > 0),
        "G-OG3": bool(boot["p_mean_nonpositive"] <= 0.01),
        "G-OG4": bool(all(y in annual and annual[y]["base_total_points"] > 0
                          for y in ("2024", "2025"))),
        "G-OG5": bool(len(trades) and (trades.stop_distance_points > 0).all()
                          and trades.exit_wall_minute.le(600).all()),
        "G-OG6": bool(pace["starts"] and pace["pass_rate"] >= 0.70
                          and pace["breach_rate"] <= 0.10),
    }
    result = {"study": "H81 US500 overnight-firsthalf reversal",
              "source": source, "spec_sha256": sha256(spec), "gate_sha256": sha256(gate),
              "signal_dates": len(signals), "executed": len(trades),
              "skip_reasons": signals.loc[signals.skip_reason.ne(""), "skip_reason"].value_counts().to_dict(),
              "calibration_2021_2023": stats(calibration), "gated_2024_2026": stats(gated),
              "bootstrap": boot, "annual": annual, "challenge_180d_1pct": pace,
              "gates": gates, "verdict": "PASS" if all(gates.values()) else "FAIL",
              "limitations": "Fixed-spread BID M5 proxy, not actual Bid/Ask fills or FTMO feed; all historical US500 years previously exposed to project research."}
    signals.to_csv(OUT / "signals.csv", index=False)
    trades.to_csv(OUT / "trades.csv", index=False)
    (OUT / "run_meta.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report.write_text("REPORT: H81 US500 OVERNIGHT GAP FIRST-HALF-HOUR REVERSAL\n"
                      + json.dumps(result, indent=2, allow_nan=False)
                      + "\nPHASE-1 VERDICT: " + result["verdict"] + "\n", encoding="utf-8")
    print("VERDICT", result["verdict"])
    print("GATES", gates)
    print("GATED", result["gated_2024_2026"], boot)
    print("PACE", pace)
    print("YEAR NET", {year: row["base_total_points"] for year, row in annual.items()})


if __name__ == "__main__":
    main()
