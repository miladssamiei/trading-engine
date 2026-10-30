"""H84: fixed prior-cash-day asymmetry rule, US500 BID M5 historical screen."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/us500_prevday_asymmetry_v1"
SPEC = OUT / "SPEC_US500_PREVDAY_ASYMMETRY.txt"
GATE = OUT / "GATE_US500_PREVDAY_ASYMMETRY.txt"
ADDENDUM = OUT / "ADDENDUM_DATE_SCOPE_BEFORE_CODE.txt"
SOURCE = ROOT / "data/results/orb_2021_present/us500tick_m5.csv"
SOURCE_SHA = "a8ee68f1a0010a1be9704c0d59fd5382aa798846cdae4d519b799ae21ec66c2d"
ENTRY_MIN = 575
FINAL_MIN = 775
SPREAD = 0.51
SEED = 20260925


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cash():
    if sha256(SOURCE) != SOURCE_SHA:
        raise ValueError("Frozen BID M5 source changed")
    bars = pd.read_csv(SOURCE, dtype={"time": "int64"})
    if bars.time.duplicated().any() or not bars.time.is_monotonic_increasing:
        raise ValueError("Source M5 timestamps not unique increasing")
    vals = bars[["open", "high", "low", "close"]].to_numpy(float)
    if not (np.isfinite(vals).all() and (vals > 0).all()
            and (vals[:, 1] >= np.maximum(vals[:, 0], vals[:, 3])).all()
            and (vals[:, 2] <= np.minimum(vals[:, 0], vals[:, 3])).all()):
        raise ValueError("Invalid source OHLC")
    ny = pd.to_datetime(bars.time, unit="s", utc=True).dt.tz_convert("America/New_York")
    wall = ny.dt.hour * 60 + ny.dt.minute
    keep = (wall >= 570) & (wall <= 955)
    bars = bars.loc[keep].copy()
    bars["wall"] = wall.loc[keep].to_numpy(int)
    bars["ny_date"] = ny.loc[keep].dt.date.to_numpy()
    if bars.duplicated(["ny_date", "wall"]).any():
        raise ValueError("Duplicate NY cash bar")
    sessions = {}
    for day, group in bars.groupby("ny_date", sort=True):
        sessions[day] = {int(row.wall): row for row in group.itertuples(index=False)}
    cash_dates = sorted(day for day, group in sessions.items() if 570 in group)
    return sessions, cash_dates, {"sha256": SOURCE_SHA, "m5_rows": len(bars),
                                  "cash_dates": len(cash_dates),
                                  "first_cash_date": str(cash_dates[0]),
                                  "last_cash_date": str(cash_dates[-1])}


def previous_signal(group):
    if 570 not in group or 955 not in group or sum(570 <= m <= 955 for m in group) < 60:
        return None
    cash = [group[m] for m in sorted(group) if 570 <= m <= 955]
    op = float(group[570].open)
    high = max(float(bar.high) for bar in cash)
    low = min(float(bar.low) for bar in cash)
    if high <= low:
        return None
    asym = (high + low - 2 * op) / (high - low)
    return {"prev_open": op, "prev_high": high, "prev_low": low,
            "prev_range": high - low, "asymmetry": asym}


def one_trade(today: date, prev: date, group, info):
    entry = group.get(ENTRY_MIN)
    if entry is None:
        return None, "NO_0935_ENTRY_BAR"
    asym = info["asymmetry"]
    if asym == 0:
        return None, "ZERO_ASYMMETRY"
    side = 1 if asym < 0 else -1
    entry_bid = float(entry.open)
    distance = 0.5 * info["prev_range"]
    stop_bid = entry_bid - distance if side == 1 else entry_bid + distance
    worst_bid = entry_bid
    exit_bid = None
    reason = "UNRESOLVED_MISSING_PATH"
    exit_wall = None
    path_bars = 0
    for minute in range(ENTRY_MIN, FINAL_MIN + 1, 5):
        bar = group.get(minute)
        if bar is None:
            exit_wall = minute
            break
        path_bars += 1
        worst_bid = min(worst_bid, float(bar.low)) if side == 1 else max(worst_bid, float(bar.high))
        hit = float(bar.low) <= stop_bid if side == 1 else float(bar.high) >= stop_bid
        if hit:
            exit_bid = (min(stop_bid, float(bar.open)) if side == 1
                        else max(stop_bid, float(bar.open)))
            reason = "STOP"
            exit_wall = minute
            break
        if minute == FINAL_MIN:
            exit_bid = float(bar.close)
            reason = "TIME_1300"
            exit_wall = minute
    gross = side * (exit_bid - entry_bid) if exit_bid is not None else None
    net = gross - SPREAD if gross is not None else None
    worst_net = side * (worst_bid - entry_bid) - SPREAD
    row = {"ny_date": today.isoformat(), "year": today.year,
           "previous_cash_date": prev.isoformat(), **info,
           "side": "BUY" if side == 1 else "SELL",
           "entry_time": int(entry.time), "entry_wall": ENTRY_MIN,
           "entry_bid": entry_bid, "stop_bid": stop_bid,
           "stop_distance": distance, "exit_wall": exit_wall,
           "exit_bid": exit_bid, "exit_reason": reason, "path_bars": path_bars,
           "worst_bid": worst_bid, "gross_points": gross,
           "base_net_points": net,
           "stress_net_points": net - SPREAD if net is not None else None,
           "worst_floating_stress_points": worst_net - SPREAD,
           "realized_stress_r": (net - SPREAD) / distance if net is not None else None,
           "worst_floating_stress_r": (worst_net - SPREAD) / distance}
    return row, ""


def run(sessions, dates):
    signals, trades = [], []
    for j in range(1, len(dates)):
        today, prev = dates[j], dates[j - 1]
        prev_info = previous_signal(sessions[prev])
        record = {"ny_date": today.isoformat(), "previous_cash_date": prev.isoformat()}
        if prev_info is None:
            record["skip_reason"] = "PREVIOUS_CASH_DAY_INCOMPLETE"
        else:
            trade, skip = one_trade(today, prev, sessions[today], prev_info)
            record.update(prev_info)
            record["skip_reason"] = skip
            if trade is not None:
                record.update({"side": trade["side"], "entry_bid": trade["entry_bid"],
                               "exit_reason": trade["exit_reason"]})
                trades.append(trade)
        signals.append(record)
    return pd.DataFrame(signals), pd.DataFrame(trades)


def stats(frame):
    valid = frame[frame.exit_reason != "UNRESOLVED_MISSING_PATH"]
    if valid.empty:
        return {"trades": 0, "unresolved": int(len(frame)), "base_total_points": 0.0,
                "base_mean_points": None, "stress_mean_points": None}
    return {"trades": int(len(frame)), "resolved": int(len(valid)),
            "unresolved": int(len(frame) - len(valid)),
            "dates": int(frame.ny_date.nunique()),
            "gross_mean_points": float(valid.gross_points.mean()),
            "base_mean_points": float(valid.base_net_points.mean()),
            "base_total_points": float(valid.base_net_points.sum()),
            "stress_mean_points": float(valid.stress_net_points.mean()),
            "stress_total_points": float(valid.stress_net_points.sum()),
            "base_win_rate": float((valid.base_net_points > 0).mean()),
            "stops": int((valid.exit_reason == "STOP").sum()),
            "sides": valid.side.value_counts().to_dict(),
            "p10_p50_p90_base": np.quantile(valid.base_net_points, [.1, .5, .9]).tolist(),
            "worst_floating_stress_r_min": float(valid.worst_floating_stress_r.min())}


def bootstrap(frame):
    valid = frame[frame.exit_reason != "UNRESOLVED_MISSING_PATH"]
    if valid.empty:
        return {"p_mean_nonpositive": 1.0, "ci95_points": None}
    values = valid.base_net_points.to_numpy(float)
    rng = np.random.default_rng(SEED)
    means = np.array([values[rng.integers(len(values), size=len(values))].mean()
                      for _ in range(10000)])
    return {"p_mean_nonpositive": float(np.mean(means <= 0)),
            "ci95_points": np.quantile(means, [.025, .975]).tolist()}


def pace(frame, last_cash_date, horizon, target, min_days):
    valid = frame[frame.exit_reason != "UNRESOLVED_MISSING_PATH"].sort_values("ny_date")
    rows = valid.to_dict("records")
    out = []
    for i, start in enumerate(rows):
        day0 = date.fromisoformat(start["ny_date"])
        end = day0 + timedelta(days=horizon)
        if end > last_cash_date + timedelta(days=1):
            continue
        balance = 10000.0
        outcome, pass_days = "TIMEOUT", None
        trade_dates = set()
        for row in rows[i:]:
            day = date.fromisoformat(row["ny_date"])
            if day >= end:
                break
            floor = max(9000.0, balance - 500.0)
            units = 100.0 / row["stop_distance"]
            low_equity = balance + units * row["worst_floating_stress_points"]
            if low_equity < floor:
                outcome = "BREACH"
                break
            balance += units * row["stress_net_points"]
            if balance < floor:
                outcome = "BREACH"
                break
            trade_dates.add(day)  # NY 09:35 equals same calendar date in Prague.
            if balance >= target and len(trade_dates) >= min_days:
                outcome = "PASS"
                pass_days = (day - day0).days
                break
        out.append({"start_date": day0.isoformat(), "outcome": outcome,
                    "pass_days": pass_days, "ending_balance": balance})
    counts = Counter(row["outcome"] for row in out)
    n = len(out)
    summary = {"starts": n, **{key.lower(): counts[key] for key in ("PASS", "BREACH", "TIMEOUT")},
               "pass_rate": counts["PASS"] / n if n else None,
               "breach_rate": counts["BREACH"] / n if n else None,
               "median_pass_days": float(np.median([row["pass_days"] for row in out
                                                   if row["pass_days"] is not None]))
               if counts["PASS"] else None}
    return pd.DataFrame(out), summary


def main():
    report = OUT / "REPORT_US500_PREVDAY_ASYMMETRY.txt"
    if report.exists():
        raise FileExistsError("Frozen H84 report already exists")
    if not SPEC.exists() or not GATE.exists() or not ADDENDUM.exists():
        raise FileNotFoundError("Frozen protocol document missing")
    gate_lines = GATE.read_text(encoding="utf-8").splitlines()
    spec_lines = [line for line in SPEC.read_text(encoding="utf-8").splitlines()
                  if line.startswith("G-PA")]
    if spec_lines != gate_lines or len(gate_lines) != 6:
        raise ValueError("GATE is not verbatim six-gate SPEC extraction")
    sessions, dates, source = load_cash()
    signals, trades = run(sessions, dates)
    gated = trades[trades.ny_date.between("2024-01-01", "2026-04-30")].copy()
    calibration = trades[trades.ny_date.lt("2024-01-01")].copy()
    boot = bootstrap(gated)
    yearly = {str(year): stats(part) for year, part in trades.groupby("year")}
    last = date.fromisoformat(source["last_cash_date"])
    paid_rows, paid = pace(gated, last, 90, 11000.0, 4)
    trial_rows, trial = pace(gated, last, 14, 10500.0, 2)
    gated_resolved = gated[gated.exit_reason != "UNRESOLVED_MISSING_PATH"]
    first_date = gated_resolved.ny_date.nunique() == len(gated_resolved)
    same_day = all(date.fromisoformat(row.ny_date).year == row.year
                   for row in gated_resolved.itertuples(index=False))
    no_holes = all(row.path_bars == (row.exit_wall - ENTRY_MIN) // 5 + 1
                   for row in gated_resolved.itertuples(index=False))
    gates = {
        "G-PA1": bool(len(gated) >= 150 and gated.ny_date.nunique() >= 150
                      and gated.exit_reason.ne("UNRESOLVED_MISSING_PATH").all()),
        "G-PA2": bool(len(gated_resolved) and gated_resolved.stress_net_points.mean() > 0),
        "G-PA3": bool(boot["p_mean_nonpositive"] <= 0.01),
        "G-PA4": bool(all(y in yearly and yearly[y]["base_total_points"] > 0
                          for y in ("2024", "2025"))),
        "G-PA5": bool(paid["starts"] >= 150 and paid["pass_rate"] >= .70
                      and paid["breach_rate"] <= .10),
        "G-PA6": bool(first_date and same_day and no_holes and len(trades)
                      and (trades.stop_distance > 0).all()
                      and gated_resolved.exit_wall.le(FINAL_MIN).all()),
    }
    result = {"study": "H84 US500 previous-day cash-range asymmetry",
              "source": source, "spec_sha256": sha256(SPEC),
              "gate_sha256": sha256(GATE), "addendum_sha256": sha256(ADDENDUM),
              "signal_dates": len(signals), "executed_or_marked": len(trades),
              "skip_reasons": signals.loc[signals.skip_reason.ne(""), "skip_reason"].value_counts().to_dict(),
              "calibration_2021_2023": stats(calibration),
              "gated_2024_2026": stats(gated), "bootstrap": boot,
              "yearly": yearly, "paid_90d_1pct": paid, "free_trial_14d_1pct": trial,
              "gates": gates, "verdict": "PASS" if all(gates.values()) else "FAIL",
              "limitations": "Historical fixed-spread BID M5 proxy on non-FTMO symbol. Source PDF unavailable; rule is a declared transfer inference. No actual FTMO Bid/Ask fill, commission, size tier, EUR conversion, or forward proof."}
    signals.to_csv(OUT / "signals.csv", index=False)
    trades.to_csv(OUT / "trades.csv", index=False)
    paid_rows.to_csv(OUT / "paid_90d_outcomes.csv", index=False)
    trial_rows.to_csv(OUT / "free_trial_14d_outcomes.csv", index=False)
    (OUT / "run_meta.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with report.open("x", encoding="utf-8") as stream:
        stream.write("REPORT: H84 US500 PREVIOUS-DAY ASYMMETRY - PHASE 1\n")
        stream.write(json.dumps(result, indent=2, allow_nan=False))
        stream.write("\nFINAL PHASE-1 VERDICT: " + result["verdict"] + "\n")
    print("VERDICT", result["verdict"])
    print("GATES", gates)
    print("GATED", result["gated_2024_2026"])
    print("BOOTSTRAP", boot)
    print("PAID", paid)
    print("TRIAL", trial)
    print("YEARLY", {y: v["base_total_points"] for y, v in yearly.items()})


if __name__ == "__main__":
    main()
