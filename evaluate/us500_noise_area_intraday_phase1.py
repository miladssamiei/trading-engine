"""H71: causal price-only noise-area intraday momentum on US500 M5."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/us500_noise_area_intraday_v1"
SPEC = OUT / "SPEC_NOISE_AREA_INTRADAY.txt"
GATE = OUT / "GATE_NOISE_AREA_INTRADAY.txt"
SOURCE = ROOT / "data/results/orb_2021_present/us500tick_m5.csv"
EXPECTED_SOURCE_SHA = "a8ee68f1a0010a1be9704c0d59fd5382aa798846cdae4d519b799ae21ec66c2d"
NY = ZoneInfo("America/New_York")
SPREAD = 0.51
SEED = 20261008
N_BOOT = 10_000
CLOCKS = tuple(range(600, 931, 30))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_sessions() -> tuple[list[int], dict[int, dict[int, tuple]], dict]:
    assert digest(SOURCE) == EXPECTED_SOURCE_SHA
    raw = pd.read_csv(SOURCE, dtype={"time": "int64"})
    ny = pd.to_datetime(raw.time, unit="s", utc=True).dt.tz_convert(NY)
    wall = ny.dt.hour * 60 + ny.dt.minute
    day = ny.dt.year * 10000 + ny.dt.month * 100 + ny.dt.day
    cash = raw[(wall >= 570) & (wall < 960)].copy()
    cash["wall"] = wall[(wall >= 570) & (wall < 960)].to_numpy()
    cash["day"] = day[(wall >= 570) & (wall < 960)].to_numpy()
    assert not cash.duplicated(["day", "wall"]).any()
    by_day: dict[int, dict[int, tuple]] = {}
    for row in cash.itertuples(index=False):
        by_day.setdefault(int(row.day), {})[int(row.wall)] = (
            int(row.time), float(row.open), float(row.close))
    valid = sorted(d for d, bars in by_day.items() if 570 in bars and 955 in bars)
    funnel = {"m5_rows": len(raw), "cash_m5_rows": len(cash),
              "cash_dates_0930": sum(570 in b for b in by_day.values()),
              "cash_dates_with_1555": len(valid),
              "days_no_1555": sum(570 in b and 955 not in b for b in by_day.values()),
              "days_no_0930": sum(570 not in b for b in by_day.values()),
              "warmup_dates": min(14, len(valid)), "clocks_checked": 0,
              "skip_current_prebar": 0, "skip_current_entrybar": 0,
              "skip_prior14_prebar": 0, "signals_long": 0,
              "signals_short": 0, "entries": 0, "reversals": 0,
              "day_flat_exits": 0}
    return valid, by_day, funnel


def make_trade(position: dict, exit_bid: float, exit_s: float, exit_type: str) -> dict:
    side = int(position["side"])
    gross = side * (exit_bid - position["entry_bid"])
    net = gross - SPREAD
    return {"ny_date": position["ny_date"], "year": position["ny_date"] // 10000,
            "side": "BUY" if side == 1 else "SELL",
            "entry_epoch_s": position["entry_s"], "exit_epoch_s": exit_s,
            "entry_bid": position["entry_bid"], "exit_bid": exit_bid,
            "initial_risk_pts": position["initial_risk_pts"],
            "gross_pts": gross, "net_pts": net, "stress_pts": net - SPREAD,
            "r": net / position["initial_risk_pts"],
            "hold_hours": (exit_s - position["entry_s"]) / 3600,
            "exit_type": exit_type}


def run_machine(dates: list[int], by_day: dict[int, dict[int, tuple]], funnel: dict) -> pd.DataFrame:
    trades = []
    for j in range(14, len(dates)):
        day = dates[j]
        bars = by_day[day]
        past = dates[j - 14:j]
        today_open = bars[570][1]
        prev_close = by_day[dates[j - 1]][955][2]
        position = None
        for clock in CLOCKS:
            funnel["clocks_checked"] += 1
            prebar = bars.get(clock - 5)
            entrybar = bars.get(clock)
            if prebar is None:
                funnel["skip_current_prebar"] += 1
                continue
            if entrybar is None:
                funnel["skip_current_entrybar"] += 1
                continue
            if any(clock - 5 not in by_day[d] for d in past):
                funnel["skip_prior14_prebar"] += 1
                continue
            sigma = sum(abs(by_day[d][clock - 5][2] / by_day[d][570][1] - 1.0)
                        for d in past) / 14.0
            upper = max(today_open, prev_close) * (1.0 + sigma)
            lower = min(today_open, prev_close) * (1.0 - sigma)
            observed = prebar[2]
            signal = 1 if observed > upper else -1 if observed < lower else 0
            if signal == 1:
                funnel["signals_long"] += 1
            elif signal == -1:
                funnel["signals_short"] += 1
            if signal == 0 or (position is not None and signal == position["side"]):
                continue
            entry_s, entry_bid = entrybar[0], entrybar[1]
            if position is not None:
                trades.append(make_trade(position, entry_bid, entry_s, "REVERSE"))
                funnel["reversals"] += 1
            risk = (entry_bid + SPREAD - lower) if signal == 1 else (upper + SPREAD - entry_bid)
            assert risk > 0 and np.isfinite(risk), (day, clock, risk)
            position = {"ny_date": day, "side": signal, "entry_s": entry_s,
                        "entry_bid": entry_bid, "initial_risk_pts": risk}
            funnel["entries"] += 1
        if position is not None:
            close_s, _, close_bid = bars[955]
            trades.append(make_trade(position, close_bid, close_s + 299.999, "DAY_FLAT"))
            funnel["day_flat_exits"] += 1
    assert funnel["entries"] == len(trades)
    return pd.DataFrame(trades)


def metrics(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {"n": 0}
    curve = rows.net_pts.cumsum().to_numpy()
    peak = np.maximum.accumulate(np.r_[0.0, curve])[1:]
    return {"n": len(rows), "dates": rows.ny_date.nunique(),
            "gross_mean": float(rows.gross_pts.mean()),
            "net_mean": float(rows.net_pts.mean()),
            "net_total": float(rows.net_pts.sum()),
            "stress_mean": float(rows.stress_pts.mean()),
            "r_mean": float(rows.r.mean()),
            "win_rate": float((rows.net_pts > 0).mean()),
            "max_drawdown_pts": float(np.max(peak - curve)),
            "median_hold_hours": float(rows.hold_hours.median()),
            "p90_hold_hours": float(rows.hold_hours.quantile(0.9)),
            "side": {name: {"n": len(group), "net_total": float(group.net_pts.sum())}
                     for name, group in rows.groupby("side")},
            "exit_type": {name: int(n) for name, n in rows.exit_type.value_counts().items()}}


def date_bootstrap(rows: pd.DataFrame) -> dict:
    groups = rows.groupby("ny_date", sort=True).net_pts.agg(["sum", "size"])
    rng = np.random.default_rng(SEED)
    sample = rng.integers(0, len(groups), size=(N_BOOT, len(groups)))
    sums = groups["sum"].to_numpy()[sample].sum(axis=1)
    counts = groups["size"].to_numpy()[sample].sum(axis=1)
    means = sums / counts
    return {"date_blocks": len(groups), "p_mean_le_zero": float(np.mean(means <= 0)),
            "ci_95_low": float(np.quantile(means, .025)),
            "ci_95_high": float(np.quantile(means, .975))}


def challenge(rows: pd.DataFrame) -> dict:
    ts = pd.to_datetime(rows.entry_epoch_s, unit="s", utc=True)
    prepared = rows.copy()
    prepared["entry_ny_date"] = ts.dt.tz_convert(NY).dt.date
    prepared["entry_prague_date"] = ts.dt.tz_convert("Europe/Prague").dt.date
    prepared = prepared.sort_values(["entry_epoch_s", "exit_epoch_s"])
    last_date = prepared.entry_ny_date.max()
    starts = sorted(set(prepared.entry_ny_date))
    eligible = [d for d in starts if d + timedelta(days=180) <= last_date + timedelta(days=1)]
    counts = {"PASS": 0, "BREACH": 0, "TIMEOUT": 0}
    pass_days = []
    for start in eligible:
        end = start + timedelta(days=180)
        window = prepared[(prepared.entry_ny_date >= start) & (prepared.entry_ny_date < end)]
        balance = 10_000.0
        day_open = balance
        prior_day = None
        distinct_days = set()
        outcome = "TIMEOUT"
        for trade in window.itertuples():
            day = trade.entry_prague_date
            if day != prior_day:
                day_open = balance
                prior_day = day
            distinct_days.add(day)
            balance += 100.0 * trade.r
            if balance < 9_000.0 or balance < day_open - 500.0:
                outcome = "BREACH"
                break
            if balance >= 11_000.0 and len(distinct_days) >= 4:
                outcome = "PASS"
                pass_days.append((day - start).days)
                break
        counts[outcome] += 1
    n = len(eligible)
    return {"starts": n, "outcomes": counts,
            "pass_rate": counts["PASS"] / n if n else 0.0,
            "breach_rate": counts["BREACH"] / n if n else 0.0,
            "median_pass_days": float(np.median(pass_days)) if pass_days else None}


def main() -> None:
    report_path = OUT / "REPORT_NOISE_AREA_INTRADAY.txt"
    assert not report_path.exists(), "Frozen report exists; never overwrite"
    spec = SPEC.read_text(encoding="utf-8")
    gate_lines = GATE.read_text(encoding="utf-8").splitlines()
    assert len(gate_lines) == 6 and all(line in spec for line in gate_lines)
    dates, by_day, funnel = load_sessions()
    trades = run_machine(dates, by_day, funnel)
    trades.to_csv(OUT / "noise_area_trades.csv", index=False, float_format="%.8f")
    cal = trades[trades.year < 2024]
    test = trades[trades.year >= 2024]
    year = {str(y): metrics(group) for y, group in trades.groupby("year", sort=True)}
    boot = date_bootstrap(test) if not test.empty else None
    entry_date = pd.to_datetime(trades.entry_epoch_s, unit="s", utc=True).dt.tz_convert(NY).dt.date
    exit_date = pd.to_datetime(trades.exit_epoch_s, unit="s", utc=True).dt.tz_convert(NY).dt.date
    exit_wall = pd.to_datetime(trades.exit_epoch_s, unit="s", utc=True).dt.tz_convert(NY)
    same_day = bool((entry_date == exit_date).all() and (exit_wall.dt.hour * 60 + exit_wall.dt.minute < 960).all())
    first_five = {"G-NA1": len(test) >= 150 and test.ny_date.nunique() >= 150,
                  "G-NA2": not test.empty and float(test.stress_pts.mean()) > 0,
                  "G-NA3": boot is not None and boot["p_mean_le_zero"] <= .05,
                  "G-NA4": all(year.get(str(y), {}).get("net_total", 0) > 0 for y in (2024, 2025)),
                  "G-NA5": same_day and bool((trades.initial_risk_pts > 0).all())}
    pace = challenge(test) if all(first_five.values()) else None
    gates = {**first_five,
             "G-NA6": pace is not None and pace["starts"] > 0 and
                      pace["pass_rate"] >= .70 and pace["breach_rate"] <= .10}
    verdict = "PASS" if all(gates.values()) else "FAIL"
    result = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": digest(SOURCE), "spec_sha256": digest(SPEC),
              "gate_sha256": digest(GATE), "funnel": funnel,
              "calibration": metrics(cal), "test": metrics(test),
              "bootstrap": boot, "yearly": year, "challenge": pace,
              "gates": gates, "verdict": verdict}
    (OUT / "run_meta.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["REPORT: HYPOTHESIS 71 US500 PRICE-ONLY NOISE-AREA INTRADAY MOMENTUM",
             f"Generated UTC: {result['generated_utc']}   PHASE-1 VERDICT: {verdict}",
             "Published SPY base rule adapted causally to US500 M5 BID bars; not the volume/VWAP variant.",
             "Fixed 0.51 spread and separate extra 0.51 cost stress; M5 fills remain optimistic.",
             f"SOURCE SHA256: {result['source_sha256']}",
             f"SPEC/GATE SHA256: {result['spec_sha256']} / {result['gate_sha256']}",
             f"FUNNEL: {json.dumps(funnel, sort_keys=True)}",
             f"CALIBRATION 2021-23: {json.dumps(result['calibration'], sort_keys=True)}",
             f"GATED 2024+: {json.dumps(result['test'], sort_keys=True)}",
             f"DATE BOOTSTRAP: {json.dumps(boot, sort_keys=True)}",
             "YEARLY:"]
    lines += [f"  {y}: {json.dumps(v, sort_keys=True)}" for y, v in year.items()]
    lines += [f"CHALLENGE 180-DAY 1% PROXY: {json.dumps(pace, sort_keys=True)}",
              f"GATES: {json.dumps(gates, sort_keys=True)}",
              f"OVERALL PHASE-1 VERDICT: {verdict}",
              "FAIL is final for this exact rule; no MT5 Model=4 tester under the frozen stop rule.",
              "PASS would permit only a new frozen native-tick execution and FTMO risk SPEC/GATE, plus later forward confirmation."]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "test": result["test"], "bootstrap": boot,
                      "challenge": pace, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
