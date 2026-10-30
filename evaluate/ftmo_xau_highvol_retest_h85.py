"""H85: predeclared FTMO gold London high-volume bar retest, Phase 1 only."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/ftmo_xau_highvol_retest_h85"
SPEC = OUT / "SPEC_FTMO_XAU_HIGHVOL_RETEST_H85.txt"
GATE = OUT / "GATE_FTMO_XAU_HIGHVOL_RETEST_H85.txt"
REPORT = OUT / "REPORT_FTMO_XAU_HIGHVOL_RETEST_H85.txt"
SOURCE = ROOT / "data/results/ftmo_xauusd_london_sweep_h83/ftmo_xauusd_m15.csv"
SOURCE_HASH = "59add37f91b792ee0944780774ece65c797e799f46811b52b9772c65275e2e01"
LONDON = ZoneInfo("Europe/London")
PRAGUE = ZoneInfo("Europe/Prague")
SPREAD = 0.41
RISKS = (0.005, 0.01, 0.015, 0.02)
SEED = 20260927


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_frozen() -> None:
    if REPORT.exists():
        raise FileExistsError("H85 REPORT already exists; never overwrite it")
    gates = [line for line in SPEC.read_text(encoding="utf-8").splitlines()
             if line.startswith("G-HV")]
    assert len(gates) == 6
    assert gates == GATE.read_text(encoding="utf-8").splitlines()
    assert digest(SOURCE) == SOURCE_HASH


def load_bars() -> pd.DataFrame:
    bars = pd.read_csv(SOURCE)
    required = {"utc_epoch", "server_epoch", "open", "high", "low", "close", "tick_volume"}
    assert required.issubset(bars.columns)
    t = bars.utc_epoch.to_numpy(np.int64)
    assert len(t) >= 60000 and np.all(np.diff(t) > 0) and np.all(t % 900 == 0)
    assert np.isfinite(bars[["open", "high", "low", "close"]].to_numpy()).all()
    assert bool((bars.low > 0).all() and (bars.tick_volume > 0).all())
    assert bool((bars.low <= bars[["open", "close"]].min(axis=1)).all())
    assert bool((bars.high >= bars[["open", "close"]].max(axis=1)).all())
    local = pd.to_datetime(bars.utc_epoch, unit="s", utc=True).dt.tz_convert(LONDON)
    bars["london_date"] = local.dt.date
    bars["london_minute"] = local.dt.hour * 60 + local.dt.minute
    return bars


def signal_for_day(day: pd.DataFrame) -> tuple[dict | None, str]:
    opening = day[day.london_minute.between(480, 585)].sort_values("london_minute")
    if len(opening) != 8 or opening.london_minute.tolist() != list(range(480, 600, 15)):
        return None, "MISSING_OPENING_BAR"
    width = float(opening.high.max() - opening.low.min())
    if width <= 0.82:
        return None, "NARROW_OPENING_RANGE"
    if not float(opening.close.iloc[-1]) > float(opening.open.iloc[0]):
        return None, "NO_OPENING_ADVANCE"
    busiest = opening.loc[opening.tick_volume.idxmax()]
    reference = float((busiest.high + busiest.low) / 2)
    if not float(opening.close.iloc[-1]) > reference:
        return None, "OPEN_CLOSE_NOT_ABOVE_REFERENCE"
    scan = day[day.london_minute.between(600, 825)].sort_values("utc_epoch")
    for row in scan.itertuples(index=False):
        if row.low <= reference < row.close:
            return {"signal_epoch": int(row.utc_epoch), "reference": reference,
                    "width": width, "opening_high": float(opening.high.max()),
                    "opening_low": float(opening.low.min()),
                    "peak_volume": int(busiest.tick_volume),
                    "peak_bar_epoch": int(busiest.utc_epoch)}, "SIGNAL"
    return None, "NO_RETEST_REJECTION"


def execute_one(day: pd.DataFrame, signal: dict) -> tuple[dict | None, str]:
    entry_epoch = signal["signal_epoch"] + 900
    by_epoch = day.set_index("utc_epoch", drop=False)
    if entry_epoch not in by_epoch.index:
        return None, "MISSING_ENTRY_BAR"
    entry = by_epoch.loc[entry_epoch]
    if int(entry.london_minute) > 840:
        return None, "LATE_ENTRY"
    entry_bid = float(entry.open)
    entry_ask = entry_bid + SPREAD
    reference = signal["reference"]
    stop = reference - 0.5 * signal["width"]
    risk = entry_ask - stop
    target = entry_ask + 2 * risk
    if not (entry_bid > reference and entry_bid > stop and risk > 0 and target > entry_ask):
        return None, "INVALID_ENTRY_GEOMETRY"

    path = day[(day.utc_epoch >= entry_epoch) & (day.london_minute <= 945)]
    final_exists = bool((day.london_minute == 945).any())
    used = []
    exit_fill = None
    exit_reason = None
    exit_row = None
    prior_epoch = None
    for row in path.itertuples(index=False):
        if prior_epoch is not None and row.utc_epoch != prior_epoch + 900:
            exit_reason = "UNRESOLVED_GAP"
            break
        prior_epoch = row.utc_epoch
        used.append(row)
        if row.low <= stop:
            exit_fill, exit_reason, exit_row = min(stop, float(row.open)), "STOP", row
            break
        if row.high >= target:
            exit_fill, exit_reason, exit_row = target, "TARGET", row
            break
    if not used:
        return None, "MISSING_ENTRY_BAR"
    if exit_reason is None:
        exit_row = used[-1]
        exit_fill = float(exit_row.close)
        exit_reason = "DAY_FLAT" if final_exists and exit_row.london_minute == 945 else "UNRESOLVED_FLAT"
    if exit_reason == "UNRESOLVED_GAP":
        exit_row = used[-1]
        exit_fill = float(exit_row.close)
    entry_dt = datetime.fromtimestamp(entry_epoch, timezone.utc)
    exit_dt = datetime.fromtimestamp(int(exit_row.utc_epoch) + 900 - 0.001, timezone.utc)
    net = exit_fill - entry_ask
    worst = min(float(row.low) for row in used) - entry_ask
    result = {
        "london_date": entry_dt.astimezone(LONDON).date(),
        "year": entry_dt.astimezone(LONDON).year,
        "entry_prague_date": entry_dt.astimezone(PRAGUE).date(),
        "exit_prague_date": exit_dt.astimezone(PRAGUE).date(),
        "signal_utc": datetime.fromtimestamp(signal["signal_epoch"], timezone.utc).isoformat(),
        "entry_utc": entry_dt.isoformat(), "exit_bar_end_utc": exit_dt.isoformat(),
        "reference": reference, "width": signal["width"],
        "opening_high": signal["opening_high"], "opening_low": signal["opening_low"],
        "peak_volume": signal["peak_volume"], "peak_bar_epoch": signal["peak_bar_epoch"],
        "entry_bid": entry_bid, "entry_ask": entry_ask, "stop": stop, "target": target,
        "exit_fill": exit_fill, "exit_reason": exit_reason,
        "unresolved": exit_reason.startswith("UNRESOLVED"),
        "risk_units": risk, "gross_bid_units": exit_fill - entry_bid,
        "net_units": net, "stress_units": net - SPREAD,
        "realized_r": (net - SPREAD) / risk,
        "worst_float_r": (worst - SPREAD) / risk,
        "hold_hours_upper": (int(exit_row.utc_epoch) + 900 - entry_epoch) / 3600,
        "path_bars": len(used),
    }
    return result, exit_reason


def run_days(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    trades, days = [], []
    funnel = Counter()
    for day_date, group in bars.groupby("london_date", sort=True):
        if day_date.weekday() >= 5 or day_date < date(2024, 1, 1) or day_date > date(2026, 9, 24):
            continue
        signal, status = signal_for_day(group)
        if signal is not None:
            trade, status = execute_one(group, signal)
            if trade is not None:
                trades.append(trade)
        else:
            trade = None
        funnel[status] += 1
        days.append({"london_date": day_date, "status": status,
                     "signaled": signal is not None, "trade": trade is not None})
    return pd.DataFrame(trades), pd.DataFrame(days), dict(funnel)


def bootstrap(trades: pd.DataFrame) -> dict:
    groups = trades.groupby("london_date", sort=True).net_units.agg(["sum", "count"])
    sums = groups["sum"].to_numpy(float)
    counts = groups["count"].to_numpy(int)
    if not len(sums):
        return {"date_blocks": 0, "p_mean_le_zero": 1.0, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(SEED)
    means = np.empty(10_000)
    for left in range(0, 10_000, 200):
        right = min(left + 200, 10_000)
        picks = rng.integers(0, len(sums), size=(right - left, len(sums)))
        means[left:right] = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return {"date_blocks": len(sums), "p_mean_le_zero": float(np.mean(means <= 0)),
            "ci95_low": float(np.quantile(means, 0.025)),
            "ci95_high": float(np.quantile(means, 0.975))}


def replay_window(trades: pd.DataFrame, start: date, days: int, target: float, risk: float) -> dict:
    end = start + timedelta(days=days)
    balance = 10_000.0
    day_open = balance
    last_day = None
    trade_days = set()
    for trade in trades.itertuples(index=False):
        if trade.london_date < start or trade.london_date >= end:
            continue
        if trade.entry_prague_date != last_day:
            day_open = balance
            last_day = trade.entry_prague_date
        trade_days.add(trade.entry_prague_date)
        floor = max(9000.0, day_open - 500.0)
        floating = balance + 10_000.0 * risk * trade.worst_float_r
        if floating < floor:
            return {"outcome": "BREACH", "days": (trade.london_date - start).days}
        balance += 10_000.0 * risk * trade.realized_r
        if balance < floor:
            return {"outcome": "BREACH", "days": (trade.london_date - start).days}
        if balance >= target and len(trade_days) >= 4:
            return {"outcome": "PASS", "days": (trade.london_date - start).days}
    return {"outcome": "TIMEOUT", "days": days}


def challenge(trades: pd.DataFrame, last_date: date, days: int, target: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    starts = sorted({d for d in trades.london_date
                     if d + timedelta(days=days) <= last_date + timedelta(days=1)})
    outcomes = [{"risk_pct": risk * 100, "start_date": start.isoformat(),
                 **replay_window(trades, start, days, target, risk)}
                for risk in RISKS for start in starts]
    frame = pd.DataFrame(outcomes)
    summary = []
    for pct, group in frame.groupby("risk_pct", sort=True):
        passed = group.loc[group.outcome.eq("PASS"), "days"]
        summary.append({"risk_pct": pct, "starts": len(group), "pass": len(passed),
                        "breach": int(group.outcome.eq("BREACH").sum()),
                        "timeout": int(group.outcome.eq("TIMEOUT").sum()),
                        "pass_rate": float(group.outcome.eq("PASS").mean()),
                        "breach_rate": float(group.outcome.eq("BREACH").mean()),
                        "median_pass_days": float(passed.median()) if len(passed) else None})
    return frame, pd.DataFrame(summary)


def main() -> None:
    check_frozen()
    bars = load_bars()
    trades, roster, funnel = run_days(bars)
    if trades.empty:
        raise ValueError("Zero trades; must still produce an H85 REPORT")
    test = trades[trades.london_date >= date(2025, 1, 1)].copy()
    complete = test[~test.unresolved].copy()
    inference = bootstrap(complete)
    last_date = bars.london_date.iloc[-1]
    paid_outcomes, paid_pace = challenge(complete, last_date, 90, 11000.0)
    trial_outcomes, trial_pace = challenge(complete, last_date, 14, 10500.0)
    yearly = {str(y): {"n": len(group), "base_total_units": float(group.net_units.sum()),
                       "stress_total_units": float(group.stress_units.sum())}
              for y, group in trades.groupby("year", sort=True)}
    signals = int(roster.loc[roster.london_date >= date(2025, 1, 1), "signaled"].sum())
    missing_entry = int(((roster.london_date >= date(2025, 1, 1))
                         & roster.status.eq("MISSING_ENTRY_BAR")).sum())
    unresolved = int(test.unresolved.sum())
    same_day = bool((test.entry_prague_date == test.exit_prague_date).all()
                    and (test.hold_hours_upper < 16).all()
                    and (test.risk_units > 0).all()
                    and (test.target > test.entry_ask).all())
    qualifying_pace = paid_pace[(paid_pace.starts >= 150) & (paid_pace.pass_rate >= 0.70)
                                & (paid_pace.breach_rate <= 0.10)]
    gates = {
        "G-HV1": bool(signals >= 150 and len(test) >= 150
                      and test.london_date.nunique() >= 150 and missing_entry == 0 and unresolved == 0),
        "G-HV2": bool(unresolved == 0 and complete.stress_units.mean() > 0),
        "G-HV3": bool(unresolved == 0 and inference["p_mean_le_zero"] <= 0.05),
        "G-HV4": bool(unresolved == 0 and yearly.get("2025", {}).get("base_total_units", 0) > 0
                      and yearly.get("2026", {}).get("base_total_units", 0) > 0),
        "G-HV5": bool(unresolved == 0 and len(qualifying_pace) > 0),
        "G-HV6": bool(same_day and digest(SOURCE) == SOURCE_HASH and len(bars) >= 60000),
    }
    trades.to_csv(OUT / "trades.csv", index=False, float_format="%.8f")
    roster.to_csv(OUT / "day_roster.csv", index=False)
    paid_outcomes.to_csv(OUT / "paid_90d_outcomes.csv", index=False)
    paid_pace.to_csv(OUT / "paid_90d_pace.csv", index=False)
    trial_outcomes.to_csv(OUT / "free_trial_14d_outcomes.csv", index=False)
    trial_pace.to_csv(OUT / "free_trial_14d_pace.csv", index=False)
    net = complete.net_units.to_numpy(float)
    equity = np.r_[0.0, np.cumsum(net)]
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": digest(SPEC), "gate_sha256": digest(GATE),
        "source_sha256": digest(SOURCE), "source_rows": len(bars),
        "source_first_utc": datetime.fromtimestamp(int(bars.utc_epoch.iloc[0]), timezone.utc).isoformat(),
        "source_last_utc": datetime.fromtimestamp(int(bars.utc_epoch.iloc[-1]), timezone.utc).isoformat(),
        "funnel": funnel, "years": yearly,
        "gated": {"signals": signals, "trades": len(test), "complete": len(complete),
                  "dates": int(test.london_date.nunique()), "missing_entry": missing_entry,
                  "unresolved": unresolved, "base_mean_units": float(complete.net_units.mean()),
                  "base_total_units": float(complete.net_units.sum()),
                  "stress_mean_units": float(complete.stress_units.mean()),
                  "mean_stressed_r": float(complete.realized_r.mean()),
                  "win_rate": float(np.mean(net > 0)),
                  "max_drawdown_units": float(np.max(np.maximum.accumulate(equity) - equity)),
                  "median_hold_hours_upper": float(complete.hold_hours_upper.median()),
                  "exit_reasons": test.exit_reason.value_counts().to_dict()},
        "bootstrap": inference,
        "paid_90d_pace": paid_pace.astype(object).where(pd.notna(paid_pace), None).to_dict("records"),
        "free_trial_14d_pace": trial_pace.astype(object).where(pd.notna(trial_pace), None).to_dict("records"),
        "pace_provisional_if_unresolved": bool(unresolved),
        "gates": gates, "verdict": "PASS" if all(gates.values()) else "FAIL",
        "limitations": "Historical FTMO native M15 BID bar proxy with fixed 0.41 Ask spread plus 0.41 stress; no true tick-level Ask, commission, slippage, EUR conversion, lot rounding, or independent post-discovery sample. Not a deployable EA.",
    }
    (OUT / "run_meta.json").write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    REPORT.write_text("H85 FTMO GOLD HIGH-VOLUME RETEST PHASE-1 REPORT\n"
                      + json.dumps(payload, indent=2, allow_nan=False, default=str)
                      + "\nOVERALL VERDICT: " + payload["verdict"] + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("verdict", "gated", "years", "bootstrap", "paid_90d_pace", "free_trial_14d_pace", "gates")},
                     indent=2, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
