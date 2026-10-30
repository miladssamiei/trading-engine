"""H72: fixed UTC 00-08 XAUUSD long, M1 bid proxy and FTMO pace screen."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/xauusd_asia_session_long_v1"
SOURCE = ROOT / "data/results/xauusd_onboarding/xauusdtick_m1.csv"
SPEC = OUT / "SPEC_ASIA_SESSION_LONG.txt"
GATE = OUT / "GATE_ASIA_SESSION_LONG.txt"
SOURCE_SHA = "a8c4c1d6551d27f3796f8d9b88d332172c41c76c151844e759492837152e7b1f"
SPREAD = 0.41
N_BOOT = 10_000
SEED = 20261009
PRAGUE = ZoneInfo("Europe/Prague")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def utc_date(epoch_s: int) -> date:
    return datetime.fromtimestamp(epoch_s, timezone.utc).date()


def make_trades() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    assert sha256(SOURCE) == SOURCE_SHA
    raw = pd.read_csv(SOURCE, usecols=["time", "open", "low"],
                      dtype={"time": "int64", "open": "float64", "low": "float64"})
    assert raw.time.is_monotonic_increasing and raw.time.is_unique
    raw["utc_day"] = raw.time // 86_400
    raw["minute"] = (raw.time % 86_400) // 60
    all_days = range(int(raw.utc_day.min()), int(raw.utc_day.max()) + 1)
    relevant = raw[raw.minute <= 480]
    grouped = {int(day): group.set_index("minute", verify_integrity=True)
               for day, group in relevant.groupby("utc_day", sort=True)}
    trades, unavailable = [], []
    reason_count = {"MISSING_0000": 0, "MISSING_0800": 0, "INCOMPLETE_PATH": 0}
    for day in all_days:
        bars = grouped.get(day)
        label = utc_date(day * 86_400)
        if bars is None or 0 not in bars.index:
            reason = "MISSING_0000"
        elif 480 not in bars.index:
            reason = "MISSING_0800"
        elif len(bars) != 481 or not np.array_equal(bars.index.to_numpy(), np.arange(481)):
            reason = "INCOMPLETE_PATH"
        else:
            reason = None
        if reason is not None:
            reason_count[reason] += 1
            unavailable.append({"utc_date": str(label), "reason": reason})
            continue
        entry_s = int(bars.loc[0, "time"])
        exit_s = int(bars.loc[480, "time"])
        entry_bid = float(bars.loc[0, "open"])
        exit_bid = float(bars.loc[480, "open"])
        low_bid = float(bars.loc[np.arange(480), "low"].min())
        assert entry_s == day * 86_400 and exit_s == entry_s + 28_800
        assert low_bid <= entry_bid and entry_bid > 0 and exit_bid > 0
        entry_ask = entry_bid + SPREAD
        gross = exit_bid - entry_bid
        net = exit_bid - entry_ask
        ounces = 10_000.0 / entry_ask
        prague_entry = datetime.fromtimestamp(entry_s, timezone.utc).astimezone(PRAGUE).date()
        prague_exit = datetime.fromtimestamp(exit_s, timezone.utc).astimezone(PRAGUE).date()
        assert label == prague_entry == prague_exit
        trades.append({"utc_date": str(label), "prague_date": str(prague_entry),
                       "year": label.year, "entry_s": entry_s, "exit_s": exit_s,
                       "entry_bid": entry_bid, "entry_ask": entry_ask,
                       "exit_bid": exit_bid, "min_bid": low_bid,
                       "gross_units": gross, "net_units": net,
                       "stress_units": net - SPREAD,
                       "ounces_1x": ounces, "pnl_usd_1x": ounces * net,
                       "min_pnl_usd_1x": ounces * (low_bid - entry_ask)})
    result = pd.DataFrame(trades)
    missing = pd.DataFrame(unavailable)
    funnel = {"m1_rows": len(raw), "utc_calendar_dates": len(all_days),
              "eligible_dates": len(result), "unavailable_dates": len(missing),
              "unavailable_reasons": reason_count,
              "first_utc_date": str(utc_date(int(raw.utc_day.min()) * 86_400)),
              "last_utc_date": str(utc_date(int(raw.utc_day.max()) * 86_400))}
    return result, missing, funnel


def metrics(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {"n": 0}
    curve = rows.net_units.cumsum().to_numpy()
    peak = np.maximum.accumulate(np.r_[0.0, curve])[1:]
    return {"n": len(rows), "dates": rows.utc_date.nunique(),
            "gross_mean_units": float(rows.gross_units.mean()),
            "net_mean_units": float(rows.net_units.mean()),
            "net_total_units": float(rows.net_units.sum()),
            "stress_mean_units": float(rows.stress_units.mean()),
            "net_win_rate": float((rows.net_units > 0).mean()),
            "max_drawdown_units": float(np.max(peak - curve)),
            "mean_pnl_usd_1x": float(rows.pnl_usd_1x.mean()),
            "total_pnl_usd_1x": float(rows.pnl_usd_1x.sum()),
            "worst_session_floating_usd_1x": float(rows.min_pnl_usd_1x.min())}


def bootstrap(rows: pd.DataFrame) -> dict:
    values = rows.net_units.to_numpy()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return {"date_blocks": len(values),
            "p_mean_le_zero": float((means <= 0).mean()),
            "ci_95_low": float(np.quantile(means, .025)),
            "ci_95_high": float(np.quantile(means, .975))}


def challenge(rows: pd.DataFrame, days: int) -> tuple[pd.DataFrame, dict]:
    ordered = rows.copy().sort_values("entry_s")
    ordered["day_obj"] = pd.to_datetime(ordered.utc_date).dt.date
    last = ordered.day_obj.max()
    starts = sorted(set(ordered.day_obj))
    eligible = [d for d in starts if d + timedelta(days=days) <= last + timedelta(days=1)]
    windows = []
    for start in eligible:
        balance = 10_000.0
        distinct = set()
        outcome = "TIMEOUT"
        end = start + timedelta(days=days)
        for trade in ordered.itertuples():
            if trade.day_obj < start or trade.day_obj >= end:
                continue
            day_open = balance
            distinct.add(trade.prague_date)
            floating_equity_min = balance + trade.min_pnl_usd_1x
            if floating_equity_min < 9_000.0 or floating_equity_min < day_open - 500.0:
                outcome = "BREACH"
                event_day = trade.day_obj
                break
            balance += trade.pnl_usd_1x
            if balance < 9_000.0 or balance < day_open - 500.0:
                outcome = "BREACH"
                event_day = trade.day_obj
                break
            if balance >= 11_000.0 and len(distinct) >= 4:
                outcome = "PASS"
                event_day = trade.day_obj
                break
        if outcome == "TIMEOUT":
            event_day = end
        windows.append({"start": str(start), "horizon_days": days,
                        "outcome": outcome, "event_day": str(event_day),
                        "days_elapsed": (event_day - start).days,
                        "balance": balance, "trading_days": len(distinct)})
    out = pd.DataFrame(windows)
    count = out.outcome.value_counts()
    passes = out[out.outcome == "PASS"]
    n = len(out)
    summary = {"starts": n, "pass": int(count.get("PASS", 0)),
               "breach": int(count.get("BREACH", 0)),
               "timeout": int(count.get("TIMEOUT", 0)),
               "pass_rate": float((out.outcome == "PASS").mean()) if n else 0.0,
               "breach_rate": float((out.outcome == "BREACH").mean()) if n else 0.0,
               "median_days_to_pass": float(passes.days_elapsed.median()) if len(passes) else None}
    return out, summary


def main() -> None:
    report_path = OUT / "REPORT_ASIA_SESSION_LONG.txt"
    assert not report_path.exists(), "Frozen REPORT exists"
    spec = SPEC.read_text(encoding="utf-8")
    gate_lines = GATE.read_text(encoding="utf-8").splitlines()
    assert len(gate_lines) == 6 and all(line in spec for line in gate_lines)
    trades, missing, funnel = make_trades()
    trades.to_csv(OUT / "asia_session_trades.csv", index=False, float_format="%.8f")
    missing.to_csv(OUT / "unavailable_dates.csv", index=False)
    cal, test = trades[trades.year < 2024], trades[trades.year >= 2024]
    year = {str(y): metrics(group) for y, group in trades.groupby("year", sort=True)}
    boot = bootstrap(test) if len(test) else None
    windows_180, pace_180 = challenge(test, 180)
    windows_90, pace_90 = challenge(test, 90)
    pd.concat([windows_90, windows_180]).to_csv(OUT / "challenge_windows.csv", index=False)
    dates_unique = trades.utc_date.is_unique and (trades.utc_date == trades.prague_date).all()
    path_complete = len(trades) + len(missing) == funnel["utc_calendar_dates"]
    gates = {"G-AS1": len(test) >= 150 and test.utc_date.nunique() >= 150,
             "G-AS2": len(test) > 0 and float(test.stress_units.mean()) > 0,
             "G-AS3": boot is not None and boot["p_mean_le_zero"] <= .05,
             "G-AS4": all(year.get(str(y), {}).get("net_total_units", 0) > 0 for y in (2024, 2025)),
             "G-AS5": bool(dates_unique and path_complete and
                           ((trades.exit_s - trades.entry_s) == 28_800).all()),
             "G-AS6": pace_180["starts"] > 0 and pace_180["pass_rate"] >= .70
                      and pace_180["breach_rate"] <= .10}
    verdict = "PASS" if all(gates.values()) else "FAIL"
    result = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": sha256(SOURCE), "spec_sha256": sha256(SPEC),
              "gate_sha256": sha256(GATE), "funnel": funnel,
              "calibration": metrics(cal), "test": metrics(test),
              "bootstrap": boot, "yearly": year,
              "challenge_90": pace_90, "challenge_180": pace_180,
              "gates": gates, "verdict": verdict}
    (OUT / "run_meta.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["REPORT: HYPOTHESIS 72 XAUUSD 00:00-08:00 UTC LONG EVERY ELIGIBLE DATE",
             f"Generated UTC: {result['generated_utc']}   PHASE-1 VERDICT: {verdict}",
             "M1 bar-open BID/ASK+0.41 proxy, plus extra 0.41 cost stress; no stop or session filter.",
             "One gold notional x $10000 Challenge simulation is diagnostic, not local $10-risk-policy compliant.",
             f"SOURCE SHA256: {result['source_sha256']}",
             f"SPEC/GATE SHA256: {result['spec_sha256']} / {result['gate_sha256']}",
             f"FUNNEL: {json.dumps(funnel, sort_keys=True)}",
             f"CALIBRATION 2020-23: {json.dumps(result['calibration'], sort_keys=True)}",
             f"GATED 2024+: {json.dumps(result['test'], sort_keys=True)}",
             f"DATE BOOTSTRAP: {json.dumps(boot, sort_keys=True)}",
             "YEARLY:"]
    lines += [f"  {y}: {json.dumps(v, sort_keys=True)}" for y, v in year.items()]
    lines += [f"CHALLENGE 90-DAY 1X: {json.dumps(pace_90, sort_keys=True)}",
              f"CHALLENGE 180-DAY 1X: {json.dumps(pace_180, sort_keys=True)}",
              f"GATES: {json.dumps(gates, sort_keys=True)}",
              f"OVERALL PHASE-1 VERDICT: {verdict}",
              "PASS only permits a newly frozen native Bid/Ask replay and later Model=4 if that passes.",
              "FAIL final for this artifact; no post-hoc clock, side, stop, size or year rescue."]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "test": result["test"],
                      "bootstrap": boot, "challenge_180": pace_180,
                      "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
