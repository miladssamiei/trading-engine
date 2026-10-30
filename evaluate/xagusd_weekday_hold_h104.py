"""Frozen H104 native FTMO silver Phase 1. Never places orders."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/results/xagusd_weekday_source_qualification/xagusd_ftmo_native_m1_2023_2026.csv"
OUT = ROOT / "data/results/xagusd_weekday_hold_h104"
EXPECTED = "c73194dba207ff8a71fd7adce4b60ed40dce60c954dd01991ace981aff153f9c"
COMMISSION_RATE = 0.000007
SWAP_NORMAL = -0.01925
EXTRA_STRESS = 0.060


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def cash_days(frame: pd.DataFrame) -> pd.DataFrame:
    local = pd.to_datetime(frame.minute_epoch_s, unit="s", utc=True).dt.tz_convert("America/New_York")
    frame = frame.copy()
    frame["ny_date"] = local.dt.date.astype(str)
    frame["ny_clock"] = local.dt.strftime("%H:%M")
    frame["cash"] = (frame.ny_clock >= "09:30") & (frame.ny_clock < "16:00")
    cash = frame[frame.cash]
    grouped = cash.groupby("ny_date", sort=True)
    daily = grouped.agg(high=("bid_high", "max"), low=("bid_low", "min"))
    first = cash[cash.ny_clock == "09:30"].drop_duplicates("ny_date").set_index("ny_date")
    last = cash[cash.ny_clock == "15:55"].drop_duplicates("ny_date").set_index("ny_date")
    daily = daily.join(first[["bid_open"]].rename(columns={"bid_open": "open"}))
    daily = daily.join(last[["bid_close"]].rename(columns={"bid_close": "close"}))
    daily = daily.dropna().copy()
    prev = daily.close.shift(1)
    daily["tr"] = pd.concat([(daily.high - daily.low), (daily.high - prev).abs(),
                             (daily.low - prev).abs()], axis=1).max(axis=1)
    daily["atr20"] = daily.tr.rolling(20, min_periods=20).mean()
    daily["prior20_close"] = daily.close.shift(20)
    return daily


def utc_epoch(date: pd.Timestamp, clock: str) -> int:
    local = pd.Timestamp(f"{date.date()} {clock}", tz="America/New_York")
    return int(local.timestamp())


def swap_weight(start_epoch: int, end_epoch: int) -> int:
    a = pd.Timestamp(start_epoch, unit="s", tz="UTC")
    b = pd.Timestamp(end_epoch, unit="s", tz="UTC")
    la = a.tz_convert("Europe/Athens")
    lb = b.tz_convert("Europe/Athens")
    weight = 0
    for date in pd.date_range(la.date(), lb.date(), freq="D"):
        midnight = date.tz_localize("Europe/Athens").tz_convert("UTC")
        if a < midnight < b:
            weight += 3 if date.weekday() == 2 else 1
    return weight


def evaluate(frame: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    index = frame.set_index("minute_epoch_s", verify_integrity=True).sort_index()
    times = index.index.to_numpy(dtype=np.int64)
    dates = daily.index.to_numpy()
    rows = []
    counts = {"planned": 0, "covered": 0, "feature_ready": 0, "signaled": 0,
              "unresolved": 0, "normal_hours_gaps": 0}
    for monday in pd.date_range("2023-01-02", "2026-09-21", freq="W-MON"):
        for leg, day, entry_clock, exit_day, exit_clock in (
            ("MON_WED", monday, "09:30", monday + pd.Timedelta(days=2), "09:25"),
            ("WED_THU", monday + pd.Timedelta(days=2), "09:30", monday + pd.Timedelta(days=3), "15:55"),
        ):
            year = int(day.year)
            entry_epoch = utc_epoch(day, entry_clock)
            end_epoch = utc_epoch(exit_day, exit_clock)
            if year >= 2024:
                counts["planned"] += 1
            eligible = entry_epoch in index.index and end_epoch in index.index
            if eligible and year >= 2024:
                counts["covered"] += 1
            prior_pos = np.searchsorted(dates, str(day.date()), side="left") - 1
            if prior_pos < 20:
                continue
            prior = daily.iloc[prior_pos]
            age = (day.date() - pd.Timestamp(dates[prior_pos]).date()).days
            if age > 4 or not np.isfinite(prior.atr20) or not np.isfinite(prior.prior20_close):
                continue
            if year >= 2024:
                counts["feature_ready"] += 1
            if not prior.close > prior.prior20_close:
                continue
            if year >= 2024:
                counts["signaled"] += 1
            if not eligible:
                if year >= 2024:
                    counts["unresolved"] += 1
                rows.append({"monday": str(monday.date()), "year": year, "leg": leg,
                             "entry_epoch": entry_epoch, "scheduled_exit_epoch": end_epoch,
                             "status": "UNRESOLVED_ENDPOINT"})
                continue
            entry = index.loc[entry_epoch]
            stop = float(entry.ask_open) - 1.5 * float(prior.atr20)
            a = np.searchsorted(times, entry_epoch, side="left")
            b = np.searchsorted(times, end_epoch, side="left")
            path = index.iloc[a:b]
            # The broker trades silver around the clock on weekdays; exclude
            # the brief 17:00 NY maintenance hour from quote-gap checks.
            all_min = pd.date_range(pd.Timestamp(entry_epoch, unit="s", tz="UTC"),
                                    pd.Timestamp(end_epoch - 60, unit="s", tz="UTC"), freq="min")
            local = all_min.tz_convert("America/New_York")
            expected_minutes = all_min[(local.weekday < 5) & (local.strftime("%H:%M") < "17:00") |
                                       ((local.weekday < 5) & (local.strftime("%H:%M") >= "18:00"))]
            gaps = int((~np.isin(expected_minutes.asi8 // 10**9, times[a:b])).sum())
            if gaps and year >= 2024:
                counts["normal_hours_gaps"] += gaps
            hit = path[(path.bid_open <= stop) | (path.bid_low <= stop)]
            if len(hit):
                first_hit = hit.iloc[0]
                exit_epoch = int(hit.index[0])
                exit_bid = min(float(first_hit.bid_open), stop)
                status = "STOP"
            else:
                exit_epoch = end_epoch
                exit_bid = float(index.loc[end_epoch].bid_open)
                status = "TIME"
            weight = swap_weight(entry_epoch, exit_epoch)
            swap = SWAP_NORMAL * weight
            commission = COMMISSION_RATE * (float(entry.ask_open) + exit_bid)
            net = exit_bid - float(entry.ask_open) + swap - commission
            rows.append({"monday": str(monday.date()), "year": year, "leg": leg,
                         "entry_epoch": entry_epoch, "scheduled_exit_epoch": end_epoch,
                         "status": status, "exit_epoch": exit_epoch,
                         "entry_ask": float(entry.ask_open), "entry_bid": float(entry.bid_open),
                         "exit_bid": exit_bid, "stop": stop, "atr20": float(prior.atr20),
                         "trend_close": float(prior.close), "trend_prior20": float(prior.prior20_close),
                         "entry_spread": float(entry.ask_open - entry.bid_open),
                         "gaps": gaps, "swap_weight": weight, "swap": swap,
                         "commission": commission, "net": net, "stress_net": net - EXTRA_STRESS})
    return pd.DataFrame(rows), counts


def bootstrap(trades: pd.DataFrame) -> float:
    ordered = trades.sort_values("entry_epoch")
    returns = ordered.stress_net.to_numpy()
    weeks = ordered.monday.to_numpy()
    unique_weeks = np.unique(weeks)
    blocks = [returns[np.isin(weeks, unique_weeks[(np.arange(i, i + 5) % len(unique_weeks))])]
              for i in range(len(unique_weeks))]
    rng = np.random.default_rng(20261004)
    bad = 0
    for _ in range(10000):
        samples = []
        length = 0
        while length < len(returns):
            part = blocks[int(rng.integers(len(blocks)))]
            samples.append(part)
            length += len(part)
        if np.concatenate(samples)[:len(returns)].mean() <= 0:
            bad += 1
    return bad / 10000


def main() -> None:
    report = OUT / "REPORT_XAGUSD_WEEKDAY_H104.txt"
    ledger = OUT / "trades.csv"
    if report.exists() or ledger.exists():
        raise FileExistsError("H104 historical artifacts already exist")
    if digest(SOURCE) != EXPECTED:
        raise ValueError("Frozen source hash mismatch")
    frame = pd.read_csv(SOURCE)
    daily = cash_days(frame)
    trades, counts = evaluate(frame, daily)
    trades.to_csv(ledger, index=False)
    oos = trades[(trades.year >= 2024) & (trades.status != "UNRESOLVED_ENDPOINT")]
    p = bootstrap(oos) if len(oos) else 1.0
    years = {str(y): {"n": int(len(g)), "stress_sum": float(g.stress_net.sum()),
                      "stress_mean": float(g.stress_net.mean())}
             for y, g in oos.groupby("year")}
    gates = {"G-H104-1": len(oos) >= 150 and counts["unresolved"] == 0 and counts["normal_hours_gaps"] == 0,
             "G-H104-2": len(oos) > 0 and float(oos.stress_net.mean()) > 0,
             "G-H104-3": p <= 0.05,
             "G-H104-4": all(years.get(str(y), {}).get("stress_sum", 0) > 0 for y in (2024, 2025, 2026)),
             "G-H104-5": True, "G-H104-6": False}
    verdict = "PHASE1_PASS_PHASE2_REQUIRED" if all(gates[k] for k in list(gates)[:5]) else "PHASE1_FAIL"
    result = {"verdict": verdict, "source_hash": EXPECTED, "counts": counts,
              "resolved_oos": len(oos), "bootstrap_p_nonpositive": p,
              "oos_stress_mean": float(oos.stress_net.mean()) if len(oos) else None,
              "years": years, "gates": gates,
              "oos_swap_sum": float(oos.swap.sum()), "oos_commission_sum": float(oos.commission.sum()),
              "oos_entry_spread_median": float(oos.entry_spread.median()) if len(oos) else None,
              "ledger_sha256": digest(ledger), "phase2": "NOT_RUN"}
    report.write_text("H104 XAGUSD PHASE 1 " + verdict + "\n" +
                      json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
