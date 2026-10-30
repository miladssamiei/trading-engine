"""Frozen H106 Phase-1 proxy replay. No MT5 connection or orders."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/cot_gold_intraweek_h106"
SPEC = OUT / "SPEC_COT_GOLD_INTRAWEEK_H106.txt"
GATE = OUT / "GATE_COT_GOLD_INTRAWEEK_H106.txt"
PRICE = ROOT / "data/results/xauusd_onboarding/xauusdtick_m1.csv"
COT_DIR = ROOT / "data/raw/cftc_gold_cot"
PRICE_SHA = "A8C4C1D6551D27F3796F8D9B88D332172C41C76C151844E759492837152E7B1F"
COT_SHA = {
    2023: "9FE75DBA0256819D476BB0D01F9E8E5AC6EC2C4390F609A7E3259E85796B79CE",
    2024: "5E40F60993120D26C116AB347BEAF93702F5DA60C7BDCCBBEE43CB3D0A91D01A",
    2025: "17AC2FEF1B53303D01E486BBF5768A4E062308A3C34A2970871F3EAB89829CBC",
    2026: "CA88B14A88386BA250E4C48FA786162566EFEDB3FD5D63C5E87DC49A9DBD39DA",
}
NY = ZoneInfo("America/New_York")
PRAGUE = ZoneInfo("Europe/Prague")
SPREAD = 0.41
STRESS = 0.41
COMMISSION_RATE = 0.000007
SWAP = {1: -0.83, -1: -0.083}
SEED = 20260926


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def source_guard() -> tuple[str, str]:
    if (OUT / "REPORT_COT_GOLD_INTRAWEEK_H106.txt").exists():
        raise RuntimeError("Frozen H106 report already exists")
    sh, gh = sha(SPEC), sha(GATE)
    gates = [s for s in SPEC.read_text(encoding="utf-8").splitlines() if s.startswith("G-H106-")]
    gate_lines = [s for s in GATE.read_text(encoding="utf-8-sig").splitlines() if s]
    if len(gates) != 5 or gate_lines != gates:
        raise RuntimeError("GATE is not verbatim SPEC extraction")
    if sha(PRICE) != PRICE_SHA:
        raise RuntimeError("Gold BID M1 source hash changed")
    for year, expected in COT_SHA.items():
        if sha(COT_DIR / f"fut_disagg_txt_{year}.zip") != expected:
            raise RuntimeError(f"CFTC {year} ZIP hash changed")
    return sh, gh


def load_cot() -> pd.DataFrame:
    parts = []
    cols = [
        "Market_and_Exchange_Names", "Report_Date_as_YYYY-MM-DD",
        "CFTC_Contract_Market_Code", "Open_Interest_All",
        "Prod_Merc_Positions_Long_All", "Prod_Merc_Positions_Short_All",
    ]
    for year in COT_SHA:
        with zipfile.ZipFile(COT_DIR / f"fut_disagg_txt_{year}.zip") as z:
            with z.open("f_year.txt") as f:
                frame = pd.read_csv(f, usecols=cols, low_memory=False)
        frame = frame.loc[
            (frame["CFTC_Contract_Market_Code"].astype(str).str.zfill(6) == "088691")
            & (frame["Market_and_Exchange_Names"] == "GOLD - COMMODITY EXCHANGE INC.")
        ].copy()
        parts.append(frame)
    cot = pd.concat(parts, ignore_index=True)
    cot["report_date"] = pd.to_datetime(cot["Report_Date_as_YYYY-MM-DD"]).dt.date
    if cot["report_date"].duplicated().any() or len(cot) != 195:
        raise RuntimeError("Gold contract/date source uniqueness failed")
    cot = cot.sort_values("report_date").reset_index(drop=True)
    for col in cols[-3:]:
        cot[col] = pd.to_numeric(cot[col], errors="raise")
    cot["producer_net"] = cot["Prod_Merc_Positions_Long_All"] - cot["Prod_Merc_Positions_Short_All"]
    cot["prior_net"] = cot["producer_net"].shift(1)
    cot["prior_oi"] = cot["Open_Interest_All"].shift(1)
    cot["q"] = (cot["producer_net"] - cot["prior_net"]) / cot["prior_oi"]
    return cot


def excluded(d: date) -> bool:
    return (
        date(2023, 1, 31) <= d <= date(2023, 3, 14)
        or d == date(2025, 1, 7)
        or date(2025, 9, 30) <= d <= date(2026, 1, 13)
    )


def following_monday(d: date) -> date:
    offset = (7 - d.weekday()) % 7
    return d + timedelta(days=offset or 7)


def thursday_closed(d: date) -> bool:
    if d.weekday() != 3:
        raise ValueError("Expected Thursday")
    thanksgiving = d.month == 11 and 22 <= d.day <= 28
    return thanksgiving or (d.month, d.day) in {(1, 1), (7, 4), (12, 25)}


def load_price() -> pd.DataFrame:
    px = pd.read_csv(PRICE, usecols=["time", "open", "high", "low", "close"])
    px = px.loc[px.time >= int(pd.Timestamp("2023-01-01", tz="UTC").timestamp())].copy()
    if not px.time.is_monotonic_increasing or px.time.duplicated().any():
        raise RuntimeError("Price M1 chronology or uniqueness failed")
    px["utc"] = pd.to_datetime(px.time, unit="s", utc=True)
    px["ny"] = px.utc.dt.tz_convert(NY)
    px["ny_date"] = px.ny.dt.date
    px["ny_minute"] = px.ny.dt.hour * 60 + px.ny.dt.minute
    return px.reset_index(drop=True)


def price_bar_index(px: pd.DataFrame, d: date, minute: int) -> int | None:
    row = px.index[(px.ny_date == d) & (px.ny_minute == minute)]
    return int(row[0]) if len(row) else None


def abnormal_gap(hold: pd.DataFrame, entry_date: date, exit_date: date) -> bool:
    # Gold's daily maintenance break is outside this 01:00-16:00 NY window.
    normal = hold.loc[(hold.ny_minute >= 60) & (hold.ny_minute < 960)]
    for d in (entry_date + timedelta(days=i) for i in range((exit_date - entry_date).days + 1)):
        day = normal.loc[normal.ny_date == d]
        if d.weekday() < 5 and len(day) == 0:
            return True
        if len(day) > 1 and bool((day.time.diff().dropna() > 31 * 60).any()):
            return True
    return False


def rollover_units(entry_utc: pd.Timestamp, exit_utc: pd.Timestamp, monday: date, side: int) -> tuple[float, int]:
    units = 0
    for i in range(3):
        d = monday + timedelta(days=i)
        roll = datetime(d.year, d.month, d.day, 23, 0, tzinfo=PRAGUE).astimezone(ZoneInfo("UTC"))
        if entry_utc.to_pydatetime() < roll <= exit_utc.to_pydatetime():
            units += 3 if i == 2 else 1
    return SWAP[side] * units, units


def replay(hold: pd.DataFrame, side: int, monday: date) -> dict:
    first = hold.iloc[0]
    last = hold.iloc[-1]
    entry_bid = float(first.open)
    entry = entry_bid + SPREAD if side == 1 else entry_bid
    stop = entry * (1 - 0.015 * side)
    exit_px = float(last.open) if side == 1 else float(last.open) + SPREAD
    exit_row = last
    stopped = False
    for row in hold.itertuples(index=False):
        hit = float(row.low) <= stop if side == 1 else float(row.high) + SPREAD >= stop
        if hit:
            exit_px = min(float(row.open), stop) if side == 1 else max(float(row.open) + SPREAD, stop)
            exit_row = row
            stopped = True
            break
    gross_after_spread = side * (exit_px - entry)
    swap, rollover_count = rollover_units(first.utc, exit_row.utc, monday, side)
    commission = COMMISSION_RATE * (entry + exit_px)
    base = gross_after_spread + swap - commission
    return {
        "side": side, "entry_price": entry, "exit_price": exit_px,
        "stop_price": stop, "exit_utc": exit_row.utc,
        "stopped": stopped, "gross_after_spread": gross_after_spread,
        "swap": swap, "rollover_units": rollover_count,
        "commission": commission, "base_net": base,
        "stress_net": base - STRESS,
    }


def block_p(roster: pd.DataFrame, column: str) -> float:
    vals = roster[column].to_numpy(dtype=float)
    n = len(vals)
    if n < 5:
        return math.nan
    rng = np.random.default_rng(SEED)
    draws = 10_000
    bad = 0
    blocks_needed = math.ceil(n / 5)
    for _ in range(draws):
        starts = rng.integers(0, n - 4, size=blocks_needed)
        sample = np.concatenate([vals[s:s + 5] for s in starts])[:n]
        sample = sample[~np.isnan(sample)]
        if len(sample) == 0 or sample.mean() <= 0:
            bad += 1
    return bad / draws


def main() -> None:
    spec_sha, gate_sha = source_guard()
    cot = load_cot()
    px = load_price()
    last_source_date = px.ny_date.iloc[-1]
    rows: list[dict] = []
    for r in cot.itertuples(index=False):
        report_date = r.report_date
        monday = following_monday(report_date)
        thursday = monday + timedelta(days=3)
        if thursday > last_source_date:
            continue
        out = {"report_date": report_date, "monday": monday, "thursday": thursday,
               "q": float(r.q) if pd.notna(r.q) else math.nan}
        if excluded(report_date):
            out["status"] = "EXCLUDED_PUBLICATION_DELAY"
        elif thursday_closed(thursday):
            out["status"] = "PREDECLARED_THURSDAY_HOLIDAY"
        elif not math.isfinite(out["q"]) or float(r.prior_oi) <= 0:
            out["status"] = "NO_VALID_PRIOR_REPORT"
        elif out["q"] == 0:
            out["status"] = "ZERO_POSITION_CHANGE"
        else:
            ei = price_bar_index(px, monday, 570)
            xi = price_bar_index(px, thursday, 955)
            if ei is None:
                out["status"] = "NO_REAL_ENTRY"
            elif xi is None or xi <= ei:
                out["status"] = "NO_REAL_EXIT"
            else:
                hold = px.iloc[ei:xi + 1]
                if abnormal_gap(hold, monday, thursday):
                    out["status"] = "ABNORMAL_NORMAL_HOURS_GAP"
                else:
                    side = 1 if out["q"] > 0 else -1
                    actual = replay(hold, side, monday)
                    benchmark = replay(hold, 1, monday)
                    out.update(actual)
                    out["benchmark_stress_net"] = benchmark["stress_net"]
                    out["paired_difference"] = actual["stress_net"] - benchmark["stress_net"]
                    out["entry_utc"] = hold.iloc[0].utc
                    out["status"] = "EXECUTED"
        rows.append(out)
    roster = pd.DataFrame(rows).sort_values("monday").reset_index(drop=True)
    trades = roster.loc[roster.status == "EXECUTED"].copy()
    planned = roster.loc[~roster.status.isin([
        "EXCLUDED_PUBLICATION_DELAY", "PREDECLARED_THURSDAY_HOLIDAY",
        "NO_VALID_PRIOR_REPORT", "ZERO_POSITION_CHANGE",
    ])]
    coverage = len(trades) / len(planned) if len(planned) else 0.0
    p_mean = block_p(roster, "stress_net")
    p_delta = block_p(roster, "paired_difference")
    annual = {
        y: {"n": int(len(g)), "stress_sum": float(g.stress_net.sum())}
        for y, g in trades.groupby(pd.to_datetime(trades.monday).dt.year)
    }
    gates = {
        "G-H106-1": len(trades) >= 150 and coverage >= 0.97 and not (roster.status == "ABNORMAL_NORMAL_HOURS_GAP").any(),
        "G-H106-2": len(trades) > 0 and float(trades.stress_net.mean()) > 0 and p_mean <= 0.05,
        "G-H106-3": all(annual.get(y, {}).get("stress_sum", -math.inf) > 0 for y in (2023, 2024, 2025)),
        "G-H106-4": len(trades) > 0 and float(trades.paired_difference.mean()) > 0 and p_delta <= 0.05,
        "G-H106-5": "NOT_RUN",
    }
    passed = all(gates[k] is True for k in ("G-H106-1", "G-H106-2", "G-H106-3", "G-H106-4"))
    verdict = "PHASE1_PASS" if passed else "FINAL_FAIL"
    roster.to_csv(OUT / "coverage_roster.csv", index=False)
    trades.to_csv(OUT / "trades.csv", index=False)
    result = {
        "verdict": verdict, "spec_sha256": spec_sha, "gate_sha256": gate_sha,
        "price_sha256": PRICE_SHA, "cftc_zip_sha256": COT_SHA,
        "price_last_ny_date": str(last_source_date),
        "roster_status": roster.status.value_counts().to_dict(),
        "planned_nonexcluded": int(len(planned)), "resolved_trades": int(len(trades)),
        "coverage": coverage,
        "long_count": int((trades.side == 1).sum()),
        "short_count": int((trades.side == -1).sum()),
        "base_mean": float(trades.base_net.mean()),
        "stress_mean": float(trades.stress_net.mean()),
        "benchmark_stress_mean": float(trades.benchmark_stress_net.mean()),
        "paired_stress_mean": float(trades.paired_difference.mean()),
        "p_nonpositive_stress_mean_5_calendar_week_block": p_mean,
        "p_nonpositive_paired_difference_5_calendar_week_block": p_delta,
        "annual": annual, "gates": gates,
        "trades_sha256": sha(OUT / "trades.csv"),
        "coverage_roster_sha256": sha(OUT / "coverage_roster.csv"),
        "phase2": "NOT_RUN",
    }
    (OUT / "REPORT_COT_GOLD_INTRAWEEK_H106.txt").write_text(
        "H106 CFTC PRODUCER POSITION GOLD INTRAWEEK - " + verdict + "\n"
        + json.dumps(result, indent=2, default=str) + "\n"
        + "Dukascopy synthetic spread and current FTMO swap snapshot are exploratory proxy costs. "
          "No native FTMO execution, historical swap posting or account order.\n",
        encoding="utf-8",
    )
    print(json.dumps({k: result[k] for k in (
        "verdict", "roster_status", "resolved_trades", "coverage", "stress_mean",
        "paired_stress_mean", "p_nonpositive_stress_mean_5_calendar_week_block",
        "p_nonpositive_paired_difference_5_calendar_week_block", "annual", "gates",
    )}, indent=2))


if __name__ == "__main__":
    main()
