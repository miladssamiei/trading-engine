"""Check FTMO native silver weekday quote coverage without returns."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/xagusd_weekday_source_qualification"
TERMINAL = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
SYMBOL = "XAGUSD"


def dates() -> list[pd.Timestamp]:
    result = []
    for year in (2024, 2025):
        for month in (1, 4, 7, 10):
            day = pd.Timestamp(year, month, 8)
            result.append(day + pd.Timedelta(days=(-day.weekday()) % 7))
    return result


def minute(day: pd.Timestamp, weekday_offset: int, hour: int, minute_value: int) -> dict:
    local = (day + pd.Timedelta(days=weekday_offset)).tz_localize("America/New_York")
    local += pd.Timedelta(hours=hour, minutes=minute_value)
    start = local.tz_convert("UTC")
    end = start + pd.Timedelta(seconds=59, milliseconds=999)
    row = {"week_monday_ny": str(day.date()), "endpoint_ny": local.isoformat(),
           "endpoint_kind": "MONDAY_ENTRY" if weekday_offset == 0 else "THURSDAY_EXIT"}
    raw = mt5.copy_ticks_range(SYMBOL, start.to_pydatetime(), end.to_pydatetime(), mt5.COPY_TICKS_ALL)
    if raw is None:
        row.update(status="API_NONE", source_ticks=0, error=str(mt5.last_error()))
        return row
    row["source_ticks"] = int(len(raw))
    row["source_sha256"] = hashlib.sha256(raw.tobytes()).hexdigest()
    if len(raw) == 0:
        row["status"] = "EMPTY"
        return row
    valid = np.isfinite(raw["bid"]) & np.isfinite(raw["ask"])
    valid &= (raw["bid"] > 0) & (raw["ask"] >= raw["bid"])
    row["invalid_ticks"] = int((~valid).sum())
    row["valid_ticks"] = int(valid.sum())
    if not valid.any():
        row["status"] = "NO_VALID_QUOTE"
        return row
    q = raw[valid]
    first = q[0]
    delay = (int(first["time_msc"]) - int(start.timestamp() * 1000)) / 1000
    spreads = q["ask"] - q["bid"]
    row["first_delay_seconds"] = round(delay, 3)
    row["first_spread_units"] = round(float(spreads[0]), 6)
    row["median_spread_units"] = round(float(np.median(spreads)), 6)
    row["p90_spread_units"] = round(float(np.quantile(spreads, 0.9)), 6)
    row["status"] = "COVERED" if 0 <= delay < 60 and row["invalid_ticks"] == 0 else "INVALID_OR_LATE"
    return row


def main() -> None:
    output = OUT / "SOURCE_WINDOWS_2026-09-26.jsonl"
    report = OUT / "SOURCE_REPORT_2026-09-26.txt"
    if output.exists() or report.exists():
        raise FileExistsError("Source artifact already exists")
    if not mt5.initialize(TERMINAL):
        raise RuntimeError(f"FTMO initialize failed: {mt5.last_error()}")
    was_visible = None
    try:
        account = mt5.account_info()
        if account is None or account.server != "FTMO-Demo" or account.trade_mode != 0:
            raise RuntimeError("Connected account/server is not FTMO-Demo")
        info = mt5.symbol_info(SYMBOL)
        if info is None or info.custom:
            raise RuntimeError("XAGUSD missing or custom")
        was_visible = bool(info.visible)
        if not mt5.symbol_select(SYMBOL, True):
            raise RuntimeError("Could not select XAGUSD")
        rows = [minute(day, offset, hour, value)
                for day in dates() for offset, hour, value in ((0, 9, 30), (3, 15, 55))]
    finally:
        if was_visible is False:
            mt5.symbol_select(SYMBOL, False)
        mt5.shutdown()
    output.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    covered_weeks = []
    for day in dates():
        group = [row for row in rows if row["week_monday_ny"] == str(day.date())]
        if len(group) == 2 and all(row["status"] == "COVERED" for row in group):
            covered_weeks.append(str(day.date()))
    spreads = [row["first_spread_units"] for row in rows if "first_spread_units" in row]
    verdict = "PASS" if len(covered_weeks) >= 7 else "FAIL"
    lines = ["FTMO XAGUSD WEEKDAY SOURCE VERDICT: " + verdict,
             "Frozen fixed weeks: " + ", ".join(str(day.date()) for day in dates()),
             f"Covered complete weeks: {len(covered_weeks)}/8; " + ", ".join(covered_weeks),
             "Status counts: " + json.dumps({status: sum(row["status"] == status for row in rows)
                                         for status in sorted({row["status"] for row in rows})}),
             "First-spread units median/p90: " +
             (f"{np.median(spreads):.6f}/{np.quantile(spreads, 0.9):.6f}" if spreads else "none"),
             "Window stream SHA256: " + hashlib.sha256(output.read_bytes()).hexdigest(),
             "No silver price change, trading return, PnL or account order calculated."]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
