"""Qualify native FTMO FX rollover quotes without calculating returns."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/ftmo_fx_daily_reopen_source"
TERMINAL = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
SYMBOLS = {"EURUSD": 0.0001, "USDJPY": 0.01}


def month_dates() -> list[pd.Timestamp]:
    result = []
    for first in pd.date_range("2024-01-01", "2025-12-01", freq="MS"):
        result.append(first + pd.Timedelta(days=(1 - first.weekday()) % 7))
    assert len(result) == 24 and len({str(x.date()) for x in result}) == 24
    return result


def local_time(day: pd.Timestamp, hour: int, minute: int) -> pd.Timestamp:
    return day.tz_localize("America/New_York") + pd.Timedelta(hours=hour, minutes=minute)


def window(symbol: str, pip: float, day: pd.Timestamp) -> dict:
    a = local_time(day, 16, 45).tz_convert("UTC")
    b = local_time(day, 18, 10).tz_convert("UTC")
    boundary = int(local_time(day, 17, 0).timestamp() * 1000)
    pre_begin = int(local_time(day, 16, 55).timestamp() * 1000)
    first_minute_end = boundary + 60000
    exit_begin = int(local_time(day, 18, 0).timestamp() * 1000)
    exit_end = exit_begin + 60000
    row = {"symbol": symbol, "ny_date": str(day.date()),
           "query_start_utc": a.isoformat(), "query_end_utc": b.isoformat(),
           "status": "PENDING"}
    raw = mt5.copy_ticks_range(symbol, a.to_pydatetime(), b.to_pydatetime(), mt5.COPY_TICKS_ALL)
    if raw is None:
        row.update(status="API_NONE", api_error=str(mt5.last_error()), source_ticks=0)
        return row
    row["source_ticks"] = int(len(raw))
    row["source_sha256"] = hashlib.sha256(raw.tobytes()).hexdigest()
    if len(raw) == 0:
        row["status"] = "EMPTY"
        return row
    ms = raw["time_msc"]
    if np.any(np.diff(ms) < 0):
        row["status"] = "UNORDERED"
        return row
    inside = (ms >= int(a.timestamp() * 1000)) & (ms <= int(b.timestamp() * 1000))
    valid = inside & np.isfinite(raw["bid"]) & np.isfinite(raw["ask"])
    valid &= (raw["bid"] > 0) & (raw["ask"] >= raw["bid"])
    row["valid_ticks"] = int(valid.sum())
    q = raw[valid]
    if len(q) == 0:
        row["status"] = "NO_VALID_QUOTES"
        return row
    t = q["time_msc"]
    row["first_valid_utc"] = pd.Timestamp(int(t[0]), unit="ms", tz="UTC").isoformat()
    row["last_valid_utc"] = pd.Timestamp(int(t[-1]), unit="ms", tz="UTC").isoformat()
    before = q[(t >= pre_begin) & (t < boundary)]
    after = q[t >= boundary]
    exit_quotes = q[(t >= exit_begin) & (t < exit_end)]
    row["pre_quote_present"] = len(before) > 0
    row["post_quote_present"] = len(after) > 0
    row["exit_minute_present"] = len(exit_quotes) > 0
    if len(before) == 0 or len(after) == 0:
        row["status"] = "MISSING_BOUNDARY_QUOTE"
        return row
    pre = before[-1]
    post = after[0]
    row["pre_quote_utc"] = pd.Timestamp(int(pre["time_msc"]), unit="ms", tz="UTC").isoformat()
    row["post_quote_utc"] = pd.Timestamp(int(post["time_msc"]), unit="ms", tz="UTC").isoformat()
    row["quote_break_seconds"] = round((int(post["time_msc"]) - int(pre["time_msc"])) / 1000, 3)
    row["post_delay_seconds"] = round((int(post["time_msc"]) - boundary) / 1000, 3)
    row["first_post_spread_pips"] = round((float(post["ask"]) - float(post["bid"])) / pip, 5)
    minute = after[after["time_msc"] < first_minute_end]
    spreads = (minute["ask"] - minute["bid"]) / pip
    row["first_minute_quote_count"] = int(len(minute))
    row["first_minute_spread_median_pips"] = round(float(np.median(spreads)), 5) if len(spreads) else None
    row["first_minute_spread_p90_pips"] = round(float(np.quantile(spreads, 0.9)), 5) if len(spreads) else None
    row["status"] = "SOURCE_PASS_DATE" if (row["quote_break_seconds"] >= 300
              and row["post_delay_seconds"] <= 900
              and row["exit_minute_present"]) else "SOURCE_FAIL_DATE"
    return row


def main() -> None:
    output = OUT / "source_windows.jsonl"
    report = OUT / "SOURCE_REPORT_2026-09-26.txt"
    if output.exists() or report.exists():
        raise FileExistsError("Source artifact already exists")
    if not mt5.initialize(TERMINAL):
        raise RuntimeError(f"FTMO terminal initialize failed: {mt5.last_error()}")
    visibility = {}
    try:
        account = mt5.account_info()
        if account is None or account.server != "FTMO-Demo" or account.trade_mode != 0:
            raise RuntimeError("Connected account/server is not FTMO-Demo")
        for symbol in SYMBOLS:
            info = mt5.symbol_info(symbol)
            if info is None or info.custom:
                raise RuntimeError(f"Missing native symbol: {symbol}")
            visibility[symbol] = bool(info.visible)
            if not mt5.symbol_select(symbol, True):
                raise RuntimeError(f"Could not select {symbol}")
        rows = [window(symbol, pip, day) for symbol, pip in SYMBOLS.items()
                for day in month_dates()]
    finally:
        for symbol, was_visible in visibility.items():
            if not was_visible:
                mt5.symbol_select(symbol, False)
        mt5.shutdown()
    output.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    summary = {}
    for symbol in SYMBOLS:
        group = [row for row in rows if row["symbol"] == symbol]
        passing = [row for row in group if row["status"] == "SOURCE_PASS_DATE"]
        first_spreads = [row["first_post_spread_pips"] for row in group if "first_post_spread_pips" in row]
        summary[symbol] = {"dates": len(group), "passing_dates": len(passing),
                           "source_pass": len(passing) >= 18,
                           "median_first_post_spread_pips": round(float(np.median(first_spreads)), 5) if first_spreads else None,
                           "p90_first_post_spread_pips": round(float(np.quantile(first_spreads, 0.9)), 5) if first_spreads else None,
                           "statuses": {name: sum(row["status"] == name for row in group)
                                        for name in sorted({row["status"] for row in group})}}
    verdict = "PASS" if any(item["source_pass"] for item in summary.values()) else "FAIL"
    lines = ["FTMO FX DAILY-REOPEN SOURCE VERDICT: " + verdict,
             "Frozen plan: SOURCE_PLAN_2026-09-26.txt",
             "Native FTMO-Demo Bid/Ask only; no gap sizes, returns, PnL or orders.",
             "Window stream SHA256: " + hashlib.sha256(output.read_bytes()).hexdigest()]
    for symbol, item in summary.items():
        lines.append(symbol + ": " + json.dumps(item, sort_keys=True))
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
