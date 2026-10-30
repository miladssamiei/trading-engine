"""Isolated USDJPY source-only revalidation after MT5 IPC failure."""
from __future__ import annotations

import hashlib
import json

import MetaTrader5 as mt5
import numpy as np

from ftmo_fx_daily_reopen_source import OUT, TERMINAL, month_dates, window


def connect() -> bool:
    if not mt5.initialize(TERMINAL):
        raise RuntimeError(f"FTMO initialize failed: {mt5.last_error()}")
    account = mt5.account_info()
    if account is None or account.server != "FTMO-Demo" or account.trade_mode != 0:
        mt5.shutdown()
        raise RuntimeError("Connected account/server is not FTMO-Demo")
    info = mt5.symbol_info("USDJPY")
    if info is None or info.custom:
        mt5.shutdown()
        raise RuntimeError("USDJPY missing or custom")
    was_visible = bool(info.visible)
    if not mt5.symbol_select("USDJPY", True):
        mt5.shutdown()
        raise RuntimeError("Could not select USDJPY")
    return was_visible


def main() -> None:
    output = OUT / "usdjpy_revalidation_windows.jsonl"
    report = OUT / "USDJPY_REVALIDATION_REPORT_2026-09-26.txt"
    if output.exists() or report.exists():
        raise FileExistsError("USDJPY revalidation artifact already exists")
    rows = []
    original_visible = connect()
    try:
        for day in month_dates():
            first = window("USDJPY", 0.01, day)
            if first["status"] != "API_NONE":
                first["retry_count"] = 0
                rows.append(first)
                continue
            first_error = first.get("api_error")
            mt5.shutdown()
            connect()
            second = window("USDJPY", 0.01, day)
            second["retry_count"] = 1
            second["initial_api_error"] = first_error
            rows.append(second)
    finally:
        if original_visible is False:
            mt5.symbol_select("USDJPY", False)
        mt5.shutdown()
    output.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    passing = [row for row in rows if row["status"] == "SOURCE_PASS_DATE"]
    spreads = [row["first_post_spread_pips"] for row in rows if "first_post_spread_pips" in row]
    breaks = [row["quote_break_seconds"] for row in rows if "quote_break_seconds" in row]
    verdict = "PASS" if len(passing) >= 18 else "FAIL"
    summary = {"verdict": verdict, "dates": len(rows), "passing_dates": len(passing),
               "retry_dates": sum(row["retry_count"] for row in rows),
               "statuses": {status: sum(row["status"] == status for row in rows)
                            for status in sorted({row["status"] for row in rows})},
               "median_break_seconds": float(np.median(breaks)) if breaks else None,
               "max_break_seconds": float(max(breaks)) if breaks else None,
               "median_first_post_spread_pips": float(np.median(spreads)) if spreads else None,
               "p90_first_post_spread_pips": float(np.quantile(spreads, 0.9)) if spreads else None}
    lines = ["USDJPY NATIVE REOPEN SOURCE VERDICT: " + verdict,
             "Original two-symbol source FAIL unchanged; recovery plan frozen before this run.",
             "No gap size, direction, one-hour return, PnL or account order calculated.",
             "Window stream SHA256: " + hashlib.sha256(output.read_bytes()).hexdigest(),
             json.dumps(summary, sort_keys=True)]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
