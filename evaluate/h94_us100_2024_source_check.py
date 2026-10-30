"""Fixed-date native FTMO US100 source availability, no trade returns."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/project_status/H94_US100_2024_NATIVE_SOURCE_REPORT_2026-09-26.txt"
TERMINAL = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
SYMBOL = "US100.cash"
NY = ZoneInfo("America/New_York")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if not mt5.initialize(TERMINAL):
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")
    info = None
    prior_visible = None
    try:
        account = mt5.account_info()
        if account is None or account.server != "FTMO-Demo" or account.trade_mode != 0:
            raise RuntimeError("Wrong MT5 account/server")
        info = mt5.symbol_info(SYMBOL)
        if info is None or info.custom:
            raise RuntimeError("Native US100.cash unavailable")
        prior_visible = bool(info.visible)
        if not mt5.symbol_select(SYMBOL, True):
            raise RuntimeError("Could not select native US100.cash")
        rows = []
        for monday in ("2024-01-08", "2024-07-15", "2024-12-16"):
            base = datetime.fromisoformat(monday)
            for kind, day, hour, minute in (("entry", base, 9, 30),
                                           ("exit", base + timedelta(days=3), 11, 0)):
                start = day.replace(hour=hour, minute=minute, tzinfo=NY)
                end = start + timedelta(minutes=15)
                ticks = mt5.copy_ticks_range(SYMBOL, start, end - timedelta(milliseconds=1),
                                             mt5.COPY_TICKS_ALL)
                row = {"monday": monday, "kind": kind,
                       "window_ny": start.isoformat() + "/" + end.isoformat(),
                       "api_error": str(mt5.last_error()) if ticks is None else None,
                       "count": 0 if ticks is None else int(len(ticks))}
                if ticks is not None and len(ticks):
                    valid = ticks[(ticks["bid"] > 0) & (ticks["ask"] >= ticks["bid"])]
                    row.update(first_msc=int(ticks["time_msc"][0]),
                               last_msc=int(ticks["time_msc"][-1]),
                               valid_count=int(len(valid)),
                               first_valid_spread=float(valid["ask"][0] - valid["bid"][0])
                               if len(valid) else None)
                rows.append(row)
        valid = sum(row["count"] > 0 and row.get("valid_count", 0) > 0 for row in rows)
        result = {"verdict": "PASS" if valid == 6 else "FAIL",
                  "valid_windows": valid, "planned_windows": 6,
                  "rows": rows,
                  "meaning": "Source availability only; no H94 native PnL or account replay."}
        OUT.write_text("H94 US100 2024 FTMO NATIVE SOURCE CHECK\n" +
                       json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(OUT.read_text(encoding="utf-8"))
    finally:
        if prior_visible is False:
            mt5.symbol_select(SYMBOL, False)
        mt5.shutdown()


if __name__ == "__main__":
    main()
