"""Read-only, preplanned FTMO historical quote spread observations."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "data/results/project_status/FTMO_SPREAD_STRATIFIED_PLAN_2026-09-25.txt"
OUT = ROOT / "data/results/project_status/ftmo_spread_stratified_20260925"
TERMINAL = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
SYMBOLS = ("EURUSD", "XAUUSD", "US500.cash")
LONDON = ZoneInfo("Europe/London")
NEW_YORK = ZoneInfo("America/New_York")
BROKER = ZoneInfo("Europe/Athens")


def first_wednesday(year: int, month: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(2 - first.weekday()) % 7)


def planned_dates():
    for year in (2024, 2025, 2026):
        for month in range(1, 13 if year < 2026 else 9):
            yield first_wednesday(year, month)


def sample_one(symbol: str, label: str, local_target: datetime) -> dict:
    target = local_target.astimezone(timezone.utc)
    broker_wall = target.astimezone(BROKER).replace(tzinfo=timezone.utc)
    offset_seconds = int(broker_wall.timestamp() - target.timestamp())
    window_end = broker_wall + timedelta(seconds=10)
    ticks = mt5.copy_ticks_range(symbol, broker_wall, window_end, mt5.COPY_TICKS_ALL)
    row = {"sample_date": local_target.date().isoformat(), "window": label,
           "symbol": symbol, "target_utc": target.isoformat(),
           "broker_offset_hours": offset_seconds / 3600,
           "status": "MISSING", "raw_broker_tick_ms": "", "corrected_tick_utc": "",
           "seconds_after_target": "", "bid": "", "ask": "", "spread": "",
           "invalid_quotes": 0}
    if ticks is None:
        row["status"] = "API_ERROR"
        return row
    for tick in ticks:
        ms = int(tick["time_msc"])
        delta = ms / 1000 - broker_wall.timestamp()
        if delta < 0 or delta > 10:
            continue
        bid, ask = float(tick["bid"]), float(tick["ask"])
        if not (math.isfinite(bid) and math.isfinite(ask) and 0 < bid <= ask):
            row["invalid_quotes"] += 1
            continue
        corrected = datetime.fromtimestamp(ms / 1000 - offset_seconds, timezone.utc)
        row.update(status="OK", raw_broker_tick_ms=ms,
                   corrected_tick_utc=corrected.isoformat(timespec="milliseconds"),
                   seconds_after_target=round(delta, 3), bid=bid, ask=ask,
                   spread=round((ask - bid) * (10_000 if symbol == "EURUSD" else 1), 5))
        break
    return row


def nearest_rank(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * quantile) - 1)]


def main() -> None:
    if not PLAN.is_file() or OUT.exists():
        raise FileExistsError("Predeclared plan missing or historical sample already exists")
    if not mt5.initialize(TERMINAL):
        raise RuntimeError("Cannot connect to target FTMO terminal")
    try:
        account = mt5.account_info()
        if account is None or account.server != "FTMO-Demo" or "ftmo" not in str(account.company).lower():
            raise RuntimeError("Unexpected connected broker/server")
        original_visibility = {}
        for symbol in SYMBOLS:
            info = mt5.symbol_info(symbol)
            if info is None or info.custom:
                raise RuntimeError("Missing or custom broker symbol: " + symbol)
            original_visibility[symbol] = bool(info.visible)
            if not mt5.symbol_select(symbol, True):
                raise RuntimeError("Could not select symbol: " + symbol)
        try:
            rows = []
            for sample_date in planned_dates():
                clocks = (
                    ("london_0900", datetime(sample_date.year, sample_date.month,
                                             sample_date.day, 9, 0, tzinfo=LONDON)),
                    ("ny_0930", datetime(sample_date.year, sample_date.month,
                                         sample_date.day, 9, 30, tzinfo=NEW_YORK)),
                    ("ny_1555", datetime(sample_date.year, sample_date.month,
                                         sample_date.day, 15, 55, tzinfo=NEW_YORK)),
                )
                for label, clock in clocks:
                    for symbol in SYMBOLS:
                        current = mt5.account_info()
                        if current is None or current.login != account.login or current.server != account.server:
                            raise RuntimeError("Connected account changed; stop sampling")
                        rows.append(sample_one(symbol, label, clock))
        finally:
            current = mt5.account_info()
            if current is not None and current.login == account.login:
                for symbol, visible in original_visibility.items():
                    if not visible:
                        mt5.symbol_select(symbol, False)
    finally:
        mt5.shutdown()
    if len(rows) != 32 * 3 * 3:
        raise ValueError("Unexpected predeclared sample roster size")
    OUT.mkdir(parents=True)
    csv_path = OUT / "observations.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = {}
    for row in rows:
        key = (row["symbol"], row["window"])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (symbol, window), group in sorted(grouped.items()):
        values = [float(r["spread"]) for r in group if r["status"] == "OK"]
        summary.append({"symbol": symbol, "window": window,
                        "planned": len(group), "valid": len(values),
                        "missing": len(group) - len(values),
                        "median_spread": median(values) if values else None,
                        "p90_spread": nearest_rank(values, 0.90) if values else None,
                        "max_spread": max(values) if values else None})
    payload = {"plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
               "observations_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
               "planned_windows": len(rows), "valid_windows": sum(r["status"] == "OK" for r in rows),
               "status_counts": {status: sum(r["status"] == status for r in rows)
                                 for status in sorted({r["status"] for r in rows})},
               "spread_units": {"EURUSD": "pips", "XAUUSD": "gold price units",
                                "US500.cash": "index points"},
               "summary": summary,
               "caveat": "Sparse historical first-tick spot sample, not full spread distribution, current quote, fill, slippage, commission or strategy evidence."}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
