"""Export native FTMO XAGUSD quote M1 for source qualification only."""
from __future__ import annotations

import csv
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
COLS = ("minute_epoch_s", "bid_open", "bid_high", "bid_low", "bid_close",
        "ask_open", "ask_high", "ask_low", "ask_close",
        "first_tick_msc", "last_tick_msc", "tick_count")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def connect() -> bool:
    if not mt5.initialize(TERMINAL):
        raise RuntimeError(f"FTMO initialization failed: {mt5.last_error()}")
    account = mt5.account_info()
    if account is None or account.server != "FTMO-Demo" or account.trade_mode != 0:
        mt5.shutdown()
        raise RuntimeError("Connected account/server is not FTMO-Demo")
    info = mt5.symbol_info(SYMBOL)
    if info is None or info.custom:
        mt5.shutdown()
        raise RuntimeError("Native XAGUSD unavailable")
    was_visible = bool(info.visible)
    if not mt5.symbol_select(SYMBOL, True):
        mt5.shutdown()
        raise RuntimeError("Could not select XAGUSD")
    return was_visible


def request_month(start: pd.Timestamp, end: pd.Timestamp) -> tuple[object, list[str]]:
    errors = []
    raw = mt5.copy_ticks_range(SYMBOL, start.to_pydatetime(),
                               (end - pd.Timedelta(milliseconds=1)).to_pydatetime(),
                               mt5.COPY_TICKS_ALL)
    if raw is None:
        errors.append(str(mt5.last_error()))
        mt5.shutdown()
        connect()
        raw = mt5.copy_ticks_range(SYMBOL, start.to_pydatetime(),
                                   (end - pd.Timedelta(milliseconds=1)).to_pydatetime(),
                                   mt5.COPY_TICKS_ALL)
        if raw is None:
            errors.append(str(mt5.last_error()))
    return raw, errors


def aggregate_month(raw: object, start_ms: int, end_ms: int) -> tuple[pd.DataFrame | None, dict]:
    ms = raw["time_msc"]
    inside = (ms >= start_ms) & (ms < end_ms)
    finite = np.isfinite(raw["bid"]) & np.isfinite(raw["ask"])
    valid = finite & (raw["bid"] > 0) & (raw["ask"] >= raw["bid"])
    bad_order = int((np.diff(ms) < 0).sum())
    quality = {"out_of_range": int((~inside).sum()),
               "invalid_bid_ask": int((~valid).sum()),
               "unordered_adjacent": bad_order}
    if not valid.any() or bad_order:
        return None, quality
    q = raw[valid & inside]
    if len(q) == 0:
        return None, quality
    frame = pd.DataFrame({"minute_epoch_s": (q["time_msc"] // 60000) * 60,
                          "bid": q["bid"], "ask": q["ask"], "time_msc": q["time_msc"]})
    group = frame.groupby("minute_epoch_s", sort=True)
    bids = group["bid"].agg(["first", "max", "min", "last"])
    asks = group["ask"].agg(["first", "max", "min", "last"])
    times = group["time_msc"].agg(["first", "last", "size"])
    bars = pd.DataFrame({"minute_epoch_s": bids.index.astype("int64"),
                         "bid_open": bids["first"].to_numpy(),
                         "bid_high": bids["max"].to_numpy(),
                         "bid_low": bids["min"].to_numpy(),
                         "bid_close": bids["last"].to_numpy(),
                         "ask_open": asks["first"].to_numpy(),
                         "ask_high": asks["max"].to_numpy(),
                         "ask_low": asks["min"].to_numpy(),
                         "ask_close": asks["last"].to_numpy(),
                         "first_tick_msc": times["first"].to_numpy(),
                         "last_tick_msc": times["last"].to_numpy(),
                         "tick_count": times["size"].to_numpy()})
    return bars, quality


def coverage(rows: pd.DataFrame) -> tuple[list[dict], dict]:
    local = pd.to_datetime(rows.minute_epoch_s, unit="s", utc=True).dt.tz_convert("America/New_York")
    slots = set(zip(local.dt.date.astype(str), local.dt.strftime("%H:%M")))
    result = []
    for monday in pd.date_range("2024-01-01", "2026-09-21", freq="W-MON"):
        wed = monday + pd.Timedelta(days=2)
        thu = monday + pd.Timedelta(days=3)
        for leg, entry, exit_ in (
            ("MON_WED", (str(monday.date()), "09:30"), (str(wed.date()), "09:25")),
            ("WED_THU", (str(wed.date()), "09:30"), (str(thu.date()), "15:55")),
        ):
            result.append({"monday_ny": str(monday.date()), "year": int(monday.year),
                           "leg": leg, "entry_ny": " ".join(entry), "exit_ny": " ".join(exit_),
                           "entry_present": int(entry in slots), "exit_present": int(exit_ in slots),
                           "covered": int(entry in slots and exit_ in slots)})
    gated = result
    stats = {"planned_pairs": len(gated), "covered_pairs": sum(x["covered"] for x in gated),
             "covered_share": sum(x["covered"] for x in gated) / len(gated),
             "by_year": {str(year): {"planned": sum(x["year"] == year for x in gated),
                                     "covered": sum(x["year"] == year and x["covered"] for x in gated)}
                         for year in (2024, 2025, 2026)}}
    return result, stats


def main() -> None:
    m1_path = OUT / "xagusd_ftmo_native_m1_2023_2026.csv"
    manifest_path = OUT / "FULL_SOURCE_MANIFEST_2026-09-26.json"
    coverage_path = OUT / "FULL_SOURCE_COVERAGE_2026-09-26.csv"
    report_path = OUT / "FULL_SOURCE_REPORT_2026-09-26.txt"
    if any(path.exists() for path in (m1_path, manifest_path, coverage_path, report_path)):
        raise FileExistsError("Full source artifact already exists")
    prior_visible = connect()
    manifest = []
    try:
        with m1_path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLS)
            writer.writeheader()
            for start in pd.date_range("2023-01-01", "2026-09-01", freq="MS", tz="UTC"):
                end = min(start + pd.DateOffset(months=1), pd.Timestamp("2026-09-26", tz="UTC"))
                raw, errors = request_month(start, end)
                item = {"month_utc": start.strftime("%Y-%m"), "query_start_utc": start.isoformat(),
                        "query_end_utc": end.isoformat(), "api_errors": errors}
                if raw is None:
                    item.update(status="API_NONE", source_ticks=0)
                    manifest.append(item)
                    continue
                item["source_ticks"] = int(len(raw))
                item["raw_sha256"] = hashlib.sha256(raw.tobytes()).hexdigest()
                if len(raw) == 0:
                    item["status"] = "EMPTY"
                    manifest.append(item)
                    continue
                item["first_time_msc"] = int(raw["time_msc"][0])
                item["last_time_msc"] = int(raw["time_msc"][-1])
                bars, quality = aggregate_month(raw, int(start.timestamp() * 1000), int(end.timestamp() * 1000))
                item.update(quality)
                item["m1_rows"] = 0 if bars is None else len(bars)
                item["status"] = "OK" if bars is not None and not any(quality.values()) else "QUALITY_FAIL"
                if bars is not None:
                    writer.writerows(bars.to_dict("records"))
                manifest.append(item)
                print(f"{item['month_utc']} {item['status']} ticks={item['source_ticks']} M1={item['m1_rows']}", flush=True)
    finally:
        if not prior_visible:
            mt5.symbol_select(SYMBOL, False)
        mt5.shutdown()
    frame = pd.read_csv(m1_path)
    covered, counts = coverage(frame) if len(frame) else ([], {"planned_pairs": 0, "covered_pairs": 0, "covered_share": 0})
    with coverage_path.open("x", newline="", encoding="utf-8") as stream:
        if covered:
            writer = csv.DictWriter(stream, fieldnames=list(covered[0]))
            writer.writeheader()
            writer.writerows(covered)
    year = pd.to_datetime(frame.minute_epoch_s, unit="s", utc=True).dt.year if len(frame) else []
    spreads = {}
    for y in (2023, 2024, 2025, 2026):
        sample = (frame.ask_open - frame.bid_open)[year == y] if len(frame) else []
        spreads[str(y)] = {"m1_rows": len(sample),
                           "median": float(np.median(sample)) if len(sample) else None,
                           "p90": float(np.quantile(sample, 0.9)) if len(sample) else None}
    source_ok = len(manifest) == 45 and all(item["status"] == "OK" for item in manifest)
    verdict = "PASS" if source_ok and counts["covered_pairs"] >= 200 and counts["covered_share"] >= 0.95 else "FAIL"
    result = {"verdict": verdict, "months": manifest, "coverage": counts,
              "spread_by_utc_year": spreads,
              "m1_rows": len(frame), "m1_sha256": sha256(m1_path),
              "coverage_sha256": sha256(coverage_path)}
    manifest_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["FTMO XAGUSD FULL NATIVE M1 SOURCE VERDICT: " + verdict,
             f"Months={len(manifest)}; clean={sum(item['status']=='OK' for item in manifest)}; M1={len(frame)}",
             "Coverage: " + json.dumps(counts, sort_keys=True),
             "First-tick M1 spread silver units by year: " + json.dumps(spreads, sort_keys=True),
             "M1 SHA256: " + result["m1_sha256"],
             "No signal, price return, PnL or account order calculated."]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
