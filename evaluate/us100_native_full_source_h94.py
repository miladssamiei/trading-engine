"""Full FTMO US100 native quote source for H94, no returns or orders."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import xagusd_native_m1_full_source as shared


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/us100_native_full_source_h94"
SYMBOL = "US100.cash"
shared.SYMBOL = SYMBOL


def coverage(frame: pd.DataFrame) -> tuple[list[dict], dict]:
    ny = pd.to_datetime(frame.minute_epoch_s, unit="s", utc=True).dt.tz_convert("America/New_York")
    minutes = set(zip(ny.dt.date.astype(str), ny.dt.strftime("%H:%M")))
    rows = []
    for monday in pd.date_range("2024-01-01", "2026-09-21", freq="W-MON"):
        entry = (str(monday.date()), "09:30")
        exit_ = (str((monday + pd.Timedelta(days=3)).date()), "11:00")
        rows.append({"monday_ny": str(monday.date()), "year": int(monday.year),
                     "entry_present": int(entry in minutes), "exit_present": int(exit_ in minutes),
                     "covered": int(entry in minutes and exit_ in minutes)})
    stats = {"planned": len(rows), "covered": sum(x["covered"] for x in rows),
             "share": sum(x["covered"] for x in rows) / len(rows),
             "by_year": {str(y): {"planned": sum(x["year"] == y for x in rows),
                                  "covered": sum(x["year"] == y and x["covered"] for x in rows)}
                         for y in (2024, 2025, 2026)}}
    return rows, stats


def main() -> None:
    m1 = OUT / "us100cash_ftmo_native_m1_2023_2026.csv"
    manifest_path = OUT / "SOURCE_MANIFEST_2026-09-26.json"
    coverage_path = OUT / "SOURCE_COVERAGE_2026-09-26.csv"
    report_path = OUT / "SOURCE_REPORT_2026-09-26.txt"
    if any(path.exists() for path in (m1, manifest_path, coverage_path, report_path)):
        raise FileExistsError("US100 full source artifact already exists")
    OUT.mkdir(parents=True, exist_ok=True)
    prior_visible = shared.connect()
    manifest = []
    try:
        with m1.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=shared.COLS)
            writer.writeheader()
            for start in pd.date_range("2023-01-01", "2026-09-01", freq="MS", tz="UTC"):
                end = min(start + pd.DateOffset(months=1), pd.Timestamp("2026-09-26", tz="UTC"))
                raw, errors = shared.request_month(start, end)
                item = {"month_utc": start.strftime("%Y-%m"), "query_start_utc": start.isoformat(),
                        "query_end_utc": end.isoformat(), "api_errors": errors}
                if raw is None:
                    item.update(status="API_NONE", source_ticks=0)
                    manifest.append(item)
                    print(f"{item['month_utc']} API_NONE", flush=True)
                    continue
                item["source_ticks"] = int(len(raw))
                item["raw_sha256"] = hashlib.sha256(raw.tobytes()).hexdigest()
                if len(raw) == 0:
                    item["status"] = "EMPTY"
                    manifest.append(item)
                    print(f"{item['month_utc']} EMPTY", flush=True)
                    continue
                item["first_time_msc"] = int(raw["time_msc"][0])
                item["last_time_msc"] = int(raw["time_msc"][-1])
                bars, quality = shared.aggregate_month(raw, int(start.timestamp() * 1000),
                                                       int(end.timestamp() * 1000))
                item.update(quality)
                item["m1_rows"] = 0 if bars is None else int(len(bars))
                item["status"] = "OK" if bars is not None and not any(quality.values()) else "QUALITY_FAIL"
                if bars is not None:
                    writer.writerows(bars.to_dict("records"))
                manifest.append(item)
                print(f"{item['month_utc']} {item['status']} ticks={item['source_ticks']} M1={item['m1_rows']}",
                      flush=True)
    finally:
        if not prior_visible:
            mt5.symbol_select(SYMBOL, False)
        mt5.shutdown()
    frame = pd.read_csv(m1)
    rows, stats = coverage(frame) if len(frame) else ([], {"planned": 0, "covered": 0, "share": 0})
    with coverage_path.open("x", newline="", encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    utc_year = pd.to_datetime(frame.minute_epoch_s, unit="s", utc=True).dt.year if len(frame) else []
    spreads = {}
    for y in (2023, 2024, 2025, 2026):
        sample = (frame.ask_open - frame.bid_open)[utc_year == y] if len(frame) else []
        spreads[str(y)] = {"m1_rows": len(sample),
                           "median": float(np.median(sample)) if len(sample) else None,
                           "p90": float(np.quantile(sample, 0.9)) if len(sample) else None}
    clean = len(manifest) == 45 and all(item["status"] == "OK" for item in manifest)
    verdict = "PASS" if clean and stats["covered"] >= 135 and stats["share"] >= 0.95 else "FAIL"
    result = {"verdict": verdict, "months": manifest, "coverage": stats,
              "first_tick_m1_spread_by_utc_year": spreads,
              "m1_rows": len(frame), "m1_sha256": shared.sha256(m1),
              "coverage_sha256": shared.sha256(coverage_path)}
    manifest_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["H94 FTMO US100.CASH FULL NATIVE SOURCE: " + verdict,
             f"Months={len(manifest)} clean={sum(item['status']=='OK' for item in manifest)} M1={len(frame)}",
             "Coverage: " + json.dumps(stats, sort_keys=True),
             "First-tick M1 spreads: " + json.dumps(spreads, sort_keys=True),
             "M1 SHA256: " + result["m1_sha256"],
             "No trade signal, return, stop, account order or PnL calculated."]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
