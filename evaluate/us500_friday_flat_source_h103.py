"""Check cash endpoint coverage for a Friday-flat frozen signal; no returns."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/us500_friday_flat_reversal_h103"
TRADES = ROOT / "data/results/short_term_reversal/g2_trades.csv"
M5 = ROOT / "data/results/orb_2021_present/us500tick_m5.csv"
HASHES = {
    TRADES: "9046835d29af192f98426b086e3299f9fb02adb3b5f452fc973b625ce595d214",
    M5: "a8ee68f1a0010a1be9704c0d59fd5382aa798846cdae4d519b799ae21ec66c2d",
}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    for path, expected in HASHES.items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen source hash mismatch: {path}")
    output = OUT / "SOURCE_COVERAGE_2026-09-26.csv"
    report = OUT / "SOURCE_REPORT_2026-09-26.txt"
    if output.exists() or report.exists():
        raise FileExistsError("Source artifact already exists")
    trades = pd.read_csv(TRADES, usecols=["entry", "exit"])
    ticks = pd.read_csv(M5, usecols=["time"])
    local = pd.to_datetime(ticks["time"], unit="s", utc=True).dt.tz_convert("America/New_York")
    date_clock = set(zip(local.dt.date.astype(str), local.dt.strftime("%H:%M")))
    rows = []
    for index, trade in trades.iterrows():
        start = pd.Timestamp(trade.entry, tz="UTC") + pd.Timedelta(days=1)
        end = pd.Timestamp(trade.exit, tz="UTC") + pd.Timedelta(days=1)
        if end <= start:
            continue
        start_date = start.tz_convert("America/New_York").date()
        end_date = end.tz_convert("America/New_York").date()
        for friday in pd.date_range(start_date, end_date, freq="W-FRI"):
            local_friday = friday.tz_localize("America/New_York") + pd.Timedelta(hours=15, minutes=55)
            friday_utc = local_friday.tz_convert("UTC")
            if not (start <= friday_utc < end):
                continue
            day = str(friday.date())
            friday_present = (day, "15:55") in date_clock
            next_day = None
            for days_after in (3, 4):
                candidate = str((friday + pd.Timedelta(days=days_after)).date())
                if (candidate, "09:30") in date_clock:
                    next_day = candidate
                    break
            rows.append({"trade_index": int(index), "entry_signal_date": str(trade.entry),
                         "exit_signal_date": str(trade.exit), "friday_ny_date": day,
                         "friday_1555_present": int(friday_present),
                         "next_cash_date": next_day or "",
                         "next_cash_0930_present_by_tuesday": int(next_day is not None),
                         "status": "COVERED" if friday_present and next_day else "MISSING_ENDPOINT"})
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else
                                ["trade_index", "friday_ny_date", "status"])
        writer.writeheader()
        writer.writerows(rows)
    gated = [row for row in rows if "2024-01-01" <= row["friday_ny_date"] <= "2026-04-30"]
    missing = [row for row in gated if row["status"] != "COVERED"]
    verdict = "PASS" if not missing and gated else "FAIL"
    years = {year: {"fridays": sum(row["friday_ny_date"].startswith(year) for row in gated),
                    "missing": sum(row["friday_ny_date"].startswith(year) for row in missing)}
             for year in ("2024", "2025", "2026")}
    lines = ["US500 FRIDAY-FLAT SOURCE VERDICT: " + verdict,
             f"Frozen trades: {len(trades)}; all active-Friday endpoints: {len(rows)}",
             f"Gated 2024-26 endpoints: {len(gated)}; missing: {len(missing)}",
             f"Gated active trades: {len({row['trade_index'] for row in gated})}",
             f"Year counts: {years}",
             "Coverage only: no OHLC values, returns, PnL or orders calculated.",
             "Coverage CSV SHA256: " + sha256(output)]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
