"""Read-only chronological M1 integrity audit of frozen H22 US500 longs."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/project_status"
PLAN = OUT / "H22_US500_SAME_BAR_LIMIT_SEQUENCE_PLAN_2026-09-25.txt"
TRADES = ROOT / "data/results/youtube_hybrid_scalping/hybrid_scalp_trades.csv"
M1 = ROOT / "data/results/vwap_2021_2026/us500tick_m1.csv"
M5 = ROOT / "data/results/orb_2021_present/us500tick_m5.csv"
DETAIL = OUT / "H22_US500_SAME_BAR_LIMIT_SEQUENCE_DETAIL_2026-09-25.csv"
REPORT = OUT / "H22_US500_SAME_BAR_LIMIT_SEQUENCE_AUDIT_2026-09-25.txt"
NY = ZoneInfo("America/New_York")
EPS = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(row, minutes, times):
    start = int(row.entry_time)
    lo = np.searchsorted(times, start, side="left")
    hi = np.searchsorted(times, start + 300, side="left")
    bars = minutes.iloc[lo:hi]
    if bars.empty:
        return "NO_REAL_MINUTE", 0, 0, -1, -1
    lows = bars.low.to_numpy(float)
    highs = bars.high.to_numpy(float)
    entry = float(row.entry_price)
    target = float(row.exit_price)
    stop = entry - 1.5 * float(row.atr_ref)
    ent = np.flatnonzero(lows <= entry + EPS)
    tar = np.flatnonzero(highs >= target - EPS)
    if len(ent) == 0:
        return "NO_ENTRY_TOUCH", len(bars), len(tar), -1, int(tar[0]) if len(tar) else -1
    first_ent = int(ent[0])
    if len(tar) == 0:
        return "NO_TARGET_TOUCH", len(bars), 0, first_ent, -1
    after = tar[tar >= first_ent]
    if len(after) == 0:
        return "TARGET_BEFORE_ENTRY", len(bars), len(tar), first_ent, int(tar[0])
    later = after[after > first_ent]
    if len(later) == 0:
        return "SAME_MINUTE_ONLY", len(bars), len(tar), first_ent, int(after[0])
    first_later = int(later[0])
    if bool(np.any(lows[first_ent:first_later + 1] <= stop + EPS)):
        return "LATER_MINUTE_WITH_STOP_RISK", len(bars), len(tar), first_ent, first_later
    return "LATER_MINUTE_PLAUSIBLE", len(bars), len(tar), first_ent, first_later


def main():
    if not PLAN.exists() or DETAIL.exists() or REPORT.exists():
        raise RuntimeError("Plan missing or immutable output already exists")
    trades = pd.read_csv(TRADES)
    m1 = pd.read_csv(M1)
    m5 = pd.read_csv(M5)
    for name, frame in (("M1", m1), ("M5", m5)):
        if not frame.time.is_monotonic_increasing or frame.time.duplicated().any():
            raise ValueError(name + " timestamps are not unique and increasing")
    all_longs = trades[(trades.instrument == "US500") & (trades.model == "Long")].copy()
    same = all_longs[all_longs.entry_time == all_longs.exit_time].copy()
    same_targets = same[same.exit_reason == "target"].copy()
    same_stops = same[same.exit_reason == "stop"].copy()
    minute_times = m1.time.to_numpy(np.int64)
    m5_times = m5.time.to_numpy(np.int64)
    rows = []
    source_match = 0
    for row in same_targets.itertuples(index=False):
        index = int(np.searchsorted(m5_times, int(row.entry_time)))
        if index < len(m5_times) and m5_times[index] == int(row.entry_time):
            source_match += 1
        result, n_minutes, n_targets, entry_minute, target_minute = classify(row, m1, minute_times)
        ny = datetime.fromtimestamp(int(row.entry_time), timezone.utc).astimezone(NY)
        wall = ny.hour * 60 + ny.minute
        rows.append({"entry_time": int(row.entry_time), "ny_date": str(int(row.ny_date)),
                     "year": ny.year, "cash": int(575 <= wall <= 955),
                     "class": result, "real_minutes": n_minutes,
                     "target_minutes": n_targets, "first_entry_minute": entry_minute,
                     "first_eligible_target_minute": target_minute,
                     "frozen_net_pts": float(row.pnl_net_pts)})
    with DETAIL.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(r["class"] for r in rows)
    cash_counts = Counter(r["class"] for r in rows if r["cash"])
    year_counts = defaultdict(Counter)
    net = defaultdict(float)
    for r in rows:
        year_counts[r["year"]][r["class"]] += 1
        net[r["class"]] += r["frozen_net_pts"]
    lines = [
        "H22 US500 LONG LIMIT SAME-BAR CHRONOLOGY AUDIT - 2026-09-25",
        "READ-ONLY; NOT A CORRECTED BACKTEST OR TRADE RECOMMENDATION",
        "Plan SHA256: " + sha256(PLAN),
        "Frozen H22 trade CSV SHA256: " + sha256(TRADES),
        "Verified M1 SHA256: " + sha256(M1),
        "Verified M5 SHA256: " + sha256(M5),
        f"All US500 Long trades: {len(all_longs)}; same-M5-bar exits: {len(same)} "
        f"({len(same)/len(all_longs):.2%}); same-bar target: {len(same_targets)}; "
        f"same-bar stop: {len(same_stops)}.",
        f"Same-bar targets with exact M5 source timestamp: {source_match}/{len(same_targets)}.",
        f"Same-bar targets with fewer than 5 real M1 bars: "
        f"{sum(r['real_minutes'] < 5 for r in rows)}.",
        "All same-bar target classes (count; frozen net-point contribution):",
    ]
    for key in sorted(counts):
        lines.append(f"  {key}: {counts[key]}; {net[key]:+.3f}")
    lines.append(f"Cash session 09:35..15:55 NY targets: {sum(r['cash'] for r in rows)}")
    for key in sorted(cash_counts):
        lines.append(f"  {key}: {cash_counts[key]}")
    lines.append("Year counts:")
    for year, bins in sorted(year_counts.items()):
        lines.append(f"  {year}: " + ", ".join(f"{key}={bins[key]}" for key in sorted(bins)))
    lines.extend([
        "Chronological M1 plausibility is only necessary, never sufficient: within-minute",
        "ordering, actual ASK limit fill, queue/slippage, commission and FTMO prop",
        "limits require a distinct executable native tick study. Original H22 frozen",
        "SPEC/GATE/REPORT and verdict have not been modified.",
        "Detail SHA256: " + sha256(DETAIL),
    ])
    with REPORT.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
