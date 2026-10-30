# -*- coding: utf-8 -*-
r"""
virgin_assets_onboarding.py : one-pass onboarding of the three virgin
Dukascopy-style M1 BID bar exports into standard M1 tapes under
data/results/virgin_assets/ (directory created by the script). Standalone,
argument-less CLI; every path is a constant below.

Sources (C:\Users\Milad\download, header timestamp,open,high,low,close,
epoch MILLISECONDS UTC, chronological order asserted, not assumed):
  - usatechidxusd-m1-bid-2020-01-01-2026-09-24.csv -> ustech_m1.csv
  - btcusd-m1-bid-2020-01-01-2026-09-24.csv        -> btc_m1.csv
  - ethusd-m1-bid-2020-01-01-2026-09-24.csv        -> eth_m1.csv

These are BAR exports (one row per vendor minute, flat minutes included),
not tick exports: no ask/bid spread exists, so NO cost floor is measured
here (cost floors for the #54 study are owner-supplied estimates), and no
tick volume exists (tick_volume = 0 written to preserve the standard
schema; declared, never interpreted downstream).

Spec areas (per instrument, mirroring the eurusd_onboarding discipline):
1  Integrity. Monotonic timestamp inversions vs previous raw row are
   tallied and HARD-ASSERTED post-pass (exit 2, no artifacts). Non-positive
   OHLC rows (any of open/high/low/close <= 0) and malformed bars
   (high < low, high < open, high < close, low > open, low > close) are
   excluded and tallied. Exact same-ts duplicates (same timestamp AND same
   OHLC as previous raw row) are dropped and tallied; conflicting same-ts
   rows are kept, tallied, and resolved by slot aggregation
   (first/max/min/last).
2  Gaps and weekend rule. Consecutive kept-minute deltas >= 3h are
   classified: gap is weekend closure iff total extent outside
   [Fri 17:00, Sun 17:00] America/New_York (DST-aware) <= 2h. Weekend gaps
   counted only; every non-weekend gap recorded (UTC/NY endpoints, hours,
   weekday). (BTCUSD/ETHUSD trade 24/7, so USTECH closures are expected to
   dominate the weekend bucket and crypto gaps to appear as non-weekend.)
3  Aggregation. M1 slots keyed by ts//60000 -> time = slot*60 (bucket open
   UTC epoch seconds), open=first, high=max, low=min, close=last over kept
   rows; tick_volume = 0 (declared above). Post-pass HARD ASSERT: M5 built
   from the WRITTEN M1 tape must equal M5 built directly from kept source
   rows (row count and OHLC), else exit 3.
4  Artifacts (data/results/virgin_assets/): ustech_m1.csv, btc_m1.csv,
   eth_m1.csv (columns time,open,high,low,close,tick_volume; index off;
   default float formatting = lossless round-trip), onboarding_stats.json,
   REPORT_VIRGIN_ONBOARDING.txt (plain text, no markdown).

Full run:

  & "C:\Users\Milad\miniconda3\envs\reinforce_rl\python.exe" evaluate\virgin_assets_onboarding.py
"""

import hashlib
import json
import os
import time as timemod
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = [
    dict(asset="USTECH", src=Path(r"C:\Users\Milad\download\usatechidxusd-m1-bid-2020-01-01-2026-09-24.csv"), out="ustech_m1.csv"),
    dict(asset="BTCUSD", src=Path(r"C:\Users\Milad\download\btcusd-m1-bid-2020-01-01-2026-09-24.csv"), out="btc_m1.csv"),
    dict(asset="ETHUSD", src=Path(r"C:\Users\Milad\download\ethusd-m1-bid-2020-01-01-2026-09-24.csv"), out="eth_m1.csv"),
]
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "data" / "results" / "virgin_assets"
GAP_MIN = 180
WEEKEND_SLACK_MS = 2 * 3600 * 1000
NY = "America/New_York"
WEEKEND_RULE_TEXT = ("gap is weekend closure iff total extent outside "
                     "[Fri 17:00, Sun 17:00] America/New_York (DST-aware) "
                     "<= 2h")
DTYPES = {"timestamp": "int64", "open": "float64", "high": "float64",
          "low": "float64", "close": "float64"}


def sha256_of(path, block=1 << 26):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def classify_weekend(a_ms, b_ms):
    ta = pd.Timestamp(a_ms, unit="ms", tz="UTC").tz_convert(NY)
    days_back = (ta.dayofweek - 4) % 7
    dd = ta.date() - timedelta(days=days_back)
    f = pd.Timestamp(dd.year, dd.month, dd.day, 17, tz=NY)
    sun = dd + timedelta(days=2)
    s = pd.Timestamp(sun.year, sun.month, sun.day, 17, tz=NY)
    fms = f.timestamp() * 1000.0
    sms = s.timestamp() * 1000.0
    excess = max(0.0, fms - a_ms) + max(0.0, b_ms - sms)
    return (b_ms > fms) and (a_ms < sms) and (excess <= WEEKEND_SLACK_MS)


def onboard(inst):
    asset, src = inst["asset"], inst["src"]
    if not src.exists() or src.stat().st_size == 0:
        raise SystemExit(f"missing or empty source file: {src}")
    t0 = timemod.time()
    df = pd.read_csv(src, dtype=DTYPES, usecols=list(DTYPES))
    raw_total = len(df)
    ts = df["timestamp"].to_numpy(np.int64)
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)

    d = np.empty(raw_total, np.int64)
    d[1:] = ts[1:] - ts[:-1]
    d[0] = 1
    mono_viol = int((d < 0).sum())
    if mono_viol > 0:
        print(f"{asset}: HARD ASSERT 1 FAILED: monotonic violations = {mono_viol}")
        raise SystemExit(2)

    eq = d == 0
    drop = np.zeros(raw_total, bool)
    dup_conflict = 0
    dup_exact = 0
    if eq.any():
        same = eq & (np.roll(o, 1) == o) & (np.roll(h, 1) == h) & (np.roll(l, 1) == l) & (np.roll(c, 1) == c)
        same[0] = False
        dup_exact = int(same.sum())
        dup_conflict = int((eq & ~same).sum())
        drop = same

    positive = (o > 0) & (h > 0) & (l > 0) & (c > 0)
    bad_px = int((~positive).sum())
    well_formed = (h >= l) & (h >= o) & (h >= c) & (l <= o) & (l <= c)
    malformed = int(positive.sum() - (positive & well_formed).sum())
    keep = positive & well_formed & ~drop
    kept_total = int(keep.sum())

    slot = ts[keep] // 60_000
    gk = pd.DataFrame({"k": slot, "o": o[keep], "h": h[keep], "l": l[keep], "c": c[keep]}).groupby("k", sort=True)
    m1 = pd.DataFrame({
        "time": gk.size().index.to_numpy(np.int64) * 60,
        "open": gk.first()["o"].to_numpy(),
        "high": gk.max()["h"].to_numpy(),
        "low": gk.min()["l"].to_numpy(),
        "close": gk.last()["c"].to_numpy(),
    })
    rows_per_slot = gk.size().to_numpy()
    slots_with_multi = int((rows_per_slot > 1).sum())

    s1 = m1["time"].to_numpy(np.int64)
    dk = np.empty(len(s1), np.int64)
    dk[1:] = s1[1:] - s1[:-1]
    dk[0] = 0
    gm = dk >= (GAP_MIN * 60)
    gaps = [(int(s1[i - 1]), int(s1[i])) for i in np.flatnonzero(gm)] if gm.any() else []

    weekend_count = 0
    non_weekend = []
    for a_s, b_s in sorted(gaps):
        a_ms, b_ms = a_s * 1000, (b_s - 60) * 1000
        if classify_weekend(a_ms, b_ms):
            weekend_count += 1
            continue
        ta = pd.Timestamp(a_s, unit="s", tz="UTC").tz_convert(NY)
        tb = pd.Timestamp(b_s, unit="s", tz="UTC").tz_convert(NY)
        non_weekend.append({
            "start_utc": a_s, "end_utc": b_s,
            "hours": round((b_s - a_s) / 3600.0, 2),
            "start_ny": str(ta), "end_ny": str(tb),
            "start_weekday": ta.day_name()})

    ny = pd.DatetimeIndex(pd.to_datetime(s1, unit="s", utc=True)).tz_convert(NY)
    per_year = {}
    uy, cy = np.unique(ny.year.to_numpy(), return_counts=True)
    for y, cnt in zip(uy, cy):
        per_year[int(y)] = int(cnt)

    m1 = m1.assign(tick_volume=0)
    m1_path = OUT / inst["out"]
    m1.to_csv(m1_path, index=False)

    chk = pd.read_csv(m1_path, usecols=["time", "open", "high", "low", "close"])
    chk["t5"] = (chk["time"].to_numpy(np.int64) // 300) * 300
    r5 = chk.groupby("t5").agg(open=("open", "first"), high=("high", "max"),
                               low=("low", "min"), close=("close", "last")).reset_index()
    direct = pd.DataFrame({"t5": (slot * 60 // 300) * 300, "o": o[keep], "h": h[keep], "l": l[keep], "c": c[keep]})
    d5 = direct.groupby("t5").agg(open=("o", "first"), high=("h", "max"), low=("l", "min"), close=("c", "last")).reset_index()
    mism = 0
    if len(r5) != len(d5):
        mism += abs(len(r5) - len(d5)) * 4
    else:
        mg = r5.merge(d5, on="t5", suffixes=("_w", "_d"))
        for col in ("open", "high", "low", "close"):
            mism += int((mg[col + "_w"] != mg[col + "_d"]).sum())
    if mism > 0:
        print(f"{asset}: HARD ASSERT 2 FAILED: M1->M5 vs direct M5 mismatches = {mism}")
        raise SystemExit(3)

    first_utc = pd.Timestamp(int(s1[0]), unit="s", tz="UTC")
    last_utc = pd.Timestamp(int(s1[-1]), unit="s", tz="UTC")
    stats = {
        "symbol": asset,
        "source": {"file": str(src), "size_bytes": int(src.stat().st_size),
                   "sha256": sha256_of(src), "rows": int(raw_total)},
        "integrity": {"monotonic_violations": mono_viol,
                      "dup_exact_dropped": dup_exact,
                      "dup_conflicting_aggregated": dup_conflict,
                      "slots_with_multiple_rows": slots_with_multi,
                      "bad_price_rows": bad_px,
                      "malformed_ohlc_rows": malformed},
        "bars": {"kept_minutes": kept_total, "m1_rows_written": int(len(m1))},
        "first_bar_utc": str(first_utc), "last_bar_utc": str(last_utc),
        "gaps": {"threshold_hours": 3.0, "weekend_rule": WEEKEND_RULE_TEXT,
                 "weekend_excluded_count": weekend_count,
                 "non_weekend_count": len(non_weekend),
                 "non_weekend": non_weekend},
        "bars_per_ny_year": per_year,
        "conventions": {"price": "Dukascopy M1 BID bars, no ask side -> no spread measurement, cost floors are study-level owner estimates",
                        "tick_volume": "0 (vendor export carries no volume; schema-preserving placeholder)"},
        "m1_m5_consistency": {"m5_from_written": int(len(r5)), "m5_direct": int(len(d5)),
                              "value_mismatches": int(mism), "asserted_identical": True},
        "elapsed_s": round(timemod.time() - t0, 1),
    }
    return stats, m1_path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_stats = {}
    paths = {}
    for inst in SOURCES:
        stats, m1_path = onboard(inst)
        all_stats[inst["asset"]] = stats
        paths[inst["asset"]] = m1_path
        print(f"[{inst['asset']}] {stats['source']['rows']:,} raw rows -> {stats['bars']['m1_rows_written']:,} M1 bars "
              f"({stats['first_bar_utc']} .. {stats['last_bar_utc']}); gaps w/e={stats['gaps']['weekend_excluded_count']} "
              f"non-w/e={stats['gaps']['non_weekend_count']}; {stats['elapsed_s']}s", flush=True)

    with open(OUT / "onboarding_stats.json", "w") as fh:
        json.dump(all_stats, fh, indent=1)

    rep = []
    rep.append("VIRGIN ASSETS ONBOARDING REPORT (USTECH / BTCUSD / ETHUSD)")
    rep.append(f"generated_utc: {pd.Timestamp.now(tz='UTC').isoformat()}")
    rep.append("tool: evaluate/virgin_assets_onboarding.py")
    rep.append("")
    for inst in SOURCES:
        s = all_stats[inst["asset"]]
        rep.append(f"=== {inst['asset']} ===")
        rep.append(f"source: {s['source']['file']}")
        rep.append(f"size_bytes: {s['source']['size_bytes']:,}  sha256: {s['source']['sha256']}")
        rep.append(f"raw rows: {s['source']['rows']:,}  kept minutes: {s['bars']['kept_minutes']:,}  M1 written: {s['bars']['m1_rows_written']:,}")
        rep.append(f"first bar {s['first_bar_utc']}   last bar {s['last_bar_utc']}")
        rep.append("integrity: monotonic violations=%d (HARD ASSERT 1 PASSED), exact dups dropped=%d, conflicting same-ts aggregated=%d (slots affected=%d), bad price rows=%d, malformed OHLC rows=%d"
                   % (s["integrity"]["monotonic_violations"], s["integrity"]["dup_exact_dropped"],
                      s["integrity"]["dup_conflicting_aggregated"], s["integrity"]["slots_with_multiple_rows"],
                      s["integrity"]["bad_price_rows"], s["integrity"]["malformed_ohlc_rows"]))
        rep.append("bars per NY year: " + ", ".join(f"{y}: {n:,}" for y, n in sorted(s["bars_per_ny_year"].items())))
        rep.append(f"gaps >= 3h: weekend closures excluded={s['gaps']['weekend_excluded_count']}, non-weekend={s['gaps']['non_weekend_count']}")
        for g in s["gaps"]["non_weekend"]:
            rep.append(f"   {pd.Timestamp(g['start_utc'], unit='s', tz='UTC')} -> {pd.Timestamp(g['end_utc'], unit='s', tz='UTC')}  {g['hours']}h  start {g['start_weekday']}  NY {g['start_ny']}")
        if not s["gaps"]["non_weekend"]:
            rep.append("   none")
        rep.append(f"M1->M5 vs direct-M5 HARD ASSERT 2 PASSED (rows {s['m1_m5_consistency']['m5_from_written']:,}, mismatches 0)")
        rep.append(f"conventions: {s['conventions']['price']}; tick_volume={s['conventions']['tick_volume']}")
        rep.append(f"artifact: {paths[inst['asset']]}")
        rep.append("")
    (OUT / "REPORT_VIRGIN_ONBOARDING.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
