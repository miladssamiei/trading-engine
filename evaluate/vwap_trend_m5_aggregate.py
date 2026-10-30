# -*- coding: utf-8 -*-
"""
vwap_trend_m5_aggregate.py — data prep for the VWAP trend-pullback signal
study (NOT part of Phase 1 analysis; runs before it, per GATE.md).

The ORB-verified M5 export (data/results/orb_2021_present/us500tick_m5.csv,
364,330 bars, 64/64 months clean) carries NO volume column, and the session
VWAP needs tick_volume. Instead of re-pulling M5 from the terminal (whose
per-month M5 range queries short-served 2021-01 during the VWAP probe), this
script aggregates the ALREADY-VERIFIED full-range M1 export
(data/results/vwap_2021_2026/us500tick_m1.csv — 1,791,504 bars, 64/64
months, 0 multi-day gaps, date set identical to the M5 reference) into M5
bars:

    bucket  = floor(t/300)*300   (bar-open UTC epoch)
    open    = first M1 open in bucket      (CSV is strictly time-ordered)
    high/low= max/min of M1 highs/lows
    close   = last M1 close in bucket
    volume  = sum of M1 tick_volume  (each M1 tick_volume = ticks in that
                                       minute, so the sum = ticks in the
                                       5-min window = native M5 tick_volume)

VERIFICATION (gate-relevant, declared in GATE.md): the aggregated OHLC is
compared bar-for-bar against ORB's terminal-native M5 export at 1e-6
tolerance. Mismatch rate > 0.05% of common bars aborts Phase 1.

Output: data/results/vwap_trend_pullback_signal_quality/us500tick_m5_vol.csv
        (time,open,high,low,close,tick_volume)
"""

import csv
from pathlib import Path

M1_CSV = Path(r"C:\Users\Milad\Desktop\RR\data\results\vwap_2021_2026\us500tick_m1.csv")
ORB_M5 = Path(r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present\us500tick_m5.csv")
OUT_DIR = Path(r"C:\Users\Milad\Desktop\RR\data\results\vwap_trend_pullback_signal_quality")
OUT_CSV = OUT_DIR / "us500tick_m5_vol.csv"

TOL = 1e-6
ABORT_RATE = 0.0005


def aggregate():
    buckets = {}
    with open(M1_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            t = int(row["time"])
            b = t - t % 300
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])
            v = int(row["tick_volume"])
            e = buckets.get(b)
            if e is None:
                buckets[b] = [o, h, l, c, v]
            else:
                if h > e[1]:
                    e[1] = h
                if l < e[2]:
                    e[2] = l
                e[3] = c
                e[4] += v
    return buckets


def verify(buckets):
    orb = {}
    with open(ORB_M5, newline="") as fh:
        for row in csv.DictReader(fh):
            orb[int(row["time"])] = (float(row["open"]), float(row["high"]),
                                     float(row["low"]), float(row["close"]))
    common = sorted(set(buckets) & set(orb))
    only_agg = sorted(set(buckets) - set(orb))
    only_orb = sorted(set(orb) - set(buckets))
    bad = []
    maxdiff = 0.0
    for t in common:
        o, h, l, c, _v = buckets[t]
        oo, oh, ol, oc = orb[t]
        d = max(abs(o - oo), abs(h - oh), abs(l - ol), abs(c - oc))
        maxdiff = max(maxdiff, d)
        if d > TOL:
            bad.append((t, d))
    return len(orb), common, only_agg, only_orb, bad, maxdiff


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    buckets = aggregate()
    print(f"aggregated M5 bars: {len(buckets)}")
    n_orb, common, only_agg, only_orb, bad, maxdiff = verify(buckets)
    print(f"ORB terminal-native M5 bars: {n_orb}")
    print(f"common: {len(common)}   only-in-aggregated: {len(only_agg)}   "
          f"only-in-ORB: {len(only_orb)}")
    print(f"OHLC mismatches (> {TOL}): {len(bad)} / {len(common)} "
          f"({100.0 * len(bad) / max(1, len(common)):.4f}%)   max|diff|={maxdiff:.2e}")
    if only_orb:
        import datetime as dt
        ts = [dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M") for t in only_orb[:10]]
        print(f"  only-in-ORB (first 10): {ts}")
    if only_agg:
        import datetime as dt
        ts = [dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M") for t in only_agg[:10]]
        print(f"  only-in-aggregated (first 10): {ts}")

    rate = len(bad) / max(1, len(common))
    ok = (len(common) == n_orb == len(buckets)) and rate <= ABORT_RATE
    print(f"\nverification: {'PASS' if ok else 'FAIL'} "
          f"(exact bar-set match: {len(common) == n_orb == len(buckets)}, "
          f"mismatch rate {rate:.4%} <= {ABORT_RATE:.2%})")
    if not ok:
        raise SystemExit("AGGREGATION VERIFICATION FAILED — Phase 1 aborted per GATE.md")

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "open", "high", "low", "close", "tick_volume"])
        for t in sorted(buckets):
            o, h, l, c, v = buckets[t]
            w.writerow([t, o, h, l, c, v])
    print(f"wrote {len(buckets)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
