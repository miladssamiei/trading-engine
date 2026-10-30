# -*- coding: utf-8 -*-
r"""
eurusd_onboarding.py : one-pass onboarding of the EURUSD tick tape into
data/results/eurusd_onboarding/ (directory created by the script). Standalone,
argument-less CLI; every path is a constant below.

Source: C:\Users\Milad\download\eurusd-tick-2020-01-01-2026-09-20.csv, header
timestamp,askPrice,bidPrice, epoch MILLISECONDS UTC, chronological,
171,175,617 data rows. Read in 5,000,000-row chunks in a single streaming
pass; every filter decision is tallied, nothing is silently dropped.

Spec areas:
1  Integrity. Non-positive quotes (ask<=0 or bid<=0) and spread inversions
   (ask<bid among positive) are excluded and tallied. Monotonic timestamp
   inversions (delta vs previous raw row, carried across chunks, first-ever
   row exempt) are tallied and HARD-ASSERTED post-pass: violations > 0 prints
   the count plus up to 10 (ts_prev, ts) example pairs and exits 2 with no
   artifacts written. Exact same-timestamp duplicates (same ts AND ask AND
   bid as the previous raw row, carried across chunks) are dropped and
   tallied; conflicting same-ts quotes are kept and tallied.
2  Gaps and weekend rule. Consecutive kept-tick deltas >= 3h are collected,
   then each gap is classified: gap is weekend closure iff total extent
   outside [Fri 17:00, Sun 17:00] America/New_York (DST-aware) <= 2h.
   Weekend gaps are counted only; every non-weekend gap is recorded (UTC
   endpoints, NY endpoints, hours, ISO weekday name of the start).
3  Spread cost floor. Integer spread histogram in points
   (1 point = 0.00001, 1 pip = 0.00010, 5-decimal pricing) over kept ticks
   for the full tape and the London/NY overlap [08:00,11:00) ET wall clock.
   Stats: n, median, mean (4dp), p10, p90, p99, max in integer points, plus
   median, mean, p90 in pips. The overlap median is the quoted intraday
   cost floor.
4  Aggregation (BID bars from kept ticks only). M1 keyed by ts//60000:
   time (bucket open UTC epoch seconds), open, high, low, close,
   tick_volume. M5 keyed by ts//300000: OHLC only, built directly from
   ticks. D1: day D = (D-1 17:00 ET, D 17:00 ET], labeled by NY date of the
   17:00 ET close (label = NY calendar date of tick_ms + 7h), columns date
   (int YYYYMMDD), OHLC, tick_volume. A trailing partial M5 bucket / partial
   D1 label is held back per chunk and merged with the next chunk.
   Post-pass HARD ASSERT: re-reading the written M1 csv and re-aggregating to
   5 minutes must reproduce the written M5 csv exactly (row count and OHLC);
   failure exits 3.
5  Artifacts (data/results/eurusd_onboarding/): eurusdtick_m1.csv,
   eurusdtick_m5.csv, eurusdtick_d1.csv (float_format %.5f, index off),
   onboarding_stats.json (indent 1, plain Python types) and
   REPORT_EURUSD_ONBOARDING.txt (plain text, no markdown).

Smoke mode: environment variable EURUSD_ONBOARD_SMOKE=<N> processes only the
first N chunks, writes no artifacts, prints tallies and the monotonic
assertion result. PowerShell example:

  $env:EURUSD_ONBOARD_SMOKE="2"
  & "C:\Users\Milad\miniconda3\envs\reinforce_rl\python.exe" evaluate\eurusd_onboarding.py
  Remove-Item Env:\EURUSD_ONBOARD_SMOKE

Full run:

  & "C:\Users\Milad\miniconda3\envs\reinforce_rl\python.exe" evaluate\eurusd_onboarding.py
"""

import hashlib
import json
import os
import time as timemod
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(r"C:\Users\Milad\download\eurusd-tick-2020-01-01-2026-09-20.csv")
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "data/results/eurusd_onboarding"
CHUNK = 5_000_000
GAP_MS = 3 * 3600 * 1000
POINT = 1e-5
PIP = 1e-4
HIST_N = 200_000
NY = "America/New_York"
OVERLAP_LO, OVERLAP_HI = 480, 660
WEEKEND_SLACK_MS = 2 * 3600 * 1000
WEEKEND_RULE_TEXT = ("gap is weekend closure iff total extent outside "
                     "[Fri 17:00, Sun 17:00] America/New_York (DST-aware) "
                     "<= 2h")
D1_CONVENTION = ("day D = (D-1 17:00 ET, D 17:00 ET], labeled by NY date "
                 "of the 17:00 ET close")
SPREAD_CONVENTION = ("1 pip = 0.00010, 1 point/pipette = 0.00001 "
                     "(5-decimal pricing)")
DTYPES = {"timestamp": "int64", "askPrice": "float64", "bidPrice": "float64"}


def sha256_of(path, block=1 << 26):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def estimate_nchunks(path):
    size = path.stat().st_size
    with open(path, "rb") as fh:
        head = fh.read(1 << 20)
    lines = head.count(b"\n")
    if lines < 2:
        return 1
    est_rows = int(size / (len(head) / lines))
    return max(1, -(-est_rows // CHUNK))


def hist_stats(hist):
    n = int(hist.sum())
    if n == 0:
        return {"n": 0, "median": 0, "mean": 0.0, "p10": 0, "p90": 0,
                "p99": 0, "max": 0, "median_pips": 0.0, "mean_pips": 0.0,
                "p90_pips": 0.0}
    cs = np.cumsum(hist)
    mean_pts = float(np.dot(np.arange(len(hist)), hist) / n)

    def q(p):
        return int(np.searchsorted(cs, p * n, side="left"))

    nz = np.nonzero(hist)[0]
    med, p90 = q(0.5), q(0.90)
    pipf = POINT / PIP
    return {"n": n, "median": med, "mean": round(mean_pts, 4),
            "p10": q(0.10), "p90": p90, "p99": q(0.99), "max": int(nz[-1]),
            "median_pips": round(med * pipf, 4),
            "mean_pips": round(mean_pts * pipf, 4),
            "p90_pips": round(p90 * pipf, 4)}


def new_store(with_v):
    s = {"k": [], "o": [], "h": [], "l": [], "c": []}
    if with_v:
        s["v"] = []
    return s


def append_store(store, keys, bid):
    g = pd.DataFrame({"k": keys, "b": bid}).groupby("k", sort=True)
    store["k"].extend(g.size().index.to_numpy(np.int64).tolist())
    store["o"].extend(g.first()["b"].to_numpy().tolist())
    store["h"].extend(g.max()["b"].to_numpy().tolist())
    store["l"].extend(g.min()["b"].to_numpy().tolist())
    store["c"].extend(g.last()["b"].to_numpy().tolist())
    if "v" in store:
        store["v"].extend(g.size().to_numpy().tolist())


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


def tick_bounds(ms):
    u = pd.Timestamp(ms, unit="ms", tz="UTC")
    return {"epoch_ms": int(ms), "utc": u.isoformat(),
            "ny": u.tz_convert(NY).isoformat()}


def main():
    t0 = timemod.time()
    if not SRC.exists() or SRC.stat().st_size == 0:
        raise SystemExit(f"missing or empty source file: {SRC}")
    smoke_raw = os.environ.get("EURUSD_ONBOARD_SMOKE", "").strip()
    smoke_n = int(smoke_raw) if smoke_raw else None
    nchunks = estimate_nchunks(SRC)

    hist_full = np.zeros(HIST_N, np.int64)
    hist_ov = np.zeros(HIST_N, np.int64)
    gaps = []
    mono_examples = []
    per_year = {}
    raw_total = kept_total = 0
    dup_exact = dup_conflict = 0
    bad_px = inversions = clipped = 0
    ov_total = mono_viol = 0
    prev_ts = prev_ask = prev_bid = None
    prev_kept_ts = None
    first_kept = last_kept = None
    held_ts = held_bid = held_lab = None
    m1s, m5s, d1s = new_store(True), new_store(False), new_store(True)

    reader = pd.read_csv(SRC, chunksize=CHUNK, dtype=DTYPES,
                         usecols=list(DTYPES))
    for ci, df in enumerate(reader, start=1):
        if smoke_n is not None and ci > smoke_n:
            break
        ts = df["timestamp"].to_numpy(np.int64)
        ask = df["askPrice"].to_numpy(np.float64)
        bid = df["bidPrice"].to_numpy(np.float64)
        n = len(ts)
        raw_total += n

        d = np.empty(n, np.int64)
        d[1:] = ts[1:] - ts[:-1]
        d[0] = ts[0] - prev_ts if prev_ts is not None else 1
        neg = d < 0
        mono_viol += int(neg.sum())
        if neg.any():
            for i in np.flatnonzero(neg):
                if len(mono_examples) >= 10:
                    break
                mono_examples.append((int(ts[i - 1]) if i > 0 else int(prev_ts),
                                      int(ts[i])))
        eq = d == 0
        drop = np.zeros(n, bool)
        if eq.any():
            pa = np.empty(n)
            pb = np.empty(n)
            pa[1:] = ask[:-1]
            pb[1:] = bid[:-1]
            if prev_ask is not None:
                pa[0], pb[0] = prev_ask, prev_bid
            else:
                pa[0], pb[0] = ask[0], bid[0]
            same = (pa == ask) & (pb == bid)
            dup_exact += int((eq & same).sum())
            dup_conflict += int((eq & ~same).sum())
            drop = eq & same

        positive = (ask > 0) & (bid > 0)
        bad_px += int((~positive).sum())
        inv = (ask < bid) & positive
        inversions += int(inv.sum())
        keep = positive & ~inv & ~drop
        ts_k, ask_k, bid_k = ts[keep], ask[keep], bid[keep]
        kept_total += int(len(ts_k))
        if len(ts_k) == 0:
            prev_ts, prev_ask, prev_bid = int(ts[-1]), float(ask[-1]), float(bid[-1])
            print(f"[chunk {ci}/{nchunks}] rows={n:,} kept=0 "
                  f"elapsed={timemod.time() - t0:.0f}s", flush=True)
            continue

        if first_kept is None:
            first_kept = int(ts_k[0])
        last_kept = int(ts_k[-1])

        dk = np.empty(len(ts_k), np.int64)
        dk[1:] = ts_k[1:] - ts_k[:-1]
        dk[0] = ts_k[0] - prev_kept_ts if prev_kept_ts is not None else 0
        gm = dk >= GAP_MS
        if gm.any():
            for i in np.flatnonzero(gm):
                gaps.append((int(ts_k[i - 1]) if i > 0 else int(prev_kept_ts),
                             int(ts_k[i])))
        prev_kept_ts = int(ts_k[-1])

        sp = np.rint((ask_k - bid_k) / POINT).astype(np.int64)
        sp[sp < 0] = 0
        clipm = sp >= HIST_N
        clipped += int(clipm.sum())
        sp[clipm] = HIST_N - 1
        hist_full += np.bincount(sp, minlength=HIST_N)[:HIST_N]

        ny = pd.DatetimeIndex(pd.to_datetime(ts_k, unit="ms", utc=True)
                              ).tz_convert(NY)
        wall = (ny.hour * 60 + ny.minute).to_numpy()
        ovm = (wall >= OVERLAP_LO) & (wall < OVERLAP_HI)
        hist_ov += np.bincount(sp[ovm], minlength=HIST_N)[:HIST_N]
        ov_total += int(ovm.sum())
        uy, cy = np.unique(ny.year.to_numpy(), return_counts=True)
        for y, c in zip(uy, cy):
            per_year[int(y)] = per_year.get(int(y), 0) + int(c)

        shift = pd.DatetimeIndex(
            pd.to_datetime(ts_k + 7 * 3600 * 1000, unit="ms", utc=True)
        ).tz_convert(NY)
        lab = (shift.year * 10000 + shift.month * 100 + shift.day).to_numpy(np.int64)

        if held_ts is None:
            held_ts, held_bid, held_lab = ts_k, bid_k, lab
        else:
            held_ts = np.concatenate([held_ts, ts_k])
            held_bid = np.concatenate([held_bid, bid_k])
            held_lab = np.concatenate([held_lab, lab])
        m1b = held_ts // 60_000
        m5b = held_ts // 300_000
        cut = min(int(np.flatnonzero(m5b == m5b[-1])[0]),
                  int(np.flatnonzero(held_lab == held_lab[-1])[0]))
        if cut > 0:
            append_store(m1s, m1b[:cut], held_bid[:cut])
            append_store(m5s, m5b[:cut], held_bid[:cut])
            append_store(d1s, held_lab[:cut], held_bid[:cut])
        held_ts, held_bid, held_lab = held_ts[cut:], held_bid[cut:], held_lab[cut:]

        prev_ts, prev_ask, prev_bid = int(ts[-1]), float(ask[-1]), float(bid[-1])
        print(f"[chunk {ci}/{nchunks}] rows={n:,} kept={len(ts_k):,} "
              f"elapsed={timemod.time() - t0:.0f}s", flush=True)

    if held_ts is not None and len(held_ts) > 0:
        append_store(m1s, held_ts // 60_000, held_bid)
        append_store(m5s, held_ts // 300_000, held_bid)
        append_store(d1s, held_lab, held_bid)

    if smoke_n is not None:
        print(json.dumps({
            "smoke_chunks": smoke_n, "raw": raw_total, "kept": kept_total,
            "monotonic_violations": mono_viol,
            "dup_exact_dropped": dup_exact,
            "dup_conflicting_kept": dup_conflict,
            "bad_price_rows": bad_px, "spread_inversions": inversions,
            "spread_clipped_rows": clipped, "gaps_ge_3h_found": len(gaps),
            "overlap_ticks": ov_total}, indent=1))
        print(f"SMOKE monotonic assertion: "
              f"{'PASS' if mono_viol == 0 else 'FAIL'} "
              f"(violations={mono_viol})")
        print(f"SMOKE DONE elapsed={timemod.time() - t0:.0f}s "
              f"(no artifacts written)", flush=True)
        return

    if mono_viol > 0:
        print(f"HARD ASSERT 1 FAILED: monotonic violations = {mono_viol}")
        for a, b in mono_examples[:10]:
            print(f"  {pd.Timestamp(a, unit='ms', tz='UTC')} -> "
                  f"{pd.Timestamp(b, unit='ms', tz='UTC')}")
        raise SystemExit(2)

    weekend_count = 0
    non_weekend = []
    for a_ms, b_ms in sorted(gaps):
        if classify_weekend(a_ms, b_ms):
            weekend_count += 1
            continue
        ta = pd.Timestamp(a_ms, unit="ms", tz="UTC").tz_convert(NY)
        tb = pd.Timestamp(b_ms, unit="ms", tz="UTC").tz_convert(NY)
        non_weekend.append({
            "start_utc_ms": int(a_ms), "end_utc_ms": int(b_ms),
            "hours": round((b_ms - a_ms) / 3.6e6, 2),
            "start_ny": str(ta), "end_ny": str(tb),
            "start_weekday": ta.day_name()})

    sp_full = hist_stats(hist_full)
    sp_ov = hist_stats(hist_ov)
    if kept_total > 0:
        sp_ov["share_pct"] = round(100.0 * int(hist_ov.sum()) / kept_total, 4)

    m1_df = pd.DataFrame({
        "time": np.array(m1s["k"], np.int64) * 60,
        "open": m1s["o"], "high": m1s["h"], "low": m1s["l"],
        "close": m1s["c"], "tick_volume": np.array(m1s["v"], np.int64)})
    m5_df = pd.DataFrame({
        "time": np.array(m5s["k"], np.int64) * 300,
        "open": m5s["o"], "high": m5s["h"], "low": m5s["l"],
        "close": m5s["c"]})
    d1_df = pd.DataFrame({
        "date": np.array(d1s["k"], np.int64),
        "open": d1s["o"], "high": d1s["h"], "low": d1s["l"],
        "close": d1s["c"], "tick_volume": np.array(d1s["v"], np.int64)})

    OUT.mkdir(parents=True, exist_ok=True)
    m1_path = OUT / "eurusdtick_m1.csv"
    m5_path = OUT / "eurusdtick_m5.csv"
    d1_path = OUT / "eurusdtick_d1.csv"
    m1_df.to_csv(m1_path, index=False, float_format="%.5f")
    m5_df.to_csv(m5_path, index=False, float_format="%.5f")
    d1_df.to_csv(d1_path, index=False, float_format="%.5f")

    chk = pd.read_csv(m1_path)
    chk["t5"] = (chk["time"].to_numpy(np.int64) // 300) * 300
    r = chk.groupby("t5").agg(open=("open", "first"), high=("high", "max"),
                              low=("low", "min"),
                              close=("close", "last")).reset_index()
    m5c = pd.read_csv(m5_path).rename(columns={"time": "t5"})
    mg = m5c.merge(r, on="t5", suffixes=("", "_r"))
    mism = 0
    for col in ("open", "high", "low", "close"):
        mism += int((mg[col] != mg[col + "_r"]).sum())
    consistency = {"m5_rows": int(len(m5c)), "m5_from_m1_rows": int(len(r)),
                   "value_mismatches": int(mism),
                   "asserted_identical": bool(len(m5c) == len(r) and mism == 0)}
    if not consistency["asserted_identical"]:
        print(f"HARD ASSERT 2 FAILED: M1 -> M5 rows {len(m5c)} vs {len(r)}, "
              f"value mismatches {mism}")
        raise SystemExit(3)

    sha_t0 = timemod.time()
    sha = sha256_of(SRC)
    size_bytes = SRC.stat().st_size
    print(f"sha256 elapsed={timemod.time() - sha_t0:.1f}s", flush=True)

    stats = {
        "symbol": "eurusd",
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source": {"file": str(SRC), "size_bytes": int(size_bytes),
                   "sha256": sha, "rows": int(raw_total)},
        "ticks": {"raw": int(raw_total), "kept": int(kept_total)},
        "first_last_tick": {
            "first": tick_bounds(first_kept) if first_kept is not None else None,
            "last": tick_bounds(last_kept) if last_kept is not None else None},
        "integrity": {
            "monotonic_violations": int(mono_viol),
            "dup_exact_dropped": int(dup_exact),
            "dup_conflicting_kept": int(dup_conflict),
            "bad_price_rows": int(bad_px),
            "spread_inversions": int(inversions),
            "spread_clipped_rows": int(clipped)},
        "gaps": {
            "threshold_hours": 3.0,
            "weekend_rule": WEEKEND_RULE_TEXT,
            "weekend_excluded_count": int(weekend_count),
            "non_weekend_count": int(len(non_weekend)),
            "non_weekend": non_weekend},
        "spread": {
            "convention": SPREAD_CONVENTION,
            "full_tape": sp_full,
            "overlap_0800_1100_ET": sp_ov},
        "aggregation": {
            "m1_bars": int(len(m1_df)), "m5_bars": int(len(m5_df)),
            "d1_days": int(len(d1_df)),
            "d1_convention": D1_CONVENTION,
            "m1_m5_consistency": consistency},
        "ticks_per_ny_year": {str(y): int(per_year[y])
                               for y in sorted(per_year)},
    }
    with open(OUT / "onboarding_stats.json", "w") as fh:
        json.dump(stats, fh, indent=1)

    rep = []
    rep.append("EURUSD TICK ONBOARDING REPORT")
    rep.append(f"generated_utc: {stats['generated_utc']}")
    rep.append("tool: evaluate/eurusd_onboarding.py")
    rep.append("")
    rep.append("=== SOURCE ===")
    rep.append(f"file: {SRC}")
    rep.append(f"size_bytes: {size_bytes:,}")
    rep.append(f"sha256: {sha}")
    rep.append(f"rows: {raw_total:,}")
    rep.append("")
    rep.append("=== VOLUME AND BOUNDS ===")
    rep.append(f"raw ticks: {raw_total:,}")
    rep.append(f"kept ticks: {kept_total:,}")
    rep.append(f"exact same-ts duplicates dropped: {dup_exact:,}")
    rep.append(f"conflicting same-ts quotes kept: {dup_conflict:,}")
    if first_kept is not None:
        fu = pd.Timestamp(first_kept, unit="ms", tz="UTC")
        lu = pd.Timestamp(last_kept, unit="ms", tz="UTC")
        rep.append(f"first tick UTC: {fu}   NY: {fu.tz_convert(NY)}")
        rep.append(f"last tick UTC: {lu}   NY: {lu.tz_convert(NY)}")
    rep.append("kept ticks per NY year:")
    for y in sorted(per_year):
        rep.append(f"  {y}: {per_year[y]:,}")
    rep.append("")
    rep.append("=== INTEGRITY ===")
    rep.append("HARD ASSERT 1 PASSED: monotonic violations == 0")
    rep.append(f"non-positive quote rows excluded: {bad_px:,}")
    rep.append(f"spread inversions (ask<bid) excluded: {inversions:,}")
    rep.append(f"spread rows clipped at histogram ceiling: {clipped:,}")
    rep.append("")
    rep.append("=== GAPS >= 3h ===")
    rep.append(f"rule: {WEEKEND_RULE_TEXT}")
    rep.append(f"threshold: 3.0 h")
    rep.append(f"weekend gaps excluded: {weekend_count}")
    rep.append(f"non-weekend gaps: {len(non_weekend)}")
    rep.append("start UTC | end UTC | hours | start weekday | NY start -> end")
    for g in non_weekend:
        rep.append(
            f"{pd.Timestamp(g['start_utc_ms'], unit='ms', tz='UTC')} | "
            f"{pd.Timestamp(g['end_utc_ms'], unit='ms', tz='UTC')} | "
            f"{g['hours']} | {g['start_weekday']} | "
            f"{g['start_ny']} -> {g['end_ny']}")
    if not non_weekend:
        rep.append("none")
    rep.append("")
    rep.append("=== SPREAD - COST FLOOR ===")
    rep.append(f"convention: {SPREAD_CONVENTION}")
    rep.append(f"full tape points: n={sp_full['n']:,} median={sp_full['median']} "
               f"mean={sp_full['mean']} p10={sp_full['p10']} "
               f"p90={sp_full['p90']} p99={sp_full['p99']} max={sp_full['max']}")
    rep.append(f"full tape pips: median={sp_full['median_pips']} "
               f"mean={sp_full['mean_pips']} p90={sp_full['p90_pips']}")
    rep.append(f"overlap [08:00,11:00) ET points: n={sp_ov['n']:,} "
               f"share_pct={sp_ov.get('share_pct', 0.0)} "
               f"median={sp_ov['median']} mean={sp_ov['mean']} "
               f"p10={sp_ov['p10']} p90={sp_ov['p90']} p99={sp_ov['p99']} "
               f"max={sp_ov['max']}")
    rep.append(f"overlap [08:00,11:00) ET pips: median={sp_ov['median_pips']} "
               f"mean={sp_ov['mean_pips']} p90={sp_ov['p90_pips']}")
    rep.append(f"optimal intraday cost floor (London/NY overlap median): "
               f"{sp_ov['median']} points = {sp_ov['median_pips']} pips")
    rep.append("")
    rep.append("=== AGGREGATION ===")
    rep.append(f"M1 bars: {len(m1_df):,} (BID, keyed by ts//60000, time = "
               "bucket open UTC epoch seconds, OHLC + tick_volume)")
    rep.append(f"M5 bars: {len(m5_df):,} (BID, keyed by ts//300000, built "
               "directly from kept ticks, OHLC only)")
    rep.append(f"D1 days: {len(d1_df):,}")
    rep.append(f"D1 convention: {D1_CONVENTION}")
    rep.append("HARD ASSERT 1 PASSED: monotonic violations == 0")
    rep.append(f"HARD ASSERT 2 PASSED: M1 -> M5 identical "
               f"(m5_rows={consistency['m5_rows']:,}, "
               f"m5_from_m1_rows={consistency['m5_from_m1_rows']:,}, "
               f"value_mismatches=0)")
    rep.append("")
    rep.append("=== ARTIFACTS ===")
    rep.append(f"{m1_path} rows={len(m1_df):,}")
    rep.append(f"{m5_path} rows={len(m5_df):,}")
    rep.append(f"{d1_path} rows={len(d1_df):,}")
    rep.append(f"{OUT / 'onboarding_stats.json'} (1 json object)")
    rep.append(f"{OUT / 'REPORT_EURUSD_ONBOARDING.txt'} (this report)")
    (OUT / "REPORT_EURUSD_ONBOARDING.txt").write_text(
        "\n".join(rep) + "\n", encoding="utf-8")

    print(f"DONE elapsed={timemod.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
