# -*- coding: utf-8 -*-
"""
onboard_ticks.py — USA30 / XAUUSD tick onboarding for the data-domain
re-tests (MOKAMEL_FLIP USA30 native re-test, SP2L XAUUSD native re-test).

Stitches Dukascopy tick CSVs (header: timestamp,askPrice,bidPrice; epoch ms
UTC, chronological files supplied in order), verifies them to the US500-tick
standard, and builds M1/M5 BID-bar exports in the exact format of the
verified US500 exports (time,open,high,low,close[,tick_volume]; open-time
epoch seconds UTC; rows only for buckets containing >=1 tick; M5 built
directly from ticks, never from M1). Nothing is silently dropped: every
duplicate, overlap, inversion and gap is tallied into onboarding_stats.json.

Measurements:
- per-file rows / first / last timestamp; global monotonic violations;
  same-timestamp duplicate ticks (identical vs conflicting quotes);
- price sanity (non-positive quotes, crossed ask<bid), largest mid jumps;
- exact spread distribution via integer 0.001-unit histograms for the full
  tape, the NY cash window [09:30,16:00) and overnight (the per-instrument
  cost-floor basis, Q4 — US500's 0.51 pt floor is never reused);
- tick-level gap list (>= 3h) with weekday labels;
- per-UTC-day tick counts and daily session-hour profile per year;
- raw-text decimal-digit sampling per file (digits-by-era, Q2);
- day-universe CSV (ny_date;session_open;prev_close) derived from M5:
  session_open = open of the NY wall-570 M5 bar (0.0 if absent);
  prev_close   = last M5 close of the previous NY date with bars (0.0 if
  none) — consumed by mokamel_flip_stage_a/b exactly like ORB mt5_days.csv
  (only the >0 universe filter is used there);
- M1 -> M5 agreement check (M1 resampled to 5 min vs direct-from-tick M5).

Usage:
  python evaluate/onboard_ticks.py --symbol xauusd \
    --tick-files f1.csv f2.csv patch.csv f3.csv \
    --out-dir data/results/xauusd_onboarding
"""

import argparse
import hashlib
import json
import time as timemod
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK = 5_000_000
HIST_UNITS = 1000
HIST_N = 400_000
GAP_MS = 3 * 3600 * 1000
CASH_LO, CASH_HI = 570, 960
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


def sample_digits(path, n_pos=14, n_lines=300):
    size = path.stat().st_size
    rows = []
    with open(path, "r") as fh:
        for frac in np.linspace(0.0, 0.999, n_pos):
            fh.seek(int(size * frac))
            fh.readline()
            for _ in range(n_lines):
                line = fh.readline()
                if not line:
                    break
                parts = line.strip().split(",")
                if len(parts) != 3 or parts[0] == "timestamp":
                    continue
                try:
                    ts = int(parts[0])
                except ValueError:
                    continue
                dec = [len(p.split(".")[1]) if "." in p else 0
                       for p in parts[1:3]]
                rows.append((ts, dec[0], dec[1]))
    return rows


def hist_stats(hist):
    tot = int(hist.sum())
    if tot == 0:
        return None
    cs = np.cumsum(hist)
    mean = float(np.dot(np.arange(HIST_N), hist) / tot) / HIST_UNITS

    def q(p):
        return float(np.searchsorted(cs, p * tot, side="left")) / HIST_UNITS
    nz = np.nonzero(hist)[0]
    return {"n": tot, "mean": round(mean, 4), "median": round(q(0.5), 4),
            "p10": round(q(0.10), 4), "p90": round(q(0.90), 4),
            "p99": round(q(0.99), 4),
            "max": round(float(nz[-1]) / HIST_UNITS, 4)}


def merge_bucket(carry, o, h, l, c, v):
    return dict(o=carry["o"], h=max(carry["h"], h), l=min(carry["l"], l),
                c=c, v=carry["v"] + v)


def agg_buckets(ts, bid, div_ms, carry, rows, with_v):
    mins = ts // div_ms
    g = pd.DataFrame({"m": mins, "b": bid}).groupby("m", sort=True)
    km = g.size().index.to_numpy(np.int64)
    o = g.first()["b"].to_numpy()
    c = g.last()["b"].to_numpy()
    h = g.max()["b"].to_numpy()
    lo = g.min()["b"].to_numpy()
    v = g.size().to_numpy()
    last = len(km) - 1
    for k in range(len(km)):
        m = int(km[k])
        if k == 0 and carry is not None and carry["m"] != m:
            rows["m"].append(carry["m"])
            rows["o"].append(carry["o"])
            rows["h"].append(carry["h"])
            rows["l"].append(carry["l"])
            rows["c"].append(carry["c"])
            if with_v:
                rows["v"].append(carry["v"])
            carry = None
        bo, bh, bl, bc, bv = float(o[k]), float(h[k]), float(lo[k]), float(c[k]), int(v[k])
        if carry is not None and carry["m"] == m:
            bo, bh, bl, bc, bv = (lambda mb: (mb["o"], mb["h"], mb["l"], mb["c"], mb["v"]))(
                merge_bucket(carry, bo, bh, bl, bc, bv))
        if k == last:
            carry = dict(m=m, o=bo, h=bh, l=bl, c=bc, v=bv)
            break
        rows["m"].append(m)
        rows["o"].append(bo)
        rows["h"].append(bh)
        rows["l"].append(bl)
        rows["c"].append(bc)
        if with_v:
            rows["v"].append(bv)
    return carry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--tick-files", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    t0 = timemod.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = [Path(f) for f in args.tick_files]
    for f in files:
        if not f.exists() or f.stat().st_size == 0:
            raise SystemExit(f"missing or empty input: {f}")

    hist_full = np.zeros(HIST_N, np.int64)
    hist_cash = np.zeros(HIST_N, np.int64)
    hist_on = np.zeros(HIST_N, np.int64)
    day_first, day_last, day_n = {}, {}, {}
    gaps = []
    jumps = []
    prev_ts = prev_ask = prev_bid = prev_mid = None
    mono_neg = 0
    dup_ts = dup_identical = dup_conflicting = 0
    mono_examples = []
    n_bad_px = n_inverted = n_clipped = 0
    rows_total = 0
    m1_rows = {"m": [], "o": [], "h": [], "l": [], "c": [], "v": []}
    m5_rows = {"m": [], "o": [], "h": [], "l": [], "c": []}
    carry1 = carry5 = None
    per_file = []

    for fi, path in enumerate(files):
        pf_rows = 0
        pf_first = pf_last = None
        reader = pd.read_csv(path, chunksize=CHUNK, dtype=DTYPES,
                             usecols=list(DTYPES))
        for df in reader:
            ts = df["timestamp"].to_numpy(np.int64)
            ask = df["askPrice"].to_numpy(float)
            bid = df["bidPrice"].to_numpy(float)
            n = len(ts)
            rows_total += n
            pf_rows += n
            if pf_first is None:
                pf_first = int(ts[0])
            pf_last = int(ts[-1])

            if prev_ts is not None:
                d = np.empty(n, np.int64)
                d[0] = ts[0] - prev_ts
                d[1:] = ts[1:] - ts[:-1]
                neg = d < 0
                mono_neg += int(neg.sum())
                if mono_examples is not None and neg.any():
                    for i in np.flatnonzero(neg)[:5]:
                        mono_examples.append(
                            [int(ts[i - 1]) if i > 0 else prev_ts, int(ts[i])])
                    if len(mono_examples) > 50:
                        mono_examples = None
                eq = d == 0
                if eq.any():
                    pa = np.empty(n)
                    pa[0] = prev_ask
                    pa[1:] = ask[:-1]
                    pb = np.empty(n)
                    pb[0] = prev_bid
                    pb[1:] = bid[:-1]
                    same = (pa == ask) & (pb == bid)
                    dup_ts += int(eq.sum())
                    dup_identical += int((eq & same).sum())
                    dup_conflicting += int((eq & ~same).sum())
                gm = d >= GAP_MS
                if gm.any():
                    for i in np.flatnonzero(gm):
                        gaps.append((int(ts[i - 1]) if i > 0 else prev_ts, int(ts[i])))

            mid = (ask + bid) * 0.5
            bad = (ask <= 0) | (bid <= 0)
            n_bad_px += int(bad.sum())
            inv = ask < bid
            n_inverted += int(inv.sum())
            dm = np.abs(np.diff(np.concatenate(([prev_mid], mid))) if prev_mid is not None
                        else np.abs(np.diff(mid, prepend=mid[0])))
            dm = np.nan_to_num(dm, nan=0.0)
            if n > 5:
                top = np.argpartition(dm, -5)[-5:]
            else:
                top = np.argsort(dm)
            for i in top:
                if dm[i] > 0:
                    jumps.append((float(dm[i]), int(ts[i])))
            jumps = sorted(jumps, reverse=True)[:10]

            sp = np.rint((ask - bid) * HIST_UNITS).astype(np.int64)
            sp[sp < 0] = 0
            clip = sp >= HIST_N
            n_clipped += int(clip.sum())
            sp[clip] = HIST_N - 1
            hist_full += np.bincount(sp, minlength=HIST_N)[:HIST_N]

            ny = pd.DatetimeIndex(pd.to_datetime(ts, unit="ms", utc=True)
                                  ).tz_convert("America/New_York")
            wall = (ny.hour * 60 + ny.minute).to_numpy()
            cashm = (wall >= CASH_LO) & (wall < CASH_HI)
            hist_cash += np.bincount(sp[cashm], minlength=HIST_N)[:HIST_N]
            hist_on += np.bincount(sp[~cashm], minlength=HIST_N)[:HIST_N]

            dday = (ts // 86_400_000).astype(np.int64)
            uq, first_ix = np.unique(dday, return_index=True)
            _, li = np.unique(dday[::-1], return_index=True)
            last_ix = n - 1 - li
            bc = np.bincount(dday - int(dday[0]),
                             minlength=int(dday[-1]) - int(dday[0]) + 1)
            for d0, ix in zip(uq, first_ix):
                d0 = int(d0)
                day_first[d0] = min(day_first.get(d0, 1 << 62), int(ts[ix]))
            for d0, ix in zip(uq, last_ix):
                d0 = int(d0)
                day_last[d0] = max(day_last.get(d0, -1), int(ts[ix]))
            for off, c in enumerate(bc):
                if c:
                    day_n[int(dday[0]) + off] = day_n.get(int(dday[0]) + off, 0) + int(c)

            carry1 = agg_buckets(ts, bid, 60_000, carry1, m1_rows, True)
            carry5 = agg_buckets(ts, bid, 300_000, carry5, m5_rows, False)

            prev_ts, prev_ask, prev_bid, prev_mid = int(ts[-1]), float(ask[-1]), float(bid[-1]), float(mid[-1])
        per_file.append(dict(file=str(path), sha256=sha256_of(path),
                             rows=pf_rows, first_ts=pf_first, last_ts=pf_last))
        print(f"[{args.symbol}] file {fi+1}/{len(files)} done: {path.name} "
              f"{pf_rows:,} rows ({timemod.time()-t0:.0f}s)", flush=True)

    for carry, rows in ((carry1, m1_rows), (carry5, m5_rows)):
        if carry is not None:
            rows["m"].append(carry["m"])
            rows["o"].append(carry["o"])
            rows["h"].append(carry["h"])
            rows["l"].append(carry["l"])
            rows["c"].append(carry["c"])
            if "v" in rows:
                rows["v"].append(carry["v"])

    m1 = pd.DataFrame({"time": np.array(m1_rows["m"], np.int64) * 60,
                       "open": m1_rows["o"], "high": m1_rows["h"],
                       "low": m1_rows["l"], "close": m1_rows["c"],
                       "tick_volume": m1_rows["v"]})
    m5 = pd.DataFrame({"time": np.array(m5_rows["m"], np.int64) * 300,
                       "open": m5_rows["o"], "high": m5_rows["h"],
                       "low": m5_rows["l"], "close": m5_rows["c"]})
    m1_path = out / f"{args.symbol}tick_m1.csv"
    m5_path = out / f"{args.symbol}tick_m5.csv"
    m1.to_csv(m1_path, index=False, float_format="%.3f")
    m5.to_csv(m5_path, index=False, float_format="%.3f")

    chk = pd.read_csv(m1_path)
    chk["time5"] = (chk.time // 300) * 300
    r = chk.groupby("time5").agg(open=("open", "first"), high=("high", "max"),
                                 low=("low", "min"),
                                 close=("close", "last")).reset_index()
    m5c = pd.read_csv(m5_path)
    mg = m5c.merge(r, on="time5", suffixes=("", "_r")) if "time5" in m5c else \
        m5c.rename(columns={"time": "time5"}).merge(r, on="time5", suffixes=("", "_r"))
    mism = 0
    for col in ("open", "high", "low", "close"):
        mism += int((mg[col] - mg[col + "_r"]).abs().gt(1e-9).sum())
    agree = {"m5_rows": int(len(m5c)), "m5_from_m1_rows": int(len(r)),
             "value_mismatches": int(mism),
             "identical": bool(mism == 0 and len(m5c) == len(r))}

    t5 = m5["time"].to_numpy(np.int64)
    ny5 = pd.DatetimeIndex(pd.to_datetime(t5, unit="s", utc=True)
                           ).tz_convert("America/New_York")
    wall5 = (ny5.hour * 60 + ny5.minute).to_numpy(np.int32)
    nd5 = (ny5.year * 10000 + ny5.month * 100 + ny5.day).to_numpy(np.int32)
    closes5 = m5["close"].to_numpy(float)
    opens5 = m5["open"].to_numpy(float)
    last_close_by_nd, open_by_nd = {}, {}
    for nd_ in np.unique(nd5):
        idxs = np.flatnonzero(nd5 == nd_)
        last_close_by_nd[int(nd_)] = float(closes5[idxs[-1]])
        w = wall5[idxs]
        m570 = np.flatnonzero(w == CASH_LO)
        if len(m570):
            open_by_nd[int(nd_)] = float(opens5[idxs[m570[0]]])
    ordered = sorted(set(last_close_by_nd) | set(open_by_nd))
    days_out = []
    pc = 0.0
    for nd_ in ordered:
        days_out.append(dict(ny_date=int(nd_),
                             session_open=open_by_nd.get(int(nd_), 0.0),
                             prev_close=pc))
        pc = last_close_by_nd.get(int(nd_), pc)
    days_df = pd.DataFrame(days_out)
    days_path = out / f"{args.symbol}_days.csv"
    days_df.to_csv(days_path, sep=";", index=False, float_format="%.4f")

    gap_rows = []
    for a, b in sorted(gaps):
        ta = pd.Timestamp(a, unit="ms", tz="UTC").tz_convert("America/New_York")
        tb = pd.Timestamp(b, unit="ms", tz="UTC").tz_convert("America/New_York")
        gap_rows.append(dict(start_utc_ms=int(a), end_utc_ms=int(b),
                             hours=round((b - a) / 3.6e6, 2),
                             start_ny=str(ta), end_ny=str(tb),
                             start_weekday=int(ta.dayofweek)))
    years = {}
    for d0 in sorted(day_n):
        ts0 = pd.Timestamp(day_first[d0], unit="ms", tz="UTC").tz_convert("America/New_York")
        ts1 = pd.Timestamp(day_last[d0], unit="ms", tz="UTC").tz_convert("America/New_York")
        y = ts0.year
        r_ = years.setdefault(int(y), {"n_days": 0, "ticks": 0,
                                       "first_wall": [], "last_wall": []})
        r_["n_days"] += 1
        r_["ticks"] += day_n[d0]
        r_["first_wall"].append(ts0.hour * 60 + ts0.minute)
        r_["last_wall"].append(ts1.hour * 60 + ts1.minute)
    year_prof = {}
    for y, r_ in years.items():
        fw = np.array(r_["first_wall"])
        lw = np.array(r_["last_wall"])
        year_prof[y] = {"n_days": r_["n_days"], "ticks": r_["ticks"],
                        "first_wall_p05_p50_p95":
                            [int(np.percentile(fw, 5)), int(np.percentile(fw, 50)),
                             int(np.percentile(fw, 95))],
                        "last_wall_p05_p50_p95":
                            [int(np.percentile(lw, 5)), int(np.percentile(lw, 50)),
                             int(np.percentile(lw, 95))]}

    digits_rows = []
    for path in files:
        digits_rows.extend(sample_digits(path))
    ddf = pd.DataFrame(digits_rows, columns=["ts", "ask_dec", "bid_dec"])
    ddf["ym"] = pd.to_datetime(ddf.ts, unit="ms", utc=True).dt.strftime("%Y-%m")
    digits_prof = {ym: dict(max_dec=int(g[["ask_dec", "bid_dec"]].to_numpy().max()),
                            min_dec=int(g[["ask_dec", "bid_dec"]].to_numpy().min()),
                            n=len(g))
                   for ym, g in ddf.groupby("ym")}

    stats = {
        "symbol": args.symbol,
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "files": per_file,
        "rows_total": rows_total,
        "monotonic_violations": int(mono_neg),
        "monotonic_examples": mono_examples if mono_examples is not None else ">50 suppressed",
        "dup_ts_total": int(dup_ts), "dup_identical": int(dup_identical),
        "dup_conflicting": int(dup_conflicting),
        "bad_price_rows": int(n_bad_px),
        "spread_inversions": int(n_inverted),
        "spread_clipped_rows": int(n_clipped),
        "top_mid_jumps": [[round(j, 3), int(t)] for j, t in jumps],
        "spread_full_tape": hist_stats(hist_full),
        "spread_cash_0930_1600NY": hist_stats(hist_cash),
        "spread_overnight": hist_stats(hist_on),
        "gaps_ge_3h": gap_rows,
        "gap_count": len(gap_rows),
        "utc_days_with_ticks": len(day_n),
        "session_profile_by_year": year_prof,
        "digits_by_month_sampled": digits_prof,
        "m1_bars": int(len(m1)), "m5_bars": int(len(m5)),
        "m1_m5_agreement": agree,
        "day_universe_rows": int(len(days_df)),
        "first_tick_utc": per_file[0]["first_ts"],
        "last_tick_utc": per_file[-1]["last_ts"],
    }
    with open(out / "onboarding_stats.json", "w") as fh:
        json.dump(stats, fh, indent=1, default=str)

    rep = []
    rep.append(f"# {args.symbol.upper()} tick onboarding report")
    rep.append("")
    rep.append(f"Generated: {stats['generated_utc']}  (tool: evaluate/onboard_ticks.py)")
    rep.append("")
    rep.append("## Source files (stitched in this order)")
    for pf in per_file:
        a = pd.Timestamp(pf["first_ts"], unit="ms", tz="UTC")
        b = pd.Timestamp(pf["last_ts"], unit="ms", tz="UTC")
        rep.append(f"- `{pf['file']}`  rows={pf['rows']:,}  {a} -> {b}  "
                   f"sha256={pf['sha256'][:16]}...")
    rep.append(f"- total ticks: {rows_total:,}")
    rep.append(f"- tape span (UTC): "
               f"{pd.Timestamp(stats['first_tick_utc'], unit='ms', tz='UTC')} -> "
               f"{pd.Timestamp(stats['last_tick_utc'], unit='ms', tz='UTC')}")
    rep.append("")
    rep.append("## Integrity tallies (nothing silently dropped)")
    rep.append(f"- monotonic violations (ts < previous ts): {mono_neg}")
    if mono_examples:
        for a, b in mono_examples[:10]:
            rep.append(f"  - {pd.Timestamp(a, unit='ms', tz='UTC')} -> "
                       f"{pd.Timestamp(b, unit='ms', tz='UTC')}")
    rep.append(f"- same-timestamp duplicate ticks: {dup_ts} "
               f"(identical {dup_identical}, conflicting {dup_conflicting})")
    rep.append(f"- non-positive quotes: {n_bad_px}; spread inversions (ask<bid): "
               f"{n_inverted}")
    rep.append(f"- top mid-jumps: "
               f"{[(round(j, 3), str(pd.Timestamp(t, unit='ms', tz='UTC'))) for j, t in jumps[:5]]}")
    rep.append("")
    rep.append("## Gaps >= 3h (tick-level)")
    if gap_rows:
        for g in gap_rows:
            rep.append(f"- {g['start_ny']} -> {g['end_ny']}  {g['hours']}h "
                       f"(start weekday {g['start_weekday']})")
    else:
        rep.append("- none")
    rep.append("")
    rep.append("## Spread (naive ask-bid, exact integer histogram, price units)")
    for nm, key in (("full tape", "spread_full_tape"),
                    ("cash [09:30,16:00) NY", "spread_cash_0930_1600NY"),
                    ("overnight", "spread_overnight")):
        s_ = stats[key]
        if s_:
            rep.append(f"- {nm}: median={s_['median']} mean={s_['mean']} "
                       f"p10={s_['p10']} p90={s_['p90']} p99={s_['p99']} "
                       f"max={s_['max']} (n={s_['n']:,})")
    rep.append("")
    rep.append("## Digits (sampled raw text)")
    yms = sorted(digits_prof)
    rep.append(f"- first month {yms[0]}: {digits_prof[yms[0]]}; "
               f"last month {yms[-1]}: {digits_prof[yms[-1]]}")
    rep.append("- full per-month table in onboarding_stats.json")
    rep.append("")
    rep.append("## Session profile (per year, NY wall minutes of UTC-day first/last tick)")
    for y in sorted(year_prof):
        r_ = year_prof[y]
        rep.append(f"- {y}: {r_['n_days']} days, {r_['ticks']:,} ticks, "
                   f"first-tick wall p05/p50/p95 {r_['first_wall_p05_p50_p95']}, "
                   f"last-tick wall {r_['last_wall_p05_p50_p95']}")
    rep.append("")
    rep.append("## Bar exports (US500-tick format)")
    rep.append(f"- {m1_path.name}: {len(m1):,} M1 bars")
    rep.append(f"- {m5_path.name}: {len(m5):,} M5 bars")
    rep.append(f"- M1->M5 agreement: {agree}")
    rep.append(f"- {days_path.name}: {len(days_df):,} day rows "
               "(ny_date;session_open;prev_close)")
    rep.append("")
    (out / "REPORT.md").write_text("\n".join(rep), encoding="utf-8")
    print(json.dumps({k: stats[k] for k in
                      ("rows_total", "monotonic_violations", "dup_ts_total",
                       "spread_inversions", "gap_count", "m1_bars", "m5_bars",
                       "m1_m5_agreement")}, indent=1))
    print(f"DONE {timemod.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
