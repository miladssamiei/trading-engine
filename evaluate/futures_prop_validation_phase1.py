"""Frozen futures-cost / M1 prop-guard proxy study; see SPEC_FUTURES_PROP.txt."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from unseen_asset_kc_replication_phase1 import wilder_atr as kc_atr
from unseen_asset_saleh_donchian_phase1 import wilder_atr as saleh_atr, wilder_adx, supertrend

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/futures_prop_validation"
ASSETS = [
    ("GC", "XAUUSD", "data/results/xauusd_onboarding/xauusdtick_m1.csv", .13, .41),
    ("NQ", "USTECH", "data/results/virgin_assets/ustech_m1.csv", .39, 1.0),
    ("ES", "US500-tick", "data/results/vwap_2021_2026/us500tick_m1.csv", .30, .51),
    ("BTC", "BTCUSD", "data/results/virgin_assets/btc_m1.csv", 6.0, 15.0),
    ("ETH", "ETHUSD", "data/results/virgin_assets/eth_m1.csv", .8, 2.0),
]
SEED, BOOT, BLOCK = 20260926, 10000, 5
HOUR_NS = 3600 * 10**9
DAY_NS = 24 * HOUR_NS


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    d = pd.read_csv(path, usecols=["time", "open", "high", "low", "close"])
    assert d.time.is_monotonic_increasing and d.time.is_unique
    assert d[["open", "high", "low", "close"]].notna().all().all()
    return {k: d[k].to_numpy(np.int64 if k == "time" else float) for k in d.columns}


def grids(m):
    t = m["time"]
    slot = t // 3600
    g1 = pd.DataFrame({"key": slot, "o": m["open"], "h": m["high"],
                       "l": m["low"], "c": m["close"]}).groupby("key", sort=True).agg(
                           o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    grid1 = np.arange(slot[0], slot[-1] + 1, dtype=np.int64)
    r1 = g1.reindex(grid1).ffill()
    real1 = np.isin(grid1, g1.index.to_numpy())
    assert np.isfinite(r1.to_numpy()).all()
    ny1 = pd.to_datetime(grid1 * 3600, unit="s", utc=True).tz_convert("America/New_York")
    wall1 = ny1.tz_localize(None).asi8
    key4_1 = wall1 - wall1 % (4 * HOUR_NS)
    g4a = pd.DataFrame({"key": key4_1, "o": r1.o.to_numpy(), "h": r1.h.to_numpy(),
                        "l": r1.l.to_numpy(), "c": r1.c.to_numpy()}).groupby("key", sort=True).agg(
                            o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    keys4a = g4a.index.to_numpy(np.int64)
    start4a = grid1[np.searchsorted(key4_1, keys4a)] * 3600
    end4a = np.r_[start4a[1:], (grid1[-1] + 1) * 3600]
    dkey = wall1 - (wall1 - 17 * HOUR_NS) % DAY_NS
    gd = pd.DataFrame({"key": dkey, "c": r1.c.to_numpy()}).groupby("key", sort=True).agg(c=("c", "last"))

    ny_m = pd.to_datetime(t, unit="s", utc=True).tz_convert("America/New_York")
    wall_m = ny_m.tz_localize(None).asi8
    key4m = wall_m - wall_m % (4 * HOUR_NS)
    g4b = pd.DataFrame({"key": key4m, "o": m["open"], "h": m["high"],
                        "l": m["low"], "c": m["close"]}).groupby("key", sort=True).agg(
                            o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    # Each NY wall date has six buckets; start at first real bucket to avoid leading NaNs.
    days = pd.date_range(pd.Timestamp(key4m[0]).normalize(), pd.Timestamp(key4m[-1]).normalize())
    grid4b = np.repeat(days.asi8, 6) + np.tile(np.arange(0, 24, 4) * HOUR_NS, len(days))
    grid4b = grid4b[grid4b >= key4m[0]]
    r4b = g4b.reindex(grid4b).ffill()
    assert np.isfinite(r4b.to_numpy()).all()
    real4b = np.isin(grid4b, g4b.index.to_numpy())
    return dict(grid1=grid1, r1=r1, real1=real1, ny1=ny1,
                a4=g4a, akeys=keys4a, astart=start4a, aend=end4a,
                d=gd, b4=r4b, bkeys=grid4b, breal=real4b,
                key4_1=key4_1)


def engine_a(g):
    b = g["a4"]
    c, h, l = (b[k].to_numpy(float) for k in ("c", "h", "l"))
    ema = pd.Series(c).ewm(span=29, adjust=False).mean().to_numpy()
    ema[:28] = np.nan
    atr = kc_atr(h, l, c, 29)
    dc = g["d"].c.to_numpy(float)
    dema = pd.Series(dc).ewm(span=50, adjust=False).mean().to_numpy()
    dema[:49] = np.nan
    dk = g["d"].index.to_numpy(np.int64)
    keys = g["akeys"]
    completed = keys - (keys - 17 * HOUR_NS) % DAY_NS - DAY_NS
    di = np.searchsorted(dk, completed)
    valid = (di < len(dk)) & (completed >= dk[0])
    di = np.clip(di, 0, len(dk)-1)
    valid &= dk[di] == completed
    up = ema + 2.77 * atr
    lo = ema - 2.77 * atr
    side = np.zeros(len(keys), np.int8)
    arm = valid[1:] & np.isfinite(ema[:-1]) & np.isfinite(atr[:-1]) & np.isfinite(dema[di[1:]])
    side[1:] = np.where(arm & (c[:-1] > up[:-1]) & (dc[di[1:]] > dema[di[1:]]), 1,
                        np.where(arm & (c[:-1] < lo[:-1]) & (dc[di[1:]] < dema[di[1:]]), -1, 0))
    return dict(name="A_KC", side=side, start=g["astart"], end=g["aend"],
                ema=ema, atr=atr, open=b.o.to_numpy(float), close=c,
                real=np.ones(len(keys), bool))


def engine_b(g):
    b1, b4 = g["r1"], g["b4"]
    h, l, c = (b1[k].to_numpy(float) for k in ("h", "l", "c"))
    atr = saleh_atr(h, l, c, 14)
    adx = wilder_adx(pd.Series(h), pd.Series(l), pd.Series(c))
    dup = pd.Series(h).rolling(20, min_periods=20).max().to_numpy()
    dlo = pd.Series(l).rolling(20, min_periods=20).min().to_numpy()
    _, st = supertrend(*(b4[k].to_numpy(float) for k in ("h", "l", "c")))
    containing = g["key4_1"]
    mapped = np.searchsorted(g["bkeys"], containing, side="right") - 2
    anchor = np.zeros(len(c), np.int8)
    ok = mapped >= 0
    anchor[ok] = st[mapped[ok]]
    side = np.zeros(len(c), np.int8)
    i = np.arange(2, len(c))
    arm = g["real1"][i] & np.isfinite(atr[i-1]) & np.isfinite(adx[i-1]) & (adx[i-1] >= 20)
    side[i] = np.where(arm & (c[i-1] > dup[i-2]) & (anchor[i] == 1), 1,
                       np.where(arm & (c[i-1] < dlo[i-2]) & (anchor[i] == -1), -1, 0))
    start = g["grid1"] * 3600
    return dict(name="B_DONCHIAN", side=side, start=start, end=start+3600,
                atr=atr, open=b1.o.to_numpy(float), close=c, real=g["real1"])


def replay(m, e, cost):
    t, o, h, l, c = (m[k] for k in ("time", "open", "high", "low", "close"))
    left = np.searchsorted(t, e["start"])
    right = np.searchsorted(t, e["end"])
    rows, shadow_busy, shadow_cool, no_real = [], 0, 0, 0
    occupied_until, cooldown_until = -1, -1
    for i in np.flatnonzero(e["side"]):
        if i <= occupied_until:
            shadow_busy += 1
            continue
        if e["start"][i] <= cooldown_until:
            shadow_cool += 1
            continue
        if left[i] == right[i]:
            no_real += 1
            continue
        side = int(e["side"][i])
        entry_ix = int(left[i]); entry_t = int(t[entry_ix]); entry = float(o[entry_ix])
        entry_atr = float(e["atr"][i-1])
        if not np.isfinite(entry_atr) or entry_atr <= 0:
            continue
        if e["name"] == "A_KC":
            distance = max(abs(entry - e["ema"][i-1]), 1.5 * entry_atr)
        else:
            distance = 4.0 * entry_atr
        initial = entry - side * distance
        stop = initial
        peak = 0.0
        result = None
        for j in range(i, len(e["side"])):
            a, z = int(left[j]), int(right[j])
            if j == i:
                a = entry_ix
            if a < z:
                oo, hh, ll, cc = o[a:z], h[a:z], l[a:z], c[a:z]
                if side == 1:
                    stop_hit = ll <= stop
                    favorable = hh - entry - cost
                    netclose = cc - entry - cost
                elif e["name"] == "B_DONCHIAN":
                    stop_hit = hh + cost >= stop
                    favorable = entry - ll - cost
                    netclose = entry - cc - cost
                else:
                    stop_hit = hh >= stop
                    favorable = entry - ll - cost
                    netclose = entry - cc - cost
                peaks = np.maximum.accumulate(np.r_[peak, favorable])[1:]
                prop_hit = netclose <= peaks - 6.0 * entry_atr
                sidx = int(np.flatnonzero(stop_hit)[0]) if stop_hit.any() else z-a
                pidx = int(np.flatnonzero(prop_hit)[0]) if prop_hit.any() else z-a
                if sidx < z-a or pidx < z-a:
                    q = min(sidx, pidx)
                    x = a+q
                    if sidx <= pidx:
                        if side == 1:
                            fill_bid = min(stop, o[x])
                        elif e["name"] == "B_DONCHIAN":
                            fill_bid = max(stop, o[x]+cost) - cost
                        else:
                            fill_bid = max(stop, o[x])
                        reason = "SL_TRAIL" if side*(stop-initial) > 1e-9 else "SL_INIT"
                    else:
                        fill_bid = float(c[x])
                        reason = "PROP_LIQUIDATION"
                    result = (x, fill_bid, reason, float(peaks[q]))
                    break
                peak = float(peaks[-1])
            if e["name"] == "A_KC":
                if np.isfinite(e["ema"][j]):
                    stop = max(stop, e["ema"][j]) if side == 1 else min(stop, e["ema"][j])
            elif e["real"][j] and np.isfinite(e["atr"][j]):
                candidate = e["close"][j] - side * 4.0 * e["atr"][j]
                stop = max(stop, candidate) if side == 1 else min(stop, candidate)
        if result is None:
            result = (len(t)-1, float(c[-1]), "END", peak)
            j = len(e["side"])-1
        x, fill_bid, reason, peak_at_exit = result
        occupied_until = j
        if reason == "PROP_LIQUIDATION":
            cooldown_until = int(t[x]) + 86400
        gross = side*(fill_bid-entry)
        rows.append(dict(engine=e["name"], side=side, entry_time=entry_t,
                         exit_time=int(t[x]), entry=entry, exit_bid=fill_bid,
                         entry_atr=entry_atr, initial_stop=initial, final_stop=stop,
                         peak_open_equity=peak_at_exit, guard_distance=6*entry_atr,
                         exit_type=reason, gross=gross, net=gross-cost,
                         cost=cost, year=pd.Timestamp(entry_t, unit="s", tz="UTC").tz_convert("America/New_York").year))
    return rows, dict(candidates=int(np.count_nonzero(e["side"])), busy=shadow_busy,
                      cooldown=shadow_cool, no_real=no_real)


def mbb(x):
    n = len(x)
    if n < BLOCK:
        return float("nan")
    rng = np.random.default_rng(SEED)
    bad = 0
    for _ in range(BOOT):
        starts = rng.integers(0, n-BLOCK+1, size=math.ceil(n/BLOCK))
        sample = np.concatenate([x[s:s+BLOCK] for s in starts])[:n]
        bad += sample.mean() <= 0
    return bad / BOOT


def summary(rows):
    x = np.array([r["net"] for r in rows], float)
    n = len(x)
    mean = float(x.mean()) if n else float("nan")
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    sh = mean/sd if sd > 0 else float("nan")
    p = mbb(x)
    return dict(n=n, mean=mean, total=float(x.sum()), sharpe=sh, mbb_p=p,
                passed=bool(n >= 30 and sh > 0 and p <= .20),
                exits={k:sum(r["exit_type"] == k for r in rows) for k in
                       ("SL_INIT", "SL_TRAIL", "PROP_LIQUIDATION", "END")},
                yearly={str(y):dict(n=sum(r["year"] == y for r in rows),
                                    net=sum(r["net"] for r in rows if r["year"] == y))
                        for y in sorted({r["year"] for r in rows})})


def main():
    assert (OUT/"SPEC_FUTURES_PROP.txt").exists() and (OUT/"GATE_FUTURES_PROP.txt").exists()
    results, trades = [], []
    for asset, source, rel, cost, old_cost in ASSETS:
        path = ROOT/rel
        m = load(path)
        g = grids(m)
        for engine in (engine_a(g), engine_b(g)):
            rows, funnel = replay(m, engine, cost)
            for row in rows:
                row.update(asset=asset, source=source)
            trades.extend(rows)
            stats = summary(rows)
            result = dict(asset=asset, source=source, engine=engine["name"],
                          cost=cost, cfd_floor=old_cost, tape=str(path.relative_to(ROOT)),
                          tape_sha256=sha(path), m1_rows=len(m["time"]),
                          first_utc=pd.Timestamp(int(m["time"][0]), unit="s", tz="UTC").isoformat(),
                          last_utc=pd.Timestamp(int(m["time"][-1]), unit="s", tz="UTC").isoformat(),
                          funnel=funnel, stats=stats,
                          transaction_floor_saving=(old_cost-cost)*len(rows))
            results.append(result)
            print(f"{asset} {engine['name']}: n={stats['n']} mean={stats['mean']:.5f} "
                  f"Sharpe={stats['sharpe']:.4f} p={stats['mbb_p']:.4f} pass={stats['passed']}", flush=True)
        del m, g
    pd.DataFrame(trades).to_csv(OUT/"trades.csv", index=False)
    meta = dict(study="futures_prop_validation", date="2026-09-26", seed=SEED,
                bootstrap=BOOT, block=BLOCK, script_sha256=sha(Path(__file__)),
                overall_pass=all(r["stats"]["passed"] for r in results), results=results)
    (OUT/"run_meta.json").write_text(json.dumps(meta, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["FUTURES PROP VALIDATION PHASE 1 - REPORT", "VERDICT: " + ("PASS" if meta["overall_pass"] else "FAIL"),
             "Bid-only CFD proxy tapes with owner-supplied CME cost floors; no native CME futures data or account-dollar sizing.",
             "Net Sharpe is unannualized per trade. Guard is per position with distance 6 x entry ATR.", ""]
    for r in results:
        s = r["stats"]; ex = s["exits"]
        normal = ex["SL_INIT"]+ex["SL_TRAIL"]
        prop = ex["PROP_LIQUIDATION"]
        ratio = f"{normal/prop:.3f}:1" if prop else "N/A (zero prop exits)"
        lines += [f"{r['asset']} / {r['engine']} | {r['first_utc']} to {r['last_utc']} | {r['m1_rows']} M1 bars",
                  f"  Trades {s['n']}; net expectancy {s['mean']:.6f} price units/trade; net total {s['total']:.4f}; Net Sharpe {s['sharpe']:.5f}; MBB p {s['mbb_p']:.4f}; gate {'PASS' if s['passed'] else 'FAIL'}",
                  f"  Normal stops {normal} (initial {ex['SL_INIT']}, trailed {ex['SL_TRAIL']}); PROP_LIQUIDATION {prop}; normal:prop ratio {ratio}; END {ex['END']}",
                  f"  Candidates {r['funnel']['candidates']}; occupancy shadow {r['funnel']['busy']}; cooldown shadow {r['funnel']['cooldown']}; no-real-bar skip {r['funnel']['no_real']}",
                  f"  CFD-to-CME transaction floor saving {(r['cfd_floor']-r['cost']):.4f} per trade, {r['transaction_floor_saving']:.4f} total price units (not swap)",
                  "  Annual entry-cohort net: " + ", ".join(f"{y}: {v['n']} trades / {v['net']:.3f}" for y,v in s["yearly"].items()), ""]
    lines += ["TOTAL SWAP FRICTION SAVED VS CFD BASELINE: N/A. The five historical CFD studies do not define comparable overnight swap schedules or contract multipliers. Futures swap is set to zero here; a numeric saving would be fabricated.",
              "Total modeled round-trip transaction-floor saving across all ten independent runs: " +
              f"{sum(r['transaction_floor_saving'] for r in results):.4f} native price units (heterogeneous assets; not a monetary sum).",
              "This retrospective proxy study has no CME contract rolls, real bid/ask execution, account-wide trailing floor, or walk-forward confirmation. No row constitutes a live-ready result."]
    (OUT/"REPORT_FUTURES_PROP.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
