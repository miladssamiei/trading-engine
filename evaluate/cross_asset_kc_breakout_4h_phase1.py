import os
import json
import hashlib

import numpy as np
import pandas as pd

SEED = 20260928
BOOT_ITERS = 10000
BLOCK = 5
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "results", "cross_asset_kc_breakout")

INSTRUMENTS = [
    dict(asset="US500", tape=os.path.join(ROOT, "data", "results", "vwap_2021_2026", "us500tick_m1.csv"), cost=0.51, scale=1.0, unit="points"),
    dict(asset="EURUSD", tape=os.path.join(ROOT, "data", "results", "eurusd_onboarding", "eurusdtick_m1.csv"), cost=0.3, scale=10000.0, unit="pips"),
    dict(asset="XAUUSD", tape=os.path.join(ROOT, "data", "results", "xauusd_onboarding", "xauusdtick_m1.csv"), cost=0.41, scale=1.0, unit="price units"),
]

NS_H = 3600 * 1_000_000_000
P4 = 4 * NS_H
P1D = 24 * NS_H
O17 = 17 * NS_H
KC_MULT = 2.77
CLAMP_ATR = 1.5


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wilder_atr(h, l, c, n=14):
    m = len(h)
    tr = np.full(m, np.nan)
    tr[0] = h[0] - l[0]
    for i in range(1, m):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = np.full(m, np.nan)
    if m >= n:
        atr[n - 1] = tr[:n].mean()
        for i in range(n, m):
            atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr


def mbb_p_le_zero(x):
    n_tr = len(x)
    n_blocks = int(np.ceil(n_tr / BLOCK))
    starts = np.arange(0, n_tr - BLOCK + 1)
    rng = np.random.default_rng(SEED)
    sidx = rng.integers(0, len(starts), size=(BOOT_ITERS, n_blocks))
    flat = (starts[sidx][:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(BOOT_ITERS, -1)[:, :n_tr]
    means = x[flat].mean(axis=1)
    return float((means <= 0.0).mean())


def build_1h(tape):
    df = pd.read_csv(tape, usecols=["time", "open", "high", "low", "close", "tick_volume"])
    df = df.sort_values("time").reset_index(drop=True)
    slot = df["time"].to_numpy(dtype=np.int64) // 3600
    g = (
        pd.DataFrame({"slot": slot, "o": df["open"].to_numpy(float), "h": df["high"].to_numpy(float),
                      "l": df["low"].to_numpy(float), "c": df["close"].to_numpy(float),
                      "v": df["tick_volume"].to_numpy(float)})
        .groupby("slot", sort=True)
        .agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"))
    )
    s = g.index.to_numpy(dtype=np.int64)
    grid = np.arange(s[0], s[-1] + 1, dtype=np.int64)
    r = g.reindex(grid)
    r = r.ffill()
    r["v"] = r["v"].fillna(0.0)
    n_real = len(s)
    return grid, r["o"].to_numpy(float), r["h"].to_numpy(float), r["l"].to_numpy(float), r["c"].to_numpy(float), n_real, len(df)


def run_instrument(inst):
    asset, tape, cost, scale = inst["asset"], inst["tape"], inst["cost"], inst["scale"]
    grid, o1h, h1h, l1h, c1h, n_real, n_m1 = build_1h(tape)
    ny_idx1h = pd.to_datetime(pd.Index(grid * 3600), unit="s", utc=True).tz_convert("America/New_York").tz_localize(None)
    wns = ny_idx1h.values.astype("int64")

    key4 = wns - (wns % P4)
    g4 = (
        pd.DataFrame({"key": key4, "o": o1h, "h": h1h, "l": l1h, "c": c1h})
        .groupby("key", sort=True)
        .agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    )
    keys4 = g4.index.to_numpy(dtype=np.int64)
    o = g4["o"].to_numpy(float)
    h = g4["h"].to_numpy(float)
    l = g4["l"].to_numpy(float)
    c = g4["c"].to_numpy(float)
    n = len(keys4)
    ny_idx = pd.DatetimeIndex(pd.to_datetime(keys4, unit="ns"))

    dkey = wns - ((wns - O17) % P1D)
    gd = pd.DataFrame({"key": dkey, "c": c1h}).groupby("key", sort=True).agg(c=("c", "last"))
    dkeys = gd.index.to_numpy(dtype=np.int64)
    c1d = gd["c"].to_numpy(float)
    nd1 = len(dkeys)
    ema50d = pd.Series(c1d).ewm(span=50, adjust=False).mean().to_numpy()
    ema50d[:49] = np.nan
    bull = c1d > ema50d
    bear = c1d < ema50d

    lc_d = keys4 - ((keys4 - O17) % P1D) - P1D
    posd = np.searchsorted(dkeys, lc_d, side="left")
    posdc = np.clip(posd, 0, nd1 - 1)
    validd = (lc_d >= dkeys[0]) & (dkeys[posdc] == lc_d)
    bull_m = np.where(validd, bull[posdc], False)
    bear_m = np.where(validd, bear[posdc], False)

    ema29 = pd.Series(c).ewm(span=29, adjust=False).mean().to_numpy()
    ema29[:28] = np.nan
    atr29 = wilder_atr(h, l, c, 29)
    upper = ema29 + KC_MULT * atr29
    lower = ema29 - KC_MULT * atr29

    pc = np.roll(c, 1)
    pc[0] = np.nan
    p_upper = np.roll(upper, 1)
    p_upper[0] = np.nan
    p_lower = np.roll(lower, 1)
    p_lower[0] = np.nan
    p_ema = np.roll(ema29, 1)
    p_ema[0] = np.nan
    p_atr = np.roll(atr29, 1)
    p_atr[0] = np.nan

    armed = (~np.isnan(pc)) & (~np.isnan(p_upper)) & (~np.isnan(p_lower)) & (~np.isnan(p_ema)) & (~np.isnan(p_atr)) & (validd & ~np.isnan(ema50d[posdc]))
    sig_l = armed & (pc > p_upper) & bull_m
    sig_s = armed & (pc < p_lower) & bear_m
    sig_any = sig_l | sig_s

    trades = []
    ignored_open = 0
    i = 1
    while i < n:
        if not sig_any[i]:
            i += 1
            continue
        side = 1 if sig_l[i] else -1
        entry = o[i]
        dist = max(abs(entry - p_ema[i]), CLAMP_ATR * p_atr[i])
        sl = entry - dist if side == 1 else entry + dist
        exit_px = None
        reason = None
        exit_j = None
        ext = -np.inf
        adv = -np.inf
        for j in range(i, n):
            if side == 1:
                ext = max(ext, h[j] - entry)
                adv = max(adv, entry - l[j])
                if o[j] <= sl:
                    exit_px, reason, exit_j = o[j], "SL_GAP", j
                    break
                if l[j] <= sl:
                    exit_px, reason, exit_j = sl, "SL", j
                    break
                if not np.isnan(ema29[j]):
                    sl = max(sl, ema29[j])
            else:
                ext = max(ext, entry - l[j])
                adv = max(adv, h[j] - entry)
                if o[j] >= sl:
                    exit_px, reason, exit_j = o[j], "SL_GAP", j
                    break
                if h[j] >= sl:
                    exit_px, reason, exit_j = sl, "SL", j
                    break
                if not np.isnan(ema29[j]):
                    sl = min(sl, ema29[j])
        if exit_px is None:
            exit_px, reason, exit_j = c[n - 1], "FORCED_EOD", n - 1
        ignored_open += int(sig_any[i : exit_j + 1].sum())
        gross = side * (exit_px - entry) * scale
        trades.append(
            dict(
                asset=asset,
                signal_time=ny_idx[i].strftime("%Y-%m-%d %H:%M"),
                entry_time=ny_idx[i].strftime("%Y-%m-%d %H:%M"),
                exit_time=ny_idx[exit_j].strftime("%Y-%m-%d %H:%M"),
                side="LONG" if side == 1 else "SHORT",
                entry=entry,
                initial_sl=entry - dist if side == 1 else entry + dist,
                exit=exit_px,
                exit_reason=reason,
                bars_held=exit_j - i + 1,
                risk_dist=dist,
                mfe=ext * scale,
                mae=adv * scale,
                gross=gross,
                net=gross - cost,
                r_multiple=side * (exit_px - entry) / dist,
                entry_year=int(ny_idx[i].year),
            )
        )
        i = exit_j + 1

    t = pd.DataFrame(trades)
    res = dict(
        asset=asset,
        tape=dict(path=tape, sha256=sha256_file(tape), m1_rows=int(n_m1), bars_1h_base=int(len(grid)), real_1h=int(n_real), synthetic_1h=int(len(grid) - n_real),
                  bars_4h=int(n), bars_d1=int(nd1), first=str(ny_idx[0]), last=str(ny_idx[n - 1])),
        funnel=dict(armed_bars=int(armed.sum()), bull_bars=int((armed & bull_m).sum()), bear_bars=int((armed & bear_m).sum()),
                    breakout_up=int((armed & (pc > p_upper)).sum()), breakout_dn=int((armed & (pc < p_lower)).sum()),
                    cand_long=int(sig_l.sum()), cand_short=int(sig_s.sum()),
                    ignored_while_open=ignored_open, executed=len(t)),
        cost=cost, scale=scale, unit=inst["unit"],
    )
    return t, res


def stats_block(t):
    n_tr = len(t)
    blk = dict(n=n_tr, mean=np.nan, median=np.nan, total=np.nan, win=np.nan, sharpe=np.nan, p_boot=np.nan,
               mfe=np.nan, mae=np.nan, side_mix={}, exit_mix={}, yearly=None, g1=False, g2=False, g3=False, passed=False)
    if n_tr == 0:
        return blk
    g = t["net"].to_numpy(float)
    blk["mean"] = float(g.mean())
    blk["median"] = float(np.median(g))
    blk["total"] = float(g.sum())
    blk["win"] = float((g > 0).mean())
    if n_tr >= 2:
        s = g.std(ddof=1)
        blk["sharpe"] = float(g.mean() / s * np.sqrt(252.0)) if s > 0 else (np.inf if g.mean() > 0 else -np.inf)
    if n_tr >= BLOCK:
        blk["p_boot"] = mbb_p_le_zero(g)
    blk["mfe"] = float(t["mfe"].mean())
    blk["mae"] = float(t["mae"].mean())
    blk["side_mix"] = t.groupby("side")["net"].agg(["count", "mean"]).to_dict("index")
    blk["exit_mix"] = t["exit_reason"].value_counts().to_dict()
    blk["yearly"] = t.groupby("entry_year")["net"].agg(["count", "sum", "mean"])
    blk["g1"] = n_tr >= 30
    blk["g2"] = (not np.isnan(blk["sharpe"])) and blk["sharpe"] > 0
    blk["g3"] = (not np.isnan(blk["p_boot"])) and blk["p_boot"] <= 0.20
    blk["passed"] = bool(blk["g1"] and blk["g2"] and blk["g3"])
    return blk


def main():
    os.makedirs(OUT, exist_ok=True)
    all_trades = []
    results = []
    for inst in INSTRUMENTS:
        t, res = run_instrument(inst)
        blk = stats_block(t)
        res["gates"] = dict(G_KC1=bool(blk["g1"]), G_KC2=bool(blk["g2"]), G_KC3=bool(blk["g3"]))
        res["gates"]["net_sharpe"] = blk["sharpe"]
        res["gates"]["p_mbb"] = blk["p_boot"]
        res["passed"] = blk["passed"]
        all_trades.append(t)
        results.append((inst, res, blk))
        print(f"{inst['asset']}: {res['tape']['bars_4h']} 4h bars / {res['tape']['bars_d1']} D1 anchors "
              f"(1h base {res['tape']['bars_1h_base']}, {res['tape']['synthetic_1h']} synthetic); executed {blk['n']}")

    overall = all(r[2]["passed"] for r in results)
    verdict = "PASS" if overall else "FAIL"

    if any(len(t) for t in all_trades):
        pd.concat(all_trades, ignore_index=True).to_csv(os.path.join(OUT, "kc_breakout_signals.csv"), index=False)

    meta = dict(
        study="cross_asset_kc_breakout",
        hypothesis=50,
        generated_at=pd.Timestamp.utcnow().isoformat(),
        script_sha256=sha256_file(os.path.abspath(__file__)),
        seed=SEED, boot_iters=BOOT_ITERS, block=BLOCK,
        instruments=[dict(asset=r[1]["asset"], tape=r[1]["tape"], funnel=r[1]["funnel"], cost=r[1]["cost"],
                          unit=r[1]["unit"], gates=r[1]["gates"], passed=r[1]["passed"]) for r in results],
        verdict=verdict,
    )
    with open(os.path.join(OUT, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    L = []
    L.append("REPORT - HYPOTHESIS #50: DUAL-TIMEFRAME KELTNER CHANNEL BREAKOUT MOMENTUM (4h execution / D1 17:00 ET anchor)")
    L.append(f"Generated: {meta['generated_at']}  |  OVERALL VERDICT: {verdict}")
    L.append("Mechanics: prior 4h Close beyond KC(EMA29 +/- 2.77*ATR29(4h)) with D1 EMA(50) direction filter (17:00 ET")
    L.append("daily buckets from the continuous 1h grid); entry at 4h signal-bar open; initial SL = entry -/+")
    L.append("max(|Entry-EMA29|, 1.5*ATR29) then ratcheted to the EMA29 baseline at each 4h close; honest gap fills;")
    L.append("worst-case same-bar stop; continuous forward-filled grids per SPEC_KC_BREAKOUT.txt.")
    L.append("")
    for inst, res, blk in results:
        a = inst["asset"]
        L.append(f"=== {a} (cost floor {res['cost']} {res['unit']} RT) ===")
        tp = res["tape"]
        L.append(f"1. DATA: {tp['m1_rows']} M1 rows -> 1h base {tp['bars_1h_base']} ({tp['synthetic_1h']} synthetic = "
                 f"{100.0 * tp['synthetic_1h'] / tp['bars_1h_base']:.1f}%) / {tp['bars_4h']} 4h bars / {tp['bars_d1']} D1 anchors, "
                 f"{tp['first']} .. {tp['last']}")
        L.append(f"   tape sha256 {tp['sha256'][:16]}...")
        f_ = res["funnel"]
        L.append(f"2. FUNNEL: armed 4h bars={f_['armed_bars']}  D1 direction: bull={f_['bull_bars']} bear={f_['bear_bars']}")
        L.append(f"   breakouts (prior close beyond band): up={f_['breakout_up']} down={f_['breakout_dn']}")
        L.append(f"   direction-aligned candidates L/S={f_['cand_long']}/{f_['cand_short']}  ignored-while-open={f_['ignored_while_open']}  EXECUTED={f_['executed']}")
        if blk["n"] == 0:
            L.append("   no executed trades; all gates FAIL by definition")
        else:
            L.append(f"3. NET ECONOMICS ({res['unit']}): mean={blk['mean']:.3f}/trade  median={blk['median']:.3f}  total={blk['total']:.1f}  "
                     f"win={100.0 * blk['win']:.1f}%  Net Sharpe={blk['sharpe']:.3f}")
            L.append(f"   excursion asymmetry: mean MFE={blk['mfe']:.2f} vs mean MAE={blk['mae']:.2f}  (MFE/MAE={blk['mfe'] / blk['mae'] if blk['mae'] > 0 else float('nan'):.2f})")
            sm = blk["side_mix"]
            smtxt = "  ".join(f"{k}: n={int(v['count'])} mean={v['mean']:.2f}" for k, v in sm.items())
            L.append(f"   sides: {smtxt}")
            L.append(f"   exit mix: {blk['exit_mix']}")
            L.append("   year-by-year net:")
            for y, rr in blk["yearly"].iterrows():
                L.append(f"      {y}: n={int(rr['count']):4d}  total={rr['sum']:10.1f}  mean={rr['mean']:8.3f}")
        L.append(f"4. GATES (independent, frozen in GATE_KC_BREAKOUT.txt):")
        L.append(f"   G-KC1 executed >= 30          : {blk['n']:5d}          -> {'PASS' if blk['g1'] else 'FAIL'}")
        L.append(f"   G-KC2 Net Sharpe > 0           : {blk['sharpe']:9.3f}    -> {'PASS' if blk['g2'] else 'FAIL'}")
        L.append(f"   G-KC3 MBB P(mean<=0) <= 0.20   : {blk['p_boot']:9.4f}    -> {'PASS' if blk['g3'] else 'FAIL'}  (block={BLOCK}, {BOOT_ITERS} resamples, seed {SEED})")
        L.append(f"   {a} VERDICT: {'PASS' if blk['passed'] else 'FAIL'}")
        L.append("")
    L.append(f"OVERALL VERDICT: {verdict} (requires all three instruments to pass G-KC1..G-KC3 independently)")
    L.append("")
    L.append("HONESTY NOTES")
    L.append("- Continuous forward-fill convention: synthetic flat bars fill no-tick minutes, maintenance breaks, holidays")
    L.append("  and weekends; EMA flattens toward the last close, Wilder ATR decays across long gaps (narrowing channels),")
    L.append("  flat closes cannot print new breakouts, and D1 direction persists unchanged across flat weekend buckets -")
    L.append("  declared consequences of the continuity convention (SPEC 2.1/9.3), not post-hoc fixes.")
    L.append("- D1 anchors are true 17:00 ET closes aggregated at 1h resolution (never from the straddling [16:00-20:00) 4h")
    L.append("  bucket), so no daily-close lookahead leaks into the 20:00 ET 4h evaluations.")
    L.append("- Single pre-declared parameterization (owner-supplied); no optimization, no re-runs; sides, years and")
    L.append("  instruments reported as run. Phase-1 screen is in-sample by construction; MFE/MAE are exploratory")
    L.append("  descriptors, never results. Phase 2 (MT5 execution) only on overall PASS; any FAIL is final here.")
    report = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "REPORT_KC_BREAKOUT_PHASE1.txt"), "w") as f:
        f.write(report)
    print()
    print(report)


if __name__ == "__main__":
    main()
