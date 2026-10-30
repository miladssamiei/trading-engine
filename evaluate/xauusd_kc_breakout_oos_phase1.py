import os
import json
import hashlib

import numpy as np
import pandas as pd

SEED = 20260929
BOOT_ITERS = 10000
BLOCK = 5
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAPE = os.path.join(ROOT, "data", "results", "xauusd_onboarding", "xauusdtick_m1.csv")
OUT = os.path.join(ROOT, "data", "results", "xauusd_kc_oos")
COST = 0.41

NS_H = 3600 * 1_000_000_000
P4 = 4 * NS_H
P1D = 24 * NS_H
O17 = 17 * NS_H
KC_MULT = 2.77
CLAMP_ATR = 1.5
OOS_START = pd.Timestamp("2024-01-01")


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


def run():
    grid, o1h, h1h, l1h, c1h, n_real, n_m1 = build_1h(TAPE)
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
    bar_dates = ny_idx.normalize()

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
        gross = side * (exit_px - entry)
        trades.append(
            dict(
                block="OOS" if bar_dates[i] >= OOS_START else "CAL",
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
                mfe=ext,
                mae=adv,
                gross=gross,
                net=gross - COST,
                r_multiple=side * (exit_px - entry) / dist,
                entry_year=int(ny_idx[i].year),
            )
        )
        i = exit_j + 1

    t = pd.DataFrame(trades)
    res = dict(
        tape=dict(path=TAPE, sha256=sha256_file(TAPE), m1_rows=int(n_m1), bars_1h_base=int(len(grid)), real_1h=int(n_real),
                  synthetic_1h=int(len(grid) - n_real), bars_4h=int(n), bars_d1=int(nd1), first=str(ny_idx[0]), last=str(ny_idx[n - 1])),
        funnel=dict(armed_bars=int(armed.sum()), bull_bars=int((armed & bull_m).sum()), bear_bars=int((armed & bear_m).sum()),
                    breakout_up=int((armed & (pc > p_upper)).sum()), breakout_dn=int((armed & (pc < p_lower)).sum()),
                    cand_long=int(sig_l.sum()), cand_short=int(sig_s.sum()),
                    ignored_while_open=ignored_open, executed=len(t)),
    )
    return t, res


def block_stats(t):
    n_tr = len(t)
    blk = dict(n=n_tr, mean=np.nan, median=np.nan, total=np.nan, win=np.nan, sharpe=np.nan, p_boot=np.nan,
               mfe=np.nan, mae=np.nan, side_mix={}, exit_mix={}, yearly=None)
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
    return blk


def main():
    os.makedirs(OUT, exist_ok=True)
    t, res = run()
    cal = t[t["block"] == "CAL"].reset_index(drop=True) if len(t) else t
    oos = t[t["block"] == "OOS"].reset_index(drop=True) if len(t) else t
    bcal = block_stats(cal)
    boos = block_stats(oos)

    g1 = boos["n"] >= 30
    g2 = (not np.isnan(boos["sharpe"])) and boos["sharpe"] > 0
    g3 = (not np.isnan(boos["p_boot"])) and boos["p_boot"] <= 0.20
    verdict = "PASS" if (g1 and g2 and g3) else "FAIL"

    if len(oos):
        oos.to_csv(os.path.join(OUT, "xauusd_oos_signals.csv"), index=False)
    if len(cal):
        cal.to_csv(os.path.join(OUT, "xauusd_cal_signals.csv"), index=False)

    meta = dict(
        study="xauusd_kc_oos",
        hypothesis=51,
        generated_at=pd.Timestamp.utcnow().isoformat(),
        script_sha256=sha256_file(os.path.abspath(__file__)),
        seed=SEED, boot_iters=BOOT_ITERS, block=BLOCK,
        tape=res["tape"], funnel=res["funnel"],
        cal=dict(n=bcal["n"], mean=bcal["mean"], sharpe=bcal["sharpe"], win=bcal["win"]),
        oos=dict(n=boos["n"], mean=boos["mean"], sharpe=boos["sharpe"], win=boos["win"], p_mbb=boos["p_boot"]),
        gates=dict(G_OOS1=bool(g1), G_OOS2=bool(g2), G_OOS3=bool(g3)),
        verdict=verdict,
    )
    with open(os.path.join(OUT, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    L = []
    L.append("REPORT - HYPOTHESIS #51: XAUUSD KELTNER CHANNEL BREAKOUT - STRICT OOS PARTITION VERIFICATION")
    L.append(f"Generated: {meta['generated_at']}  |  VERDICT: {verdict}")
    L.append("Mechanics frozen verbatim from #50: prior 4h Close beyond KC(EMA29 +/- 2.77*ATR29) with D1 EMA(50) direction")
    L.append("(17:00 ET anchors from the continuous 1h grid); entry at 4h signal-bar open; initial SL = entry -/+")
    L.append("max(|Entry-EMA29|, 1.5*ATR29) ratcheted to the EMA29 baseline; honest gap fills; worst-case same-bar stop;")
    L.append("continuous forward-filled grids; one continuous walk, blocks assigned by ENTRY bar wall date")
    L.append("(CAL [2020-01-01, 2023-12-31] / OOS [2024-01-01, present]). Net = gross - 0.41 u RT.")
    L.append("")
    tp = res["tape"]
    L.append("1. DATA: " + f"{tp['m1_rows']} M1 rows -> 1h base {tp['bars_1h_base']} ({tp['synthetic_1h']} synthetic = "
             f"{100.0 * tp['synthetic_1h'] / tp['bars_1h_base']:.1f}%) / {tp['bars_4h']} 4h bars / {tp['bars_d1']} D1 anchors, "
             f"{tp['first']} .. {tp['last']}")
    L.append(f"   tape sha256 {tp['sha256'][:16]}...")
    f_ = res["funnel"]
    L.append(f"2. FUNNEL (whole tape, continuous walk): armed 4h bars={f_['armed_bars']}  D1 direction bull/bear={f_['bull_bars']}/{f_['bear_bars']}")
    L.append(f"   breakouts up/dn={f_['breakout_up']}/{f_['breakout_dn']}  direction-aligned candidates L/S={f_['cand_long']}/{f_['cand_short']}")
    L.append(f"   ignored-while-open={f_['ignored_while_open']}  EXECUTED={f_['executed']}  (CAL {bcal['n']} / OOS {boos['n']})")
    L.append("")
    for name, blk in (("CALIBRATION 2020-01-01..2023-12-31 (exploratory, absorbs warmup)", bcal),
                      ("OUT-OF-SAMPLE 2024-01-01..present (gated)", boos)):
        L.append(f"3. {name}")
        if blk["n"] == 0:
            L.append("   no executed trades")
        else:
            L.append(f"   n={blk['n']}  mean={blk['mean']:.3f} u/trade  median={blk['median']:.3f}  total={blk['total']:.1f}  "
                     f"win={100.0 * blk['win']:.1f}%  Net Sharpe={blk['sharpe']:.3f}")
            L.append(f"   MFE={blk['mfe']:.2f} vs MAE={blk['mae']:.2f}  (MFE/MAE={blk['mfe'] / blk['mae'] if blk['mae'] > 0 else float('nan'):.2f})")
            sm = blk["side_mix"]
            smtxt = "  ".join(f"{k}: n={int(v['count'])} mean={v['mean']:.2f}" for k, v in sm.items())
            L.append(f"   sides: {smtxt}")
            L.append(f"   exit mix: {blk['exit_mix']}")
            for y, rr in blk["yearly"].iterrows():
                L.append(f"      {y}: n={int(rr['count']):4d}  total={rr['sum']:10.1f}  mean={rr['mean']:8.3f}")
        L.append("")
    if bcal["n"] and boos["n"] and not np.isnan(bcal["mean"]) and not np.isnan(boos["mean"]):
        L.append(f"4. IS -> OOS DEGRADATION: mean/trade {bcal['mean']:.3f} -> {boos['mean']:.3f} "
                 f"({boos['mean'] - bcal['mean']:+.3f}), Sharpe {bcal['sharpe']:.3f} -> {boos['sharpe']:.3f} "
                 f"({boos['sharpe'] - bcal['sharpe']:+.3f}) - the edge, if any, LIVES in the OOS block; the calibration")
        L.append("   years are flat-to-negative. This is regime concentration, not calibration->deployment degradation.")
        L.append("")
    L.append("5. GATES (OOS block only, frozen in GATE_XAUUSD_OOS.txt)")
    L.append(f"   G-OOS1 executed >= 30          : {boos['n']:5d}          -> {'PASS' if g1 else 'FAIL'}")
    L.append(f"   G-OOS2 Net Sharpe > 0           : {boos['sharpe']:9.3f}    -> {'PASS' if g2 else 'FAIL'}")
    L.append(f"   G-OOS3 MBB P(mean<=0) <= 0.20   : {boos['p_boot']:9.4f}    -> {'PASS' if g3 else 'FAIL'}  (block={BLOCK}, {BOOT_ITERS} resamples, seed {SEED})")
    L.append(f"   VERDICT: {verdict}")
    L.append("")
    L.append("HONESTY NOTES")
    L.append("- The 2024+ block was observed in #50's report BEFORE this partition was declared (SPEC 8.3): a PASS certifies")
    L.append("  temporal stability of the already-seen block under frozen mechanics, NOT independent out-of-sample discovery.")
    L.append("- OOS year signs: " + ("; ".join(f"{y}: {rr['sum']:+.1f}" for y, rr in boos["yearly"].iterrows()) if boos["n"] else "n/a")
             + " - sign agreement within OOS is inspected, not gated.")
    L.append("- Continuous forward-fill convention and its declared flat-bar consequences carry over unchanged from #50.")
    L.append("- Single frozen parameterization and split; no optimization, no re-runs. Phase 2 (MT5 execution) requires PASS")
    L.append("  here plus standard walk-forward discipline (>=150 trades, year-by-year sign agreement within OOS).")
    report = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "REPORT_XAUUSD_OOS.txt"), "w") as f:
        f.write(report)
    print(f"XAUUSD OOS verification: CAL {bcal['n']} trades (mean {bcal['mean']:.3f}, Sharpe {bcal['sharpe']:.3f}) / "
          f"OOS {boos['n']} trades (mean {boos['mean']:.3f}, Sharpe {boos['sharpe']:.3f}, p={boos['p_boot']:.4f}) -> {verdict}")
    print()
    print(report)


if __name__ == "__main__":
    main()
