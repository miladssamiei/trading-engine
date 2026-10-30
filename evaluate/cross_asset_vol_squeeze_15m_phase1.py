import os
import json
import hashlib

import numpy as np
import pandas as pd

SEED = 20260926
BOOT_ITERS = 10000
BLOCK = 5
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "results", "cross_asset_vol_squeeze")

INSTRUMENTS = [
    dict(asset="US500", tape=os.path.join(ROOT, "data", "results", "vwap_2021_2026", "us500tick_m1.csv"), cost=0.51, scale=1.0, unit="points"),
    dict(asset="EURUSD", tape=os.path.join(ROOT, "data", "results", "eurusd_onboarding", "eurusdtick_m1.csv"), cost=0.3, scale=10000.0, unit="pips"),
    dict(asset="XAUUSD", tape=os.path.join(ROOT, "data", "results", "xauusd_onboarding", "xauusdtick_m1.csv"), cost=0.41, scale=1.0, unit="price units"),
]

P4 = 14400 * 1_000_000_000  # 4h in wall-clock nanoseconds
LR_W = None  # set in main


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


def build_15m(tape):
    df = pd.read_csv(tape, usecols=["time", "open", "high", "low", "close", "tick_volume"])
    df = df.sort_values("time").reset_index(drop=True)
    slot = df["time"].to_numpy(dtype=np.int64) // 900
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


def wall_ns_from_slots(grid):
    idx = pd.to_datetime(pd.Index(grid * 900), unit="s", utc=True).tz_convert("America/New_York").tz_localize(None)
    return idx, idx.values.astype("int64")


def run_instrument(inst):
    asset, tape, cost, scale = inst["asset"], inst["tape"], inst["cost"], inst["scale"]
    grid, o, h, l, c, n_real, n_m1 = build_15m(tape)
    n = len(grid)
    ny_idx, wns = wall_ns_from_slots(grid)

    cur_key = wns - (wns % P4)
    lc_key = cur_key - P4
    g4 = (
        pd.DataFrame({"key": cur_key, "o": o, "h": h, "l": l, "c": c})
        .groupby("key", sort=True)
        .agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    )
    keys4 = g4.index.to_numpy(dtype=np.int64)
    h4 = g4["h"].to_numpy(float)
    l4 = g4["l"].to_numpy(float)
    c4 = g4["c"].to_numpy(float)
    k4 = len(keys4)

    sma = pd.Series(c4).rolling(20).mean().to_numpy()
    std0 = pd.Series(c4).rolling(20).std(ddof=0).to_numpy()
    ema = pd.Series(c4).ewm(span=20, adjust=False).mean().to_numpy()
    ema[:19] = np.nan
    atr4 = wilder_atr(h4, l4, c4, 14)
    bbu, bbl = sma + 2.0 * std0, sma - 2.0 * std0
    kcu, kcl = ema + 1.5 * atr4, ema - 1.5 * atr4
    squeeze4 = (bbu < kcu) & (bbl > kcl)
    global LR_W
    if LR_W is None:
        x = np.arange(20, dtype=float)
        dx = x - x.mean()
        LR_W = dx / (dx * dx).sum()
    slope4 = np.full(k4, np.nan)
    if k4 >= 20:
        slope4[19:] = np.lib.stride_tricks.sliding_window_view(c4, 20) @ LR_W
    armed4 = ~np.isnan(sma) & ~np.isnan(std0) & ~np.isnan(ema) & ~np.isnan(atr4) & ~np.isnan(slope4)
    squeeze4a = squeeze4 & armed4
    slope4a = np.where(armed4, slope4, np.nan)

    pos = np.searchsorted(keys4, lc_key, side="left")
    posc = np.clip(pos, 0, k4 - 1)
    valid = (lc_key >= keys4[0]) & (keys4[posc] == lc_key)
    sqz15 = np.where(valid, squeeze4a[posc], False)
    slope15 = np.where(valid, slope4a[posc], np.nan)

    dch = pd.Series(h).rolling(20).max().shift(2).to_numpy()
    dcl = pd.Series(l).rolling(20).min().shift(2).to_numpy()
    atr15 = wilder_atr(h, l, c, 14)
    atr_prior = np.roll(atr15, 1)
    atr_prior[0] = np.nan
    pc = np.roll(c, 1)
    pc[0] = np.nan

    armed = (~np.isnan(dch)) & (~np.isnan(dcl)) & (~np.isnan(atr_prior)) & (~np.isnan(slope15)) & (np.arange(n) >= 21)
    sig_l = armed & sqz15 & (slope15 > 0) & (pc > dch)
    sig_s = armed & sqz15 & (slope15 < 0) & (pc < dcl)
    sig_any = sig_l | sig_s

    trades = []
    ignored_open = 0
    i = 21
    while i < n:
        if not sig_any[i]:
            i += 1
            continue
        side = 1 if sig_l[i] else -1
        atr_sig = atr_prior[i]
        entry = o[i]
        sl = entry - 2.0 * atr_sig if side == 1 else entry + 2.0 * atr_sig
        exit_px = None
        reason = None
        exit_j = None
        ext = -np.inf
        adv = -np.inf
        hc = -np.inf if side == 1 else np.inf
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
                hc = max(hc, c[j])
                a = atr15[j]
                if not np.isnan(a):
                    sl = max(sl, hc - 2.5 * a)
            else:
                ext = max(ext, entry - l[j])
                adv = max(adv, h[j] - entry)
                if o[j] >= sl:
                    exit_px, reason, exit_j = o[j], "SL_GAP", j
                    break
                if h[j] >= sl:
                    exit_px, reason, exit_j = sl, "SL", j
                    break
                hc = min(hc, c[j])
                a = atr15[j]
                if not np.isnan(a):
                    sl = min(sl, hc + 2.5 * a)
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
                initial_sl=entry - 2.0 * atr_sig if side == 1 else entry + 2.0 * atr_sig,
                exit=exit_px,
                exit_reason=reason,
                bars_held=exit_j - i + 1,
                atr=atr_sig,
                mfe=ext * scale,
                mae=adv * scale,
                gross=gross,
                net=gross - cost,
                r_multiple=side * (exit_px - entry) / (2.0 * atr_sig),
                entry_year=int(ny_idx[i].year),
            )
        )
        i = exit_j + 1

    t = pd.DataFrame(trades)
    res = dict(
        asset=asset,
        tape=dict(path=tape, sha256=sha256_file(tape), m1_rows=int(n_m1), bars_15m=int(n), real_15m=int(n_real), synthetic_15m=int(n - n_real), bars_4h=int(k4),
                  first=str(ny_idx[0]), last=str(ny_idx[n - 1])),
        funnel=dict(armed_bars=int(armed.sum()), squeeze_bars=int((sqz15 & armed).sum()),
                    cand_long=int(sig_l.sum()), cand_short=int(sig_s.sum()),
                    ignored_while_open=ignored_open, executed=len(t)),
        cost=cost, scale=scale, unit=inst["unit"],
    )
    return t, res


def stats_block(t, res):
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
        blk = stats_block(t, res)
        res["gates"] = dict(G_SQ1=bool(blk["g1"]), G_SQ2=bool(blk["g2"]), G_SQ3=bool(blk["g3"]))
        res["gates"]["net_sharpe"] = blk["sharpe"]
        res["gates"]["p_mbb"] = blk["p_boot"]
        res["passed"] = blk["passed"]
        all_trades.append(t)
        results.append((inst, res, blk))
        print(f"{inst['asset']}: {res['tape']['m1_rows']} M1 rows -> {res['tape']['bars_15m']} 15m bars "
              f"({res['tape']['synthetic_15m']} synthetic) / {res['tape']['bars_4h']} 4h bars; executed {blk['n']}")

    overall = all(r[2]["passed"] for r in results)
    verdict = "PASS" if overall else "FAIL"

    if any(len(t) for t in all_trades):
        pd.concat(all_trades, ignore_index=True).to_csv(os.path.join(OUT, "cross_sqz_signals.csv"), index=False)

    meta = dict(
        study="cross_asset_vol_squeeze",
        hypothesis=47,
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
    L.append("REPORT - HYPOTHESIS #47: CROSS-ASSET MULTI-TIMEFRAME VOLATILITY SQUEEZE BREAKOUT (4h anchor / 15m execution)")
    L.append(f"Generated: {meta['generated_at']}  |  OVERALL VERDICT: {verdict}")
    L.append("Mechanics: 4h BB(20,2.0) inside KC(20EMA,1.5*ATR14) squeeze + 4h LR-slope(20) direction + 15m Donchian(20)")
    L.append("close breakout, entry at signal-bar open, initial SL 2.0*ATR15(t-1), close-ratchet trail 2.5*ATR15(j),")
    L.append("honest gap fills, worst-case same-bar stop, continuous forward-filled 15m/4h grid per SPEC_CROSS_SQZ.txt.")
    L.append("")
    for inst, res, blk in results:
        a = inst["asset"]
        L.append(f"=== {a} (cost floor {res['cost']} {res['unit']} RT) ===")
        tp = res["tape"]
        L.append(f"1. DATA: {tp['m1_rows']} M1 rows -> {tp['bars_15m']} 15m bars ({tp['synthetic_15m']} synthetic = "
                 f"{100.0 * tp['synthetic_15m'] / tp['bars_15m']:.1f}%) / {tp['bars_4h']} 4h bars, {tp['first']} .. {tp['last']}")
        L.append(f"   tape sha256 {tp['sha256'][:16]}...")
        f_ = res["funnel"]
        L.append(f"2. FUNNEL: armed bars={f_['armed_bars']}  squeeze-ON bars={f_['squeeze_bars']}  candidates: "
                 f"L={f_['cand_long']} S={f_['cand_short']}  ignored-while-open={f_['ignored_while_open']}  EXECUTED={f_['executed']}")
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
        L.append(f"4. GATES (independent, frozen in GATE_CROSS_SQZ.txt):")
        L.append(f"   G-SQ1 executed >= 30          : {blk['n']:5d}          -> {'PASS' if blk['g1'] else 'FAIL'}")
        L.append(f"   G-SQ2 Net Sharpe > 0           : {blk['sharpe']:9.3f}    -> {'PASS' if blk['g2'] else 'FAIL'}")
        L.append(f"   G-SQ3 MBB P(mean<=0) <= 0.20   : {blk['p_boot']:9.4f}    -> {'PASS' if blk['g3'] else 'FAIL'}  (block={BLOCK}, {BOOT_ITERS} resamples, seed {SEED})")
        L.append(f"   {a} VERDICT: {'PASS' if blk['passed'] else 'FAIL'}")
        L.append("")
    L.append(f"OVERALL VERDICT: {verdict} (requires all three instruments to pass G-SQ1..G-SQ3 independently)")
    L.append("")
    L.append("HONESTY NOTES")
    L.append("- Continuous forward-fill convention: synthetic flat bars fill no-tick minutes, maintenance breaks, holidays")
    L.append("  and weekends; they make Donchian breakouts impossible on flat windows and set 4h LR slope to exactly 0, but")
    L.append("  they DECAY Wilder ATR across long gaps, so early post-gap signals carry mechanically tightened stops. This is")
    L.append("  a declared consequence of the continuity convention (SPEC 2.2/3.5), not a bug fix applied post hoc.")
    L.append("- Single pre-declared parameterization; no optimization, no re-runs; sides, years and instruments reported as run.")
    L.append("- Phase-1 screen is in-sample by construction; MFE/MAE are exploratory descriptors, never results. Phase 2 (MT5")
    L.append("  execution) only on overall PASS; any FAIL is final for this artifact.")
    report = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "REPORT_CROSS_SQZ_PHASE1.txt"), "w") as f:
        f.write(report)
    print()
    print(report)


if __name__ == "__main__":
    main()
