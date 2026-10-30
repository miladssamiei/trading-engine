import os
import json
import hashlib

import numpy as np
import pandas as pd

SEED = 20260926
BOOT_ITERS = 10000
BLOCK = 5
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "results", "cross_asset_bb_squeeze")

INSTRUMENTS = [
    dict(asset="US500", tape=os.path.join(ROOT, "data", "results", "vwap_2021_2026", "us500tick_m1.csv"), cost=0.51, scale=1.0, unit="points"),
    dict(asset="EURUSD", tape=os.path.join(ROOT, "data", "results", "eurusd_onboarding", "eurusdtick_m1.csv"), cost=0.3, scale=10000.0, unit="pips"),
    dict(asset="XAUUSD", tape=os.path.join(ROOT, "data", "results", "xauusd_onboarding", "xauusdtick_m1.csv"), cost=0.41, scale=1.0, unit="price units"),
]

P4 = 14400 * 1_000_000_000  # 4h in wall-clock nanoseconds
ADX_MIN = 19.11
SL_MULT = 1.74
TRAIL_MULT = 3.71
LR_W = None  # set in run_instrument


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


def wilder_adx(h, l, c, n=14):
    m = len(h)
    tr = np.full(m, np.nan)
    pdm = np.zeros(m)
    ndm = np.zeros(m)
    tr[0] = h[0] - l[0]
    for i in range(1, m):
        up = h[i] - h[i - 1]
        dn = l[i - 1] - l[i]
        pdm[i] = up if (up > dn and up > 0.0) else 0.0
        ndm[i] = dn if (dn > up and dn > 0.0) else 0.0
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    s_tr = np.full(m, np.nan)
    s_pdm = np.full(m, np.nan)
    s_ndm = np.full(m, np.nan)
    if m >= 2 * n:
        s_tr[n] = tr[1 : n + 1].sum()
        s_pdm[n] = pdm[1 : n + 1].sum()
        s_ndm[n] = ndm[1 : n + 1].sum()
        for i in range(n + 1, m):
            s_tr[i] = s_tr[i - 1] - s_tr[i - 1] / n + tr[i]
            s_pdm[i] = s_pdm[i - 1] - s_pdm[i - 1] / n + pdm[i]
            s_ndm[i] = s_ndm[i - 1] - s_ndm[i - 1] / n + ndm[i]
        dx = np.full(m, np.nan)
        for i in range(n, m):
            if s_tr[i] > 0.0:
                pdi = 100.0 * s_pdm[i] / s_tr[i]
                ndi = 100.0 * s_ndm[i] / s_tr[i]
                if (pdi + ndi) > 0.0:
                    dx[i] = 100.0 * abs(pdi - ndi) / (pdi + ndi)
        adx = np.full(m, np.nan)
        adx[2 * n - 1] = dx[n : 2 * n].mean()
        for i in range(2 * n, m):
            adx[i] = (adx[i - 1] * (n - 1) + dx[i]) / n
        return adx
    return np.full(m, np.nan)


def mbb_p_le_zero(x):
    n_tr = len(x)
    n_blocks = int(np.ceil(n_tr / BLOCK))
    starts = np.arange(0, n_tr - BLOCK + 1)
    rng = np.random.default_rng(SEED)
    sidx = rng.integers(0, len(starts), size=(BOOT_ITERS, n_blocks))
    flat = (starts[sidx][:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(BOOT_ITERS, -1)[:, :n_tr]
    means = x[flat].mean(axis=1)
    return float((means <= 0.0).mean())


def build_30m(tape):
    df = pd.read_csv(tape, usecols=["time", "open", "high", "low", "close", "tick_volume"])
    df = df.sort_values("time").reset_index(drop=True)
    slot = df["time"].to_numpy(dtype=np.int64) // 1800
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
    global LR_W
    asset, tape, cost, scale = inst["asset"], inst["tape"], inst["cost"], inst["scale"]
    grid, o, h, l, c, n_real, n_m1 = build_30m(tape)
    n = len(grid)
    ny_idx = pd.to_datetime(pd.Index(grid * 1800), unit="s", utc=True).tz_convert("America/New_York").tz_localize(None)
    wns = ny_idx.values.astype("int64")

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

    sma = pd.Series(c4).rolling(29).mean().to_numpy()
    std0 = pd.Series(c4).rolling(29).std(ddof=0).to_numpy()
    ema = pd.Series(c4).ewm(span=29, adjust=False).mean().to_numpy()
    ema[:28] = np.nan
    atr4 = wilder_atr(h4, l4, c4, 14)
    adx4 = wilder_adx(h4, l4, c4, 14)
    bbu, bbl = sma + 1.82 * std0, sma - 1.82 * std0
    kcu, kcl = ema + 1.56 * atr4, ema - 1.56 * atr4
    sqz_valid = (~np.isnan(bbu)) & (~np.isnan(kcu)) & (~np.isnan(bbl)) & (~np.isnan(kcl))
    sqz = sqz_valid & (bbu < kcu) & (bbl > kcl)

    if LR_W is None:
        x = np.arange(11, dtype=float)
        dx = x - x.mean()
        LR_W = dx / (dx * dx).sum()
    slope4 = np.full(k4, np.nan)
    if k4 >= 11:
        slope4[10:] = np.lib.stride_tricks.sliding_window_view(c4, 11) @ LR_W

    fired4 = np.zeros(k4, dtype=bool)
    hist_ok = np.zeros(k4, dtype=bool)
    for k in range(4, k4):
        if sqz_valid[k] and sqz_valid[k - 1] and sqz_valid[k - 2] and sqz_valid[k - 3] and sqz_valid[k - 4]:
            hist_ok[k] = True
            fired4[k] = (not sqz[k]) and sqz[k - 1] and sqz[k - 2] and sqz[k - 3] and sqz[k - 4]
    armed4 = hist_ok & (~np.isnan(adx4)) & (~np.isnan(slope4)) & (~np.isnan(atr4)) & (np.arange(k4) >= 33)
    fired4a = fired4 & armed4

    pos = np.searchsorted(keys4, lc_key, side="left")
    posc = np.clip(pos, 0, k4 - 1)
    valid = (lc_key >= keys4[0]) & (keys4[posc] == lc_key)
    fired30 = np.where(valid, fired4a[posc], False)
    adx30 = np.where(valid, adx4[posc], np.nan)
    slope30 = np.where(valid, slope4[posc], np.nan)
    atr30 = np.where(valid, atr4[posc], np.nan)

    armed = valid & (~np.isnan(adx30)) & (~np.isnan(slope30)) & (~np.isnan(atr30)) & np.isin(lc_key, keys4[armed4])
    sig_l = armed & fired30 & (adx30 >= ADX_MIN) & (slope30 > 0)
    sig_s = armed & fired30 & (adx30 >= ADX_MIN) & (slope30 < 0)
    sig_any = sig_l | sig_s

    trades = []
    ignored_open = 0
    i = 0
    while i < n:
        if not sig_any[i]:
            i += 1
            continue
        side = 1 if sig_l[i] else -1
        atr4_sig = atr30[i]
        dist0 = SL_MULT * atr4_sig
        dist1 = TRAIL_MULT * atr4_sig
        entry = o[i]
        sl = entry - dist0 if side == 1 else entry + dist0
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
                sl = max(sl, hc - dist1)
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
                sl = min(sl, hc + dist1)
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
                initial_sl=entry - dist0 if side == 1 else entry + dist0,
                exit=exit_px,
                exit_reason=reason,
                bars_held=exit_j - i + 1,
                atr4=atr4_sig,
                adx=adx30[i],
                mfe=ext * scale,
                mae=adv * scale,
                gross=gross,
                net=gross - cost,
                r_multiple=side * (exit_px - entry) / dist0,
                entry_year=int(ny_idx[i].year),
            )
        )
        i = exit_j + 1

    t = pd.DataFrame(trades)
    res = dict(
        asset=asset,
        tape=dict(path=tape, sha256=sha256_file(tape), m1_rows=int(n_m1), bars_30m=int(n), real_30m=int(n_real), synthetic_30m=int(n - n_real), bars_4h=int(k4),
                  first=str(ny_idx[0]), last=str(ny_idx[n - 1])),
        funnel=dict(armed_bars=int(armed.sum()), squeeze_on_4h=int(sqz.sum()), fired_4h=int(fired4a.sum()),
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
        res["gates"] = dict(G_BB1=bool(blk["g1"]), G_BB2=bool(blk["g2"]), G_BB3=bool(blk["g3"]))
        res["gates"]["net_sharpe"] = blk["sharpe"]
        res["gates"]["p_mbb"] = blk["p_boot"]
        res["passed"] = blk["passed"]
        all_trades.append(t)
        results.append((inst, res, blk))
        print(f"{inst['asset']}: {res['tape']['m1_rows']} M1 rows -> {res['tape']['bars_30m']} 30m bars "
              f"({res['tape']['synthetic_30m']} synthetic) / {res['tape']['bars_4h']} 4h bars; "
              f"fires={res['funnel']['fired_4h']}; executed {blk['n']}")

    overall = all(r[2]["passed"] for r in results)
    verdict = "PASS" if overall else "FAIL"

    if any(len(t) for t in all_trades):
        pd.concat(all_trades, ignore_index=True).to_csv(os.path.join(OUT, "bb_sqz_signals.csv"), index=False)

    meta = dict(
        study="cross_asset_bb_squeeze",
        hypothesis=48,
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
    L.append("REPORT - HYPOTHESIS #48: MULTI-TIMEFRAME BOLLINGER SQUEEZE TREND (4h anchor / 30m execution)")
    L.append(f"Generated: {meta['generated_at']}  |  OVERALL VERDICT: {verdict}")
    L.append("Mechanics: 4h BB(29,1.82) inside KC(29EMA,1.56*ATR14) compression for >= 4 consecutive anchor bars then")
    L.append("release (fire), 4h ADX(14) >= 19.11 trend filter, 4h LR-slope(11) direction; entry at 30m signal-bar open;")
    L.append("initial SL 1.74*ATR4 and close-ratchet trail 3.71*ATR4 (ATR4 frozen at entry); honest gap fills;")
    L.append("worst-case same-bar stop; continuous forward-filled 30m/4h grid per SPEC_BB_SQZ.txt.")
    L.append("")
    for inst, res, blk in results:
        a = inst["asset"]
        L.append(f"=== {a} (cost floor {res['cost']} {res['unit']} RT) ===")
        tp = res["tape"]
        L.append(f"1. DATA: {tp['m1_rows']} M1 rows -> {tp['bars_30m']} 30m bars ({tp['synthetic_30m']} synthetic = "
                 f"{100.0 * tp['synthetic_30m'] / tp['bars_30m']:.1f}%) / {tp['bars_4h']} 4h bars, {tp['first']} .. {tp['last']}")
        L.append(f"   tape sha256 {tp['sha256'][:16]}...")
        f_ = res["funnel"]
        L.append(f"2. FUNNEL: armed 30m bars={f_['armed_bars']}  4h squeeze-ON bars={f_['squeeze_on_4h']}  4h FIRE events={f_['fired_4h']}")
        L.append(f"   candidate 30m bars: L={f_['cand_long']} S={f_['cand_short']}  ignored-while-open={f_['ignored_while_open']}  EXECUTED={f_['executed']}")
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
        L.append(f"4. GATES (independent, frozen in GATE_BB_SQZ.txt):")
        L.append(f"   G-BB1 executed >= 30          : {blk['n']:5d}          -> {'PASS' if blk['g1'] else 'FAIL'}")
        L.append(f"   G-BB2 Net Sharpe > 0           : {blk['sharpe']:9.3f}    -> {'PASS' if blk['g2'] else 'FAIL'}")
        L.append(f"   G-BB3 MBB P(mean<=0) <= 0.20   : {blk['p_boot']:9.4f}    -> {'PASS' if blk['g3'] else 'FAIL'}  (block={BLOCK}, {BOOT_ITERS} resamples, seed {SEED})")
        L.append(f"   {a} VERDICT: {'PASS' if blk['passed'] else 'FAIL'}")
        L.append("")
    L.append(f"OVERALL VERDICT: {verdict} (requires all three instruments to pass G-BB1..G-BB3 independently)")
    L.append("")
    L.append("HONESTY NOTES")
    L.append("- Continuous forward-fill convention: synthetic flat bars fill no-tick minutes, maintenance breaks, holidays")
    L.append("  and weekends; LR slope is exactly 0 on constant windows (arms neither side), but Wilder ATR/ADX decay across")
    L.append("  long gaps, so post-gap fires can carry mechanically tightened stops - a declared consequence of the continuity")
    L.append("  convention (SPEC 2.2/9.3), not a post-hoc fix.")
    L.append("- The fired flag of an anchor bar applies to every 30m bar of the following 4h window; after an exit inside a")
    L.append("  window, later bars of the same window may re-enter (SPEC 3.5).")
    L.append("- Single pre-declared parameterization (owner-supplied); no optimization, no re-runs; sides, years and")
    L.append("  instruments reported as run. Phase-1 screen is in-sample by construction; MFE/MAE are exploratory")
    L.append("  descriptors, never results. Phase 2 (MT5 execution) only on overall PASS; any FAIL is final here.")
    report = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "REPORT_BB_SQZ_PHASE1.txt"), "w") as f:
        f.write(report)
    print()
    print(report)


if __name__ == "__main__":
    main()
