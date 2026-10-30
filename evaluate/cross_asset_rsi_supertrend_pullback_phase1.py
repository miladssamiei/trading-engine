import os
import json
import hashlib

import numpy as np
import pandas as pd

SEED = 20260930
BOOT_ITERS = 10000
BLOCK = 5
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "results", "cross_asset_rsi_pullback")

INSTRUMENTS = [
    dict(asset="US500", tape=os.path.join(ROOT, "data", "results", "vwap_2021_2026", "us500tick_m1.csv"), cost=0.51, scale=1.0, unit="points"),
    dict(asset="EURUSD", tape=os.path.join(ROOT, "data", "results", "eurusd_onboarding", "eurusdtick_m1.csv"), cost=0.3, scale=10000.0, unit="pips"),
    dict(asset="XAUUSD", tape=os.path.join(ROOT, "data", "results", "xauusd_onboarding", "xauusdtick_m1.csv"), cost=0.41, scale=1.0, unit="price units"),
]

P4 = 14400 * 1_000_000_000
SL_MULT = 6.0


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


def wilder_rsi(c, n=14):
    m = len(c)
    diff = np.full(m, np.nan)
    diff[1:] = c[1:] - c[:-1]
    gain = np.where(diff > 0.0, diff, 0.0)
    loss = np.where(diff < 0.0, -diff, 0.0)
    ag = np.full(m, np.nan)
    al = np.full(m, np.nan)
    if m >= n + 1:
        ag[n] = gain[1 : n + 1].mean()
        al[n] = loss[1 : n + 1].mean()
        for i in range(n + 1, m):
            ag[i] = (ag[i - 1] * (n - 1) + gain[i]) / n
            al[i] = (al[i - 1] * (n - 1) + loss[i]) / n
    rsi = np.full(m, np.nan)
    for i in range(n, m):
        if np.isnan(ag[i]) or np.isnan(al[i]):
            continue
        if al[i] <= 0.0:
            rsi[i] = 100.0 if ag[i] > 0.0 else np.nan
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + ag[i] / al[i])
    return rsi


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


def supertrend_dir(h, l, c, n=10, mult=3.0):
    m = len(c)
    atr = wilder_atr(h, l, c, n)
    med = (h + l) / 2.0
    ub = med + mult * atr
    lb = med - mult * atr
    fub = np.full(m, np.nan)
    flb = np.full(m, np.nan)
    direction = np.full(m, np.nan)
    start = None
    for i in range(m):
        if np.isnan(atr[i]):
            continue
        if start is None:
            fub[i] = ub[i]
            flb[i] = lb[i]
            direction[i] = 1.0
            start = i
            continue
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if direction[i - 1] == 1.0:
            direction[i] = -1.0 if c[i] < flb[i] else 1.0
        else:
            direction[i] = 1.0 if c[i] > fub[i] else -1.0
    return direction


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


def run_instrument(inst):
    asset, tape, cost, scale = inst["asset"], inst["tape"], inst["cost"], inst["scale"]
    grid, o, h, l, c, n_real, n_m1 = build_15m(tape)
    n = len(grid)
    ny_idx = pd.to_datetime(pd.Index(grid * 900), unit="s", utc=True).tz_convert("America/New_York").tz_localize(None)
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

    rsi4 = wilder_rsi(c4, 14)
    st4 = supertrend_dir(h4, l4, c4, 10, 3.0)
    adx4 = wilder_adx(h4, l4, c4, 14)

    pos = np.searchsorted(keys4, lc_key, side="left")
    posc = np.clip(pos, 0, k4 - 1)
    valid = (lc_key >= keys4[0]) & (keys4[posc] == lc_key)
    m_rsi4 = np.where(valid, rsi4[posc], np.nan)
    m_st4 = np.where(valid, st4[posc], np.nan)
    m_adx4 = np.where(valid, adx4[posc], np.nan)

    sma20 = pd.Series(c).rolling(20).mean().to_numpy()
    std20 = pd.Series(c).rolling(20).std(ddof=0).to_numpy()
    bbu = sma20 + 2.0 * std20
    bbl = sma20 - 2.0 * std20
    adx15 = wilder_adx(h, l, c, 14)
    atr15 = wilder_atr(h, l, c, 14)

    p_bbu = np.roll(bbu, 1)
    p_bbu[0] = np.nan
    p_bbl = np.roll(bbl, 1)
    p_bbl[0] = np.nan
    p_adx15 = np.roll(adx15, 1)
    p_adx15[0] = np.nan
    p_atr15 = np.roll(atr15, 1)
    p_atr15[0] = np.nan

    armed = (~np.isnan(p_bbu)) & (~np.isnan(p_bbl)) & (~np.isnan(p_adx15)) & (~np.isnan(p_atr15)) & \
            (~np.isnan(m_rsi4)) & (~np.isnan(m_st4)) & (~np.isnan(m_adx4))
    setup_l = armed & (m_rsi4 > 70.0) & (m_st4 == 1.0) & (m_adx4 > 40.0) & (p_adx15 > 20.0)
    setup_s = armed & (m_rsi4 < 30.0) & (m_st4 == -1.0) & (m_adx4 > 40.0) & (p_adx15 > 20.0)
    setup_any = setup_l | setup_s

    trades = []
    setups_placed = 0
    setups_placed_l = 0
    setups_placed_s = 0
    ignored_open = 0
    i = 1
    while i < n:
        if not setup_any[i]:
            i += 1
            continue
        side = 1 if setup_l[i] else -1
        setups_placed += 1
        if side == 1:
            setups_placed_l += 1
            if not (l[i] <= p_bbl[i]):
                i += 1
                continue
            fill = min(o[i], p_bbl[i])
            sl = fill - SL_MULT * p_atr15[i]
            tp = p_bbu[i]
        else:
            setups_placed_s += 1
            if not (h[i] >= p_bbu[i]):
                i += 1
                continue
            fill = max(o[i], p_bbu[i])
            sl = fill + SL_MULT * p_atr15[i]
            tp = p_bbl[i]
        exit_px = None
        reason = None
        exit_j = None
        ext = -np.inf
        adv = -np.inf
        if side == 1:
            if l[i] <= sl:
                exit_px, reason, exit_j = sl, "SL", i
            elif h[i] >= tp:
                exit_px, reason, exit_j = tp, "TP", i
        else:
            if h[i] >= sl:
                exit_px, reason, exit_j = sl, "SL", i
            elif l[i] <= tp:
                exit_px, reason, exit_j = tp, "TP", i
        if exit_px is None:
            for j in range(i + 1, n):
                if side == 1:
                    if o[j] <= sl:
                        exit_px, reason, exit_j = o[j], "SL_GAP", j
                        break
                    if o[j] >= tp:
                        exit_px, reason, exit_j = o[j], "TP_GAP", j
                        break
                    if l[j] <= sl:
                        exit_px, reason, exit_j = sl, "SL", j
                        break
                    if h[j] >= tp:
                        exit_px, reason, exit_j = tp, "TP", j
                        break
                else:
                    if o[j] >= sl:
                        exit_px, reason, exit_j = o[j], "SL_GAP", j
                        break
                    if o[j] <= tp:
                        exit_px, reason, exit_j = o[j], "TP_GAP", j
                        break
                    if h[j] >= sl:
                        exit_px, reason, exit_j = sl, "SL", j
                        break
                    if l[j] <= tp:
                        exit_px, reason, exit_j = tp, "TP", j
                        break
        if exit_px is None:
            exit_px, reason, exit_j = c[n - 1], "FORCED_EOD", n - 1
        ignored_open += int(setup_any[i : exit_j + 1].sum())
        hs = h[i : exit_j + 1]
        ls = l[i : exit_j + 1]
        if side == 1:
            ext = float(hs.max() - fill)
            adv = float(fill - ls.min())
        else:
            ext = float(fill - ls.min())
            adv = float(hs.max() - fill)
        gross = side * (exit_px - fill) * scale
        trades.append(
            dict(
                asset=asset,
                signal_time=ny_idx[i].strftime("%Y-%m-%d %H:%M"),
                entry_time=ny_idx[i].strftime("%Y-%m-%d %H:%M"),
                exit_time=ny_idx[exit_j].strftime("%Y-%m-%d %H:%M"),
                side="LONG" if side == 1 else "SHORT",
                fill=fill,
                sl=sl,
                tp=tp,
                exit=exit_px,
                exit_reason=reason,
                bars_held=exit_j - i + 1,
                atr=p_atr15[i],
                mfe=ext * scale,
                mae=adv * scale,
                gross=gross,
                net=gross - cost,
                r_multiple=side * (exit_px - fill) / (SL_MULT * p_atr15[i]),
                entry_year=int(ny_idx[i].year),
            )
        )
        i = exit_j + 1

    t = pd.DataFrame(trades)
    res = dict(
        asset=asset,
        tape=dict(path=tape, sha256=sha256_file(tape), m1_rows=int(n_m1), bars_15m=int(n), real_15m=int(n_real), synthetic_15m=int(n - n_real), bars_4h=int(k4),
                  first=str(ny_idx[0]), last=str(ny_idx[n - 1])),
        funnel=dict(armed_bars=int(armed.sum()), setups_long=setups_placed_l, setups_short=setups_placed_s,
                    setups_total=setups_placed, unfilled=setups_placed - len(t),
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
        res["gates"] = dict(G_MR1=bool(blk["g1"]), G_MR2=bool(blk["g2"]), G_MR3=bool(blk["g3"]))
        res["gates"]["net_sharpe"] = blk["sharpe"]
        res["gates"]["p_mbb"] = blk["p_boot"]
        res["passed"] = blk["passed"]
        all_trades.append(t)
        results.append((inst, res, blk))
        print(f"{inst['asset']}: {res['tape']['bars_15m']} 15m bars ({res['tape']['synthetic_15m']} synthetic) / "
              f"{res['tape']['bars_4h']} 4h bars; setups {res['funnel']['setups_total']} -> fills {blk['n']}")

    overall = all(r[2]["passed"] for r in results)
    verdict = "PASS" if overall else "FAIL"

    if any(len(t) for t in all_trades):
        pd.concat(all_trades, ignore_index=True).to_csv(os.path.join(OUT, "rsi_pullback_signals.csv"), index=False)

    meta = dict(
        study="cross_asset_rsi_pullback",
        hypothesis=52,
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
    L.append("REPORT - HYPOTHESIS #52: TREND-ALIGNED BOLLINGER BAND PULLBACK (4h anchor / 15m limit execution)")
    L.append(f"Generated: {meta['generated_at']}  |  OVERALL VERDICT: {verdict}")
    L.append("Mechanics: strong-trend pullback - 4h RSI(14) > 70 (< 30) + Supertrend(10,3.0) direction + 4h ADX > 40 +")
    L.append("15m ADX > 20 gates; one-bar limit order at the prior 15m BB Lower/Upper (fill = min/max(Open, band),")
    L.append("equality fills); SL = fill -/+ 6.0*ATR15(t-1); TP = opposite band; worst-case same-bar ordering")
    L.append("(SL before TP), honest gap fills, FORCED_EOD included; continuous forward-filled grids per")
    L.append("SPEC_RSI_PULLBACK.txt.")
    L.append("")
    for inst, res, blk in results:
        a = inst["asset"]
        L.append(f"=== {a} (cost floor {res['cost']} {res['unit']} RT) ===")
        tp = res["tape"]
        L.append(f"1. DATA: {tp['m1_rows']} M1 rows -> {tp['bars_15m']} 15m bars ({tp['synthetic_15m']} synthetic = "
                 f"{100.0 * tp['synthetic_15m'] / tp['bars_15m']:.1f}%) / {tp['bars_4h']} 4h bars, {tp['first']} .. {tp['last']}")
        L.append(f"   tape sha256 {tp['sha256'][:16]}...")
        f_ = res["funnel"]
        L.append(f"2. FUNNEL (setups vs limit fills): armed bars={f_['armed_bars']}  setups L/S={f_['setups_long']}/{f_['setups_short']} "
                 f"(total {f_['setups_total']}, one-bar validity)")
        L.append(f"   fills (executed)={f_['executed']}  unfilled={f_['unfilled']}  fill rate="
                 f"{100.0 * f_['executed'] / f_['setups_total'] if f_['setups_total'] else float('nan'):.1f}%  "
                 f"setups ignored-while-open={f_['ignored_while_open']}")
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
        L.append(f"4. GATES (independent, frozen in GATE_RSI_PULLBACK.txt):")
        L.append(f"   G-MR1 executed >= 30          : {blk['n']:5d}          -> {'PASS' if blk['g1'] else 'FAIL'}")
        L.append(f"   G-MR2 Net Sharpe > 0           : {blk['sharpe']:9.3f}    -> {'PASS' if blk['g2'] else 'FAIL'}")
        L.append(f"   G-MR3 MBB P(mean<=0) <= 0.20   : {blk['p_boot']:9.4f}    -> {'PASS' if blk['g3'] else 'FAIL'}  (block={BLOCK}, {BOOT_ITERS} resamples, seed {SEED})")
        L.append(f"   {a} VERDICT: {'PASS' if blk['passed'] else 'FAIL'}")
        L.append("")
    L.append(f"OVERALL VERDICT: {verdict} (requires all three instruments to pass G-MR1..G-MR3 independently)")
    L.append("")
    L.append("HONESTY NOTES")
    L.append("- Continuous forward-fill convention: synthetic flat bars fill no-tick minutes, maintenance breaks, holidays")
    L.append("  and weekends; the ADX15>20 and 4h ADX>40 gates decay across flat runs and suppress degenerate collapsed-band")
    L.append("  equality fills, while Wilder ATR decay tightens the 6xATR stop for post-gap fills (SPEC 2.2/9.3) - declared")
    L.append("  consequences, not post-hoc fixes.")
    L.append("- Limit fills are same-bar with honest gap pricing (fill = min/max(Open, band)); the entry bar participates")
    L.append("  in the exit walk with worst-case SL-before-TP ordering (SPEC 4.2/5.3).")
    L.append("- Single pre-declared parameterization (owner-supplied); no optimization, no re-runs; sides, years and")
    L.append("  instruments reported as run. Phase-1 screen is in-sample by construction; MFE/MAE are exploratory")
    L.append("  descriptors, never results. Phase 2 (MT5 execution) only on overall PASS; any FAIL is final here.")
    report = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "REPORT_RSI_PULLBACK_PHASE1.txt"), "w") as f:
        f.write(report)
    print()
    print(report)


if __name__ == "__main__":
    main()
