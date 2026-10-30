import os
import json
import hashlib

import numpy as np
import pandas as pd

SEED = 20260927
BOOT_ITERS = 10000
BLOCK = 5
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "results", "cross_asset_alligator")

INSTRUMENTS = [
    dict(asset="US500", tape=os.path.join(ROOT, "data", "results", "vwap_2021_2026", "us500tick_m1.csv"), cost=0.51, scale=1.0, unit="points"),
    dict(asset="EURUSD", tape=os.path.join(ROOT, "data", "results", "eurusd_onboarding", "eurusdtick_m1.csv"), cost=0.3, scale=10000.0, unit="pips"),
    dict(asset="XAUUSD", tape=os.path.join(ROOT, "data", "results", "xauusd_onboarding", "xauusdtick_m1.csv"), cost=0.41, scale=1.0, unit="price units"),
]

P4 = 14400 * 1_000_000_000  # 4h in wall-clock nanoseconds
BRACKET_ATR = 2.0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def smma(x, n):
    m = len(x)
    out = np.full(m, np.nan)
    if m >= n:
        out[n - 1] = x[:n].mean()
        for i in range(n, m):
            out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def shift_strict(a, k):
    out = np.full(len(a), np.nan)
    if len(a) > k:
        out[k:] = a[:-k]
    return out


def alligator(h, l):
    med = (h + l) / 2.0
    return shift_strict(smma(med, 5), 3), shift_strict(smma(med, 8), 5), shift_strict(smma(med, 13), 8)


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


def cmo(c, n=14):
    m = len(c)
    diff = np.full(m, np.nan)
    diff[1:] = c[1:] - c[:-1]
    up = np.where(diff > 0.0, diff, 0.0)
    dn = np.where(diff < 0.0, -diff, 0.0)
    su = pd.Series(up).rolling(n).sum().to_numpy()
    sd = pd.Series(dn).rolling(n).sum().to_numpy()
    out = np.full(m, np.nan)
    for i in range(m):
        if not np.isnan(su[i]) and (su[i] + sd[i]) > 0.0:
            out[i] = 100.0 * (su[i] - sd[i]) / (su[i] + sd[i])
    return out


def stoch_rsi_k(c, n=14, k_smooth=3):
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
    rs = pd.Series(rsi)
    lo = rs.rolling(n).min().to_numpy()
    hi = rs.rolling(n).max().to_numpy()
    raw = np.full(m, np.nan)
    for i in range(m):
        if not np.isnan(rsi[i]) and not np.isnan(lo[i]) and hi[i] > lo[i]:
            raw[i] = 100.0 * (rsi[i] - lo[i]) / (hi[i] - lo[i])
    return pd.Series(raw).rolling(k_smooth).mean().to_numpy()


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
    grid, o, h, l, c, n_real, n_m1 = build_1h(tape)
    n = len(grid)
    ny_idx = pd.to_datetime(pd.Index(grid * 3600), unit="s", utc=True).tz_convert("America/New_York").tz_localize(None)
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

    lips4, teeth4, jaw4 = alligator(h4, l4)
    ema100 = pd.Series(c4).ewm(span=100, adjust=False).mean().to_numpy()
    ema100[:99] = np.nan

    lips1, teeth1, jaw1 = alligator(h, l)
    adx1 = wilder_adx(h, l, c, 14)
    cmo1 = cmo(c, 14)
    srsi1 = stoch_rsi_k(c, 14, 3)
    atr1 = wilder_atr(h, l, c, 14)

    pos = np.searchsorted(keys4, lc_key, side="left")
    posc = np.clip(pos, 0, k4 - 1)
    valid = (lc_key >= keys4[0]) & (keys4[posc] == lc_key)
    m_lips4 = np.where(valid, lips4[posc], np.nan)
    m_teeth4 = np.where(valid, teeth4[posc], np.nan)
    m_jaw4 = np.where(valid, jaw4[posc], np.nan)
    m_ema100 = np.where(valid, ema100[posc], np.nan)

    p_lips1 = shift_strict(lips1, 1)
    p_teeth1 = shift_strict(teeth1, 1)
    p_jaw1 = shift_strict(jaw1, 1)
    p_close = shift_strict(c, 1)
    p_adx = shift_strict(adx1, 1)
    p_cmo = shift_strict(cmo1, 1)
    p_srsi = shift_strict(srsi1, 1)
    p_atr = shift_strict(atr1, 1)

    defined = ~(np.isnan(p_lips1) | np.isnan(p_teeth1) | np.isnan(p_jaw1) | np.isnan(p_close) | np.isnan(p_adx) |
                np.isnan(p_cmo) | np.isnan(p_srsi) | np.isnan(p_atr) | np.isnan(m_lips4) | np.isnan(m_teeth4) |
                np.isnan(m_jaw4) | np.isnan(m_ema100))
    armed = defined & (np.arange(n) >= 1)

    trend1_l = (p_lips1 > p_teeth1) & (p_teeth1 > p_jaw1) & (p_close > p_lips1)
    trend1_s = (p_lips1 < p_teeth1) & (p_teeth1 < p_jaw1) & (p_close < p_lips1)
    trend4_l = (m_lips4 > m_teeth4) & (m_teeth4 > m_jaw4) & (p_close > m_ema100)
    trend4_s = (m_lips4 < m_teeth4) & (m_teeth4 < m_jaw4) & (p_close < m_ema100)
    mom_l = (p_adx > 30.0) & (p_cmo > 20.0) & (p_srsi < 20.0)
    mom_s = (p_adx > 30.0) & (p_cmo < -20.0) & (p_srsi > 80.0)

    sig_l = armed & trend1_l & trend4_l & mom_l
    sig_s = armed & trend1_s & trend4_s & mom_s
    sig_any = sig_l | sig_s

    trades = []
    ignored_open = 0
    i = 1
    while i < n:
        if not sig_any[i]:
            i += 1
            continue
        side = 1 if sig_l[i] else -1
        atr_sig = p_atr[i]
        risk = BRACKET_ATR * atr_sig
        entry = o[i]
        sl = entry - risk if side == 1 else entry + risk
        tp = entry + risk if side == 1 else entry - risk
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
                ext = max(ext, entry - l[j])
                adv = max(adv, h[j] - entry)
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
                sl=sl,
                tp=tp,
                exit=exit_px,
                exit_reason=reason,
                bars_held=exit_j - i + 1,
                atr=atr_sig,
                mfe=ext * scale,
                mae=adv * scale,
                gross=gross,
                net=gross - cost,
                r_multiple=side * (exit_px - entry) / risk,
                entry_year=int(ny_idx[i].year),
            )
        )
        i = exit_j + 1

    t = pd.DataFrame(trades)
    res = dict(
        asset=asset,
        tape=dict(path=tape, sha256=sha256_file(tape), m1_rows=int(n_m1), bars_1h=int(n), real_1h=int(n_real), synthetic_1h=int(n - n_real), bars_4h=int(k4),
                  first=str(ny_idx[0]), last=str(ny_idx[n - 1])),
        funnel=dict(armed_bars=int(armed.sum()),
                    trend1_long=int((armed & trend1_l).sum()), trend1_short=int((armed & trend1_s).sum()),
                    trend4_long=int((armed & trend1_l & trend4_l).sum()), trend4_short=int((armed & trend1_s & trend4_s).sum()),
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
        res["gates"] = dict(G_ALI1=bool(blk["g1"]), G_ALI2=bool(blk["g2"]), G_ALI3=bool(blk["g3"]))
        res["gates"]["net_sharpe"] = blk["sharpe"]
        res["gates"]["p_mbb"] = blk["p_boot"]
        res["passed"] = blk["passed"]
        all_trades.append(t)
        results.append((inst, res, blk))
        print(f"{inst['asset']}: {res['tape']['m1_rows']} M1 rows -> {res['tape']['bars_1h']} 1h bars "
              f"({res['tape']['synthetic_1h']} synthetic) / {res['tape']['bars_4h']} 4h bars; executed {blk['n']}")

    overall = all(r[2]["passed"] for r in results)
    verdict = "PASS" if overall else "FAIL"

    if any(len(t) for t in all_trades):
        pd.concat(all_trades, ignore_index=True).to_csv(os.path.join(OUT, "alligator_signals.csv"), index=False)

    meta = dict(
        study="cross_asset_alligator",
        hypothesis=49,
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
    L.append("REPORT - HYPOTHESIS #49: ALLIGATORAI TREND STRATEGY (4h anchor / 1h execution / fixed 1:1 ATR bracket)")
    L.append(f"Generated: {meta['generated_at']}  |  OVERALL VERDICT: {verdict}")
    L.append("Mechanics: 1h Alligator(5/8/13, shifts 3/5/8, SMMA of median) alignment + close vs Lips, 4h Alligator")
    L.append("alignment + close vs 4h EMA100, momentum gate ADX(14)>30 + CMO(14)>+/-20 + StochRSI_K(14,14,3) beyond")
    L.append("80/20; entry at 1h signal-bar open; fixed bracket SL/TP = +/-2.0*ATR1h(t-1) (1:1); honest gap fills;")
    L.append("worst-case same-bar default to SL; continuous forward-filled 1h/4h grid per SPEC_ALLIGATOR.txt.")
    L.append("")
    for inst, res, blk in results:
        a = inst["asset"]
        L.append(f"=== {a} (cost floor {res['cost']} {res['unit']} RT) ===")
        tp = res["tape"]
        L.append(f"1. DATA: {tp['m1_rows']} M1 rows -> {tp['bars_1h']} 1h bars ({tp['synthetic_1h']} synthetic = "
                 f"{100.0 * tp['synthetic_1h'] / tp['bars_1h']:.1f}%) / {tp['bars_4h']} 4h bars, {tp['first']} .. {tp['last']}")
        L.append(f"   tape sha256 {tp['sha256'][:16]}...")
        f_ = res["funnel"]
        L.append(f"2. FUNNEL: armed bars={f_['armed_bars']}  1h-trend L/S={f_['trend1_long']}/{f_['trend1_short']}")
        L.append(f"   +4h-trend L/S={f_['trend4_long']}/{f_['trend4_short']}  +momentum (candidates) L/S={f_['cand_long']}/{f_['cand_short']}")
        L.append(f"   ignored-while-open={f_['ignored_while_open']}  EXECUTED={f_['executed']}")
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
        L.append(f"4. GATES (independent, frozen in GATE_ALLIGATOR.txt):")
        L.append(f"   G-ALI1 executed >= 30          : {blk['n']:5d}          -> {'PASS' if blk['g1'] else 'FAIL'}")
        L.append(f"   G-ALI2 Net Sharpe > 0           : {blk['sharpe']:9.3f}    -> {'PASS' if blk['g2'] else 'FAIL'}")
        L.append(f"   G-ALI3 MBB P(mean<=0) <= 0.20   : {blk['p_boot']:9.4f}    -> {'PASS' if blk['g3'] else 'FAIL'}  (block={BLOCK}, {BOOT_ITERS} resamples, seed {SEED})")
        L.append(f"   {a} VERDICT: {'PASS' if blk['passed'] else 'FAIL'}")
        L.append("")
    L.append(f"OVERALL VERDICT: {verdict} (requires all three instruments to pass G-ALI1..G-ALI3 independently)")
    L.append("")
    L.append("HONESTY NOTES")
    L.append("- Continuous forward-fill convention: synthetic flat bars fill no-tick minutes, maintenance breaks, holidays")
    L.append("  and weekends; CMO/StochRSI go undefined on flat windows (killing weekend signals), SMMA flattens toward the")
    L.append("  last close, and Wilder ATR/ADX decay across long gaps - declared consequences of the continuity convention")
    L.append("  (SPEC 2.2/9.3), not post-hoc fixes.")
    L.append("- Single pre-declared parameterization (owner-supplied); no optimization, no re-runs; sides, years and")
    L.append("  instruments reported as run. Phase-1 screen is in-sample by construction; MFE/MAE are exploratory")
    L.append("  descriptors, never results. Phase 2 (MT5 execution) only on overall PASS; any FAIL is final here.")
    report = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "REPORT_ALLIGATOR_PHASE1.txt"), "w") as f:
        f.write(report)
    print()
    print(report)


if __name__ == "__main__":
    main()
