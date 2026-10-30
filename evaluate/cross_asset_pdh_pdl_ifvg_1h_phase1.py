import os
import json
import hashlib

import numpy as np
import pandas as pd

SEED = 20261001
BOOT_ITERS = 10000
BLOCK = 5
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "results", "cross_asset_pdh_pdl_ifvg")

INSTRUMENTS = [
    dict(asset="US500", tape=os.path.join(ROOT, "data", "results", "vwap_2021_2026", "us500tick_m1.csv"), cost=0.51, scale=1.0, unit="points"),
    dict(asset="EURUSD", tape=os.path.join(ROOT, "data", "results", "eurusd_onboarding", "eurusdtick_m1.csv"), cost=0.3, scale=10000.0, unit="pips"),
    dict(asset="XAUUSD", tape=os.path.join(ROOT, "data", "results", "xauusd_onboarding", "xauusdtick_m1.csv"), cost=0.41, scale=1.0, unit="price units"),
]

NS_H = 3600 * 1_000_000_000
P1D = 24 * NS_H
O17 = 17 * NS_H


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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

    # D1 anchor bars: NY wall [17:00 -> next 17:00) buckets from the continuous 1h grid.
    dkey = wns - ((wns - O17) % P1D)
    gd = (
        pd.DataFrame({"key": dkey, "o": o, "h": h, "l": l, "c": c})
        .groupby("key", sort=True)
        .agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    )
    dkeys = gd.index.to_numpy(dtype=np.int64)
    d_h = gd["h"].to_numpy(float)
    d_l = gd["l"].to_numpy(float)
    nd1 = len(dkeys)

    # Active FVG levels as of the OPEN of bar t (detection at t uses completed bars t-3..t-1; most-recent-wins).
    bull_new = np.full(n, np.nan)
    bear_new = np.full(n, np.nan)
    bull_new[3:] = np.where(l[2 : n - 1] > h[0 : n - 3], h[0 : n - 3], np.nan)
    bear_new[3:] = np.where(h[2 : n - 1] < l[0 : n - 3], l[0 : n - 3], np.nan)
    bull_bot = pd.Series(bull_new).ffill().to_numpy(float)
    bear_top = pd.Series(bear_new).ffill().to_numpy(float)

    # Per-bar session state, stored as of the END of each completed bar (signals at t read index t-1).
    ses = dkey
    pdh_arr = np.full(n, np.nan)
    pdl_arr = np.full(n, np.nan)
    swept_h = np.zeros(n, dtype=bool)
    swept_l = np.zeros(n, dtype=bool)
    sweep_hi = np.full(n, np.nan)
    sweep_lo = np.full(n, np.nan)

    cur = None
    pdh = np.nan
    pdl = np.nan
    swh = False
    swl = False
    shi = -np.inf
    slo = np.inf
    for i in range(n):
        if ses[i] != cur:
            cur = ses[i]
            p = np.searchsorted(dkeys, cur - P1D, side="left")
            if p < nd1 and dkeys[p] == cur - P1D:
                pdh = d_h[p]
                pdl = d_l[p]
            else:
                pdh = np.nan
                pdl = np.nan
            swh = False
            swl = False
            shi = -np.inf
            slo = np.inf
        if not np.isnan(pdh):
            if h[i] > pdh:
                swh = True
                shi = max(shi, h[i])
            if l[i] < pdl:
                swl = True
                slo = min(slo, l[i])
        pdh_arr[i] = pdh
        pdl_arr[i] = pdl
        swept_h[i] = swh
        swept_l[i] = swl
        sweep_hi[i] = shi if swh else np.nan
        sweep_lo[i] = slo if swl else np.nan

    # Signal masks at the open of bar t: strictly end-of-(t-1) state, same session as t-1.
    sig_s = np.zeros(n, dtype=bool)
    sig_l = np.zeros(n, dtype=bool)
    same = np.zeros(n, dtype=bool)
    same[1:] = ses[1:] == ses[:-1]
    sig_s[1:] = same[1:] & swept_h[:-1] & (c[:-1] < bull_bot[1:])
    sig_l[1:] = same[1:] & swept_l[:-1] & (c[:-1] > bear_top[1:])
    sig_any = sig_s | sig_l

    trades = []
    ignored_open = 0
    i = 1
    while i < n:
        if not sig_any[i]:
            i += 1
            continue
        side = -1 if sig_s[i] else 1
        entry = o[i]
        if side == -1:
            sl = sweep_hi[i - 1]
            tp = pdl_arr[i - 1]
        else:
            sl = sweep_lo[i - 1]
            tp = pdh_arr[i - 1]
        exit_px = None
        reason = None
        exit_j = None
        ext = -np.inf
        adv = -np.inf
        if side == -1:
            if o[i] >= sl:
                exit_px, reason, exit_j = o[i], "SL_GAP", i
            elif o[i] <= tp:
                exit_px, reason, exit_j = o[i], "TP_GAP", i
            elif h[i] >= sl:
                exit_px, reason, exit_j = sl, "SL", i
            elif l[i] <= tp:
                exit_px, reason, exit_j = tp, "TP", i
        else:
            if o[i] <= sl:
                exit_px, reason, exit_j = o[i], "SL_GAP", i
            elif o[i] >= tp:
                exit_px, reason, exit_j = o[i], "TP_GAP", i
            elif l[i] <= sl:
                exit_px, reason, exit_j = sl, "SL", i
            elif h[i] >= tp:
                exit_px, reason, exit_j = tp, "TP", i
        if exit_px is None:
            for j in range(i + 1, n):
                if side == -1:
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
                else:
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
        if exit_px is None:
            exit_px, reason, exit_j = c[n - 1], "FORCED_EOD", n - 1
        ignored_open += int(sig_any[i : exit_j + 1].sum())
        hs = h[i : exit_j + 1]
        ls = l[i : exit_j + 1]
        if side == 1:
            ext = float(hs.max() - entry)
            adv = float(entry - ls.min())
        else:
            ext = float(entry - ls.min())
            adv = float(hs.max() - entry)
        gross = side * (exit_px - entry) * scale
        risk = abs(entry - sl)
        trades.append(
            dict(
                asset=asset,
                signal_time=ny_idx[i].strftime("%Y-%m-%d %H:%M"),
                entry_time=ny_idx[i].strftime("%Y-%m-%d %H:%M"),
                exit_time=ny_idx[exit_j].strftime("%Y-%m-%d %H:%M"),
                side="SHORT" if side == -1 else "LONG",
                entry=entry,
                sl=sl,
                tp=tp,
                pdh=pdh_arr[i - 1],
                pdl=pdl_arr[i - 1],
                sweep_high=sweep_hi[i - 1] if side == -1 else np.nan,
                sweep_low=sweep_lo[i - 1] if side == 1 else np.nan,
                fvg_level=bull_bot[i] if side == -1 else bear_top[i],
                exit=exit_px,
                exit_reason=reason,
                bars_held=exit_j - i + 1,
                mfe=ext * scale,
                mae=adv * scale,
                gross=gross,
                net=gross - cost,
                r_multiple=side * (exit_px - entry) / risk if risk > 0.0 else np.nan,
                entry_year=int(ny_idx[i].year),
            )
        )
        i = exit_j + 1

    t = pd.DataFrame(trades)
    res = dict(
        asset=asset,
        tape=dict(path=tape, sha256=sha256_file(tape), m1_rows=int(n_m1), bars_1h=int(n), real_1h=int(n_real), synthetic_1h=int(n - n_real),
                  bars_d1=int(nd1), first=str(ny_idx[0]), last=str(ny_idx[n - 1])),
        funnel=dict(bull_fvg_detected=int(np.isfinite(bull_new).sum()), bear_fvg_detected=int(np.isfinite(bear_new).sum()),
                    pdh_sweep_bars=int(swept_h.sum()), pdl_sweep_bars=int(swept_l.sum()),
                    cand_short=int(sig_s.sum()), cand_long=int(sig_l.sum()),
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
        res["gates"] = dict(G_SMC1=bool(blk["g1"]), G_SMC2=bool(blk["g2"]), G_SMC3=bool(blk["g3"]))
        res["gates"]["net_sharpe"] = blk["sharpe"]
        res["gates"]["p_mbb"] = blk["p_boot"]
        res["passed"] = blk["passed"]
        all_trades.append(t)
        results.append((inst, res, blk))
        print(f"{inst['asset']}: {res['tape']['bars_1h']} 1h bars ({res['tape']['synthetic_1h']} synthetic) / "
              f"{res['tape']['bars_d1']} D1 anchors; candidates L/S={res['funnel']['cand_long']}/{res['funnel']['cand_short']} "
              f"-> executed {blk['n']}")

    overall = all(r[2]["passed"] for r in results)
    verdict = "PASS" if overall else "FAIL"

    if any(len(t) for t in all_trades):
        pd.concat(all_trades, ignore_index=True).to_csv(os.path.join(OUT, "pdh_ifvg_signals.csv"), index=False)

    meta = dict(
        study="cross_asset_pdh_pdl_ifvg",
        hypothesis=53,
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
    L.append("REPORT - HYPOTHESIS #53: DAILY RANGE SWEEP & INVERSION FVG (1h execution / D1 17:00 ET anchor)")
    L.append(f"Generated: {meta['generated_at']}  |  OVERALL VERDICT: {verdict}")
    L.append("Mechanics: prior-day-high (low) sweep by any 1h bar of the current 17:00 ET session, then a 1h close")
    L.append("through the far side of the most recent opposite-side 1h FVG (inversion): SHORT if Swept_PDH and")
    L.append("Close(t-1) < bullish-FVG bottom (High of candle-1); LONG if Swept_PDL and Close(t-1) > bearish-FVG top")
    L.append("(Low of candle-1). Entry at 1h signal-bar open; SL = signal-time Sweep_High/Low; TP = opposite daily")
    L.append("bound (PDL/PDH); worst-case same-bar SL-before-TP, honest gap fills, FORCED_EOD included; continuous")
    L.append("forward-filled grids per SPEC_PDH_IFVG.txt.")
    L.append("")
    for inst, res, blk in results:
        a = inst["asset"]
        L.append(f"=== {a} (cost floor {res['cost']} {res['unit']} RT) ===")
        tp = res["tape"]
        L.append(f"1. DATA: {tp['m1_rows']} M1 rows -> 1h base {tp['bars_1h']} ({tp['synthetic_1h']} synthetic = "
                 f"{100.0 * tp['synthetic_1h'] / tp['bars_1h']:.1f}%) / {tp['bars_d1']} D1 anchors, {tp['first']} .. {tp['last']}")
        L.append(f"   tape sha256 {tp['sha256'][:16]}...")
        f_ = res["funnel"]
        L.append(f"2. FUNNEL: 1h FVGs detected bull/bear={f_['bull_fvg_detected']}/{f_['bear_fvg_detected']}  "
                 f"bars-ending-swept PDH/PDL={f_['pdh_sweep_bars']}/{f_['pdl_sweep_bars']}")
        L.append(f"   signal candidates SHORT={f_['cand_short']} LONG={f_['cand_long']}  ignored-while-open={f_['ignored_while_open']}  "
                 f"EXECUTED={f_['executed']}")
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
        L.append(f"4. GATES (independent, frozen in GATE_PDH_IFVG.txt):")
        L.append(f"   G-SMC1 executed >= 30          : {blk['n']:5d}          -> {'PASS' if blk['g1'] else 'FAIL'}")
        L.append(f"   G-SMC2 Net Sharpe > 0           : {blk['sharpe']:9.3f}    -> {'PASS' if blk['g2'] else 'FAIL'}")
        L.append(f"   G-SMC3 MBB P(mean<=0) <= 0.20   : {blk['p_boot']:9.4f}    -> {'PASS' if blk['g3'] else 'FAIL'}  (block={BLOCK}, {BOOT_ITERS} resamples, seed {SEED})")
        L.append(f"   {a} VERDICT: {'PASS' if blk['passed'] else 'FAIL'}")
        L.append("")
    L.append(f"OVERALL VERDICT: {verdict} (requires all three instruments to pass G-SMC1..G-SMC3 independently)")
    L.append("")
    L.append("HONESTY NOTES")
    L.append("- Strictly causal state: PDH/PDL come only from D1 buckets completed at or before the open of the signal")
    L.append("  bar; sweep flags/extremes are read as of the end of bar t-1 within the same 17:00 session; FVG detection")
    L.append("  uses completed bars t-3..t-1 only (a bullish gap detected at t can never satisfy its own short")
    L.append("  condition, since Close(t-1) >= Low(t-1) > High(t-3)).")
    L.append("- Declared mechanic absences (not oversights): no FVG staleness/invalidation rule (a gap persists until")
    L.append("  replaced by a newer same-side gap, even after close-throughs), no sweep recency limit, no session-time")
    L.append("  filter on entries.")
    L.append("- Continuous forward-fill convention: synthetic flat bars carry the last real close; flat closes can")
    L.append("  satisfy close-through conditions and carried highs/lows extend sweeps only within the carried range;")
    L.append("  entries on synthetic bars fill at the carried price and the walk resumes at the real reopen with honest")
    L.append("  gap fills (SPEC 2.2/9.3). No bar-skipping fix applied.")
    L.append("- Structural bracket: SL = signal-time sweep extreme, TP = opposite prior-day bound; worst-case same-bar")
    L.append("  ordering (SL before TP) and honest entry/subsequent-bar gap fills; degenerate entry-bar gap exits pay")
    L.append("  the cost floor.")
    L.append("- Single pre-declared parameterization (the mechanic has no free numeric parameters); no optimization, no")
    L.append("  re-runs; sides, years and instruments reported as run. Phase-1 screen is in-sample by construction;")
    L.append("  MFE/MAE are exploratory descriptors, never results. Phase 2 (MT5 execution) only on overall PASS; any")
    L.append("  FAIL is final here.")
    report = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "REPORT_PDH_IFVG_PHASE1.txt"), "w") as f:
        f.write(report)
    print()
    print(report)


if __name__ == "__main__":
    main()
