import os
import json
import hashlib

import numpy as np
import pandas as pd

SEED = 20261003
BOOT_ITERS = 10000
BLOCK = 10
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "results", "crypto_portfolio_kc")

INSTRUMENTS = [
    dict(asset="BTCUSD", tape=os.path.join(ROOT, "data", "results", "virgin_assets", "btc_m1.csv"), cost=15.0, scale=1.0, unit="price units"),
    dict(asset="ETHUSD", tape=os.path.join(ROOT, "data", "results", "virgin_assets", "eth_m1.csv"), cost=2.0, scale=1.0, unit="price units"),
]

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
        gross = side * (exit_px - entry) * scale
        net = gross - cost
        trades.append(
            dict(
                asset=asset,
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
                mfe=ext * scale,
                mae=adv * scale,
                gross=gross,
                net=net,
                r_multiple=side * (exit_px - entry) / dist,
                net_r=net / dist,
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


def max_drawdown(cum):
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    k = int(np.argmax(dd))
    if dd[k] <= 0.0:
        return 0.0, None, None
    start = int(np.argmax(cum[: k + 1]))
    return float(dd[k]), start, k


def main():
    os.makedirs(OUT, exist_ok=True)
    per_asset = []
    ledgers = []
    for inst in INSTRUMENTS:
        t, res = run_instrument(inst)
        per_asset.append((inst, res, t))
        ledgers.append(t)
        print(f"{inst['asset']}: executed {len(t)} trades, mean net_r "
              f"{(t['net_r'].mean() if len(t) else float('nan')):.4f}")

    ledger = pd.concat(ledgers, ignore_index=True)
    ledger = ledger.sort_values(by=["entry_time", "asset"], kind="mergesort").reset_index(drop=True)

    n_tr = len(ledger)
    r = ledger["net_r"].to_numpy(float)
    gr = ledger["r_multiple"].to_numpy(float)

    cum = np.cumsum(r)
    mdd, mdd_s, mdd_e = max_drawdown(cum)
    mean_r = float(r.mean())
    median_r = float(np.median(r))
    total_r = float(r.sum())
    win = float((r > 0).mean())
    gross_mean_r = float(gr.mean())
    sharpe_r = np.nan
    if n_tr >= 2:
        s = r.std(ddof=1)
        sharpe_r = float(r.mean() / s * np.sqrt(252.0)) if s > 0 else (np.inf if r.mean() > 0 else -np.inf)
    p_boot = mbb_p_le_zero(r) if n_tr >= BLOCK else np.nan

    g1 = n_tr >= 300
    g2 = mean_r > 0.15
    g3 = (not np.isnan(p_boot)) and p_boot <= 0.10
    verdict = "PASS" if (g1 and g2 and g3) else "FAIL"

    ledger.to_csv(os.path.join(OUT, "crypto_portfolio_signals.csv"), index=False)

    yearly = ledger.groupby("entry_year")["net_r"].agg(["count", "sum", "mean"])
    asset_tbl = ledger.groupby("asset")["net_r"].agg(["count", "sum", "mean"])
    side_tbl = ledger.groupby("side")["net_r"].agg(["count", "mean"])
    exit_mix = ledger["exit_reason"].value_counts().to_dict()
    both_neg_years = [int(y) for y, rr in yearly.iterrows() if rr["sum"] < 0]

    meta = dict(
        study="crypto_portfolio_kc",
        hypothesis=55,
        generated_at=pd.Timestamp.utcnow().isoformat(),
        script_sha256=sha256_file(os.path.abspath(__file__)),
        seed=SEED, boot_iters=BOOT_ITERS, block=BLOCK,
        instruments=[dict(asset=res["asset"], tape=res["tape"], funnel=res["funnel"], cost=res["cost"],
                          unit=res["unit"]) for _, res, _ in per_asset],
        portfolio=dict(n=int(n_tr), mean_net_r=mean_r, median_net_r=median_r, total_net_r=total_r,
                       win=win, gross_mean_r=gross_mean_r, sharpe_r=sharpe_r, p_mbb=p_boot,
                       max_drawdown_r=mdd),
        gates=dict(G_PORT1=bool(g1), G_PORT2=bool(g2), G_PORT3=bool(g3)),
        verdict=verdict,
    )
    with open(os.path.join(OUT, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    L = []
    L.append("REPORT - HYPOTHESIS #55: CRYPTO PORTFOLIO KC BREAKOUT - POOLED R-MULTIPLE VALIDATION")
    L.append(f"Generated: {meta['generated_at']}  |  OVERALL VERDICT: {verdict}")
    L.append("Mechanics frozen verbatim from #51/#54 (4h KC EMA29 +/- 2.77*ATR29 breakout + D1 EMA(50) 17:00 ET")
    L.append("direction, EMA-riding stop with 1.5xATR clamp, exit checks before ratchet, honest gap fills,")
    L.append("worst-case same-bar, 1 trade/instrument); BTCUSD + ETHUSD trades pooled into one chronological")
    L.append("ledger and normalized by INITIAL risk distance: Net_R = (side*(Exit-Entry) - cost) / Initial_Risk.")
    L.append("Costs are the #54 owner-estimated floors (BTCUSD 15.0 / ETHUSD 2.0 u RT), applied in native units")
    L.append("before normalization. NOT a fresh touch of the data: the same instrument-level trade sets were")
    L.append("already reported in #54 (SPEC section 0); this is the pre-declared portfolio aggregation step.")
    L.append("")
    for inst, res, t in per_asset:
        a = inst["asset"]
        at = asset_tbl.loc[a]
        f_ = res["funnel"]
        tp = res["tape"]
        L.append(f"=== {a} leg (cost floor {res['cost']} {res['unit']} RT, owner estimate) ===")
        L.append(f"1. DATA: {tp['m1_rows']} M1 rows -> 1h base {tp['bars_1h_base']} ({tp['synthetic_1h']} synthetic) / "
                 f"{tp['bars_4h']} 4h bars / {tp['bars_d1']} D1 anchors, {tp['first']} .. {tp['last']}")
        L.append(f"   tape sha256 {tp['sha256'][:16]}...")
        L.append(f"2. FUNNEL: armed 4h bars={f_['armed_bars']}  candidates L/S={f_['cand_long']}/{f_['cand_short']}  "
                 f"ignored-while-open={f_['ignored_while_open']}  EXECUTED={f_['executed']}")
        L.append(f"3. R EXPECTANCY: n={int(at['count'])}  mean Net R={at['mean']:.4f}/trade  total Net R={at['sum']:.1f}"
                 f"  (mean gross R={t['r_multiple'].mean():.4f})")
        st = t.groupby("side")["net_r"].agg(["count", "mean"])
        L.append("   sides: " + "  ".join(f"{k}: n={int(v['count'])} meanR={v['mean']:.4f}" for k, v in st.iterrows()))
        L.append("")
    L.append("=== POOLED PORTFOLIO LEDGER (gated) ===")
    L.append(f"4. LEDGER: {n_tr} trades sorted by 4h entry time (tie-break asset name), {ledger['entry_time'].iloc[0] if n_tr else 'n/a'} .. "
             f"{ledger['entry_time'].iloc[-1] if n_tr else 'n/a'}")
    L.append(f"5. POOLED NET-R ECONOMICS: mean={mean_r:.4f} R/trade  median={median_r:.4f}  total={total_r:.1f} R  "
             f"win={100.0 * win:.1f}%  per-trade R Sharpe={sharpe_r:.3f} (informational)")
    L.append(f"   gross mean R={gross_mean_r:.4f} vs pooled cost drag (net vs gross mean R={mean_r - gross_mean_r:+.4f})")
    L.append(f"   EVENT-INDEXED equity curve (cumulative Net R in entry order): final={cum[-1]:.1f} R; "
             f"max drawdown={mdd:.1f} R"
             + (f" (trades {mdd_s + 1}->{mdd_e + 1}: {ledger['entry_time'].iloc[mdd_s]} .. {ledger['entry_time'].iloc[mdd_e]})" if mdd_s is not None else ""))
    L.append("   per-year pooled Net R:")
    for y, rr in yearly.iterrows():
        L.append(f"      {y}: n={int(rr['count']):4d}  total={rr['sum']:8.1f} R  mean={rr['mean']:7.4f}")
    L.append("   negative years: " + (", ".join(str(y) for y in both_neg_years) if both_neg_years else "none"))
    L.append("   sides: " + "  ".join(f"{k}: n={int(v['count'])} meanR={v['mean']:.4f}" for k, v in side_tbl.iterrows()))
    L.append(f"   exit mix: {exit_mix}")
    L.append("6. GATES (pooled ledger, frozen in GATE_CRYPTO_PORTFOLIO.txt):")
    L.append(f"   G-PORT1 pooled trades >= 300        : {n_tr:5d}          -> {'PASS' if g1 else 'FAIL'}")
    L.append(f"   G-PORT2 mean Net R > 0.15           : {mean_r:9.4f}    -> {'PASS' if g2 else 'FAIL'}")
    L.append(f"   G-PORT3 MBB P(mean<=0) <= 0.10      : {p_boot:9.4f}    -> {'PASS' if g3 else 'FAIL'}  (block={BLOCK}, {BOOT_ITERS} resamples, seed {SEED})")
    L.append(f"   OVERALL VERDICT: {verdict}")
    L.append("")
    L.append("HONESTY NOTES")
    L.append("- NOT a fresh touch: the identical instrument-level trade sets were already observed in #54; pooling")
    L.append("  and R-normalization are the pre-declared new content (SPEC section 0). A PASS certifies portfolio-")
    L.append("  level robustness of already-seen results, not independent discovery, and does not by itself unblock")
    L.append("  Phase 2 (validation_discipline).")
    L.append("- R denominators use INITIAL risk distance, frozen at entry, never re-based when the EMA trail moves;")
    L.append("  cost floors enter in native units before normalization (BTC 15.0 / ETH 2.0), so each asset's cost")
    L.append("  drag reflects its own floor estimate; gross R printed alongside net R (SPEC 7.3).")
    L.append("- Equity curve is EVENT-INDEXED cumulative Net R by entry order (declared simplification); concurrent")
    L.append("  overlapping trades (max one per instrument) appear sequentially, so drawdown timing is approximate.")
    L.append("- Continuous forward-fill conventions and owner-estimated cost floors carry over unchanged from #54;")
    L.append("  single frozen parameterization, no optimization, no re-runs; assets, years, sides reported as run.")
    L.append("- Phase 2 (MT5 execution) only via a NEW pre-declared forward-tracking / strict-partition SPEC meeting")
    L.append("  the standard discipline (>=150 trades, year-by-year sign agreement, gates before the run).")
    report = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "REPORT_CRYPTO_PORTFOLIO.txt"), "w") as f:
        f.write(report)
    print()
    print(report)


if __name__ == "__main__":
    main()
