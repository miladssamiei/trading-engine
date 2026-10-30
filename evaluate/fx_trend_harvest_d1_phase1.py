import os
import json
import hashlib

import numpy as np
import pandas as pd

SEED = 20260925
BOOT_ITERS = 10000
BLOCK = 5
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAPE = os.path.join(ROOT, "data", "results", "eurusd_onboarding", "eurusdtick_d1.csv")
OUT = os.path.join(ROOT, "data", "results", "fx_trend_harvest")


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


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(TAPE)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df = df.sort_values("date").reset_index(drop=True)
    if df["date"].duplicated().any():
        raise SystemExit("duplicate dates in tape; aborting")
    d = df["date"].to_numpy()
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    m = len(df)

    atr = wilder_atr(h, l, c, 14)
    atr_prior = np.roll(atr, 1)
    atr_prior[0] = np.nan
    dch = pd.Series(h).rolling(20).max().shift(1).to_numpy()
    dcl = pd.Series(l).rolling(20).min().shift(1).to_numpy()

    armed = np.arange(m) >= 21
    armed &= ~np.isnan(dch) & ~np.isnan(dcl) & ~np.isnan(atr_prior)

    trades = []
    sig_long = sig_short = ignored_open = 0
    void_last = 0
    i = 21
    while i < m:
        if not armed[i] or not (c[i] > dch[i] or c[i] < dcl[i]):
            i += 1
            continue
        side = 1 if c[i] > dch[i] else -1
        if side == 1:
            sig_long += 1
        else:
            sig_short += 1
        if i == m - 1:
            void_last += 1
            break
        atrv = atr_prior[i]
        ei = i + 1
        entry = o[ei]
        sl = entry - 1.5 * atrv if side == 1 else entry + 1.5 * atrv
        tp = entry + 3.0 * atrv if side == 1 else entry - 3.0 * atrv
        exit_px = None
        reason = None
        exit_j = None
        for j in range(ei, m):
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
            exit_px, reason, exit_j = c[m - 1], "FORCED_EOD", m - 1
        for k in range(ei, exit_j):
            if armed[k] and (c[k] > dch[k] or c[k] < dcl[k]):
                ignored_open += 1
        gross = side * (exit_px - entry) * 10000.0
        rmult = side * (exit_px - entry) / (1.5 * atrv)
        trades.append(
            dict(
                signal_date=pd.Timestamp(d[i]).strftime("%Y-%m-%d"),
                entry_date=pd.Timestamp(d[ei]).strftime("%Y-%m-%d"),
                side="LONG" if side == 1 else "SHORT",
                atr=atrv,
                entry=entry,
                sl=sl,
                tp=tp,
                exit=exit_px,
                exit_reason=reason,
                gross_pips=gross,
                r_multiple=rmult,
                bars_held=exit_j - ei + 1,
                entry_year=pd.Timestamp(d[ei]).year,
            )
        )
        i = exit_j

    t = pd.DataFrame(trades)
    g = t["gross_pips"].to_numpy(dtype=float) if len(t) else np.array([])
    n_tr = len(t)
    mean_g = float(g.mean()) if n_tr else float("nan")
    win = float((g > 0).mean()) if n_tr else float("nan")
    p_boot = mbb_p_le_zero(g) if n_tr >= BLOCK else float("nan")
    yr = t.groupby("entry_year")["gross_pips"].agg(["count", "sum", "mean"]) if n_tr else None
    reason_mix = t["exit_reason"].value_counts().to_dict() if n_tr else {}
    lvl_mix = dict(
        TP=reason_mix.get("TP", 0) + reason_mix.get("TP_GAP", 0),
        SL=reason_mix.get("SL", 0) + reason_mix.get("SL_GAP", 0),
        FORCED=reason_mix.get("FORCED_EOD", 0),
    )
    side_mix = t["side"].value_counts().to_dict() if n_tr else {}

    g1 = n_tr >= 30
    g2 = n_tr > 0 and mean_g > 0
    g3 = n_tr >= BLOCK and p_boot <= 0.20
    verdict = "PASS" if (g1 and g2 and g3) else "FAIL"

    if n_tr:
        t.to_csv(os.path.join(OUT, "fx_trend_harvest_signals.csv"), index=False)

    meta = dict(
        study="fx_trend_harvest",
        hypothesis=46,
        generated_at=pd.Timestamp.utcnow().isoformat(),
        script_sha256=sha256_file(os.path.abspath(__file__)),
        seed=SEED,
        boot_iters=BOOT_ITERS,
        block=BLOCK,
        tape=dict(path=TAPE, sha256=sha256_file(TAPE), bars=int(m), first=str(pd.Timestamp(d[0]).date()), last=str(pd.Timestamp(d[-1]).date())),
        funnel=dict(armed_days=int(armed.sum()), sig_long=sig_long, sig_short=sig_short, ignored_while_open=ignored_open, executed=n_tr, void_last_bar=void_last),
        gates=dict(G_FXH1=bool(g1), G_FXH2=bool(g2), G_FXH3=bool(g3), mean_gross=mean_g, p_mbb=p_boot),
        verdict=verdict,
    )
    with open(os.path.join(OUT, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    L = []
    L.append("REPORT - HYPOTHESIS #46: EURUSD D1 TREND HARVEST (DONCHIAN 20 + 1.5xATR SL / +2R TP)")
    L.append(f"Generated: {meta['generated_at']}  |  Verdict: {verdict}")
    L.append("")
    L.append("1. DATA INTEGRITY")
    L.append(f"   Tape: {m} D1 bars {meta['tape']['first']}..{meta['tape']['last']} (sha256 {meta['tape']['sha256'][:16]}...)")
    L.append(f"   Indicators: Donchian 20 strictly prior; Wilder ATR(14) strictly prior; armed from index 21")
    L.append("")
    L.append("2. FUNNEL")
    f_ = meta["funnel"]
    L.append(f"   armed days={f_['armed_days']}  signals: long={f_['sig_long']} short={f_['sig_short']} total={f_['sig_long'] + f_['sig_short']}")
    L.append(f"   ignored while open={f_['ignored_while_open']}  EXECUTED={f_['executed']}  void(last bar)={f_['void_last_bar']}")
    L.append("")
    L.append("3. GROSS PERFORMANCE (pips; costs excluded per SPEC 5)")
    L.append(f"   trades={n_tr}  mean={mean_g:.3f} pips  total={g.sum() if n_tr else float('nan'):.1f}  win={100.0 * win if n_tr else float('nan'):.1f}%")
    if n_tr:
        w_ = g[g > 0]
        lo_ = g[g <= 0]
        L.append(f"   avg win={w_.mean() if len(w_) else float('nan'):.1f} pips (n={len(w_)})  avg loss={lo_.mean() if len(lo_) else float('nan'):.1f} pips (n={len(lo_)})")
        L.append(f"   R multiples: mean={t['r_multiple'].mean():.3f}  median={t['r_multiple'].median():.3f}  avg ATR={t['atr'].mean() * 10000:.1f} pips  avg hold={t['bars_held'].mean():.1f} bars")
    L.append(f"   sides={side_mix}")
    L.append(f"   exit mix (level): {lvl_mix}   granular: {reason_mix}")
    L.append("")
    L.append("4. YEAR-BY-YEAR GROSS (sign inspection, SPEC 7.3)")
    if n_tr:
        for y, r in yr.iterrows():
            L.append(f"   {y}: n={int(r['count']):3d}  total={r['sum']:9.1f} pips  mean={r['mean']:7.3f}")
    L.append("")
    L.append("5. GATES (frozen in GATE_FX_HARVEST_D1.txt)")
    L.append(f"   G-FXH1 executed >= 30                : {n_tr:4d}      -> {'PASS' if g1 else 'FAIL'}")
    L.append(f"   G-FXH2 mean gross > 0 (strict, pips) : {mean_g:8.3f} -> {'PASS' if g2 else 'FAIL'}")
    L.append(f"   G-FXH3 MBB P(mean<=0) <= 0.20        : {p_boot:8.4f} -> {'PASS' if g3 else 'FAIL'}  (block={BLOCK}, {BOOT_ITERS} resamples, seed {SEED})")
    L.append(f"   VERDICT: {verdict}")
    L.append("")
    L.append("6. HONESTY NOTES")
    L.append("   Gross-only accounting per owner designation (verified EURUSD floor ~0.3 pip RT).")
    L.append("   Lineage: #42 (same signal family, trailing exit) FAILED; this fixed-harvest variant is a")
    L.append("   pre-declared new mechanic, single run, no parameter search. Null result is final here.")
    report = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "REPORT_FX_HARVEST_D1_PHASE1.txt"), "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
