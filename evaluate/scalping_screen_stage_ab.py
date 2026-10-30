# -*- coding: utf-8 -*-
"""
scalping_screen_stage_ab.py — Hypothesis #14 scalping/technical-indicator
screening funnel, Stages A and B.

Implements SPEC_SCALPING_SCREEN.txt / GATE_SCALPING_SCREEN.txt exactly
(frozen 2026-09-03 BEFORE this script was written):
  Stage A: exit-free direction-native MFE/MAE per pattern+filter combo,
           ATR-scaled thresholds {0.5,0.75,1.0}xATR14, day-block bootstrap
           (day means, 10000 draws, seed 20260903), GATE A1.
  Stage B: 16-cell M0 R:R grid (stop {0.5,0.75,1.0,1.5}, target
           {1.0,1.5,2.0,3.0} x ATR14) on the M1 tape with SP2L/Bollinger
           conventions (first-crossing, SL-first ambiguity, gap fills at
           open, 15:55-bar-open forced exit, no overnight), strict
           neighborhood GATE B1.

Outputs (data/results/scalping_screen/):
  funnel_master.csv, matrix_master.csv, matrix_<pattern>.csv,
  stage_a_descriptive.csv, signals/<combo>.csv, stage_ab_stats.json
"""

import json
import time as timemod
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\scalping_screen")
M5 = Path(r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present\us500tick_m5.csv")
M1 = Path(r"C:\Users\Milad\Desktop\RR\data\results\vwap_2021_2026\us500tick_m1.csv")
DAYS = Path(r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present\mt5_days.csv")

WARMUP = 200
WIN_LO, WIN_HI = 480, 690            # [08:00, 11:30) NY wall union window
DESC_LO, DESC_HI = 480, 960          # excursion coverage for descriptive buckets
HORIZON_S = 3600                     # 60-minute Stage-A walk
THR_ATR = (0.50, 0.75, 1.00)
THR_PCT = (0.10, 0.15, 0.20)         # descriptive only
N_BOOT, SEED = 10000, 20260903
COST_PRIMARY, COST_LEGACY = 0.51, 0.80
STOPS = (0.50, 0.75, 1.00, 1.50)
TARGETS = (1.00, 1.50, 2.00, 3.00)
FORCE_WALL = 955                     # 15:55 NY
MIN_N = 150
BOOT_ALPHA = 0.20


# ---------------------------------------------------------------- indicators
def ema(a, span):
    return pd.Series(a).ewm(span=span, adjust=False).mean().to_numpy(float)


def rsi14(c):
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / 14, adjust=False).mean().to_numpy(float)
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / 14, adjust=False).mean().to_numpy(float)
    rs = np.divide(up, dn, out=np.full_like(up, np.inf), where=dn > 0)
    return 100.0 - 100.0 / (1.0 + rs)


def atr14(h, l, c):
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy(float)


def macd_pack(c):
    line = ema(c, 12) - ema(c, 26)
    sig = pd.Series(line).ewm(span=9, adjust=False).mean().to_numpy(float)
    return line, sig, line - sig


def slow_k(h, l, c, n=14, k_smooth=3):
    hh = pd.Series(h).rolling(n).max().to_numpy(float)
    ll = pd.Series(l).rolling(n).min().to_numpy(float)
    rng = hh - ll
    raw = np.divide(100.0 * (c - ll), rng, out=np.full_like(c, np.nan), where=rng > 0)
    return pd.Series(raw).rolling(k_smooth).mean().to_numpy(float), raw


class Tape:
    def __init__(self, path, tf_sec):
        df = pd.read_csv(path)
        self.tf = tf_sec
        self.t = df["time"].to_numpy(np.int64)
        self.o = df["open"].to_numpy(float)
        self.h = df["high"].to_numpy(float)
        self.l = df["low"].to_numpy(float)
        self.c = df["close"].to_numpy(float)
        self.n = len(df)
        idx = pd.DatetimeIndex(pd.to_datetime(self.t, unit="s", utc=True))
        ny = idx.tz_convert("America/New_York")
        self.wall = (ny.hour * 60 + ny.minute).to_numpy(np.int32)
        self.nd = (ny.year * 10000 + ny.month * 100 + ny.day).to_numpy(np.int32)
        self.year = ny.year.to_numpy(np.int32)
        # forced-exit structures: per bar, the 15:55-bar-open exit of its date
        self.force_idx = np.full(self.n, -1, np.int64)
        self.force_px = np.full(self.n, np.nan)
        bounds = np.flatnonzero(np.r_[True, np.diff(self.nd) != 0, True])
        for b in range(len(bounds) - 1):
            s, e = bounds[b], bounds[b + 1] - 1
            m = self.wall[s:e + 1] == FORCE_WALL
            if m.any():
                fi = s + int(np.argmax(m))
                self.force_idx[s:e + 1] = fi
                self.force_px[s:e + 1] = self.o[fi]
            else:
                self.force_idx[s:e + 1] = e
                self.force_px[s:e + 1] = self.c[e]


def build_signals(tv, ind, sub):
    """Raw signals for one sub-variant. Returns dict of arrays over signals:
    i (pattern-TF bar), d (+1/-1), fill (0=close, 1=next-open)."""
    o, h, l, c = tv.o, tv.h, tv.l, tv.c
    n = tv.n
    sig_i, sig_d, sig_f = [], [], []

    def add(mask, d, fill):
        idx = np.flatnonzero(mask)
        sig_i.extend(idx.tolist())
        sig_d.extend([d] * len(idx))
        sig_f.extend([fill] * len(idx))

    if sub == "P1a" or sub == "P1b":
        e8, e21 = ind.ema8, ind.ema21
        up = np.zeros(n, bool)
        dn = np.zeros(n, bool)
        up[1:] = (e8[1:] > e21[1:]) & (e8[:-1] <= e21[:-1])
        dn[1:] = (e8[1:] < e21[1:]) & (e8[:-1] >= e21[:-1])
        add(up, +1, 0)
        add(dn, -1, 0)

    elif sub == "P2":
        body = np.abs(c - o)
        rng = h - l
        uw = h - np.maximum(o, c)
        lw = np.minimum(o, c) - l
        ph = np.roll(c, 6)          # close[t-6]
        prevh = np.roll(h, 1)
        prevl = np.roll(l, 1)
        ok = (rng > 0) & (body >= 0.10 * rng)
        trend_up = np.sign(c - ph) > 0
        trend_dn = np.sign(c - ph) < 0
        bear = ok & trend_up & (uw >= 2 * body) & (c < prevh) & (c > prevl)
        bull = ok & trend_dn & (lw >= 2 * body) & (c > prevl) & (c < prevh)
        add(bear, -1, 1)
        add(bull, +1, 1)

    elif sub == "P3":
        prev_up = np.roll(c, 1) > np.roll(o, 1)
        prev_dn = np.roll(c, 1) < np.roll(o, 1)
        cur_dn = c < o
        cur_up = c > o
        pb = np.abs(np.roll(c, 1) - np.roll(o, 1)) > 0
        bear = prev_up & cur_dn & pb & (o >= np.roll(c, 1)) & (c <= np.roll(o, 1))
        bull = prev_dn & cur_up & pb & (o <= np.roll(c, 1)) & (c >= np.roll(o, 1))
        add(bear, -1, 1)
        add(bull, +1, 1)

    elif sub in ("P4a", "P4b", "P4c", "P4d"):
        nbars = 10 if sub in ("P4a", "P4b") else 20
        fill = 0 if sub in ("P4a", "P4c") else 1
        hh = pd.Series(h).rolling(nbars).max().shift(1).to_numpy(float)
        ll = pd.Series(l).rolling(nbars).min().shift(1).to_numpy(float)
        add(c > hh, +1, fill)
        add(c < ll, -1, fill)

    elif sub in ("P5", "P6", "P7"):
        if sub == "P5":
            osc = ind.rsi
        elif sub == "P6":
            osc = ind.macd_line
        else:
            osc = ind.slowk
        ph = np.zeros(n, bool)
        pl = np.zeros(n, bool)
        ph[2:-2] = (h[2:-2] > h[0:-4]) & (h[2:-2] > h[1:-3]) & \
                   (h[2:-2] > h[3:-1]) & (h[2:-2] > h[4:])
        pl[2:-2] = (l[2:-2] < l[0:-4]) & (l[2:-2] < l[1:-3]) & \
                   (l[2:-2] < l[3:-1]) & (l[2:-2] < l[4:])
        hi_idx = np.flatnonzero(ph)
        lo_idx = np.flatnonzero(pl)
        ev = []  # (confirmation_bar, direction)
        for arr, d in ((hi_idx, -1), (lo_idx, +1)):
            for k in range(1, len(arr)):
                p1, p2 = arr[k - 1], arr[k]
                if p1 < WARMUP:
                    continue
                if d == -1 and h[p2] > h[p1] and osc[p2] < osc[p1]:
                    ev.append((p2 + 2, -1))
                if d == +1 and l[p2] < l[p1] and osc[p2] > osc[p1]:
                    ev.append((p2 + 2, +1))
        conf = {}
        for tbar, d in ev:
            conf.setdefault(tbar, []).append(d)
        for tbar, ds in sorted(conf.items()):
            if len(ds) > 1:
                continue  # same-bar conflict, skipped (tallied in stats)
            sig_i.append(tbar)
            sig_d.append(ds[0])
            sig_f.append(0)

    elif sub in ("P8a", "P8b"):
        ma = ind.sma50 if sub == "P8a" else ind.ema50
        atrv = ind.atr
        ext_hi = pd.Series(h - ma).rolling(12).max().shift(1).to_numpy(float)
        ext_lo = pd.Series(l - ma).rolling(12).min().shift(1).to_numpy(float)
        a_m1 = np.roll(atrv, 1)
        bear = (ext_hi >= 1.0 * a_m1) & (np.roll(c, 1) > np.roll(ma, 1)) & \
               (np.roll(l, 1) > np.roll(ma, 1)) & (l <= ma)
        bull = (ext_lo <= -1.0 * a_m1) & (np.roll(c, 1) < np.roll(ma, 1)) & \
               (np.roll(h, 1) < np.roll(ma, 1)) & (h >= ma)
        add(bear, -1, 0)
        add(bull, +1, 0)

    order = np.argsort(np.array(sig_i, dtype=np.int64), kind="stable")
    return (np.array(sig_i, np.int64)[order],
            np.array(sig_d, np.int8)[order],
            np.array(sig_f, np.int8)[order])


# ---------------------------------------------------------------- Stage A walk
def excursions(m1, entry, d, i0):
    """Direction-native MFE/MAE in POINTS over the 60-min M1 walk from i0.
    E-XTM first-bar convention."""
    i1 = int(np.searchsorted(m1.t, m1.t[i0] + HORIZON_S, side="left")) - 1
    if i1 < i0:
        return None
    if d == 1:
        fav = max(m1.c[i0] - entry, 0.0)
        adv = max(entry - m1.l[i0], 0.0)
        if i1 > i0:
            fav = max(fav, float(m1.h[i0 + 1:i1 + 1].max()) - entry)
            adv = max(adv, entry - float(m1.l[i0 + 1:i1 + 1].min()))
    else:
        fav = max(entry - m1.c[i0], 0.0)
        adv = max(m1.h[i0] - entry, 0.0)
        if i1 > i0:
            fav = max(fav, entry - float(m1.l[i0 + 1:i1 + 1].min()))
            adv = max(adv, float(m1.h[i0 + 1:i1 + 1].max()) - entry)
    return max(fav, 0.0), max(adv, 0.0)


# ---------------------------------------------------------------- Stage B sim
def sim_m0(o, h, l, i0, fidx, fpx, entry, d, sl, tp):
    """M0 fixed SL/TP walk on the M1 tape. Returns
    (pnl_pts, exit_ts, reason, mfe_pts, mae_pts). SP2L/Bollinger conventions:
    first-crossing; same-bar both-level ambiguity SL-first; gap through a
    level fills at the open; 15:55-bar-open forced exit has priority."""
    mfe = mae = 0.0
    for j in range(i0, fidx):
        hj, lj, oj = h[j], l[j], o[j]
        if d == 1:
            mfe = max(mfe, hj - entry)
            mae = max(mae, entry - lj)
            if oj <= sl:
                return oj - entry, j, "stop_gap", mfe, mae
            if oj >= tp:
                return oj - entry, j, "target_gap", mfe, mae
            if lj <= sl and hj >= tp:
                return sl - entry, j, "stop_loss", mfe, mae
            if lj <= sl:
                return sl - entry, j, "stop_loss", mfe, mae
            if hj >= tp:
                return tp - entry, j, "target", mfe, mae
        else:
            mfe = max(mfe, entry - lj)
            mae = max(mae, hj - entry)
            if oj >= sl:
                return entry - oj, j, "stop_gap", mfe, mae
            if oj <= tp:
                return entry - oj, j, "target_gap", mfe, mae
            if hj >= sl and lj <= tp:
                return entry - sl, j, "stop_loss", mfe, mae
            if hj >= sl:
                return entry - sl, j, "stop_loss", mfe, mae
            if lj <= tp:
                return entry - tp, j, "target", mfe, mae
    px = fpx
    if d == 1:
        mfe = max(mfe, px - entry)
        mae = max(mae, entry - px)
    else:
        mfe = max(mfe, entry - px)
        mae = max(mae, px - entry)
    return (d == 1) * (px - entry) + (d == -1) * (entry - px), fidx, \
        "time_exit", mfe, mae


WF_YEARS = (2022, 2023, 2024, 2025, 2026)


def cell_metrics(pnls, nds, year, universe_days, day_pos):
    pnls = np.asarray(pnls, float)
    n = len(pnls)
    net = pnls - COST_PRIMARY
    win = float((net > 0).mean()) if n else np.nan
    daily = {}
    for p, d0 in zip(net, nds):   # daily PnL NET of primary cost
        daily[d0] = daily.get(d0, 0.0) + p
    dd = np.array([daily.get(d, 0.0) for d in universe_days])
    sd = dd.std(ddof=1) if len(dd) > 1 else np.nan
    sharpe = float(dd.mean() / sd * np.sqrt(252)) if sd and sd > 0 else 0.0
    cum = np.cumsum(dd)
    peak = np.maximum.accumulate(np.r_[0.0, cum])[:-1]
    maxdd = float((peak - cum).max() / 10000.0 * 100.0) if n else np.nan
    yearly = {}
    for p, y0 in zip(net, year):
        if y0 in WF_YEARS:
            yearly[y0] = yearly.get(y0, 0.0) + p
    total = float(net.sum())
    years_pos = int(sum(1 for v in yearly.values() if v > 0))
    best_share = (max(yearly.values()) / total * 100.0) if total > 0 else np.nan
    return {"n_trades": n, "win_rate": round(win, 4),
            "expectancy_pts": round(float(pnls.mean()), 4) if n else np.nan,
            "expectancy_after_cost": round(float(pnls.mean() - COST_PRIMARY), 4) if n else np.nan,
            "expectancy_after_cost_legacy08": round(float(pnls.mean() - COST_LEGACY), 4) if n else np.nan,
            "Sharpe": round(sharpe, 4), "max_DD_pct": round(maxdd, 4),
            "best_year_share_pct": round(best_share, 2) if total > 0 else "",
            "years_positive_of_5": years_pos,
            "total_net_pts": round(total, 2)}


def boot_p_le0(e, nd):
    per_day = {}
    for x, d0 in zip(e, nd):
        per_day.setdefault(d0, []).append(x)
    days = np.array([np.mean(v) for v in per_day.values()])
    rng = np.random.default_rng(SEED)
    means = rng.choice(days, size=(N_BOOT, len(days)), replace=True).mean(axis=1)
    return float((means <= 0).mean()), len(days)


def main():
    t0 = timemod.time()
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "signals").mkdir(exist_ok=True)

    print("loading tapes ...", flush=True)
    m5 = Tape(M5, 300)
    m1 = Tape(M1, 60)
    print(f"  M5 {m5.n} bars, M1 {m1.n} bars ({timemod.time()-t0:.0f}s)", flush=True)

    days_df = pd.read_csv(DAYS, sep=";")
    universe_days = days_df.loc[(days_df.session_open > 0) | (days_df.prev_close > 0),
                                "ny_date"].astype(np.int64).to_numpy()
    universe_days.sort()
    day_pos = {int(d): k for k, d in enumerate(universe_days)}

    for tv, name in ((m5, "M5"), (m1, "M1")):
        tv.rsi = rsi14(tv.c)
        assert 0.0 <= tv.rsi.min() <= tv.rsi.max() <= 100.0
        assert np.ptp(tv.rsi) > 50.0, "degenerate RSI"
        tv.macd_line, tv.macd_sig, tv.macd_hist = macd_pack(tv.c)
        tv.atr = atr14(tv.h, tv.l, tv.c)
        tv.ema8, tv.ema21 = ema(tv.c, 8), ema(tv.c, 21)
        tv.sma50 = pd.Series(tv.c).rolling(50).mean().to_numpy(float)
        tv.ema50 = ema(tv.c, 50)
        tv.slowk, _ = slow_k(tv.h, tv.l, tv.c)
    print(f"indicators done ({timemod.time()-t0:.0f}s)", flush=True)

    subs = {"P1a": m1, "P1b": m5, "P2": m5, "P3": m5, "P4a": m5, "P4b": m5,
            "P4c": m5, "P4d": m5, "P5": m5, "P6": m5, "P7": m5,
            "P8a": m5, "P8b": m5}
    allowed_filters = {s: ["F0", "F1", "F2", "F3"] for s in subs}
    allowed_filters["P5"] = ["F0", "F2", "F3"]   # skips own-oscillator arm
    allowed_filters["P6"] = ["F0", "F1", "F3"]

    o1 = m1.o.tolist()
    h1 = m1.h.tolist()
    l1 = m1.l.tolist()

    funnel_rows, matrix_rows, desc_rows = [], [], []
    stats = {"combos": {}, "tallies": {}}

    for sub, tv in subs.items():
        pattern = sub[:2]
        ind = tv
        si, sd, sf = build_signals(tv, ind, sub)
        keep = si >= WARMUP
        si, sd, sf = si[keep], sd[keep], sf[keep]
        stats["tallies"][sub] = {"raw_signals": int(len(si))}
        if len(si) == 0:
            for f in allowed_filters[sub]:
                funnel_rows.append(dict(combo_id=f"{sub}_{f}", pattern=pattern,
                                        subvariant=sub, filter=f,
                                        stage_reached="A", status="DROP-A",
                                        reason="no signals", n_signals=0))
            continue

        # ---- execution resolution on the M1 tape
        theo = tv.t[si] + tv.tf
        ex = np.searchsorted(m1.t, theo, side="left")
        valid = ex < m1.n
        si, sd, sf, ex = si[valid], sd[valid], sf[valid], ex[valid]
        wall = m1.wall[ex]
        nd = m1.nd[ex]
        atr_sig = tv.atr[si]
        ok_atr = np.isfinite(atr_sig) & (atr_sig > 0)
        entry = np.where(sf == 0, tv.c[si], m1.o[ex])
        in_universe = np.isin(nd, universe_days)
        elig = ok_atr & in_universe & (wall >= WIN_LO) & (wall < WIN_HI)

        # ---- Stage-A excursions for ALL signals (complete descriptive
        # coverage; the binding gate uses only the eligible window subset)
        cover = ok_atr & np.isfinite(entry)
        mfe_pts = np.full(len(si), np.nan)
        mae_pts = np.full(len(si), np.nan)
        cov_idx = np.flatnonzero(cover)
        for k in cov_idx:
            r = excursions(m1, float(entry[k]), int(sd[k]), int(ex[k]))
            if r is not None:
                mfe_pts[k], mae_pts[k] = r
        e_atr = (mfe_pts - mae_pts) / atr_sig
        mfe_atr = mfe_pts / atr_sig
        mae_atr = mae_pts / atr_sig

        # ---- descriptive buckets (F0 arm, whole signal population in band)
        buckets = [(0, 480), (480, 570), (570, 690), (690, 960), (960, 1440)]
        for lo, hi in buckets:
            m = cover & (wall >= lo) & (wall < hi) & np.isfinite(e_atr)
            desc_rows.append(dict(subvariant=sub, window=f"{lo:04d}-{hi:04d}",
                                  n_signals=int(m.sum()),
                                  mean_e_atr=round(float(np.nanmean(e_atr[m])), 4)
                                  if m.any() else ""))

        # ---- filter flags at the signal bar (pattern TF)
        f1_ok = np.where(sd == 1, ind.rsi[si] < 70, ind.rsi[si] > 30)
        f2_ok = np.where(sd == 1, ind.macd_hist[si] > 0, ind.macd_hist[si] < 0)

        for f in allowed_filters[sub]:
            cid = f"{sub}_{f}"
            m = elig & np.isfinite(e_atr)
            if f == "F1":
                m &= f1_ok
            elif f == "F2":
                m &= f2_ok
            elif f == "F3":
                m &= f1_ok & f2_ok
            kk = np.flatnonzero(m)
            n = len(kk)
            cs = stats["combos"].setdefault(cid, {})
            if n == 0:
                funnel_rows.append(dict(combo_id=cid, pattern=pattern,
                                        subvariant=sub, filter=f,
                                        stage_reached="A", status="DROP-A",
                                        reason="no eligible signals", n_signals=0))
                continue
            e = e_atr[kk]
            cnt_ok = all(int((mfe_atr[kk] >= t).sum()) > int((mae_atr[kk] >= t).sum())
                         for t in THR_ATR)
            p_boot, n_days = boot_p_le0(e, nd[kk])
            pct = {t: (int((mfe_pts[kk] >= t / 100 * entry[kk]).sum()),
                       int((mae_pts[kk] >= t / 100 * entry[kk]).sum()))
                   for t in THR_PCT}
            cs["stage_a"] = {"n": int(n), "n_days": int(n_days),
                             "mean_e_atr": round(float(e.mean()), 4),
                             "counts_mfe_gt_mae_all": bool(cnt_ok),
                             "p_boot": round(p_boot, 4),
                             "pct_thresholds_descriptive": pct}
            a_pass = (n >= MIN_N) and cnt_ok and (p_boot <= BOOT_ALPHA)
            if not a_pass:
                why = []
                if n < MIN_N:
                    why.append(f"n={n}<150")
                if not cnt_ok:
                    why.append("MFE>MAE fails at some threshold")
                if p_boot > BOOT_ALPHA:
                    why.append(f"p_boot={p_boot:.3f}>0.20")
                funnel_rows.append(dict(combo_id=cid, pattern=pattern,
                                        subvariant=sub, filter=f,
                                        stage_reached="A", status="DROP-A",
                                        reason="; ".join(why), n_signals=n,
                                        p_boot=round(p_boot, 4),
                                        mean_e_atr=round(float(e.mean()), 4)))
                continue

            # ================= STAGE B: 16-cell M0 grid ================
            exs, nds, ys = ex[kk], nd[kk], m1.year[ex[kk]]
            entries = entry[kk]
            dirs = sd[kk]
            atrs = atr_sig[kk]
            best, best_cell = -1e18, None
            grid = {}
            grid_reasons = {}
            for si_, st in enumerate(STOPS):
                for ti_, tg in enumerate(TARGETS):
                    pnls = np.empty(n)
                    reasons = []
                    for q in range(n):
                        d = int(dirs[q])
                        slp = entries[q] - st * atrs[q] * d
                        tpp = entries[q] + tg * atrs[q] * d
                        pnl, j, reason, mfe_p, mae_p = sim_m0(
                            o1, h1, l1, int(exs[q]),
                            int(m1.force_idx[exs[q]]), float(m1.force_px[exs[q]]),
                            float(entries[q]), d, slp, tpp)
                        pnls[q] = pnl
                        # validation invariants (vwap-trend convention)
                        if reason in ("stop_loss", "stop_gap"):
                            assert mae_p >= st * atrs[q] - 1e-6, (cid, q)
                        elif reason in ("target", "target_gap"):
                            assert mfe_p >= tg * atrs[q] - 1e-6, (cid, q)
                        else:
                            assert mae_p < st * atrs[q] + 1e-6 and \
                                   mfe_p < tg * atrs[q] + 1e-6, (cid, q)
                        reasons.append(reason)
                    grid_reasons[(si_, ti_)] = reasons
                    met = cell_metrics(pnls, nds, ys, universe_days, day_pos)
                    met.update(pattern=pattern, subvariant=sub, filter=f,
                               MM_variant="M0", stop_ATR=st, target_ATR=tg,
                               combo_id=cid, source_stage="B_M0")
                    grid[(si_, ti_)] = met
                    matrix_rows.append(met)
                    if met["expectancy_after_cost"] > best:
                        best, best_cell = met["expectancy_after_cost"], met
            # strict neighborhood
            nb_pass = False
            for (si_, ti_), met in grid.items():
                if met["expectancy_after_cost"] <= 0 or met["n_trades"] < MIN_N:
                    continue
                oks = []
                for ds_, dt_ in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    js, jt = si_ + ds_, ti_ + dt_
                    if 0 <= js < 4 and 0 <= jt < 4:
                        oks.append(grid[(js, jt)]["expectancy_after_cost"] > 0)
                if all(oks):
                    nb_pass = True
                    break
            b_pass = nb_pass
            cs["stage_b"] = {"b1_pass": bool(b_pass),
                             "best_cell": {k: best_cell[k] for k in
                                           ("stop_ATR", "target_ATR",
                                            "expectancy_after_cost", "Sharpe",
                                            "n_trades")},
                             "best_cell_exit_reasons": {
                                 r: grid_reasons[
                                     (STOPS.index(best_cell["stop_ATR"]),
                                      TARGETS.index(best_cell["target_ATR"]))].count(r)
                                 for r in ("stop_loss", "stop_gap", "target",
                                           "target_gap", "time_exit")}}
            row = dict(combo_id=cid, pattern=pattern, subvariant=sub, filter=f,
                       stage_reached="B", status="PASS-B" if b_pass else "DROP-B",
                       reason="strict-neighborhood cell found" if b_pass
                              else "no cell with positive strict neighborhood",
                       n_signals=n, p_boot=round(p_boot, 4),
                       mean_e_atr=round(float(e.mean()), 4),
                       best_stop=best_cell["stop_ATR"],
                       best_target=best_cell["target_ATR"],
                       best_expectancy_after_cost=best)
            funnel_rows.append(row)
            pd.DataFrame({"ts": m1.t[exs], "nd": nds, "dir": dirs,
                          "entry": entries, "atr": atrs,
                          "e_atr": e_atr[kk], "mfe_atr": mfe_atr[kk],
                          "mae_atr": mae_atr[kk]}
                         ).to_csv(BASE / "signals" / f"{cid}.csv", index=False)
            print(f"  {cid}: A-pass n={n} p_boot={p_boot:.3f} "
                  f"B1={'PASS' if b_pass else 'FAIL'} "
                  f"best={best:.3f} ({timemod.time()-t0:.0f}s)", flush=True)

    pd.DataFrame(funnel_rows).to_csv(BASE / "funnel_master.csv", index=False)
    md = pd.DataFrame(matrix_rows)
    md = md.sort_values("expectancy_after_cost", ascending=False)
    md.to_csv(BASE / "matrix_master.csv", index=False)
    for p in sorted(set(r["pattern"] for r in matrix_rows)):
        md[md.pattern == p].to_csv(BASE / f"matrix_{p}.csv", index=False)
    pd.DataFrame(desc_rows).to_csv(BASE / "stage_a_descriptive.csv", index=False)
    with open(BASE / "stage_ab_stats.json", "w") as fh:
        json.dump(stats, fh, indent=1, default=str)
    n_a_pass = sum(1 for r in funnel_rows if r["status"] in ("PASS-B", "DROP-B"))
    n_b_pass = sum(1 for r in funnel_rows if r["status"] == "PASS-B")
    print(f"DONE {timemod.time()-t0:.0f}s | combos={len(funnel_rows)} "
          f"A-pass={n_a_pass} B-pass={n_b_pass}", flush=True)


if __name__ == "__main__":
    main()
