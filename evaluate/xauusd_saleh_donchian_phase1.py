# -*- coding: utf-8 -*-
"""
xauusd_saleh_donchian_phase1.py -- Hypothesis #57: Saleh 4H/1H Donchian
Supertrend, native XAUUSD Phase 1 screen.

SPEC/GATE pre-declared and frozen BEFORE this script ran:
  data/results/xauusd_saleh_donchian/SPEC_SALEH_DONCHIAN.txt
  data/results/xauusd_saleh_donchian/GATE_SALEH_DONCHIAN.txt
(one configuration: 4H Supertrend(10,3.0) anchor, 1H Donchian(20) breakout at
the 1H open, Wilder ADX14 >= 20, 4xATR(14) initial stop + close-based 4xATR
trailing stop, fixed spread 0.41; no ablation, no sweep).

Tape: data/results/xauusd_onboarding/xauusdtick_m1.csv (verified onboarding
2026-09-04; 2,337,538 M1 bars, 2020-01-01..2026-09-01). Ask = Bid + 0.41
everywhere (onboarding full-tape median spread).

Outputs (data/results/xauusd_saleh_donchian/):
  saleh_donchian_signals.csv        one row per executed trade
  REPORT_SALEH_DONCHIAN_PHASE1.txt  full report + gate verdicts
  run_meta.json                     lineage + funnel + gates
"""

import csv
import datetime as dt
import hashlib
import json
import time as timemod
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\xauusd_saleh_donchian")
M1_CSV = Path(r"C:\Users\Milad\Desktop\RR\data\results\xauusd_onboarding\xauusdtick_m1.csv")

SPREAD = 0.41          # fixed round-trip spread, price units (SPEC 5)
ST_ATR = 10            # 4H Supertrend ATR period
ST_MULT = 3.0          # 4H Supertrend multiplier
DC_N = 20              # 1H Donchian channel length (SPEC 3)
ADX_PERIOD = 14        # 1H Wilder ADX (SPEC 3)
MIN_ADX = 20.0         # SPEC 4, L3
ATR_PERIOD = 14        # 1H Wilder ATR (SPEC 3)
SL_ATR_MULT = 4.0      # initial + trailing stop multiple (SPEC 5)

SEED = 20261005        # GATE_SALEH_DONCHIAN.txt
BOOT_ITERS = 10000
BLOCK = 5
UNIT_USD = 100.0       # $/price-unit/lot, display only (SPEC 5)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- data ----

def wilder_atr(h, l, c, period):
    h = pd.Series(h)
    l = pd.Series(l)
    c = pd.Series(c)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean().to_numpy(float)


def wilder_adx(h, l, c, period=ADX_PERIOD):
    """Source-exact Wilder ADX (#56 calc_adx)."""
    up = h.diff()
    down = -l.diff()
    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=h.index)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()],
                   axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr
    den = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / den
    return dx.ewm(alpha=1.0 / period, adjust=False).mean().to_numpy(float)


def supertrend(h, l, c, period=ST_ATR, mult=ST_MULT):
    """Standard (TradingView) Supertrend on the continuous 4H grid (SPEC 3).
    Returns (line, direction) with direction +1 bullish / -1 bearish."""
    atr = wilder_atr(h, l, c, period)
    hl2 = (h + l) / 2.0
    n = len(c)
    bub = hl2 + mult * atr
    blb = hl2 - mult * atr
    fub = np.copy(bub)
    flb = np.copy(blb)
    for i in range(1, n):
        if bub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]:
            fub[i] = bub[i]
        else:
            fub[i] = fub[i - 1]
        if blb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]:
            flb[i] = blb[i]
        else:
            flb[i] = flb[i - 1]
    line = np.copy(fub)                      # seed: bearish
    assert np.isfinite(fub).all() and np.isfinite(flb).all(), \
        "Supertrend band recursion hit NaN (leading-NaN grid poison)"
    direction = np.full(n, -1, dtype=np.int8)
    for i in range(1, n):
        if direction[i - 1] == -1:
            if c[i] > fub[i]:
                direction[i] = 1
                line[i] = flb[i]
            else:
                direction[i] = -1
                line[i] = fub[i]
        else:
            if c[i] < flb[i]:
                direction[i] = -1
                line[i] = fub[i]
            else:
                direction[i] = 1
                line[i] = flb[i]
    return line, direction


def build_grids():
    """SPEC 2: continuous 1H (epoch hour slots) and 4H (NY wall buckets)
    grids from the M1 tape, per-field ffill, REAL/FILL masks."""
    df = pd.read_csv(M1_CSV, usecols=["time", "open", "high", "low", "close"])
    t = df["time"].to_numpy(np.int64)
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n_m1 = len(df)

    # --- 1H: epoch top-of-hour slots -----------------------------------
    slot = t // 3600
    g1 = (
        pd.DataFrame({"slot": slot, "o": o, "h": h, "l": l, "c": c})
        .groupby("slot", sort=True)
        .agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    )
    grid1 = np.arange(int(slot.min()), int(slot.max()) + 1, dtype=np.int64)
    r1 = g1.reindex(grid1).ffill()
    real1 = np.isin(grid1, g1.index.to_numpy(np.int64))

    ny1 = pd.to_datetime(pd.Index(grid1 * 3600), unit="s", utc=True) \
        .tz_convert("America/New_York")
    d1 = (ny1.year * 10000 + ny1.month * 100 + ny1.day).to_numpy(np.int64)
    h1w = ny1.hour.to_numpy(np.int32)

    # --- 4H: NY wall buckets [00,04),[04,08),... ------------------------
    ny_m1 = pd.to_datetime(t, unit="s", utc=True).tz_convert("America/New_York")
    k4 = ((ny_m1.year * 10000 + ny_m1.month * 100 + ny_m1.day).to_numpy(np.int64)
          * 100 + ((ny_m1.hour.to_numpy(np.int32) // 4) * 4))
    g4 = (
        pd.DataFrame({"key": k4, "o": o, "h": h, "l": l, "c": c})
        .groupby("key", sort=True)
        .agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    )
    days = pd.date_range(ny_m1[0].date(), ny_m1[-1].date(), freq="D")
    grid4 = np.array(
        [int(dd.strftime("%Y%m%d")) * 100 + bh for dd in days
         for bh in (0, 4, 8, 12, 16, 20)], dtype=np.int64)
    r4 = g4.reindex(grid4).ffill()
    real4 = np.isin(grid4, g4.index.to_numpy(np.int64))
    # DEFECT FIX (first execution 2026-09-24): leading wall buckets before the
    # tape's first M1 bar stay NaN after ffill and permanently poison the
    # Supertrend band recursion. Start the grid at the first REAL bucket
    # (same convention as the 1H grid, which starts at its first real slot).
    first4 = int(np.flatnonzero(real4)[0])
    grid4 = grid4[first4:]
    real4 = real4[first4:]
    r4 = r4.iloc[first4:].reset_index(drop=True)

    return {
        "n_m1": n_m1,
        "grid1": grid1, "real1": real1,
        "o1": r1["o"].to_numpy(float), "h1": r1["h"].to_numpy(float),
        "l1": r1["l"].to_numpy(float), "c1": r1["c"].to_numpy(float),
        "d1": d1, "h1w": h1w,
        "grid4": grid4, "real4": real4,
        "o4": r4["o"].to_numpy(float), "h4": r4["h"].to_numpy(float),
        "l4": r4["l"].to_numpy(float), "c4": r4["c"].to_numpy(float),
    }


def load_ctx():
    g = build_grids()
    o1, h1, l1, c1 = g["o1"], g["h1"], g["l1"], g["c1"]
    o4, h4, l4, c4 = g["o4"], g["h4"], g["l4"], g["c4"]
    n1 = len(o1)

    # 1H indicators (continuous grid)
    s_h = pd.Series(h1)
    s_l = pd.Series(l1)
    s_c = pd.Series(c1)
    adx1 = wilder_adx(s_h, s_l, s_c)
    atr1 = wilder_atr(h1, l1, c1, ATR_PERIOD)
    dcup = s_h.rolling(DC_N, min_periods=DC_N).max().to_numpy(float)
    dclo = s_l.rolling(DC_N, min_periods=DC_N).min().to_numpy(float)

    # 4H Supertrend + strict-completion mapping (SPEC 4)
    st_line, st_dir = supertrend(h4, l4, c4)
    containing = g["d1"] * 100 + (g["h1w"] // 4) * 4
    mapped4 = np.searchsorted(g["grid4"], containing, side="right") - 2
    anchor = np.zeros(n1, dtype=np.int8)
    ok = mapped4 >= 0
    anchor[ok] = st_dir[mapped4[ok]]

    return {
        **g, "n1": n1, "adx1": adx1, "atr1": atr1,
        "dcup": dcup, "dclo": dclo,
        "st_line": st_line, "st_dir": st_dir, "mapped4": mapped4,
        "anchor": anchor,
    }


# ------------------------------------------------------ signals + machine ----

def detect_signals(ctx):
    """SPEC 4 signal masks + funnel tallies (pre-occupancy)."""
    c1, adx1, atr1 = ctx["c1"], ctx["adx1"], ctx["atr1"]
    dcup, dclo, anchor, real1 = ctx["dcup"], ctx["dclo"], ctx["anchor"], ctx["real1"]
    n1 = ctx["n1"]
    idx = np.arange(2, n1)
    p1, p2 = idx - 1, idx - 2
    fin_up = dcup[p2] == dcup[p2]
    fin_lo = dclo[p2] == dclo[p2]
    brk_l = fin_up & (c1[p1] > dcup[p2])
    brk_s = fin_lo & (c1[p1] < dclo[p2])
    adx_ok = (adx1[p1] == adx1[p1]) & (adx1[p1] >= MIN_ADX)
    atr_ok = atr1[p1] == atr1[p1]
    exe_ok = adx_ok & atr_ok & real1[idx]
    sig_l = brk_l & (anchor[idx] == 1) & exe_ok
    sig_s = brk_s & (anchor[idx] == -1) & exe_ok
    both = sig_l & sig_s
    if both.any():
        sig_l &= ~both
        sig_s &= ~both
    li = idx[sig_l]
    si = idx[sig_s]
    cands = sorted([(int(i), 1) for i in li] + [(int(i), -1) for i in si])
    funnel = {
        "brk_long": int(brk_l.sum()), "brk_short": int(brk_s.sum()),
        "brk_both_raw": int((brk_l & brk_s).sum()),
        "cand_long": int(sig_l.sum()), "cand_short": int(sig_s.sum()),
        "anchor_block_l": int((brk_l & exe_ok & (anchor[idx] != 1)).sum()),
        "anchor_block_s": int((brk_s & exe_ok & (anchor[idx] != -1)).sum()),
        "adx_block": int((((brk_l & (anchor[idx] == 1))
                           | (brk_s & (anchor[idx] == -1))) & ~adx_ok).sum()),
    }
    return cands, funnel


def walk_exit(i, dr, ctx):
    """SPEC 5 excursion walk. Returns trade dict."""
    o1, h1, l1, c1 = ctx["o1"], ctx["h1"], ctx["l1"], ctx["c1"]
    atr1, real1 = ctx["atr1"], ctx["real1"]
    n1 = ctx["n1"]
    eps = 1e-9
    if dr == 1:
        ask_e = o1[i] + SPREAD
        sl = float(o1[i] - SL_ATR_MULT * atr1[i - 1])
        init = sl
        j = i
        while j < n1:
            if real1[j]:
                if l1[j] <= sl:                      # stop on Bid
                    fill = min(sl, o1[j])            # honest adverse gap
                    return j, init, sl, "SL_TRAIL" if sl > init + eps else "SL_INIT", \
                        fill, fill - ask_e
                sl = max(sl, c1[j] - SL_ATR_MULT * atr1[j])   # ratchet at close
            j += 1
        fill = float(c1[n1 - 1])
        return n1 - 1, init, sl, "END", fill, fill - ask_e
    bid_e = o1[i]
    sl = float(o1[i] + SL_ATR_MULT * atr1[i - 1])
    init = sl
    j = i
    while j < n1:
        if real1[j]:
            if h1[j] >= sl - SPREAD:                 # stop on Ask
                fill_ask = max(sl, o1[j] + SPREAD)   # honest adverse gap
                return j, init, sl, "SL_TRAIL" if sl < init - eps else "SL_INIT", \
                    fill_ask, bid_e - fill_ask
            sl = min(sl, c1[j] + SL_ATR_MULT * atr1[j])
        j += 1
    fill_ask = float(c1[n1 - 1]) + SPREAD
    return n1 - 1, init, sl, "END", fill_ask, bid_e - fill_ask


def run_machine(ctx, cands):
    """SPEC 4 occupancy: single trade, signals <= exit bar shadowed."""
    trades = []
    cur = -1
    shadowed = 0
    for i, dr in cands:
        if i <= cur:
            shadowed += 1
            continue
        v, init, fin, xtype, xexec, pnl = walk_exit(i, dr, ctx)
        cur = v
        trades.append({
            "dir": dr, "i": i, "v": v, "init_sl": init, "fin_sl": fin,
            "xtype": xtype, "xexec": xexec, "pnl": pnl,
        })
    return trades, shadowed


# ---------------------------------------------------------------- stats ----

def mbb_p_le_zero(x):
    n_tr = len(x)
    if n_tr < BLOCK:
        return float("nan")
    n_blocks = int(np.ceil(n_tr / BLOCK))
    starts = np.arange(0, n_tr - BLOCK + 1)
    rng = np.random.default_rng(SEED)
    sidx = rng.integers(0, len(starts), size=(BOOT_ITERS, n_blocks))
    flat = (starts[sidx][:, :, None] + np.arange(BLOCK)[None, None, :]) \
        .reshape(BOOT_ITERS, -1)[:, :n_tr]
    means = x[flat].mean(axis=1)
    return float((means <= 0.0).mean())


def block_stats(pn):
    n = len(pn)
    if n == 0:
        return {"n": 0, "win_rate": float("nan"), "mean": float("nan"),
                "total": 0.0, "sharpe": float("nan"), "pf": float("nan"),
                "median": float("nan")}
    wins = pn[pn > 0.0]
    losses = pn[pn < 0.0]
    sd = float(pn.std(ddof=1)) if n > 1 else float("nan")
    return {
        "n": int(n),
        "win_rate": float(len(wins)) / n,
        "mean": float(pn.mean()),
        "median": float(np.median(pn)),
        "total": float(pn.sum()),
        "sharpe": float(pn.mean() / sd) if n > 1 and sd > 0 else float("nan"),
        "pf": float(wins.sum() / abs(losses.sum())) if losses.size else float("inf"),
    }


def fts(epoch):
    return dt.datetime.fromtimestamp(int(epoch), tz=dt.timezone.utc) \
        .strftime("%Y.%m.%d %H:%M")


def write_signals(trades, ctx):
    grid1, d1 = ctx["grid1"], ctx["d1"]
    with open(BASE / "saleh_donchian_signals.csv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["dir", "entry_time", "exit_time", "ny_date", "block",
                    "entry_open", "entry_exec", "initial_sl", "exit_sl",
                    "exit_type", "exit_exec", "pnl_units", "pnl_usd",
                    "gross_units", "bars_held", "hours_held", "anchor_dir",
                    "adx_tmin1", "atr_tmin1", "break_level"])
        for t in trades:
            i, dr = t["i"], t["dir"]
            entry_exec = ctx["o1"][i] + (SPREAD if dr == 1 else 0.0)
            brk = ctx["dcup"][i - 2] if dr == 1 else ctx["dclo"][i - 2]
            year = int(d1[i]) // 10000
            w.writerow(["BUY" if dr == 1 else "SELL",
                        fts(grid1[i] * 3600), fts(grid1[t["v"]] * 3600),
                        int(d1[i]), "CAL" if year <= 2023 else "OOS",
                        f"{ctx['o1'][i]:.3f}", f"{entry_exec:.3f}",
                        f"{t['init_sl']:.3f}", f"{t['fin_sl']:.3f}",
                        t["xtype"], f"{t['xexec']:.3f}",
                        f"{t['pnl']:.4f}", f"{t['pnl'] * UNIT_USD:.2f}",
                        f"{t['pnl'] + SPREAD:.4f}",
                        t["v"] - i, int(grid1[t["v"]] - grid1[i]),
                        int(ctx["anchor"][i]),
                        f"{ctx['adx1'][i - 1]:.2f}", f"{ctx['atr1'][i - 1]:.3f}",
                        f"{brk:.3f}"])


def main():
    t0 = timemod.time()
    ctx = load_ctx()
    n1 = ctx["n1"]
    print(f"grids: {n1} 1H bars ({int(ctx['real1'].sum())} real / "
          f"{int((~ctx['real1']).sum())} fill), {len(ctx['grid4'])} 4H bars "
          f"({int(ctx['real4'].sum())} real / {int((~ctx['real4']).sum())} fill) "
          f"from {ctx['n_m1']} M1 bars [{timemod.time() - t0:.1f}s]")

    cands, funnel = detect_signals(ctx)
    print(f"candidates: {len(cands)} (L {funnel['cand_long']} / "
          f"S {funnel['cand_short']}) [{timemod.time() - t0:.1f}s]")

    trades, shadowed = run_machine(ctx, cands)
    funnel["shadowed_by_occupancy"] = shadowed
    funnel["executed"] = len(trades)
    print(f"machine: executed {len(trades)}, shadowed {shadowed} "
          f"[{timemod.time() - t0:.1f}s]")

    pn = np.array([t["pnl"] for t in trades], dtype=float)
    n = len(pn)
    full = block_stats(pn)
    full["p_boot"] = mbb_p_le_zero(pn) if n else float("nan")

    d1 = ctx["d1"]
    cal_pn = pn[np.array([int(d1[t["i"]]) // 10000 <= 2023 for t in trades])] if n else pn
    oos_pn = pn[np.array([int(d1[t["i"]]) // 10000 >= 2024 for t in trades])] if n else pn
    cal_s = block_stats(cal_pn)
    oos_s = block_stats(oos_pn)

    g1 = n >= 30
    g2 = n > 0 and full["sharpe"] == full["sharpe"] and full["sharpe"] > 0.0
    g3 = n >= BLOCK and full["p_boot"] == full["p_boot"] and full["p_boot"] <= 0.20
    passed = g1 and g2 and g3

    # character diagnostics (not gated)
    sides, exit_mix, yearly, holds, risks = {}, {}, {}, [], []
    for t in trades:
        k = "BUY" if t["dir"] == 1 else "SELL"
        s_ = sides.setdefault(k, [0, 0.0])
        s_[0] += 1
        s_[1] += t["pnl"]
        exit_mix[t["xtype"]] = exit_mix.get(t["xtype"], 0) + 1
        y = int(d1[t["i"]]) // 10000
        yy = yearly.setdefault(y, [0, 0.0])
        yy[0] += 1
        yy[1] += t["pnl"]
        holds.append(int(ctx["grid1"][t["v"]] - ctx["grid1"][t["i"]]))
        risks.append(abs(ctx["o1"][t["i"]] - t["init_sl"]))

    write_signals(trades, ctx)

    tape_lo = fts(int(ctx["grid1"][0]) * 3600)
    tape_hi = fts(int(ctx["grid1"][n1 - 1]) * 3600 + 3540)
    meta = {
        "study": "xauusd_saleh_donchian", "hypothesis": 57,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "script_sha256": sha256_file(str(Path(__file__).resolve())),
        "tape": str(M1_CSV), "tape_sha256": sha256_file(str(M1_CSV)),
        "tape_span_utc": [tape_lo, tape_hi], "m1_bars": int(ctx["n_m1"]),
        "grids": {"h1_total": n1, "h1_real": int(ctx["real1"].sum()),
                  "h1_fill": int((~ctx["real1"]).sum()),
                  "h4_total": int(len(ctx["grid4"])),
                  "h4_real": int(ctx["real4"].sum()),
                  "h4_fill": int((~ctx["real4"]).sum())},
        "params": {"SPREAD": SPREAD, "ST_ATR": ST_ATR, "ST_MULT": ST_MULT,
                   "DC_N": DC_N, "ADX_PERIOD": ADX_PERIOD, "MIN_ADX": MIN_ADX,
                   "ATR_PERIOD": ATR_PERIOD, "SL_ATR_MULT": SL_ATR_MULT,
                   "seed": SEED, "boot_iters": BOOT_ITERS, "block": BLOCK},
        "funnel": funnel,
        "stats_full": full, "stats_cal": cal_s, "stats_oos": oos_s,
        "gates": {"G-SAL1": bool(g1), "G-SAL2": bool(g2), "G-SAL3": bool(g3)},
        "verdict": "PASS" if passed else "FAIL",
    }
    with open(BASE / "run_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    # ------------------------------------------------ report ----
    L = []
    L.append("REPORT - HYPOTHESIS #57: SALEH 4H/1H DONCHIAN SUPERTREND - XAUUSD PHASE 1")
    L.append(f"Generated: {meta['generated_at']}  |  OVERALL VERDICT: "
             f"{'PASS' if passed else 'FAIL'}")
    L.append("")
    L.append("Mechanics (frozen in SPEC_SALEH_DONCHIAN.txt before the run): 4H NY-wall")
    L.append("Supertrend(10, 3.0) direction anchor (standard final-band recursion,")
    L.append("Wilder ATR10); 1H execution on epoch top-of-hour slots; entry at the 1H")
    L.append("bar-t open when close(t-1) breaks the Donchian(20) channel of t-2 in the")
    L.append("anchor's direction with Wilder ADX14(t-1) >= 20; the anchor 4H bar is the")
    L.append("last one strictly completed before the 1H open (previous wall bucket).")
    L.append("Initial SL = open -/+ 4xATR14(t-1); trailing ratchet at every REAL 1H")
    L.append("close: SL = max/min(prev SL, close -/+ 4xATR). No TP. Spread 0.41 with")
    L.append("#56 mechanics (long fills at Ask, stops on Bid; short fills at Bid, stops")
    L.append("on Ask with -0.41 trigger offset); worst-case stop check from the entry")
    L.append("bar; honest gap fills; single trade, signals <= exit bar shadowed;")
    L.append("continuous fill grids, execution on REAL bars only; 1.0 lot, PnL in price")
    L.append("units ($100/unit/lot); Net PnL == bid-to-bid gross - 0.41 exactly.")
    L.append("")
    L.append(f"1. TAPE: {ctx['n_m1']} M1 bars -> {n1} 1H grid bars "
             f"({int(ctx['real1'].sum())} real / {int((~ctx['real1']).sum())} fill), "
             f"{len(ctx['grid4'])} 4H grid bars "
             f"({int(ctx['real4'].sum())} real / {int((~ctx['real4']).sum())} fill), "
             f"UTC {tape_lo} .. {tape_hi} (xauusd_onboarding tape, verified 2026-09-04;")
    L.append("   full tape, single in-sample pass; CAL/OOS split reporting only).")
    L.append("")
    L.append(f"2. FUNNEL: breakout closes L {funnel['brk_long']} / S {funnel['brk_short']}"
             f" -> anchor-blocked L {funnel['anchor_block_l']} / "
             f"S {funnel['anchor_block_s']}, ADX-blocked {funnel['adx_block']}"
             f" -> candidates {len(cands)} (L {funnel['cand_long']} / "
             f"S {funnel['cand_short']}) -> shadowed by occupancy {shadowed}"
             f" -> EXECUTED {n}.")
    L.append("")
    L.append("3. NET ECONOMICS (price units, 1.0 lot):")
    L.append(f"   FULL TAPE : n={n}  win {100.0 * full['win_rate']:.1f}%  "
             f"expectancy {full['mean']:+.4f} u/trade (median {full['median']:+.4f}, "
             f"total {full['total']:+.1f} u = {full['total'] * UNIT_USD:+,.0f} $)  "
             f"PF {full['pf']:.3f}  Net Sharpe {full['sharpe']:.3f}  "
             f"MBB P(mean<=0) {full['p_boot']:.4f}")
    L.append(f"   gross mean (net + 0.41): {full['mean'] + SPREAD:+.4f} u/trade "
             f"(floor share {100.0 * SPREAD / max(abs(full['mean'] + SPREAD), 1e-9):.1f}% "
             f"of gross)" if n else "   no executed trades")
    if n:
        L.append(f"   CAL 2020-2023: n={cal_s['n']}  win {100.0 * cal_s['win_rate']:.1f}%  "
                 f"mean {cal_s['mean']:+.4f}  total {cal_s['total']:+.1f} u  "
                 f"Sharpe {cal_s['sharpe']:.3f}  PF {cal_s['pf']:.3f}")
        L.append(f"   OOS 2024-pres: n={oos_s['n']}  win {100.0 * oos_s['win_rate']:.1f}%  "
                 f"mean {oos_s['mean']:+.4f}  total {oos_s['total']:+.1f} u  "
                 f"Sharpe {oos_s['sharpe']:.3f}  PF {oos_s['pf']:.3f}")
    L.append("")
    L.append("4. CHARACTER DIAGNOSTICS (not gated):")
    if n:
        sbtxt = "  ".join(f"{k}: n={v[0]} sum={v[1]:+.1f} u" for k, v in sorted(sides.items()))
        L.append(f"   sides    : {sbtxt}")
        L.append(f"   exits    : {exit_mix}")
        L.append(f"   holding  : median {int(np.median(holds))} h, mean {np.mean(holds):.1f} h, "
                 f"p90 {int(np.percentile(holds, 90))} h, max {int(np.max(holds))} h")
        L.append(f"   init risk: median {np.median(risks):.3f} u, "
                 f"p10 {np.percentile(risks, 10):.3f}, p90 {np.percentile(risks, 90):.3f} u")
        L.append("   yearly   : " + "; ".join(
            f"{y}: n={v[0]} sum={v[1]:+.1f} u" for y, v in sorted(yearly.items())))
    else:
        L.append("   n/a (no trades)")
    L.append("")
    L.append("5. GATES (frozen in GATE_SALEH_DONCHIAN.txt before the run):")
    L.append(f"   G-SAL1 executed trades >= 30        : {n:6d}      -> {'PASS' if g1 else 'FAIL'}")
    L.append(f"   G-SAL2 Net Sharpe > 0               : {full['sharpe']:9.3f}  -> {'PASS' if g2 else 'FAIL'}")
    L.append(f"   G-SAL3 MBB P(mean<=0) <= 0.20       : {full['p_boot']:9.4f}  -> {'PASS' if g3 else 'FAIL'}")
    L.append(f"   OVERALL VERDICT: {'PASS' if passed else 'FAIL'}"
             f"{'' if passed else ' - final for this artifact; no re-runs'}")
    L.append("")
    L.append("HONESTY NOTES")
    L.append("- Single pre-declared configuration; no variant, parameter, or cost floor")
    L.append("  was run before this verdict; the fixed 0.41 spread is the pre-declared")
    L.append("  cost model, embedded in the fills (net == bid-to-bid gross - 0.41).")
    L.append("- Known optimism, declared in SPEC 8: bar-open entries and level-touch stop")
    L.append("  fills are best-case on an H1 bar reconstruction; the exit-mix / holding /")
    L.append("  trailed-vs-initial statistics exist to expose any hollow pass.")
    L.append("- Full-tape in-sample single screen: under validation_discipline a PASS here")
    L.append("  is NOT a validated result (no walk-forward, no year-sign requirement) and")
    L.append("  does not by itself authorize Phase 2 (MT5 execution); a Phase 2 needs a new")
    L.append("  pre-declared SPEC meeting the standard discipline.")
    report = "\n".join(L) + "\n"
    with open(BASE / "REPORT_SALEH_DONCHIAN_PHASE1.txt", "w") as fh:
        fh.write(report)
    print(report)
    print(f"artifacts written to {BASE} (signals {n} rows) "
          f"[{timemod.time() - t0:.1f}s total]")


if __name__ == "__main__":
    main()
