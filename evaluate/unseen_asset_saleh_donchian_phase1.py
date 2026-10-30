# -*- coding: utf-8 -*-
"""
unseen_asset_saleh_donchian_phase1.py -- Hypothesis #58: Saleh 4H/1H Donchian
Supertrend cross-asset replication on the virgin tapes (BTCUSD, ETHUSD,
USTECH), Phase 1 screen.

SPEC/GATE pre-declared and frozen BEFORE this script ran:
  data/results/unseen_asset_saleh_donchian/SPEC_SALEH_REPLICATION.txt
  data/results/unseen_asset_saleh_donchian/GATE_SALEH_REPLICATION.txt
(#57 mechanics frozen verbatim; per-instrument frozen cost floors from the
virgin_assets onboarding; gates independent per instrument on the full tape;
no ablation, no sweep.)

Tapes: data/results/virgin_assets/{btc,eth,ustech}_m1.csv (verified
onboarding 2026-09-24 for #54; M1 BID bars 2020-01-01..2026-09-23, no ask
side -> owner-estimated cost floors).

Outputs (data/results/unseen_asset_saleh_donchian/):
  saleh_replication_signals.csv    one row per executed trade (all assets)
  REPORT_SALEH_REPLICATION.txt     full report + per-instrument gates
  run_meta.json                    lineage + funnels + gates
"""

import csv
import datetime as dt
import hashlib
import json
import time as timemod
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\unseen_asset_saleh_donchian")
VIRGIN = Path(r"C:\Users\Milad\Desktop\RR\data\results\virgin_assets")

INSTRUMENTS = [
    {"asset": "BTCUSD", "tape": VIRGIN / "btc_m1.csv", "cost": 15.0},
    {"asset": "ETHUSD", "tape": VIRGIN / "eth_m1.csv", "cost": 2.0},
    {"asset": "USTECH", "tape": VIRGIN / "ustech_m1.csv", "cost": 1.0},
]

ST_ATR = 10            # 4H Supertrend ATR period
ST_MULT = 3.0          # 4H Supertrend multiplier
DC_N = 20              # 1H Donchian channel length
ADX_PERIOD = 14        # 1H Wilder ADX
MIN_ADX = 20.0
ATR_PERIOD = 14        # 1H Wilder ATR
SL_ATR_MULT = 4.0      # initial + trailing stop multiple

SEED = 20261006        # GATE_SALEH_REPLICATION.txt (per instrument)
BOOT_ITERS = 10000
BLOCK = 5


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
    """Source-exact Wilder ADX (#56/#57 implementation)."""
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
    """Standard (TradingView) Supertrend on the continuous 4H grid."""
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
    assert np.isfinite(fub).all() and np.isfinite(flb).all(), \
        "Supertrend band recursion hit NaN (leading-NaN grid poison)"
    line = np.copy(fub)                      # seed: bearish
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


def build_grids(tape):
    """SPEC 2: continuous 1H (epoch hour slots) and 4H (NY wall buckets)
    grids from an M1 tape, per-field ffill, REAL/FILL masks, both starting
    at their first REAL bucket (#57 execution-1 defect fix)."""
    df = pd.read_csv(tape, usecols=["time", "open", "high", "low", "close"])
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


def load_ctx(inst):
    g = build_grids(inst["tape"])
    o1, h1, l1, c1 = g["o1"], g["h1"], g["l1"], g["c1"]
    o4, h4, l4, c4 = g["o4"], g["h4"], g["l4"], g["c4"]
    n1 = len(o1)

    s_h = pd.Series(h1)
    s_l = pd.Series(l1)
    s_c = pd.Series(c1)
    adx1 = wilder_adx(s_h, s_l, s_c)
    atr1 = wilder_atr(h1, l1, c1, ATR_PERIOD)
    dcup = s_h.rolling(DC_N, min_periods=DC_N).max().to_numpy(float)
    dclo = s_l.rolling(DC_N, min_periods=DC_N).min().to_numpy(float)

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
        "anchor": anchor, "cost": inst["cost"],
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
    """SPEC 5 excursion walk with the instrument cost floor C."""
    o1, h1, l1, c1 = ctx["o1"], ctx["h1"], ctx["l1"], ctx["c1"]
    atr1, real1 = ctx["atr1"], ctx["real1"]
    C = ctx["cost"]
    n1 = ctx["n1"]
    eps = 1e-9
    if dr == 1:
        ask_e = o1[i] + C
        sl = float(o1[i] - SL_ATR_MULT * atr1[i - 1])
        init = sl
        j = i
        while j < n1:
            if real1[j]:
                if l1[j] <= sl:                      # stop on Bid
                    fill = min(sl, o1[j])            # honest adverse gap
                    return j, init, sl, \
                        "SL_TRAIL" if sl > init + eps else "SL_INIT", \
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
            if h1[j] >= sl - C:                      # stop on Ask
                fill_ask = max(sl, o1[j] + C)        # honest adverse gap
                return j, init, sl, \
                    "SL_TRAIL" if sl < init - eps else "SL_INIT", \
                    fill_ask, bid_e - fill_ask
            sl = min(sl, c1[j] + SL_ATR_MULT * atr1[j])
        j += 1
    fill_ask = float(c1[n1 - 1]) + C
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


def write_signals(all_rows):
    with open(BASE / "saleh_replication_signals.csv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["asset", "cost_floor", "dir", "entry_time", "exit_time",
                    "ny_date", "block", "entry_open", "entry_exec",
                    "initial_sl", "exit_sl", "exit_type", "exit_exec",
                    "pnl_units", "gross_units", "bars_held", "hours_held",
                    "anchor_dir", "adx_tmin1", "atr_tmin1", "break_level"])
        for r in all_rows:
            w.writerow(r)


def main():
    t0 = timemod.time()
    results = {}
    all_rows = []

    for inst in INSTRUMENTS:
        asset = inst["asset"]
        ctx = load_ctx(inst)
        n1 = ctx["n1"]
        cands, funnel = detect_signals(ctx)
        trades, shadowed = run_machine(ctx, cands)
        funnel["shadowed_by_occupancy"] = shadowed
        funnel["executed"] = len(trades)

        pn = np.array([t["pnl"] for t in trades], dtype=float)
        n = len(pn)
        full = block_stats(pn)
        full["p_boot"] = mbb_p_le_zero(pn) if n else float("nan")

        d1 = ctx["d1"]
        if n:
            yrs = np.array([int(d1[t["i"]]) // 10000 for t in trades])
            cal_pn = pn[yrs <= 2023]
            oos_pn = pn[yrs >= 2024]
        else:
            cal_pn, oos_pn = pn, pn
        cal_s = block_stats(cal_pn)
        oos_s = block_stats(oos_pn)

        g1 = n >= 30
        g2 = n > 0 and full["sharpe"] == full["sharpe"] and full["sharpe"] > 0.0
        g3 = n >= BLOCK and full["p_boot"] == full["p_boot"] \
            and full["p_boot"] <= 0.20
        passed = g1 and g2 and g3

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

        C = ctx["cost"]
        for t in trades:
            i, dr = t["i"], t["dir"]
            entry_exec = ctx["o1"][i] + (C if dr == 1 else 0.0)
            brk = ctx["dcup"][i - 2] if dr == 1 else ctx["dclo"][i - 2]
            year = int(d1[i]) // 10000
            all_rows.append([
                asset, f"{C:.1f}", "BUY" if dr == 1 else "SELL",
                fts(ctx["grid1"][i] * 3600), fts(ctx["grid1"][t["v"]] * 3600),
                int(d1[i]), "CAL" if year <= 2023 else "OOS",
                f"{ctx['o1'][i]:.3f}", f"{entry_exec:.3f}",
                f"{t['init_sl']:.3f}", f"{t['fin_sl']:.3f}",
                t["xtype"], f"{t['xexec']:.3f}",
                f"{t['pnl']:.4f}", f"{t['pnl'] + C:.4f}",
                t["v"] - i, int(ctx["grid1"][t["v"]] - ctx["grid1"][i]),
                int(ctx["anchor"][i]),
                f"{ctx['adx1'][i - 1]:.2f}", f"{ctx['atr1'][i - 1]:.3f}",
                f"{brk:.3f}"])

        tape_lo = fts(int(ctx["grid1"][0]) * 3600)
        tape_hi = fts(int(ctx["grid1"][n1 - 1]) * 3600 + 3540)
        results[asset] = {
            "inst": inst, "ctx": ctx, "funnel": funnel, "trades": trades,
            "n": n, "full": full, "cal": cal_s, "oos": oos_s,
            "gates": (g1, g2, g3), "passed": passed,
            "sides": sides, "exit_mix": exit_mix, "yearly": yearly,
            "holds": holds, "risks": risks,
            "span": [tape_lo, tape_hi], "m1": ctx["n_m1"],
        }
        print(f"{asset}: {n1} 1H bars ({int(ctx['real1'].sum())} real), "
              f"{len(ctx['grid4'])} 4H ({int(ctx['real4'].sum())} real) | "
              f"candidates {len(cands)} -> executed {n}, shadowed {shadowed} "
              f"[{timemod.time() - t0:.1f}s]")

    write_signals(all_rows)

    overall = all(r["passed"] for r in results.values())

    meta = {
        "study": "unseen_asset_saleh_donchian", "hypothesis": 58,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "script_sha256": sha256_file(str(Path(__file__).resolve())),
        "instruments": {
            a: {"tape": str(r["inst"]["tape"]),
                "tape_sha256": sha256_file(str(r["inst"]["tape"])),
                "cost_floor": r["inst"]["cost"],
                "m1_bars": int(r["m1"]),
                "grids": {"h1_total": r["ctx"]["n1"],
                          "h1_real": int(r["ctx"]["real1"].sum()),
                          "h4_total": int(len(r["ctx"]["grid4"])),
                          "h4_real": int(r["ctx"]["real4"].sum())},
                "span_utc": r["span"]}
            for a, r in results.items()},
        "params": {"ST_ATR": ST_ATR, "ST_MULT": ST_MULT, "DC_N": DC_N,
                   "ADX_PERIOD": ADX_PERIOD, "MIN_ADX": MIN_ADX,
                   "ATR_PERIOD": ATR_PERIOD, "SL_ATR_MULT": SL_ATR_MULT,
                   "seed": SEED, "boot_iters": BOOT_ITERS, "block": BLOCK},
        "funnels": {a: r["funnel"] for a, r in results.items()},
        "stats": {a: {"full": r["full"], "cal": r["cal"], "oos": r["oos"]}
                  for a, r in results.items()},
        "gates": {a: {"G-SALREP1": bool(r["gates"][0]),
                      "G-SALREP2": bool(r["gates"][1]),
                      "G-SALREP3": bool(r["gates"][2]),
                      "instrument_pass": bool(r["passed"])}
                  for a, r in results.items()},
        "verdict": "PASS" if overall else "FAIL",
    }
    with open(BASE / "run_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    # ------------------------------------------------ report ----
    L = []
    L.append("REPORT - HYPOTHESIS #58: SALEH 4H/1H DONCHIAN SUPERTREND - "
             "UNSEEN-ASSET REPLICATION PHASE 1")
    L.append(f"Generated: {meta['generated_at']}  |  OVERALL VERDICT: "
             f"{'PASS' if overall else 'FAIL'}")
    L.append("")
    L.append("Mechanics (frozen in SPEC_SALEH_REPLICATION.txt before the run): #57")
    L.append("verbatim - 4H NY-wall Supertrend(10, 3.0) direction anchor; 1H epoch")
    L.append("top-of-hour execution; entry at the 1H bar-t open when close(t-1) breaks")
    L.append("the Donchian(20) channel of t-2 in the anchor's direction with Wilder")
    L.append("ADX14(t-1) >= 20; anchor = last strictly-completed 4H wall bucket;")
    L.append("initial SL = open -/+ 4xATR14(t-1); trailing ratchet at every REAL 1H")
    L.append("close (SL = max/min(prev, close -/+ 4xATR), never loosens); no TP;")
    L.append("#56/#57 spread architecture with per-instrument frozen cost floors")
    L.append("(long fills at Ask = Bid + C, stops on Bid; short fills at Bid, stops on")
    L.append("Ask with -C trigger offset); worst-case stop check from the entry bar;")
    L.append("honest gap fills; single trade per instrument, signals <= exit bar")
    L.append("shadowed; continuous fill grids starting at first REAL bucket, execution")
    L.append("on REAL bars only; 1.0 lot; net PnL == bid-to-bid gross - C. Cost floors")
    L.append("are the #54 virgin-onboarding owner estimates (no ask side in the tapes).")
    L.append("Gates independent per instrument on the FULL tape (seed 20261006);")
    L.append("CAL [2020..2023] / OOS [2024..present] reporting only.")
    L.append("")
    for a, r in results.items():
        C = r["inst"]["cost"]
        f = r["funnel"]
        fl = r["full"]
        g1, g2, g3 = r["gates"]
        L.append(f"== {a} (cost floor {C:.1f} RT) == "
                 f"{r['m1']} M1 bars, {r['ctx']['n1']} 1H grid "
                 f"({int(r['ctx']['real1'].sum())} real), "
                 f"{len(r['ctx']['grid4'])} 4H grid "
                 f"({int(r['ctx']['real4'].sum())} real), UTC "
                 f"{r['span'][0]} .. {r['span'][1]}")
        L.append(f"   FUNNEL: breakout closes L {f['brk_long']} / S {f['brk_short']}"
                 f" -> anchor-blocked L {f['anchor_block_l']} / "
                 f"S {f['anchor_block_s']}, ADX-blocked {f['adx_block']}"
                 f" -> candidates {f['cand_long'] + f['cand_short']} "
                 f"(L {f['cand_long']} / S {f['cand_short']})"
                 f" -> shadowed {f['shadowed_by_occupancy']} -> EXECUTED {r['n']}.")
        L.append(f"   FULL TAPE : n={r['n']}  win {100.0 * fl['win_rate']:.1f}%  "
                 f"net expectancy {fl['mean']:+.4f} u/trade "
                 f"(median {fl['median']:+.4f}, total {fl['total']:+.1f} u)  "
                 f"PF {fl['pf']:.3f}  Net Sharpe {fl['sharpe']:.3f}  "
                 f"MBB P(mean<=0) {fl['p_boot']:.4f}")
        if r["n"]:
            gm = fl["mean"] + C
            band = "  [INSIDE FLOOR-UNCERTAINTY BAND]" \
                if 0.0 < gm < C else ""
            L.append(f"   gross mean (net + {C:.1f}): {gm:+.4f} u/trade "
                     f"(floor share {100.0 * C / max(abs(gm), 1e-9):.1f}% of "
                     f"gross){band}")
            cs, os_ = r["cal"], r["oos"]
            deg = "DEGRADES" if (cs["sharpe"] == cs["sharpe"]
                                 and os_["sharpe"] == os_["sharpe"]
                                 and os_["sharpe"] < cs["sharpe"]) else "IMPROVES/HOLDS"
            L.append(f"   CAL 2020-2023: n={cs['n']}  win {100.0 * cs['win_rate']:.1f}%  "
                     f"mean {cs['mean']:+.4f}  total {cs['total']:+.1f} u  "
                     f"Sharpe {cs['sharpe']:.3f}  PF {cs['pf']:.3f}")
            L.append(f"   OOS 2024-pres: n={os_['n']}  win {100.0 * os_['win_rate']:.1f}%  "
                     f"mean {os_['mean']:+.4f}  total {os_['total']:+.1f} u  "
                     f"Sharpe {os_['sharpe']:.3f}  PF {os_['pf']:.3f}  "
                     f"-> CAL->OOS {deg}")
            sbtxt = "  ".join(f"{k}: n={v[0]} sum={v[1]:+.1f} u"
                              for k, v in sorted(r["sides"].items()))
            L.append(f"   sides: {sbtxt}  |  exits: {r['exit_mix']}")
            L.append(f"   holding: median {int(np.median(r['holds']))} h, "
                     f"mean {np.mean(r['holds']):.1f} h, "
                     f"p90 {int(np.percentile(r['holds'], 90))} h, "
                     f"max {int(np.max(r['holds']))} h  |  "
                     f"init risk: median {np.median(r['risks']):.3f} u "
                     f"(p10 {np.percentile(r['risks'], 10):.3f}, "
                     f"p90 {np.percentile(r['risks'], 90):.3f})")
            L.append("   yearly   : " + "; ".join(
                f"{y}: n={v[0]} sum={v[1]:+.1f} u"
                for y, v in sorted(r["yearly"].items())))
        L.append(f"   GATES: G-SALREP1 trades >= 30 : {r['n']:6d} -> "
                 f"{'PASS' if g1 else 'FAIL'}  |  G-SALREP2 Net Sharpe > 0 : "
                 f"{fl['sharpe']:8.3f} -> {'PASS' if g2 else 'FAIL'}  |  "
                 f"G-SALREP3 MBB P <= 0.20 : {fl['p_boot']:8.4f} -> "
                 f"{'PASS' if g3 else 'FAIL'}")
        L.append(f"   INSTRUMENT VERDICT: {'PASS' if r['passed'] else 'FAIL'}")
        L.append("")
    L.append(f"OVERALL VERDICT: {'PASS' if overall else 'FAIL'} "
             f"(rule: OVERALL PASS only if ALL THREE instruments pass; "
             f"passing: "
             f"{', '.join(a for a, r in results.items() if r['passed']) or 'none'}"
             f"){'' if overall else ' - final for this artifact; no re-runs'}")
    L.append("")
    L.append("HONESTY NOTES")
    L.append("- Single pre-declared configuration per instrument (#57 frozen values,")
    L.append("  verbatim mechanics); no variant, parameter, or cost floor was run")
    L.append("  before this verdict; net PnL embeds each floor in the fills")
    L.append("  (net == bid-to-bid gross - C).")
    L.append("- Cost floors are #54 virgin-onboarding OWNER ESTIMATES (no ask side in")
    L.append("  the M1 BID exports); verdicts with positive gross mean below the floor")
    L.append("  are flagged inside the floor-uncertainty band (SPEC 6).")
    L.append("- Known optimism, declared in SPEC 8 (inherited from #57): bar-open")
    L.append("  entries and level-touch stop fills are best-case on an H1 bar")
    L.append("  reconstruction; exit-mix / holding / trailed-vs-initial statistics are")
    L.append("  computed so hollow passes are detectable.")
    L.append("- In-sample full-tape first-touch replication per instrument: a PASS is")
    L.append("  NOT a validated result under validation_discipline (no walk-forward,")
    L.append("  no year-sign gate) and does not authorize Phase 2; Phase 2 requires a")
    L.append("  NEW pre-declared SPEC meeting the standard discipline.")
    report = "\n".join(L) + "\n"
    with open(BASE / "REPORT_SALEH_REPLICATION.txt", "w") as fh:
        fh.write(report)
    print(report)
    print(f"artifacts written to {BASE} "
          f"(signals {len(all_rows)} rows) [{timemod.time() - t0:.1f}s total]")


if __name__ == "__main__":
    main()
