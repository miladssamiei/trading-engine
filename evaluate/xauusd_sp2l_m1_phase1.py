# -*- coding: utf-8 -*-
"""
xauusd_sp2l_m1_phase1.py -- Hypothesis #56: SP2L Advanced M1 Scalper,
native XAUUSD Phase 1 signal-quality + spread-sensitivity screen.

SPEC/GATE pre-declared and frozen BEFORE this script ran:
  data/results/xauusd_sp2l_m1/SPEC_SP2L_M1.txt
  data/results/xauusd_sp2l_m1/GATE_SP2L_M1.txt
(one configuration: all filters enabled, fixed spread 0.41, session
[01:00, 05:00) ET; no ablation, no sweep).

Tape: data/results/xauusd_onboarding/xauusdtick_m1.csv (verified onboarding
2026-09-04; 2,337,538 M1 bars, 2020-01-01..2026-09-01). Ask = Bid + 0.41
everywhere (onboarding full-tape median spread).

Outputs (data/results/xauusd_sp2l_m1/):
  sp2l_m1_signals.csv            one row per executed trade
  REPORT_SP2L_M1_PHASE1.txt      full report + gate verdicts
  run_meta.json                  lineage + funnel + gates
"""

import csv
import datetime as dt
import hashlib
import json
import time as timemod
from array import array
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\xauusd_sp2l_m1")
M1_CSV = Path(r"C:\Users\Milad\Desktop\RR\data\results\xauusd_onboarding\xauusdtick_m1.csv")

SPREAD = 0.41          # fixed round-trip spread, price units (SPEC 2)
PGAP = 1.0             # P-gap imbalance, price units (SPEC 3, B4)
MAX_SL = 10.0          # max risk, price units (SPEC 4, E4)
SPIKE_RATIO = 1.5      # SPEC 3, B5
EMA_PERIOD = 60        # SPEC 4, E5
ADX_PERIOD = 14        # SPEC 4, E6
MIN_ADX = 20.0
MAX_OPP = 1            # SPEC 4, E1
TP_R = 1.0             # SPEC 6
SESS_LO = 60           # 01:00 ET wall minute, inclusive (SPEC 4, E7)
SESS_HI = 300          # 05:00 ET wall minute, exclusive

SEED = 20261004        # GATE_SP2L_M1.txt
BOOT_ITERS = 10000
BLOCK = 5
UNIT_USD = 100.0       # $/price-unit/lot, display only (SPEC 2)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- data ----

def calc_adx(df, period=ADX_PERIOD):
    """Source-exact Wilder ADX (SP2L_Advanced_Bot.calculate_adx)."""
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()],
                   axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr
    den = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / den
    return dx.ewm(alpha=1.0 / period, adjust=False).mean().to_numpy(float)


def load_ctx():
    df = pd.read_csv(M1_CSV)
    t = pd.to_datetime(df["time"], unit="s", utc=True)
    ny = t.dt.tz_convert("America/New_York")
    ny_wall = (ny.dt.hour * 60 + ny.dt.minute).to_numpy(np.int32)
    ny_date = (ny.dt.year * 10000 + ny.dt.month * 100 + ny.dt.day).to_numpy(np.int64)

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    ema = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean().to_numpy(float)
    adx = calc_adx(df)

    N = len(df)
    starts = np.flatnonzero(np.r_[True, ny_date[1:] != ny_date[:-1]])
    ends = np.r_[starts[1:], N]
    blk_of = np.repeat(np.arange(len(starts), dtype=np.int32), ends - starts)

    return {
        "N": N, "time": df["time"].to_numpy(np.int64),
        "o": o, "h": h, "l": l, "c": c, "ema": ema, "adx": adx,
        "ny_wall": ny_wall, "ny_date": ny_date,
        "blk_starts": starts, "blk_ends": ends, "nblk": len(starts),
        "A_o": array("d", o), "A_h": array("d", h),
        "A_l": array("d", l), "A_c": array("d", c),
        "A_ema": array("d", ema), "A_adx": array("d", adx),
        "A_wall": array("i", ny_wall), "A_blkof": array("i", blk_of),
        "A_blkend": array("i", ends.astype(np.int32)),
    }


# ------------------------------------------------------------ detection ----

def detect_setups(ctx):
    """SPEC 3 on completed bars; detection bar d, pattern d-3/d-2/d-1."""
    o, h, l, c = ctx["o"], ctx["h"], ctx["l"], ctx["c"]
    N = ctx["N"]
    body = np.abs(c - o)
    idx = np.arange(3, N)                 # detection bars d
    p1, p2, p3 = idx - 1, idx - 2, idx - 3  # t-2, t-3, t-4
    buy = (
        (c[p1] > o[p1]) & (c[p2] > o[p2]) & (c[p3] > o[p3])
        & (c[p1] > c[p2]) & (c[p2] > c[p3])
        & (o[p1] > o[p2]) & (o[p2] > o[p3])
        & (l[p1] > h[p3] + PGAP)
        & (body[p2] > SPIKE_RATIO * body[p1])
        & (body[p2] > SPIKE_RATIO * body[p3])
    )
    sell = (
        (c[p1] < o[p1]) & (c[p2] < o[p2]) & (c[p3] < o[p3])
        & (c[p1] < c[p2]) & (c[p2] < c[p3])
        & (o[p1] < o[p2]) & (o[p2] < o[p3])
        & (h[p1] < l[p3] - PGAP)
        & (body[p2] > SPIKE_RATIO * body[p1])
        & (body[p2] > SPIKE_RATIO * body[p3])
    )
    both = buy & sell
    buy &= ~both
    sell &= ~both
    bi = idx[buy]
    si = idx[sell]
    setups = [(int(i), 1, float(l[i - 3])) for i in bi] \
        + [(int(i), -1, float(h[i - 3])) for i in si]
    setups.sort()
    return setups, int(bi.size), int(si.size), int(both.size)


# ------------------------------------------------- pending machine + exits ----

def walk_exit(d, u, trig, sl, risk, o, h, l, c, N):
    """SPEC 6 exit walk. Returns (exit_bar, exit_type, exit_exec, pnl)."""
    if d == 1:
        ask_e = trig + SPREAD
        sl_bid = sl
        tp_bid = ask_e + TP_R * risk
        for v in range(u, N):
            if v == u:                     # entry at the bar's low; SL unreachable
                if h[v] >= tp_bid:
                    return v, "TP", tp_bid, tp_bid - ask_e
                continue
            if l[v] <= sl_bid:             # worst-case SL first
                fill = min(sl_bid, o[v])   # honest gap fill (adverse)
                return v, "SL", fill, fill - ask_e
            if h[v] >= tp_bid:
                fill = max(tp_bid, o[v])   # gap open beyond target (favorable)
                return v, "TP", fill, fill - ask_e
        v = N - 1
        fill = float(c[v])
        return v, "END", fill, fill - ask_e
    bid_e = trig
    sl_bid_lvl = sl - SPREAD               # Bid level where Ask = structural SL
    tp_bid_lvl = bid_e - TP_R * risk - SPREAD
    tp_ask = bid_e - TP_R * risk
    for v in range(u, N):
        if v == u:                         # entry at the bar's high
            if l[v] <= sl_bid_lvl:
                return v, "SL", sl, bid_e - sl
            if l[v] <= tp_bid_lvl:
                return v, "TP", tp_ask, TP_R * risk
            continue
        if h[v] >= sl_bid_lvl:             # worst-case SL first
            fill_ask = max(sl, o[v] + SPREAD)
            return v, "SL", fill_ask, bid_e - fill_ask
        if l[v] <= tp_bid_lvl:
            fill_ask = min(tp_ask, o[v] + SPREAD)
            return v, "TP", fill_ask, bid_e - fill_ask
    v = N - 1
    fill_ask = float(c[v]) + SPREAD
    return v, "END", fill_ask, bid_e - fill_ask


def run_machine(ctx, setups):
    """SPEC 4 pending machine. Returns (trades, tallies)."""
    o = ctx["A_o"]; h = ctx["A_h"]; l = ctx["A_l"]; c = ctx["A_c"]
    ema = ctx["A_ema"]; adx = ctx["A_adx"]
    wall = ctx["A_wall"]; blk_of = ctx["A_blkof"]; blk_end = ctx["A_blkend"]
    N = ctx["N"]
    ns = len(setups)

    trades = []
    tallies = {"setups": 0, "entered": 0, "killed": 0, "expired": 0,
               "trend_dead": 0, "risk_nonpos_skips": 0}
    cur = -1
    si = 0
    while True:
        while si < ns and setups[si][0] <= cur:
            si += 1
        if si >= ns:
            break
        d, dr, sl = setups[si]
        si += 1
        tallies["setups"] += 1
        e0 = blk_end[blk_of[d]]            # pending dies at its NY date end
        u = d + 1
        opp = 0
        dead = False
        outcome = None
        trig = None
        risk = None
        while u < e0:
            w = wall[u]
            if w >= SESS_HI:               # past 05:00 ET: entry impossible today
                outcome = ("expired", e0 - 1)
                break
            if dr == 1:                    # trend structure, window (d, u]
                if h[u] > h[u - 1]:
                    opp = 0
                else:
                    opp += 1
                    if opp > MAX_OPP and not dead:
                        dead = True
                        tallies["trend_dead"] += 1
            else:
                if l[u] < l[u - 1]:
                    opp = 0
                else:
                    opp += 1
                    if opp > MAX_OPP and not dead:
                        dead = True
                        tallies["trend_dead"] += 1
            if dead:                       # window only grows -> permanently blocked
                outcome = ("expired", e0 - 1)
                break
            if dr == 1:                    # E2/E3: pullback break + risk
                if l[u] >= l[u - 1]:
                    u += 1
                    continue
                trig = l[u]
                risk = trig - sl
            else:
                if h[u] <= h[u - 1]:
                    u += 1
                    continue
                trig = h[u]
                risk = sl - trig
            if risk <= 0.0:                # pullback not deep enough yet
                tallies["risk_nonpos_skips"] += 1
                u += 1
                continue
            if risk > MAX_SL:              # E4: source INVALID
                outcome = ("killed", u)
                break
            if dr == 1:                    # E5: EMA
                if c[u] <= ema[u]:
                    u += 1
                    continue
            else:
                if c[u] >= ema[u]:
                    u += 1
                    continue
            a = adx[u]                     # E6: ADX
            if not (a == a) or a < MIN_ADX:
                u += 1
                continue
            if w < SESS_LO:                # E7: session (before 01:00 ET)
                u += 1
                continue
            outcome = ("entered", u)       # E8: execute
            break
        if outcome is None:
            outcome = ("expired", e0 - 1)
        kind, obar = outcome
        tallies[kind] += 1
        cur = obar
        if kind != "entered":
            continue
        vb, xtype, xexec, pnl = walk_exit(
            dr, u, trig, sl, risk, o, h, l, c, N)
        cur = vb                           # trade occupies through its exit bar
        trades.append({
            "dir": dr, "d": d, "u": u, "v": vb, "sl": sl, "risk": risk,
            "trig": trig, "xtype": xtype, "xexec": xexec, "pnl": pnl,
        })
    return trades, tallies


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


def fts(epoch):
    return dt.datetime.fromtimestamp(int(epoch), tz=dt.timezone.utc) \
        .strftime("%Y.%m.%d %H:%M")


def write_signals(trades, ctx):
    nd = ctx["ny_date"]
    tm = ctx["time"]
    with open(BASE / "sp2l_m1_signals.csv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["dir", "setup_time", "entry_time", "exit_time",
                    "ny_date", "entry_wall_min", "entry_trigger",
                    "entry_exec", "sl_trigger_bid", "tp_trigger_bid",
                    "risk", "exit_type", "exit_exec", "pnl_units",
                    "pnl_usd", "bars_held", "same_bar"])
        for t in trades:
            dr, u = t["dir"], t["u"]
            if dr == 1:
                entry_exec = t["trig"] + SPREAD
                sl_bid = t["sl"]
                tp_bid = entry_exec + TP_R * t["risk"]
            else:
                entry_exec = t["trig"]
                sl_bid = t["sl"] - SPREAD
                tp_bid = t["trig"] - TP_R * t["risk"] - SPREAD
            w.writerow(["BUY" if dr == 1 else "SELL",
                        fts(tm[t["d"]]), fts(tm[u]), fts(tm[t["v"]]),
                        int(nd[u]), int(ctx["ny_wall"][u]),
                        f"{t['trig']:.3f}", f"{entry_exec:.3f}",
                        f"{sl_bid:.3f}", f"{tp_bid:.3f}",
                        f"{t['risk']:.3f}", t["xtype"],
                        f"{t['xexec']:.3f}", f"{t['pnl']:.4f}",
                        f"{t['pnl'] * UNIT_USD:.2f}",
                        t["v"] - u, int(t["v"] == u)])


def main():
    t0 = timemod.time()
    ctx = load_ctx()
    print(f"loaded {ctx['N']} M1 bars, {ctx['nblk']} NY dates "
          f"({timemod.time() - t0:.1f}s)")

    setups, nb, ns_, nboth = detect_setups(ctx)
    print(f"setups: {len(setups)} (buy {nb} / sell {ns_}, "
          f"both-fired dropped {nboth}) [{timemod.time() - t0:.1f}s]")

    trades, tallies = run_machine(ctx, setups)
    print(f"machine: {tallies} [{timemod.time() - t0:.1f}s]")

    pn = np.array([t["pnl"] for t in trades], dtype=float)
    n = len(pn)
    stats = {}
    if n:
        wins = pn[pn > 0.0]
        losses = pn[pn < 0.0]
        stats = {
            "n": n,
            "win_rate": float(len(wins)) / n,
            "mean": float(pn.mean()),
            "median": float(np.median(pn)),
            "total": float(pn.sum()),
            "std": float(pn.std(ddof=1)) if n > 1 else float("nan"),
            "pf": float(wins.sum() / abs(losses.sum())) if losses.size else float("inf"),
            "sharpe": float(pn.mean() / pn.std(ddof=1)) if n > 1 and pn.std(ddof=1) > 0 else float("nan"),
        }
        stats["p_boot"] = mbb_p_le_zero(pn)
    else:
        stats = {"n": 0, "win_rate": float("nan"), "mean": float("nan"),
                 "median": float("nan"), "total": 0.0, "std": float("nan"),
                 "pf": float("nan"), "sharpe": float("nan"),
                 "p_boot": float("nan")}

    g1 = n >= 30
    g2 = n > 0 and stats["sharpe"] == stats["sharpe"] and stats["sharpe"] > 0.0
    g3 = n >= BLOCK and stats["p_boot"] == stats["p_boot"] and stats["p_boot"] <= 0.20
    passed = g1 and g2 and g3

    # character diagnostics (not gated)
    sides = {}
    exit_mix = {}
    yearly = {}
    same_bar = 0
    holds = []
    risks = []
    for t in trades:
        k = "BUY" if t["dir"] == 1 else "SELL"
        s_ = sides.setdefault(k, [0, 0.0])
        s_[0] += 1
        s_[1] += t["pnl"]
        exit_mix[t["xtype"]] = exit_mix.get(t["xtype"], 0) + 1
        y = int(ctx["ny_date"][t["u"]]) // 10000
        yy = yearly.setdefault(y, [0, 0.0])
        yy[0] += 1
        yy[1] += t["pnl"]
        same_bar += int(t["v"] == t["u"])
        holds.append(t["v"] - t["u"])
        risks.append(t["risk"])

    write_signals(trades, ctx)

    tape_lo = fts(ctx["time"][0])
    tape_hi = fts(ctx["time"][ctx["N"] - 1])
    meta = {
        "study": "xauusd_sp2l_m1", "hypothesis": 56,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "script_sha256": sha256_file(str(Path(__file__).resolve())),
        "tape": str(M1_CSV), "tape_sha256": sha256_file(str(M1_CSV)),
        "tape_span_utc": [tape_lo, tape_hi], "bars": int(ctx["N"]),
        "params": {"SPREAD": SPREAD, "PGAP": PGAP, "MAX_SL": MAX_SL,
                   "SPIKE_RATIO": SPIKE_RATIO, "EMA_PERIOD": EMA_PERIOD,
                   "ADX_PERIOD": ADX_PERIOD, "MIN_ADX": MIN_ADX,
                   "MAX_OPP": MAX_OPP, "TP_R": TP_R,
                   "SESS_LO_ET": "01:00", "SESS_HI_ET": "05:00",
                   "seed": SEED, "boot_iters": BOOT_ITERS, "block": BLOCK},
        "funnel": {"setups_buy": nb, "setups_sell": ns_,
                   "setups_both_dropped": nboth, **tallies},
        "stats": stats,
        "gates": {"G-SP1": bool(g1), "G-SP2": bool(g2), "G-SP3": bool(g3)},
        "verdict": "PASS" if passed else "FAIL",
    }
    with open(BASE / "run_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    # ------------------------------------------------ report ----
    L = []
    L.append("REPORT - HYPOTHESIS #56: SP2L ADVANCED M1 SCALPER - NATIVE XAUUSD PHASE 1")
    L.append(f"Generated: {meta['generated_at']}  |  OVERALL VERDICT: "
             f"{'PASS' if passed else 'FAIL'}")
    L.append("")
    L.append("Mechanics (frozen in SPEC_SP2L_M1.txt before the run): M1 3-candle expansion/")
    L.append("imbalance detection on bars t-4/t-3/t-2 evaluated at completed bar t-1")
    L.append("(3 aligned candles, strictly monotone opens/closes, P-gap 1.0 price unit across")
    L.append("the spike, spike body > 1.5x both neighbors, SL = extreme of t-4); pending-order")
    L.append("pullback entry (breaking bar's own extreme as Trigger_Price, risk clamp (0, 10.0]);")
    L.append("all filters enabled: EMA60 direction, Wilder ADX14 >= 20, trend structure max 1")
    L.append("opposite move, NY session [01:00, 05:00) ET at entry; spread-aware brackets with")
    L.append("fixed 0.41 spread (long at Ask = Bid+0.41, TP_Bid = Ask+1R, SL_Bid = structural;")
    L.append("short at Bid, stops/targets offset by the spread on the Ask side); worst-case")
    L.append("same-bar SL-before-TP; honest gap fills; max 1 open trade; pending expires at its")
    L.append("NY date end; 1.0 lot, PnL in price units ($100/unit/lot).")
    L.append("")
    L.append(f"1. TAPE: {ctx['N']} M1 bars, {ctx['nblk']} NY dates, UTC {tape_lo} .. {tape_hi}")
    L.append(f"   (data/results/xauusd_onboarding/xauusdtick_m1.csv, verified 2026-09-04; full")
    L.append("   tape, single pass, in-sample screen).")
    L.append("")
    L.append("2. FUNNEL: setups "
             f"buy {nb} / sell {ns_} (both-fired dropped {nboth}); pendings processed "
             f"{tallies['setups']}; entered {tallies['entered']}; "
             f"killed (risk > 10.0) {tallies['killed']}; "
             f"expired {tallies['expired']} (of which trend-dead "
             f"{tallies['trend_dead']}); break-bars skipped with risk <= 0: "
             f"{tallies['risk_nonpos_skips']}.")
    L.append("")
    L.append("3. NET ECONOMICS (price units, 1.0 lot):")
    if n:
        L.append(f"   executed trades   : {n}")
        L.append(f"   win rate          : {100.0 * stats['win_rate']:.1f}%")
        L.append(f"   net expectancy    : {stats['mean']:+.4f} u/trade "
                 f"(median {stats['median']:+.4f}, total {stats['total']:+.1f} u, "
                 f"{stats['total'] * UNIT_USD:+,.0f} $)")
        L.append(f"   profit factor     : {stats['pf']:.3f}")
        L.append(f"   Net Sharpe        : {stats['sharpe']:.3f} "
                 f"(per-trade, ddof=1, not annualized)")
        L.append(f"   MBB P(mean<=0)    : {stats['p_boot']:.4f} "
                 f"(block={BLOCK}, {BOOT_ITERS} resamples, seed {SEED})")
    else:
        L.append("   no executed trades")
    L.append("")
    L.append("4. CHARACTER DIAGNOSTICS (not gated; the #52/#53 hollow-pass check):")
    if n:
        sbtxt = "  ".join(f"{k}: n={v[0]} sum={v[1]:+.1f} u" for k, v in sorted(sides.items()))
        L.append(f"   sides   : {sbtxt}")
        L.append(f"   exits   : {exit_mix}")
        L.append(f"   same-bar-exit share: {100.0 * same_bar / n:.1f}%  "
                 f"(median bars held {int(np.median(holds))}, "
                 f"p90 {int(np.percentile(holds, 90))})")
        L.append(f"   risk    : median {np.median(risks):.3f} u, "
                 f"p10 {np.percentile(risks, 10):.3f}, p90 {np.percentile(risks, 90):.3f} u")
        L.append("   yearly  : " + "; ".join(
            f"{y}: n={v[0]} sum={v[1]:+.1f} u" for y, v in sorted(yearly.items())))
    else:
        L.append("   n/a (no trades)")
    L.append("")
    L.append("5. GATES (frozen in GATE_SP2L_M1.txt before the run):")
    L.append(f"   G-SP1 executed trades >= 30        : {n:6d}      -> {'PASS' if g1 else 'FAIL'}")
    L.append(f"   G-SP2 Net Sharpe > 0               : {stats['sharpe']:9.3f}  -> {'PASS' if g2 else 'FAIL'}")
    L.append(f"   G-SP3 MBB P(mean<=0) <= 0.20       : {stats['p_boot']:9.4f}  -> {'PASS' if g3 else 'FAIL'}")
    L.append(f"   OVERALL VERDICT: {'PASS' if passed else 'FAIL'}"
             f"{'' if passed else ' - final for this artifact; no re-runs'}")
    L.append("")
    L.append("HONESTY NOTES")
    L.append("- Single pre-declared configuration; no variant, parameter, or cost floor was")
    L.append("  run before this verdict; the fixed 0.41 spread is the pre-declared cost model")
    L.append("  (applied inside the fills) and no spread sweep is part of this screen.")
    L.append("- Known optimism, declared in SPEC 8: entry fills at the breaking bar's extreme")
    L.append("  (source-faithful E-XTM) and same-bar TP fills are best-case on a bar-close")
    L.append("  reconstruction; the section-4 character diagnostics exist to expose any hollow")
    L.append("  pass (same-bar round trips, degenerate micro-risk entries).")
    L.append("- Full-tape in-sample single screen: under validation_discipline a PASS here is")
    L.append("  NOT a validated result (no walk-forward, no year-sign agreement requirement)")
    L.append("  and does not by itself authorize Phase 2 (MT5 execution); a Phase 2 would need")
    L.append("  a new pre-declared SPEC meeting the standard discipline.")
    L.append("- Declared deviations from the source bot (SPEC 3/4): detection omits the")
    L.append("  forming-bar pullback/spike clauses (pullback is exclusively the entry trigger);")
    L.append("  pending expires at its NY date end (source leaves it indefinitely alive /")
    L.append("  replaced); indicators are continuous full-tape recursions (source: 10k-bar")
    L.append("  poll windows); single-pending occupancy with shadowing.")
    report = "\n".join(L) + "\n"
    with open(BASE / "REPORT_SP2L_M1_PHASE1.txt", "w") as fh:
        fh.write(report)
    print(report)
    print(f"artifacts written to {BASE} "
          f"(signals {n} rows) [{timemod.time() - t0:.1f}s total]")


if __name__ == "__main__":
    main()
