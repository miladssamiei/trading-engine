# -*- coding: utf-8 -*-
"""
sp2l_phase1.py — Phase 1 signal-quality + filter/session ablation for the
SP2L spike/gap/continuation hypothesis on US500-tick M1, 2021-01..2026-04.

Step 0 spec: data/results/sp2l_ablation/SPEC.md (unit conversions, pending
machine semantics, session variants). Gates PRE-DECLARED in
data/results/sp2l_ablation/GATE.md BEFORE this script ran.

48 configurations = {EMA, TREND, ADX, SECOND} on/off x {LON, NY, ALL24}
sessions. Stage A = light-touch subset (ALL24, <=1 filter on) reported
first. PGAP_PCT sensitivity {0.02, 0.08}% informational only (no-filter
ALL24); gates evaluated only at 0.04%.

Both entry conventions per SPEC §4: E-XTM (entry = breaking bar's extreme,
source-faithful) and E-CLS (entry = breaking bar's close, pessimistic).
G1 must hold under BOTH; G2 counts first-entry signals only.

Outputs:
  data/results/sp2l_ablation/phase1_ablation.csv   (one row per config)
  data/results/sp2l_ablation/signals/<cfg>.csv     (per-signal MFE/MAE)
  data/results/sp2l_ablation/phase1_stats.json     (setup/pending tallies)
"""

import csv
import datetime as dt
import json
import time as timemod
from array import array
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\sp2l_ablation")
SIGDIR = BASE / "signals"
M1_CSV = Path(r"C:\Users\Milad\Desktop\RR\data\results\vwap_2021_2026\us500tick_m1.csv")

SPIKE_RATIO = 1.5
PGAP_PCT = 0.04
PGAP_SENS = (0.02, 0.08)
MAX_SL_PCT = 0.40
EMA_PERIOD = 60
ADX_PERIOD = 14
MIN_ADX = 20.0
MAX_OPP = 1

THRESHOLDS = (0.10, 0.15, 0.20)
INFO_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
GATE_MIN_SIGNALS = 150

# session: (zone, entry_wall_lo, entry_wall_hi, walk_last_wall, final_wall)
SESSIONS = {
    "LON": ("lon", 480, 960, 984, 985),     # London 08:00-16:00, walk <=16:24 +16:25 open
    "NY":  ("ny", 570, 930, 954, 955),      # NY 09:30-15:30, walk <=15:54 +15:55 open
    "24":  ("ny", -1, 1380, 10 ** 6, None), # any bar, last entry 23:00 NY, walk to date end
}


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
    lon = t.dt.tz_convert("Europe/London")
    ny_wall = (ny.dt.hour * 60 + ny.dt.minute).to_numpy(np.int32)
    lon_wall = (lon.dt.hour * 60 + lon.dt.minute).to_numpy(np.int32)
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

    ctx = {
        "N": N, "time": df["time"].to_numpy(np.int64),
        "o": o, "h": h, "l": l, "c": c, "ema": ema, "adx": adx,
        "ny_wall": ny_wall, "lon_wall": lon_wall, "ny_date": ny_date,
        "blk_starts": starts, "blk_ends": ends, "nblk": len(starts),
        # hot-loop array() copies
        "A_h": array("d", h), "A_l": array("d", l), "A_c": array("d", c),
        "A_ema": array("d", ema), "A_adx": array("d", adx),
        "A_nyw": array("i", ny_wall), "A_lonw": array("i", lon_wall),
        "A_blkof": array("i", blk_of),
        "A_blkend": array("i", ends.astype(np.int32)),
    }
    return ctx


def build_session(sname, ctx):
    """Per-date walk anchor + suffix excursion arrays for one session."""
    zone, wlo, whi, wlast, wfinal = SESSIONS[sname]
    wall = ctx["lon_wall"] if zone == "lon" else ctx["ny_wall"]
    h, l, o = ctx["h"], ctx["l"], ctx["o"]
    N = ctx["N"]
    nblk = ctx["nblk"]
    starts, ends = ctx["blk_starts"], ctx["blk_ends"]
    we = np.zeros(nblk, np.int64)
    fpv = np.full(nblk, np.nan)
    smax = np.full(N, np.nan)
    smin = np.full(N, np.nan)
    for b in range(nblk):
        s, e = int(starts[b]), int(ends[b])
        idx = np.nonzero(wall[s:e] <= wlast)[0]
        a = s + (int(idx[-1]) if idx.size else (e - 1 - s))
        we[b] = a
        if wfinal is not None:
            j = np.nonzero(wall[s:e] == wfinal)[0]
            if j.size:
                fpv[b] = o[s + int(j[0])]
        if a >= s:
            smax[s:a + 1] = np.maximum.accumulate(h[s:a + 1][::-1])[::-1]
            smin[s:a + 1] = np.minimum.accumulate(l[s:a + 1][::-1])[::-1]
    return {"we": array("i", we.astype(np.int32)),
            "fp": array("d", fpv), "smax": smax, "smin": smin}


def detect_setups(ctx, gap_pct):
    """SPEC §3 on closed bars. Returns sorted [(bar_idx, dir, sl)]."""
    o, h, l, c = ctx["o"], ctx["h"], ctx["l"], ctx["c"]
    N = ctx["N"]
    body = c - o
    idx = np.arange(3, N)
    t1, t2, t3 = idx - 1, idx - 2, idx - 3
    g = gap_pct / 100.0
    buy = (
        (l[idx] < l[t1])
        & (c[t1] > c[t2]) & (o[t1] > o[t2])
        & (c[t2] > c[t3]) & (o[t2] > o[t3])
        & (c[t1] > o[t1]) & (c[t2] > o[t2]) & (c[t3] > o[t3])
        & (l[t1] > h[t3] * (1.0 + g))
        & (body[t2] > SPIKE_RATIO * body[t1])
        & (body[t2] > SPIKE_RATIO * body[t3])
        & (body[t2] > SPIKE_RATIO * body[idx])
    )
    sell = (
        (h[idx] > h[t1])
        & (c[t1] < c[t2]) & (o[t1] < o[t2])
        & (c[t2] < c[t3]) & (o[t2] < o[t3])
        & (c[t1] < o[t1]) & (c[t2] < o[t2]) & (c[t3] < o[t3])
        & (h[t1] < l[t3] * (1.0 - g))
        & (-body[t2] > SPIKE_RATIO * -body[t1])
        & (-body[t2] > SPIKE_RATIO * -body[t3])
        & (-body[t2] > SPIKE_RATIO * -body[idx])
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


# ------------------------------------------------------------- machine ----

def run_config(sess, ema_on, trend_on, adx_on, sec_on, ctx, sctx, setups):
    """Event-driven pending machine, SPEC §4. Returns (signals, tallies)."""
    zone, wlo, whi, wlast, wfinal = SESSIONS[sess]
    h = ctx["A_h"]; l = ctx["A_l"]; c = ctx["A_c"]
    ema = ctx["A_ema"]; adx = ctx["A_adx"]
    wall = ctx["A_lonw"] if zone == "lon" else ctx["A_nyw"]
    blk_of = ctx["A_blkof"]; blk_end = ctx["A_blkend"]
    we = sctx["we"]; fp = sctx["fp"]
    ny_date = ctx["ny_date"]
    ns = len(setups)

    signals = []
    tallies = {"setups": 0, "entered": 0, "killed": 0, "expired": 0,
               "trend_dead": 0, "attempts": 0}
    si = 0
    cur = -1
    while True:
        while si < ns and setups[si][0] <= cur:
            si += 1
        if si >= ns:
            break
        tidx, d, sl = setups[si]
        si += 1
        tallies["setups"] += 1
        e0 = blk_end[blk_of[tidx]]
        u = tidx + 1
        opp = 0
        dead = False
        outcome = None
        while u < e0:
            w = wall[u]
            if w > whi:                       # past window: linger to date end
                outcome = ("expired", e0 - 1)
                break
            if d == 1:                        # trend-structure run (SPEC §4.4)
                if h[u] > h[u - 1]:
                    opp = 0
                else:
                    opp += 1
            else:
                if l[u] < l[u - 1]:
                    opp = 0
                else:
                    opp += 1
            if opp > MAX_OPP and not dead:
                dead = True
                tallies["trend_dead"] += 1
                if trend_on:                  # permanently blocked -> linger
                    outcome = ("expired", e0 - 1)
                    break
            if d == 1:                        # break + risk (SPEC §4.1-3)
                if l[u] >= l[u - 1]:
                    u += 1
                    continue
                risk = l[u] - sl
                if risk <= 0:
                    u += 1
                    continue
                if risk > MAX_SL_PCT / 100.0 * l[u]:
                    outcome = ("killed", u)
                    break
            else:
                if h[u] <= h[u - 1]:
                    u += 1
                    continue
                risk = sl - h[u]
                if risk <= 0:
                    u += 1
                    continue
                if risk > MAX_SL_PCT / 100.0 * h[u]:
                    outcome = ("killed", u)
                    break
            tallies["attempts"] += 1
            if ema_on:                        # SPEC §4.5
                if d == 1:
                    if c[u] <= ema[u]:
                        u += 1
                        continue
                else:
                    if c[u] >= ema[u]:
                        u += 1
                        continue
            if adx_on:                        # SPEC §4.6
                a = adx[u]
                if not (a == a) or a < MIN_ADX:
                    u += 1
                    continue
            if w < wlo:                       # SPEC §4.7
                u += 1
                continue
            outcome = ("entered", u)          # SPEC §4.8
            break
        if outcome is None:
            outcome = ("expired", e0 - 1)
        kind, obar = outcome
        tallies[kind] += 1
        cur = obar
        if kind != "entered":
            continue
        blk = blk_of[tidx]
        a = we[blk]
        fpv = fp[blk]
        nd = int(ny_date[tidx])
        if d == 1:
            e_xtm = l[u]
            second_lvl = e_xtm - risk / 2.0
        else:
            e_xtm = h[u]
            second_lvl = e_xtm + risk / 2.0
        signals.append({"kind": "first", "dir": d, "t": tidx, "u": u,
                        "e_xtm": e_xtm, "e_cls": c[u], "sl": sl,
                        "risk": risk, "a": a, "fp": fpv, "nd": nd})
        if sec_on:                            # SPEC §6
            v = u + 1
            if d == 1:
                while v <= a and l[v] > second_lvl:
                    v += 1
            else:
                while v <= a and h[v] < second_lvl:
                    v += 1
            if v <= a:
                signals.append({"kind": "second", "dir": d, "t": tidx,
                                "u": v, "e_xtm": second_lvl, "e_cls": c[v],
                                "sl": sl, "risk": risk / 2.0, "a": a,
                                "fp": fpv, "nd": nd})
    return signals, tallies


# --------------------------------------------------------- excursions ----

def add_excursions(signals, ctx, sctx):
    """MFE/MAE % under both conventions (SPEC §4). Mutates signals."""
    smax, smin = sctx["smax"], sctx["smin"]
    h, l = ctx["h"], ctx["l"]
    for s in signals:
        d = s["dir"]; u = s["u"]; a = s["a"]
        fpv = s["fp"]
        has_fp = fpv == fpv
        cu = float(ctx["c"][u])
        ex, e2 = s["e_xtm"], s["e_cls"]
        if s["kind"] == "first":
            fav0 = max(cu - ex, 0.0)          # = close-extreme for firsts
            adv0 = 0.0                        # extreme IS the bar's limit
        else:                                 # second: limit fill mid-bar
            if d == 1:
                fav0 = max(cu - ex, 0.0)
                adv0 = max(ex - float(l[u]), 0.0)
            else:
                fav0 = max(ex - cu, 0.0)
                adv0 = max(float(h[u]) - ex, 0.0)
        wmax = wmin = None
        if u + 1 <= a:
            wmax = float(smax[u + 1])
            wmin = float(smin[u + 1])
        if has_fp:
            wmax = fpv if wmax is None else max(wmax, fpv)
            wmin = fpv if wmin is None else min(wmin, fpv)
        if d == 1:
            mfe_x = (max(fav0, 0.0 if wmax is None else wmax - ex)) / ex * 100.0
            mae_x = (max(adv0, 0.0 if wmin is None else ex - wmin)) / ex * 100.0
            mfe_c = max(0.0 if wmax is None else (wmax - e2) / e2 * 100.0, 0.0)
            mae_c = max(0.0 if wmin is None else (e2 - wmin) / e2 * 100.0, 0.0)
        else:
            mfe_x = (max(fav0, 0.0 if wmin is None else ex - wmin)) / ex * 100.0
            mae_x = (max(adv0, 0.0 if wmax is None else wmax - ex)) / ex * 100.0
            mfe_c = max(0.0 if wmin is None else (e2 - wmin) / e2 * 100.0, 0.0)
            mae_c = max(0.0 if wmax is None else (wmax - e2) / e2 * 100.0, 0.0)
        s["mfe_xtm"] = mfe_x
        s["mae_xtm"] = mae_x
        s["mfe_cls"] = mfe_c
        s["mae_cls"] = mae_c
    return signals


# ------------------------------------------------------------ summary ----

def summarize(cfg, sess, flt, signals):
    firsts = [s for s in signals if s["kind"] == "first"]
    seconds = [s for s in signals if s["kind"] == "second"]
    n_up = sum(1 for s in firsts if s["dir"] == 1)
    days = len({s["nd"] for s in firsts})
    row = {"cfg": cfg, "session": sess,
           "ema": int(flt[0]), "trend": int(flt[1]),
           "adx": int(flt[2]), "second": int(flt[3]),
           "n_first": len(firsts), "n_second": len(seconds),
           "n_up": n_up, "n_down": len(firsts) - n_up, "days": days}
    per_year = {}
    for s in firsts:
        y = s["nd"] // 10000
        p = per_year.setdefault(y, [0, 0.0, 0.0])
        p[0] += 1
        p[1] += s["mfe_xtm"]
        p[2] += s["mae_xtm"]
    row["per_year"] = {str(y): [v[0], round(v[1] / v[0], 3), round(v[2] / v[0], 3)]
                       for y, v in sorted(per_year.items())}
    for conv in ("xtm", "cls"):
        mfs = [s["mfe_" + conv] for s in signals]
        mas = [s["mae_" + conv] for s in signals]
        row[f"mean_mfe_{conv}"] = round(sum(mfs) / len(mfs), 4) if mfs else None
        row[f"mean_mae_{conv}"] = round(sum(mas) / len(mas), 4) if mas else None
        ok = True
        for t in THRESHOLDS:
            nm = sum(1 for v in mfs if v >= t)
            na = sum(1 for v in mas if v >= t)
            row[f"{conv}_{int(t * 100):03d}_mfe"] = nm
            row[f"{conv}_{int(t * 100):03d}_mae"] = na
            ok = ok and nm > na
        row[f"g1_{conv}"] = ok
    row["g2"] = len(firsts) >= GATE_MIN_SIGNALS
    row["pass"] = row["g1_xtm"] and row["g1_cls"] and row["g2"]
    return row


def fts(ctx, t):
    return dt.datetime.fromtimestamp(int(t), tz=dt.timezone.utc) \
        .strftime("%Y.%m.%d %H:%M")


def write_signals(cfg, signals, ctx):
    with open(SIGDIR / f"{cfg}.csv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["kind", "dir", "setup_time", "entry_time", "ny_date",
                    "entry_xtm", "entry_cls", "sl", "risk",
                    "mfe_xtm_pct", "mae_xtm_pct", "mfe_cls_pct", "mae_cls_pct"])
        for s in signals:
            w.writerow([s["kind"], "BUY" if s["dir"] == 1 else "SELL",
                        fts(ctx, ctx["time"][s["t"]]),
                        fts(ctx, ctx["time"][s["u"]]), s["nd"],
                        f"{s['e_xtm']:.2f}", f"{s['e_cls']:.2f}",
                        f"{s['sl']:.2f}", f"{s['risk']:.2f}",
                        f"{s['mfe_xtm']:.4f}", f"{s['mae_xtm']:.4f}",
                        f"{s['mfe_cls']:.4f}", f"{s['mae_cls']:.4f}"])


def print_row(row):
    g = lambda k: row[k]
    print(f"  {row['cfg']:<10} n={row['n_first']:>5}+{row['n_second']:<4} "
          f"up/dn={row['n_up']}/{row['n_down']} d={row['days']:>4} "
          f"| xtm mfe/mae@10={g('xtm_010_mfe'):>5}/{g('xtm_010_mae'):<5} "
          f"@15={g('xtm_015_mfe'):>5}/{g('xtm_015_mae'):<5} "
          f"@20={g('xtm_020_mfe'):>5}/{g('xtm_020_mae'):<5} "
          f"| cls @10={g('cls_010_mfe'):>5}/{g('cls_010_mae'):<5} "
          f"@15={g('cls_015_mfe'):>5}/{g('cls_015_mae'):<5} "
          f"@20={g('cls_020_mfe'):>5}/{g('cls_020_mae'):<5} "
          f"| {'PASS' if row['pass'] else 'fail'}")


def main():
    SIGDIR.mkdir(parents=True, exist_ok=True)
    t0 = timemod.time()
    ctx = load_ctx()
    print(f"loaded {ctx['N']} M1 bars, {ctx['nblk']} NY dates "
          f"({timemod.time() - t0:.1f}s)")

    sessions = {s: build_session(s, ctx) for s in SESSIONS}
    print(f"session anchors built ({timemod.time() - t0:.1f}s)")

    setups_by_gap = {}
    setup_stats = {}
    for gap in (PGAP_PCT,) + PGAP_SENS:
        st, nb, ns_, nboth = detect_setups(ctx, gap)
        setups_by_gap[gap] = st
        setup_stats[f"gap_{gap}"] = {"buy": nb, "sell": ns_, "both_dropped": nboth}
        print(f"PGAP {gap}%: {len(st)} setups (buy {nb} / sell {ns_}, "
              f"both-fired dropped {nboth})")

    rows = []
    all_tallies = {}
    cfg_list = []
    for sess in ("24", "NY", "LON"):
        for ema_on in (False, True):
            for trend_on in (False, True):
                for adx_on in (False, True):
                    for sec_on in (False, True):
                        flt = (ema_on, trend_on, adx_on, sec_on)
                        name = sess + "_" + "".join(
                            p for cond, p in zip(flt, ("E", "T", "A", "S")) if cond)
                        if not any(flt):
                            name = sess + "_base"
                        cfg_list.append((name, sess, flt))

    for name, sess, flt in cfg_list:
        t1 = timemod.time()
        sigs, tallies = run_config(sess, flt[0], flt[1], flt[2], flt[3],
                                   ctx, sessions[sess], setups_by_gap[PGAP_PCT])
        sigs = add_excursions(sigs, ctx, sessions[sess])
        row = summarize(name, sess, flt, sigs)
        rows.append(row)
        all_tallies[name] = tallies
        write_signals(name, sigs, ctx)
        print(f"[{timemod.time() - t0:7.1f}s] {name:<10} "
              f"({timemod.time() - t1:5.1f}s) setups={tallies['setups']:>6} "
              f"entered={tallies['entered']:>6} killed={tallies['killed']:>5} "
              f"expired={tallies['expired']:>6} signals={row['n_first']}")

    # PGAP sensitivity (informational): no-filter ALL24 at 0.02 / 0.08
    sens_rows = []
    for gap in PGAP_SENS:
        sigs, tallies = run_config("24", False, False, False, False,
                                   ctx, sessions["24"], setups_by_gap[gap])
        sigs = add_excursions(sigs, ctx, sessions["24"])
        r = summarize(f"24_base_gap{int(gap * 100)}", "24",
                      (False, False, False, False), sigs)
        r["gap_info_only"] = gap
        sens_rows.append(r)
        all_tallies[r["cfg"]] = tallies
        write_signals(r["cfg"], sigs, ctx)

    # ---------------- console: Stage A, then full grid ----------------
    print()
    print("=" * 110)
    print("Stage A — light-touch single-filter ablation (ALL24, gates at PGAP 0.04%)")
    print("=" * 110)
    stage_a = ["24_base", "24_E", "24_T", "24_A", "24_S"]
    for name in stage_a:
        print_row(next(r for r in rows if r["cfg"] == name))

    print()
    print("=" * 110)
    print("Stage B — full 48-configuration factorial (gates: G1 both conventions + G2)")
    print("=" * 110)
    for r in rows:
        print_row(r)

    print()
    print("PGAP sensitivity (INFORMATIONAL ONLY — cannot admit to Step 3):")
    for r in sens_rows:
        print_row(r)

    passers = [r["cfg"] for r in rows if r["pass"]]
    print()
    print(f"GATE-PASSING configurations (G1 xtm AND G1 cls AND G2): "
          f"{len(passers)}")
    for p in passers:
        print(f"  {p}")

    # ---------------- artifacts ----------------
    cols = ["cfg", "session", "ema", "trend", "adx", "second", "n_first",
            "n_second", "n_up", "n_down", "days", "mean_mfe_xtm",
            "mean_mae_xtm", "mean_mfe_cls", "mean_mae_cls"]
    for conv in ("xtm", "cls"):
        for t in THRESHOLDS:
            cols += [f"{conv}_{int(t * 100):03d}_mfe", f"{conv}_{int(t * 100):03d}_mae"]
    cols += [f"g1_{conv}" for conv in ("xtm", "cls")] + ["g2", "pass"]
    with open(BASE / "phase1_ablation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows + sens_rows:
            w.writerow(r)

    with open(BASE / "phase1_stats.json", "w", encoding="utf-8") as fh:
        json.dump({"setup_counts": setup_stats,
                   "pending_tallies": all_tallies,
                   "per_year_first_signals": {r["cfg"]: r["per_year"]
                                              for r in rows},
                   "passers": passers,
                   "params": {"SPIKE_RATIO": SPIKE_RATIO, "PGAP_PCT": PGAP_PCT,
                              "MAX_SL_PCT": MAX_SL_PCT, "EMA_PERIOD": EMA_PERIOD,
                              "ADX_PERIOD": ADX_PERIOD, "MIN_ADX": MIN_ADX,
                              "MAX_OPP": MAX_OPP,
                              "THRESHOLDS": THRESHOLDS,
                              "GATE_MIN_SIGNALS": GATE_MIN_SIGNALS}},
                  fh, indent=2)
    print(f"\nwrote phase1_ablation.csv, phase1_stats.json, signals/*.csv "
          f"({len(rows) + len(sens_rows)} configs)")


if __name__ == "__main__":
    main()
