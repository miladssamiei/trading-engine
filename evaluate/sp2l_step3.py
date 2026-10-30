# -*- coding: utf-8 -*-
"""
sp2l_step3.py — Step 3 for SP2L: TP_R grid (Phase A first-crossing
reconstruction, exploratory) + Phase B occupancy re-simulation with
walk-forward selection, for every Phase-1 gate-passing configuration
(passers read from data/results/sp2l_ablation/phase1_stats.json).

Protocol pre-declared in data/results/sp2l_ablation/GATE.md:
  TP_R grid {0.5, 0.75, 1.0, 1.5, 2.0}; SL = pre-spike extreme (rule-fixed).
  Occupancy: one setup chain at a time — while any leg of a position is
  open, no new setups are detected (source: Strategy() returns early while
  status). Signals while holding are skipped, not queued. Second-entry leg:
  limit at entry -/+ risk/2, fills on first touch while leg 1 is open,
  canceled at leg-1 close; 2.0 lots vs leg-1 1.0 lot.
  Exit engine (prior-project conventions): time-exit at the forced-exit
  bar's open (NY 15:55 / LON 16:25 / 24h: date's last bar close); then
  bar-open gap fills; same-bar both-level ambiguity resolved SL-FIRST
  (conservative, per SPEC §8 — stricter than the close-vs-open heuristic).
  Entry/SL/TP normalized to the 0.01 trade grid (project discipline).
  Same-bar rules: E-XTM leg-1 fills at the bar extreme (SL impossible that
  bar; TP credited only if the bar CLOSE is beyond TP — conservative);
  E-CLS leg-1 fills at bar close (no same-bar checks); leg-2 fills at the
  limit (SL taken first if the bar's range also reaches it).
  Walk-forward: burn-in 2021; folds 2022H1..2026H1 (9 half-years); per
  fold select TP_R by trailing expectancy (>=20 trailing trades, else the
  a-priori default TP_R=1.0).
  Gates on the WF result: G1 expt>0 & Sharpe>0; G2 >=150 trades (legs
  count as trades); G3 >=4/5 years positive AND best year <60% of profit.
  STRICHER THAN PRE-DECLARED (documented addition, not a relaxation): a
  configuration is CONFIRMED only if the WF gates pass under BOTH entry
  conventions (E-XTM and E-CLS), since E-XTM is known-optimistic.

Outputs: data/results/sp2l_ablation/step3/
  phaseA_grid.csv, insample_occ.csv, wf_all.csv, wf_folds.csv,
  wf_trades/<cfg>__<conv>.csv
"""

import csv
import datetime as dt
import json
import math
import sys
import time as timemod
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp2l_phase1 as p1

BASE = p1.BASE
OUT = BASE / "step3"
TP_GRID = (0.5, 0.75, 1.0, 1.5, 2.0)
DEFAULT_TP = 1.0
MIN_TRAILING = 20
# "xtm"/"cls" are the two pre-declared Phase-1/Step-3 conventions (GATE.md).
# "next" = E-NEXT (signal on completed bar, market fill at next bar's open —
# the project's EA convention). Added AFTER the first Step-3 run as a
# TIGHTENING-ONLY Step-4 buildability gate: it can disqualify a config from
# EA construction, never admit one. Confirmation per GATE.md remains
# xtm AND cls; step4_ready = confirmed AND next-pass.
CONVS = ("xtm", "cls", "next")

FOLD_BOUNDS = [
    (2022, 1), (2022, 7), (2023, 1), (2023, 7), (2024, 1), (2024, 7),
    (2025, 1), (2025, 7), (2026, 1), (2026, 5),
]


def norm(p):
    return round(round(p / 0.01) * 0.01, 2)


def fold_of(nd):
    y, m = nd // 10000, (nd // 100) % 100
    for k in range(len(FOLD_BOUNDS) - 1):
        (fy, fm), (ty, tm) = FOLD_BOUNDS[k], FOLD_BOUNDS[k + 1]
        if (y, m) >= (fy, fm) and (y, m) < (ty, tm):
            return k
    return None


def parse_cfg(name):
    sess, _, suffix = name.partition("_")
    return sess, ("E" in suffix, "T" in suffix, "A" in suffix, "S" in suffix)


# ---------------------------------------------------- exit engine ----

def simulate_leg(d, entry, sl, tp, fill_bar, same_bar, ctx, sctx, sess,
                 a, fp_bar_wall, block_end):
    """Exit simulation for one leg. same_bar: None (cls) or 'xtm' (fill at
    extreme) or 'limit' (leg-2 fill at limit level). Returns trade dict or
    None (leg-2 canceled pre-fill handled by caller)."""
    o, h, l, c = ctx["A_h_o"], ctx["A_h"], ctx["A_l"], ctx["A_c"]
    wall = ctx["A_lonw"] if SESSION_ZONE[sess] == "lon" else ctx["A_nyw"]
    times = ctx["time"]

    def mk(exit_px, bar, reason, lots):
        sign = d
        return {"nd": int(ctx["ny_date"][fill_bar]), "dir": d,
                "open_ts": int(times[fill_bar]), "close_ts": int(times[bar]),
                "entry": entry, "exit": exit_px, "reason": reason,
                "pnl": round(sign * (exit_px - entry) * lots, 2), "lots": lots}

    lots = 1  # caller overrides for leg 2 via lots param below

    # ---- same-bar (fill bar) checks
    if same_bar == "xtm":
        # SL impossible (entry is the bar extreme); TP only if close beyond
        if d == 1 and c[fill_bar] >= tp:
            return mk(tp, fill_bar, "target", 1)
        if d == -1 and c[fill_bar] <= tp:
            return mk(tp, fill_bar, "target", 1)
    elif same_bar == "limit":
        if d == 1 and l[fill_bar] <= sl:
            return mk(sl, fill_bar, "stop_loss", 1)
        if d == -1 and h[fill_bar] >= sl:
            return mk(sl, fill_bar, "stop_loss", 1)
        if d == 1 and c[fill_bar] >= tp:
            return mk(tp, fill_bar, "target", 1)
        if d == -1 and c[fill_bar] <= tp:
            return mk(tp, fill_bar, "target", 1)
    elif same_bar == "open":
        # fill at the bar's OPEN: whole bar range is post-entry (E-NEXT,
        # the project EA convention). Gap fills, then SL-first ambiguity.
        if d == 1:
            if o[fill_bar] <= sl:
                return mk(o[fill_bar], fill_bar, "stop_loss", 1)
            if o[fill_bar] >= tp:
                return mk(o[fill_bar], fill_bar, "target", 1)
            if l[fill_bar] <= sl and h[fill_bar] >= tp:
                return mk(sl, fill_bar, "stop_loss", 1)
            if l[fill_bar] <= sl:
                return mk(sl, fill_bar, "stop_loss", 1)
            if h[fill_bar] >= tp:
                return mk(tp, fill_bar, "target", 1)
        else:
            if o[fill_bar] >= sl:
                return mk(o[fill_bar], fill_bar, "stop_loss", 1)
            if o[fill_bar] <= tp:
                return mk(o[fill_bar], fill_bar, "target", 1)
            if h[fill_bar] >= sl and l[fill_bar] <= tp:
                return mk(sl, fill_bar, "stop_loss", 1)
            if h[fill_bar] >= sl:
                return mk(sl, fill_bar, "stop_loss", 1)
            if l[fill_bar] <= tp:
                return mk(tp, fill_bar, "target", 1)

    # ---- subsequent bars
    b = fill_bar + 1
    while b < block_end:
        w = wall[b]
        if fp_bar_wall is not None and w >= fp_bar_wall:
            return mk(o[b], b, "time_exit", 1)
        if d == 1:
            if o[b] <= sl:
                return mk(o[b], b, "stop_loss", 1)
            if o[b] >= tp:
                return mk(o[b], b, "target", 1)
            if l[b] <= sl and h[b] >= tp:
                return mk(sl, b, "stop_loss", 1)     # SL-first (SPEC §8)
            if l[b] <= sl:
                return mk(sl, b, "stop_loss", 1)
            if h[b] >= tp:
                return mk(tp, b, "target", 1)
        else:
            if o[b] >= sl:
                return mk(o[b], b, "stop_loss", 1)
            if o[b] <= tp:
                return mk(o[b], b, "target", 1)
            if h[b] >= sl and l[b] <= tp:
                return mk(sl, b, "stop_loss", 1)
            if h[b] >= sl:
                return mk(sl, b, "stop_loss", 1)
            if l[b] <= tp:
                return mk(tp, b, "target", 1)
        b += 1
    # still open at the date's last walkable bar -> close there
    last = block_end - 1
    return mk(c[last], last, "time_exit", 1)


SESSION_ZONE = {s: p1.SESSIONS[s][0] for s in p1.SESSIONS}
SESSION_FP_WALL = {s: p1.SESSIONS[s][4] for s in p1.SESSIONS}


# -------------------------------------------- occupancy simulation ----

def simulate_config(name, sess, flt, tp_r, conv, ctx, sessions, setups):
    ema_on, trend_on, adx_on, sec_on = flt
    _, wlo, whi, _, _ = p1.SESSIONS[sess]
    h = ctx["A_h"]; l = ctx["A_l"]; c = ctx["A_c"]
    ema = ctx["A_ema"]; adx = ctx["A_adx"]
    wall = ctx["A_lonw"] if SESSION_ZONE[sess] == "lon" else ctx["A_nyw"]
    blk_of = ctx["A_blkof"]; blk_end = ctx["A_blkend"]
    we = sessions[sess]["we"]
    ny_date = ctx["ny_date"]
    ns = len(setups)
    fp_wall = SESSION_FP_WALL[sess]

    trades = []
    si = 0
    cur = -1
    while True:
        while si < ns and setups[si][0] <= cur:
            si += 1
        if si >= ns:
            break
        tidx, d, sl = setups[si]
        si += 1
        e0 = blk_end[blk_of[tidx]]
        u = tidx + 1
        opp = 0
        dead = False
        outcome = None
        while u < e0:
            w = wall[u]
            if w > whi:
                outcome = ("expired", e0 - 1)
                break
            if d == 1:
                if h[u] > h[u - 1]:
                    opp = 0
                else:
                    opp += 1
            else:
                if l[u] < l[u - 1]:
                    opp = 0
                else:
                    opp += 1
            if opp > p1.MAX_OPP and not dead:
                dead = True
                if trend_on:
                    outcome = ("expired", e0 - 1)
                    break
            if d == 1:
                if l[u] >= l[u - 1]:
                    u += 1
                    continue
                risk = l[u] - sl
                if risk <= 0:
                    u += 1
                    continue
                if risk > p1.MAX_SL_PCT / 100.0 * l[u]:
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
                if risk > p1.MAX_SL_PCT / 100.0 * h[u]:
                    outcome = ("killed", u)
                    break
            if ema_on:
                if d == 1:
                    if c[u] <= ema[u]:
                        u += 1
                        continue
                else:
                    if c[u] >= ema[u]:
                        u += 1
                        continue
            if adx_on:
                a = adx[u]
                if not (a == a) or a < p1.MIN_ADX:
                    u += 1
                    continue
            if w < wlo:
                u += 1
                continue
            outcome = ("entered", u)
            break
        if outcome is None:
            outcome = ("expired", e0 - 1)
        kind, obar = outcome
        cur = obar
        if kind != "entered":
            continue

        sl_n = norm(sl)
        fill_bar = u
        if conv == "xtm":
            E = norm(l[u] if d == 1 else h[u])
            same = "xtm"
        elif conv == "cls":
            E = norm(c[u])
            same = None
        else:                                # "next": fill at open(u+1)
            if u + 1 >= e0:                  # next bar is a new NY date
                cur = e0 - 1
                continue
            fill_bar = u + 1
            E = norm(ctx["A_h_o"][fill_bar])
            same = "open"
        risk = (E - sl_n) if d == 1 else (sl_n - E)
        if risk <= 0:                      # grid-rounding / gap guard
            continue
        tp = norm(E + d * tp_r * risk)

        blk = blk_of[tidx]
        a = we[blk]
        t1 = simulate_leg(d, E, sl_n, tp, fill_bar, same, ctx, sessions[sess],
                          sess, a, fp_wall, e0)
        t1["leg"] = 1
        trades.append(t1)
        exit_bar1 = index_of_ts(ctx, t1["close_ts"])
        exit_bars = [exit_bar1]

        if sec_on:
            S = norm(E - d * risk / 2.0)   # BUY: E - r/2; SELL: E + r/2
            risk2 = (S - sl_n) if d == 1 else (sl_n - S)
            if risk2 > 0:
                # first touch of S in (u, min(a, exit_bar1)]
                hi = min(a, exit_bar1)
                v = u + 1
                if d == 1:
                    while v <= hi and l[v] > S:
                        v += 1
                else:
                    while v <= hi and h[v] < S:
                        v += 1
                if v <= hi:
                    tp2 = norm(S + d * tp_r * risk2)
                    t2 = simulate_leg(d, S, sl_n, tp2, v, "limit", ctx,
                                      sessions[sess], sess, a, fp_wall, e0)
                    t2["leg"] = 2
                    t2["pnl"] = round(d * (t2["exit"] - S) * 2.0, 2)
                    t2["lots"] = 2
                    trades.append(t2)
                    exit_bars.append(index_of_ts(ctx, t2["close_ts"]))
        cur = max(exit_bars)
    return trades


_TS_INDEX = {}


def index_of_ts(ctx, ts):
    key = ts
    if key not in _TS_INDEX:
        import bisect
        times = ctx["_times_list"]
        i = bisect.bisect_left(times, ts)
        _TS_INDEX[key] = i if i < len(times) and times[i] == ts else i - 1
    return _TS_INDEX[key]


# ------------------------------------------------------ Phase A ----

def phase_a(signals, tp_r, conv, ctx, sess, sessions):
    """First-crossing reconstruction over the Phase-1 signal set."""
    o, h, l, c = ctx["A_h_o"], ctx["A_h"], ctx["A_l"], ctx["A_c"]
    wall = ctx["A_lonw"] if SESSION_ZONE[sess] == "lon" else ctx["A_nyw"]
    we = sessions[sess]["we"]
    fp_wall = SESSION_FP_WALL[sess]
    res = []
    for s in signals:
        d = s["dir"]; u = s["u"]
        sl_n = norm(s["sl"])
        blk = ctx["A_blkof"][s["t"]]
        a = we[blk]
        e0 = ctx["A_blkend"][blk]
        lots = 1 if s["kind"] == "first" else 2
        fill_bar = u
        if conv == "xtm":
            E = norm(s["e_xtm"] if s["kind"] == "first" else s["e_xtm"])
            same = "xtm" if s["kind"] == "first" else "limit"
        elif conv == "cls":
            E = norm(s["e_cls"])
            same = "limit" if s["kind"] == "second" else None
        else:                                # "next"
            if u + 1 >= e0 or u + 1 > a:
                continue
            E1 = norm(ctx["A_h_o"][u + 1])
            r1 = (E1 - sl_n) if d == 1 else (sl_n - E1)
            if r1 <= 0:
                continue
            if s["kind"] == "first":
                E, fill_bar, same = E1, u + 1, "open"
            else:                            # second: limit off E_NEXT
                E = norm(E1 - d * r1 / 2.0)
                r2 = (E - sl_n) if d == 1 else (sl_n - E)
                if r2 <= 0:
                    continue
                # first touch from the leg-1 fill bar onward
                v = u + 1
                if d == 1:
                    while v <= a and l[v] > E:
                        v += 1
                else:
                    while v <= a and h[v] < E:
                        v += 1
                if v > a:
                    continue
                fill_bar, same = v, "limit"
        risk = (E - sl_n) if d == 1 else (sl_n - E)
        if risk <= 0:
            continue
        tp = norm(E + d * tp_r * risk)
        t = simulate_leg(d, E, sl_n, tp, fill_bar, same, ctx, sessions[sess],
                         sess, a, fp_wall, e0)
        t["pnl"] = round(d * (t["exit"] - E) * lots, 2)
        t["leg"] = lots
        res.append(t)
    return res


# ------------------------------------------------------ metrics ----

def metrics(trades, all_days, day_from=None, day_to=None):
    days = [d for d in all_days
            if (day_from is None or d >= day_from)
            and (day_to is None or d < day_to)]
    per_day = defaultdict(float)
    for t in trades:
        per_day[t["nd"]] += t["pnl"]
    vals = [per_day.get(d, 0.0) for d in days]
    mean = sum(vals) / len(vals) if vals else 0.0
    var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1) \
        if len(vals) > 1 else 0.0
    std = math.sqrt(var)
    sharpe = mean / std * math.sqrt(252) if std > 0 else None
    cum = peak = maxdd = 0.0
    for x in vals:
        cum += x
        peak = max(peak, cum)
        maxdd = max(maxdd, peak - cum)
    total = sum(t["pnl"] for t in trades)
    return {"trades": len(trades), "total": round(total, 2),
            "expt": round(total / len(trades), 3) if trades else None,
            "sharpe": None if sharpe is None else round(sharpe, 3),
            "maxdd": round(maxdd, 2)}


def fts(t):
    return dt.datetime.fromtimestamp(int(t), tz=dt.timezone.utc) \
        .strftime("%Y.%m.%d %H:%M")


# ---------------------------------------------------------- main ----

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "wf_trades").mkdir(exist_ok=True)
    t0 = timemod.time()
    ctx = p1.load_ctx()
    ctx["A_h_o"] = __import__("array").array("d", ctx["o"])
    ctx["_times_list"] = ctx["time"].tolist()
    sessions = {s: p1.build_session(s, ctx) for s in p1.SESSIONS}
    setups, nb, nsell, nboth = p1.detect_setups(ctx, p1.PGAP_PCT)
    print(f"ctx loaded, {len(setups)} setups ({timemod.time() - t0:.0f}s)")

    passers = json.load(open(BASE / "phase1_stats.json"))["passers"]
    all_days = sorted(int(x) for x in np_unique(ctx["ny_date"]))
    wf_start = FOLD_BOUNDS[0][0] * 10000 + FOLD_BOUNDS[0][1] * 100
    wf_end = FOLD_BOUNDS[-1][0] * 10000 + FOLD_BOUNDS[-1][1] * 100

    grid_rows = []
    insample_rows = []
    wf_rows = []
    fold_rows = []
    summary = []
    confirmed = []

    for name in passers:
        sess, flt = parse_cfg(name)
        sigs, _ = p1.run_config(sess, flt[0], flt[1], flt[2], flt[3],
                                ctx, sessions[sess], setups)
        for conv in CONVS:
            runs = {}
            for tp_r in TP_GRID:
                pa = phase_a(sigs, tp_r, conv, ctx, sess, sessions)
                n = len(pa)
                tot = sum(t["pnl"] for t in pa)
                wins = sum(1 for t in pa if t["pnl"] > 0)
                grid_rows.append({"cfg": name, "conv": conv, "tp_r": tp_r,
                                  "n": n,
                                  "winrate": round(100.0 * wins / n, 2) if n else None,
                                  "expt": round(tot / n, 3) if n else None,
                                  "total": round(tot, 2)})
                tr = simulate_config(name, sess, flt, tp_r, conv, ctx,
                                     sessions, setups)
                runs[tp_r] = tr
                m = metrics(tr, all_days)
                insample_rows.append({"cfg": name, "conv": conv,
                                      "tp_r": tp_r, **m})
            # walk-forward over the 5 candidates
            wf_trades = []
            for k in range(len(FOLD_BOUNDS) - 1):
                (fy, fm), (ty, tm) = FOLD_BOUNDS[k], FOLD_BOUNDS[k + 1]
                f_start = fy * 10000 + fm * 100
                f_end = ty * 10000 + tm * 100
                best, best_exp = None, None
                for tp_r in TP_GRID:
                    trail = [t for t in runs[tp_r] if t["nd"] < f_start]
                    if len(trail) < MIN_TRAILING:
                        continue
                    exp = sum(t["pnl"] for t in trail) / len(trail)
                    if best_exp is None or exp > best_exp:
                        best, best_exp = tp_r, exp
                sel = best if best is not None else DEFAULT_TP
                fold_tr = [dict(t, tp_r=sel) for t in runs[sel]
                           if f_start <= t["nd"] < f_end]
                wf_trades.extend(fold_tr)
                fpnl = sum(t["pnl"] for t in fold_tr)
                fold_rows.append({"cfg": name, "conv": conv,
                                  "fold": f"{fy}-{fm:02d}..{ty}-{tm:02d}",
                                  "selected": sel, "trades": len(fold_tr),
                                  "pnl": round(fpnl, 2)})
            m = metrics(wf_trades, all_days, wf_start, wf_end)
            per_year = defaultdict(lambda: [0, 0.0])
            for t in wf_trades:
                per_year[t["nd"] // 10000][0] += 1
                per_year[t["nd"] // 10000][1] += t["pnl"]
            py = {y: [v[0], round(v[1], 2)] for y, v in sorted(per_year.items())}
            years_pos = sum(1 for y in py if py[y][1] > 0)
            best_share = (max(v[1] for v in py.values()) / m["total"] * 100.0) \
                if m["total"] > 0 and py else None
            g1 = m["expt"] is not None and m["expt"] > 0 \
                and m["sharpe"] is not None and m["sharpe"] > 0
            g2 = m["trades"] >= 150
            g3 = years_pos >= 4 and best_share is not None and best_share < 60.0
            verdict = g1 and g2 and g3
            summary.append({"cfg": name, "conv": conv, **m,
                            "per_year": py, "years_pos": years_pos,
                            "best_share": None if best_share is None
                            else round(best_share, 1),
                            "G1": g1, "G2": g2, "G3": g3,
                            "verdict": verdict})
            with open(OUT / "wf_trades" / f"{name}__{conv}.csv", "w",
                      newline="") as fh:
                w = csv.writer(fh, delimiter=";")
                w.writerow(["ticket", "open_time", "close_time", "dir",
                            "leg", "entry", "exit", "reason", "pnl", "tp_r"])
                for i, t in enumerate(wf_trades, 1):
                    w.writerow([i, fts(t["open_ts"]), fts(t["close_ts"]),
                                "BUY" if t["dir"] == 1 else "SELL", t["leg"],
                                f"{t['entry']:.2f}", f"{t['exit']:.2f}",
                                t["reason"], f"{t['pnl']:.2f}", t.get("tp_r")])
            print(f"[{timemod.time() - t0:6.0f}s] {name:<10} {conv}: WF "
                  f"n={m['trades']:>4} total={m['total']:>9.2f} "
                  f"expt={m['expt']} sharpe={m['sharpe']} "
                  f"y+={years_pos} best%={summary[-1]['best_share']} "
                  f"G={'PASS' if verdict else 'fail'}")

    # cross-convention confirmation
    by_cfg = defaultdict(dict)
    for s in summary:
        by_cfg[s["cfg"]][s["conv"]] = s["verdict"]
    for name in passers:
        if by_cfg[name].get("xtm") and by_cfg[name].get("cls"):
            confirmed.append(name)
    step4_ready = [n for n in confirmed if by_cfg[n].get("next")]

    print()
    print("=" * 100)
    print(f"WALK-FORWARD VERDICTS — {len(passers)} Phase-1 passers "
          f"(CONFIRMED needs gates under BOTH conventions)")
    print("=" * 100)
    for s in summary:
        print(f"  {s['cfg']:<10} {s['conv']}: n={s['trades']:>4} "
              f"total={s['total']:>10.2f} expt={s['expt']:>7} "
              f"sharpe={s['sharpe']:>7} maxdd={s['maxdd']:>8.2f} "
              f"y+={s['years_pos']} best%={s['best_share']} "
              f"G1/2/3={int(s['G1'])}{int(s['G2'])}{int(s['G3'])} "
              f"-> {'PASS' if s['verdict'] else 'FAIL'}")
    print()
    print(f"CONFIRMED (both pre-declared conventions): "
          f"{len(confirmed)} {confirmed if confirmed else '— NONE'}")
    print(f"STEP-4 READY (confirmed AND E-NEXT EA-convention gates pass): "
          f"{len(step4_ready)} {step4_ready if step4_ready else '— NONE'}")

    # artifacts
    def dump(path, rows, fields):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    dump(OUT / "phaseA_grid.csv", grid_rows,
         ["cfg", "conv", "tp_r", "n", "winrate", "expt", "total"])
    dump(OUT / "insample_occ.csv", insample_rows,
         ["cfg", "conv", "tp_r", "trades", "total", "expt", "sharpe", "maxdd"])
    dump(OUT / "wf_folds.csv", fold_rows,
         ["cfg", "conv", "fold", "selected", "trades", "pnl"])
    wf_all = []
    for s in summary:
        row = dict(s)
        for y in range(2022, 2027):
            row[f"y{y}"] = s["per_year"].get(y, [0, 0.0])[1]
        wf_all.append(row)
    dump(OUT / "wf_all.csv", wf_all,
         ["cfg", "conv", "trades", "total", "expt", "sharpe", "maxdd",
          "y2022", "y2023", "y2024", "y2025", "y2026", "years_pos",
          "best_share", "G1", "G2", "G3", "verdict"])
    with open(OUT / "step3_summary.json", "w", encoding="utf-8") as fh:
        json.dump({"passers": passers, "confirmed": confirmed,
                   "step4_ready": step4_ready,
                   "wf": [{k: v for k, v in s.items()} for s in summary]},
                  fh, indent=2, default=str)
    print(f"\nwrote step3/ artifacts ({timemod.time() - t0:.0f}s)")


def np_unique(arr):
    import numpy as np
    return np.unique(arr)


if __name__ == "__main__":
    main()
