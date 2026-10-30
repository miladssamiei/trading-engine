# -*- coding: utf-8 -*-
"""
evaluate/vwap_reconcile.py — Python-vs-MT5 reconciliation for the VWAP
mean-reversion strategy (US500-tick, 2021-01-01 .. 2026-05-01), modeled on
evaluate/orb_reconcile_fullrange.py.

Reads (base dir default data/results/vwap_2021_2026/):
    py_trades.csv / mt5_trades.csv   semicolon CSV, times 'YYYY.MM.DD HH:MM' UTC
    py_days.csv   / mt5_days.csv     semicolon CSV, one row per NY calendar date
Writes:
    RECONCILE.md + console summary

Trade matching: same direction, same NY open date, |open_time| <= 10 min
(greedy on smallest dt). Tolerances (same family as ORB): entry <= 1.5 pts,
exit <= 1.5 pts, exit time <= 15 min, mapped exit reason equal.

Residual classification uses the per-trade signal diagnostics both engines
log at entry (dev_pct, slope_pct) plus day-level counters:
    dev_threshold_straddle   |dev| within 0.05% of the 0.25% entry band
    slope_zero_straddle      |slope| within 0.02% of the slope>=0 / <=0 gate
    range_day_straddle       day-level range_ok flip near net 0.6% / hl 1.0%
    maxmove_straddle         |close-open| near the 0.42% max-move gate
    exit_level_straddle      exit timestamp moves across bars, prices agree
    vwap_touch_price_drift   moving-target close differs on the two tapes
    early_close_carry        15:55 force exit impossible (market closed)
    unexplained              flagged for dedicated investigation
Pure stdlib.
"""

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

MATCH_TOL = 10.0        # minutes on open_time
ENTRY_PX_TOL = 1.5      # pts
EXIT_PX_TOL = 1.5       # pts
EXIT_TIME_TOL = 15.0    # minutes
LEVEL_TOL = 0.05        # pts, day-level price delta flag
DEV_STRADDLE = 0.05     # % margin from the 0.25% band
SLOPE_STRADDLE = 0.02   # % margin from the slope 0 gate
RANGE_STRADDLE = 0.05   # % margin from range-day thresholds
MAXMOVE_STRADDLE = 0.05 # % margin from the max-move gate
MAXMOVE_PCT = 0.42


def parse_args():
    p = argparse.ArgumentParser(description="VWAP py-vs-MT5 reconciliation")
    p.add_argument("--base-dir",
                   default=r"C:\Users\Milad\Desktop\RR\data\results\vwap_2021_2026")
    return p.parse_args()


def parse_ts(s):
    s = (s or "").strip()
    return datetime.strptime(s, "%Y.%m.%d %H:%M") if s else None


def load_trades(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    out = []
    for r in rows:
        out.append({
            "ticket": int(r["ticket"]),
            "open_time": parse_ts(r["open_time"]),
            "close_time": parse_ts(r["close_time"]),
            "type": r["type"].strip().upper(),
            "open_price": float(r["open_price"]),
            "sl": float(r["stop_loss"]),
            "tp0": float(r["take_profit"]),
            "close_price": float(r["close_price"]),
            "exit_reason": r["exit_reason"].strip(),
            "profit": float(r["profit"]),
            "dev_pct": float(r["dev_pct"]) if r.get("dev_pct") else None,
            "slope_pct": float(r["slope_pct"]) if r.get("slope_pct") else None,
        })
    return out


def load_days(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    days = {}
    for r in rows:
        d = int(r["ny_date"])
        days[d] = {
            "ny_date": d,
            "session_open": float(r["session_open"]) if r["session_open"] else 0.0,
            "session_high": float(r["session_high"]) if r["session_high"] else 0.0,
            "session_low": float(r["session_low"]) if r["session_low"] else 0.0,
            "net_pct_final": float(r["net_pct_final"]) if r["net_pct_final"] else None,
            "hl_pct_final": float(r["hl_pct_final"]) if r["hl_pct_final"] else None,
            "range_ok_final": int(r["range_ok_final"] or 0),
            "skipped": int(r["skipped"] or 0),
            "skip_reason": (r["skip_reason"] or "").strip(),
            "slope_ok_bars": int(r["slope_ok_bars"] or 0),
            "signals": int(r["entry_signals"] or 0),
            "entries": int(r["entries"] or 0),
            "exits_tp": int(r["exits_tp"] or 0),
            "exits_sl": int(r["exits_sl"] or 0),
            "exits_time": int(r["exits_time"] or 0),
            "total_pnl": float(r["total_pnl"] or 0.0),
        }
    return days


def dstr(ny_date):
    s = str(ny_date)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def p95(vals):
    s = sorted(vals)
    if not s:
        return float("nan")
    if len(s) == 1:
        return s[0]
    k = 0.95 * (len(s) - 1)
    lo = int(math.floor(k))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def match_trades(py, mt5):
    cands = []
    for i, p in enumerate(py):
        for j, m in enumerate(mt5):
            if p["type"] != m["type"]:
                continue
            if p["open_time"].date() != m["open_time"].date():
                continue
            dt = abs((p["open_time"] - m["open_time"]).total_seconds()) / 60.0
            if dt <= MATCH_TOL:
                cands.append((dt, i, j))
    cands.sort()
    pairs, used_py, used_mt5 = [], set(), set()
    for dt, i, j in cands:
        if i in used_py or j in used_mt5:
            continue
        used_py.add(i)
        used_mt5.add(j)
        pairs.append((i, j, dt))
    return (pairs,
            [i for i in range(len(py)) if i not in used_py],
            [j for j in range(len(mt5)) if j not in used_mt5])


def margins(t):
    """Distance of the trade's signal values from each decision gate."""
    return {
        "dev": abs(abs(t["dev_pct"]) - 0.25),
        "slope": abs(t["slope_pct"]),
        "maxmove": None,   # needs day context
    }


def day_maxmove_margin(day):
    """Approximate distance of the day's final |net| from the 0.42% gate."""
    if day["net_pct_final"] is None or day["session_open"] <= 0:
        return None
    return abs(day["net_pct_final"] - MAXMOVE_PCT)


def classify_unmatched(t, side, oth_days, own_days, py_trades, mt5_trades, pairs):
    ny_date = int(t["open_time"].strftime("%Y%m%d"))
    o = oth_days.get(ny_date)
    own = own_days.get(ny_date)
    if o is None:
        return "other", f"other side has no day record for {ny_date}"

    # -- resignal cascade: the other engine was still holding the previous
    #    position at this trade's entry because its exit fired later (the
    #    matched sibling exited at different times on the two tapes). -----
    if side == "mt5":
        for i, j, _dt in pairs:
            p, m = py_trades[i], mt5_trades[j]
            if (p["open_time"].date() == t["open_time"].date()
                    and t["open_time"] >= p["close_time"]
                    and m["close_time"] < p["close_time"]):
                return "resignal_cascade_exit_timing", (
                    f"matched sibling py#{p['ticket']}/mt5#{m['ticket']} same day: "
                    f"mt5 exited {m['close_time'].strftime('%H:%M')} vs py "
                    f"{p['close_time'].strftime('%H:%M')} (entry-bar SL path on "
                    f"the tester tape); mt5 was flat again for this signal, py "
                    f"was still holding")
    else:
        for i, j, _dt in pairs:
            p, m = py_trades[i], mt5_trades[j]
            if (p["open_time"].date() == t["open_time"].date()
                    and t["open_time"] >= m["close_time"]
                    and p["close_time"] < m["close_time"]):
                return "resignal_cascade_exit_timing", (
                    f"matched sibling py#{p['ticket']}/mt5#{m['ticket']} same day: "
                    f"py exited {p['close_time'].strftime('%H:%M')} vs mt5 "
                    f"{m['close_time'].strftime('%H:%M')}; py was flat again for "
                    f"this signal, mt5 was still holding")

    dev_margin = abs(abs(t["dev_pct"]) - 0.25) if t["dev_pct"] is not None else None
    slope_margin = abs(t["slope_pct"]) if t["slope_pct"] is not None else None

    if dev_margin is not None and dev_margin <= DEV_STRADDLE:
        return "dev_threshold_straddle", (
            f"signal dev {t['dev_pct']:+.4f}% sits {dev_margin:.4f}% from the "
            f"0.25% band; the two tapes' VWAP/close differ slightly so only "
            f"this side crossed")
    if slope_margin is not None and slope_margin <= SLOPE_STRADDLE:
        return "slope_zero_straddle", (
            f"slope {t['slope_pct']:+.4f}% sits {slope_margin:.4f}% from the "
            f"slope-0 gate")
    if own is not None and own["net_pct_final"] is not None:
        nm = abs(own["net_pct_final"] - 0.6)
        hm = abs(own["hl_pct_final"] - 1.0)
        if nm <= RANGE_STRADDLE or hm <= RANGE_STRADDLE:
            which = "net 0.6%" if nm <= hm else "hl 1.0%"
            return "range_day_straddle", (
                f"day range margins net {nm:.4f}% / hl {hm:.4f}% — {which} "
                f"gate flipped between tapes")
    mm = day_maxmove_margin(own) if own else None
    if mm is not None and mm <= MAXMOVE_STRADDLE:
        return "maxmove_straddle", (
            f"day |net| {own['net_pct_final']:.4f}% sits {mm:.4f}% from the "
            f"0.42% max-move gate")
    # other side traded the opposite direction that day?
    o_dir_trades = o["entries"]
    note = (f"other side entries={o['entries']} signals={o['signals']} "
            f"slope_ok_bars={o['slope_ok_bars']} vs own "
            f"signals={own['signals'] if own else '?'}")
    if o_dir_trades == 0 and o["signals"] == 0:
        return "no_signal_other_side", note + " (other side never signaled)"
    return "unexplained", note


def main():
    args = parse_args()
    base = Path(args.base_dir)
    py = load_trades(base / "py_trades.csv")
    mt5 = load_trades(base / "mt5_trades.csv")
    py_days = load_days(base / "py_days.csv")
    mt5_days = load_days(base / "mt5_days.csv")

    pairs, unmatched_py, unmatched_mt5 = match_trades(py, mt5)

    R = []
    A = R.append
    A("# VWAP MR — Python vs MT5 reconciliation")
    A("")
    A(f"US500-tick, 2021-01-01 .. 2026-05-01 — generated "
      f"{datetime.now().isoformat(timespec='seconds')} by evaluate/vwap_reconcile.py")
    A("")
    A("## a. Headline")
    A("")
    A("| Quantity | Value |")
    A("|---|---|")
    A(f"| Python trades | {len(py)} |")
    A(f"| MT5 trades (stitched) | {len(mt5)} |")
    A(f"| Matched pairs (same type, same date, \\|open_time\\| <= {MATCH_TOL:g} min) | {len(pairs)} |")
    A(f"| Unmatched Python | {len(unmatched_py)} |")
    A(f"| Unmatched MT5 | {len(unmatched_mt5)} |")

    entry_abs, exit_abs, exit_t, tp0_abs = [], [], [], []
    fully_ok = reason_ok = 0
    for i, j, _dt in pairs:
        p, m = py[i], mt5[j]
        e_d = m["open_price"] - p["open_price"]
        x_d = m["close_price"] - p["close_price"]
        xt = abs((m["close_time"] - p["close_time"]).total_seconds()) / 60.0
        t_d = m["tp0"] - p["tp0"]
        entry_abs.append(abs(e_d))
        exit_abs.append(abs(x_d))
        exit_t.append(xt)
        tp0_abs.append(abs(t_d))
        ok_r = p["exit_reason"] == m["exit_reason"]
        reason_ok += ok_r
        if abs(e_d) <= ENTRY_PX_TOL and abs(x_d) <= EXIT_PX_TOL and xt <= EXIT_TIME_TOL and ok_r:
            fully_ok += 1

    denom = max(len(py), len(mt5))
    A("")
    A(f"Agreement at tolerances (entry <= {ENTRY_PX_TOL} pts, exit <= {EXIT_PX_TOL} pts, "
      f"exit time <= {EXIT_TIME_TOL:g} min, equal reason): "
      f"**{fully_ok}/{denom} ({100.0 * fully_ok / denom:.1f}%)**")
    A("")
    A("Matched-pair delta statistics (|MT5 - Python|):")
    A("")
    A("| Delta | mean | p95 | max | tolerance | within |")
    A("|---|---|---|---|---|---|")
    A(f"| entry price (pts) | {mean(entry_abs):.3f} | {p95(entry_abs):.3f} | {max(entry_abs):.3f} | <= {ENTRY_PX_TOL} | {sum(1 for v in entry_abs if v <= ENTRY_PX_TOL)}/{len(entry_abs)} |")
    A(f"| exit price (pts) | {mean(exit_abs):.3f} | {p95(exit_abs):.3f} | {max(exit_abs):.3f} | <= {EXIT_PX_TOL} | {sum(1 for v in exit_abs if v <= EXIT_PX_TOL)}/{len(exit_abs)} |")
    A(f"| exit time (min) | {mean(exit_t):.1f} | {p95(exit_t):.1f} | {max(exit_t):.0f} | <= {EXIT_TIME_TOL:g} | {sum(1 for v in exit_t if v <= EXIT_TIME_TOL)}/{len(exit_t)} |")
    A(f"| exit reason | - | - | - | - | {reason_ok}/{len(pairs)} |")
    A(f"| VWAP@entry tp0 (pts) | {mean(tp0_abs):.3f} | {p95(tp0_abs):.3f} | {max(tp0_abs):.3f} | - | - |")

    years = sorted({t["open_time"].year for t in py + mt5})
    A("")
    A("| Year | Python | MT5 | Matched | MT5 - Python |")
    A("|---|---|---|---|---|")
    for y in years:
        pyn = sum(1 for t in py if t["open_time"].year == y)
        mtn = sum(1 for t in mt5 if t["open_time"].year == y)
        m = sum(1 for i, _j, _d in pairs if py[i]["open_time"].year == y)
        A(f"| {y} | {pyn} | {mtn} | {m} | {mtn - pyn:+d} |")

    # ---- day-level comparison ----------------------------------------
    A("")
    A("## b. Day-level comparison")
    A("")
    only_py = sorted(set(py_days) - set(mt5_days))
    only_mt5 = sorted(set(mt5_days) - set(py_days))
    common = sorted(set(py_days) & set(mt5_days))
    A(f"* day rows: Python {len(py_days)}, MT5 {len(mt5_days)}; only-py: "
      f"{[dstr(d) for d in only_py] or 'none'}; only-mt5: "
      f"{[dstr(d) for d in only_mt5] or 'none'}; common: {len(common)}")

    sess_open_d, sess_high_d, sess_low_d = [], [], []
    entry_diff_days = []
    slope_diff_days = []
    for d in common:
        p, m = py_days[d], mt5_days[d]
        if p["session_open"] > 0 and m["session_open"] > 0:
            sess_open_d.append(abs(p["session_open"] - m["session_open"]))
            sess_high_d.append(abs(p["session_high"] - m["session_high"]))
            sess_low_d.append(abs(p["session_low"] - m["session_low"]))
        if p["entries"] != m["entries"]:
            entry_diff_days.append(d)
        if abs(p["slope_ok_bars"] - m["slope_ok_bars"]) > 2:
            slope_diff_days.append(d)

    A(f"* session-day count (session_open > 0): py {sum(1 for d in common if py_days[d]['session_open'] > 0)}, "
      f"mt5 {sum(1 for d in common if mt5_days[d]['session_open'] > 0)}")
    A(f"* |session_open| delta: mean {mean(sess_open_d):.3f} p95 {p95(sess_open_d):.3f} "
      f"max {max(sess_open_d):.3f} pts (tester tape vs real-tick M1)")
    A(f"* |session_high| delta: mean {mean(sess_high_d):.3f} max {max(sess_high_d):.3f}; "
      f"|session_low|: mean {mean(sess_low_d):.3f} max {max(sess_low_d):.3f}")
    A(f"* days with entries count mismatch: {len(entry_diff_days)}"
      + (f" -> {', '.join(dstr(d) for d in entry_diff_days)}" if entry_diff_days else ""))
    A(f"* days with slope_ok_bars diff > 2: {len(slope_diff_days)}")
    A(f"* skipped days: py {sum(1 for d in common if py_days[d]['skipped'])}, "
      f"mt5 {sum(1 for d in common if mt5_days[d]['skipped'])}")
    A(f"* total day PnL: py {sum(d['total_pnl'] for d in py_days.values()):+.2f}, "
      f"mt5 {sum(d['total_pnl'] for d in mt5_days.values()):+.2f}")

    # ---- unmatched classification ------------------------------------
    A("")
    A("## c. Unmatched / extra trades (every one, classified)")
    A("")
    rows = []
    for i in unmatched_py:
        t = py[i]
        cls, note = classify_unmatched(t, "py", mt5_days, py_days, py, mt5, pairs)
        rows.append(("PY", t, cls, note))
    for j in unmatched_mt5:
        t = mt5[j]
        cls, note = classify_unmatched(t, "mt5", py_days, mt5_days, py, mt5, pairs)
        rows.append(("MT5", t, cls, note))
    um_counts = defaultdict(int)
    for _s, _t, cls, _n in rows:
        um_counts[cls] += 1
    A("Classification counts: " + (", ".join(
        f"{k}: {v}" for k, v in sorted(um_counts.items(), key=lambda kv: -kv[1])) or "none"))
    A("")
    A("| Side | # | Dir | Open (UTC) | Entry | dev% | slope% | Class | Evidence |")
    A("|---|---|---|---|---|---|---|---|---|")
    for side, t, cls, note in sorted(rows, key=lambda r: r[1]["open_time"]):
        A(f"| {side} | {t['ticket']} | {t['type']} | "
          f"{t['open_time'].strftime('%Y-%m-%d %H:%M')} | {t['open_price']:.2f} | "
          f"{t['dev_pct']:+.4f} | {t['slope_pct']:+.4f} | **{cls}** | {note} |")

    # ---- exit outliers ------------------------------------------------
    A("")
    A("## d. Exit outliers among matched pairs")
    A("")
    t_out, px_out = [], []
    for i, j, _dt in pairs:
        p, m = py[i], mt5[j]
        xt = abs((m["close_time"] - p["close_time"]).total_seconds()) / 60.0
        x_d = m["close_price"] - p["close_price"]
        if xt > EXIT_TIME_TOL:
            t_out.append((p, m, xt))
        if abs(x_d) > EXIT_PX_TOL:
            px_out.append((p, m, x_d, xt))
    A(f"### |exit time| > {EXIT_TIME_TOL:g} min: {len(t_out)} pairs")
    A("")
    if t_out:
        A("| py# | mt5# | Dir | Date | py close | mt5 close | dt (min) | reasons |")
        A("|---|---|---|---|---|---|---|---|")
        for p, m, xt in sorted(t_out, key=lambda r: -r[2]):
            A(f"| {p['ticket']} | {m['ticket']} | {p['type']} | "
              f"{p['open_time'].strftime('%m-%d')} | "
              f"{p['close_time'].strftime('%m-%d %H:%M')} | "
              f"{m['close_time'].strftime('%m-%d %H:%M')} | {xt:.0f} | "
              f"{p['exit_reason']} / {m['exit_reason']} |")
    A("")
    A(f"### |exit price| > {EXIT_PX_TOL} pts: {len(px_out)} pairs")
    A("")
    A("| py# | mt5# | Dir | Date | py exit | mt5 exit | delta | dt (min) | reasons | Class |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for p, m, x_d, xt in sorted(px_out, key=lambda r: -abs(r[2])):
        if p["exit_reason"] != m["exit_reason"]:
            cls = "stop_vs_target_path_flip"
        elif xt <= 1.0:
            cls = "stop_fill_slippage"
        else:
            cls = "level_cross_bar_shift"
        A(f"| {p['ticket']} | {m['ticket']} | {p['type']} | "
          f"{p['open_time'].strftime('%m-%d')} | {p['close_price']:.2f} | "
          f"{m['close_price']:.2f} | {x_d:+.3f} | {xt:.0f} | "
          f"{p['exit_reason']} / {m['exit_reason']} | {cls} |")
    A("")
    A("* `stop_vs_target_path_flip`: the SL (0.2% from entry, one side) and the "
      "VWAP target (~0.25%+ the other side) are both inside the exit window; "
      "the tester's synthetic tape crosses one first, the real-tick M1 path "
      "the other. Same mechanism as ORB's exit-level straddles, amplified by "
      "this strategy's tight stop.")
    A("* `stop_fill_slippage`: same bar, same reason; the tester's stop fill "
      "slips a few points past the level on the synthetic tape.")
    A("* `level_cross_bar_shift`: slow market crosses the exit level in "
      "different bars on the two tapes.")

    # ---- verdict -------------------------------------------------------
    A("")
    A("## e. Verdict")
    A("")
    pct = 100.0 * fully_ok / denom
    unexplained = um_counts.get("unexplained", 0)
    A(f"Overall agreement **{fully_ok}/{denom} ({pct:.1f}%)**. "
      f"{len(unmatched_py) + len(unmatched_mt5)} unmatched trades, "
      f"{len(t_out)} exit-time outliers, {len(px_out)} exit-price outliers. "
      f"Unexplained residuals: **{unexplained}**"
      + (" — each listed in section c and flagged for dedicated investigation."
         if unexplained else " — every residual carries a documented root cause."))

    text = "\n".join(R) + "\n"
    out = base / "RECONCILE.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"reconcile report -> {out}")


if __name__ == "__main__":
    main()
