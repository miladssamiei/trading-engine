# -*- coding: utf-8 -*-
"""
vwap_rr_walkforward.py — Phase 2 (the ONLY phase allowed to be called a
result) of the VWAP stop/target exploration.

Full re-simulation of the VWAP mean-reversion rule with parameterized
  STOP_PCT  — stop distance % beyond entry (baseline 0.20)
  TARGET_F  — take-profit at entry + F*(VWAP_t - entry), i.e. F of the way
              back to the moving VWAP (baseline 1.00 = touch of VWAP)
Entries AND exits are re-run together (a different exit rule changes when
the strategy is next flat, hence which later entries exist) — this resolves
the fixed-entries approximation of Phase 1. Engine is a faithful
parameterization of evaluate/vwap_reference.py; the baseline config MUST
reproduce the original 281 trades / -$202.42 exactly.

Walk-forward protocol (pre-declared in GATE.md):
  burn-in 2021; 9 half-year folds 2022H1..2026H1; per fold, select the
  candidate with the best trailing expectancy/trade (data strictly before
  the fold; eligibility >= 20 trailing trades, else baseline); stitch the
  selected folds -> walk-forward result. Also reports each candidate's
  naive full-range in-sample number to expose the fit gap.

Pure stdlib (pandas only used for nothing — plain csv/collections).
"""

import csv
import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\vwap_2021_2026")
OUT = Path(r"C:\Users\Milad\Desktop\RR\data\results\vwap_rr_exploration")
M1_CSV = BASE / "us500tick_m1.csv"

NY = ZoneInfo("America/New_York")
UTC = dt.timezone.utc

# strategy constants (identical to the reference / EA defaults)
DEVIATION_PCT = 0.25
RANGE_NET_PCT = 0.6
RANGE_HL_PCT = 1.0
MAXMOVE_PCT = 0.42
SLOPE_WINDOW_MIN = 30
LOT = 1.0
CONTRACT = 1.0
TICK_GRID = 0.01

SESSION_OPEN_M = 570
SESSION_LAST_M = 959
FORCE_EXIT_M = 955
LAST_ENTRY_M = 930
WINDOW_START = dt.datetime(2021, 1, 1, tzinfo=UTC)

CANDIDATES = {
    "baseline_020_100": (0.20, 1.00),
    "A_010_125": (0.10, 1.25),
    "B_025_125": (0.25, 1.25),
}

FOLD_BOUNDS = [
    dt.datetime(2022, 1, 1, tzinfo=UTC), dt.datetime(2022, 7, 1, tzinfo=UTC),
    dt.datetime(2023, 1, 1, tzinfo=UTC), dt.datetime(2023, 7, 1, tzinfo=UTC),
    dt.datetime(2024, 1, 1, tzinfo=UTC), dt.datetime(2024, 7, 1, tzinfo=UTC),
    dt.datetime(2025, 1, 1, tzinfo=UTC), dt.datetime(2025, 7, 1, tzinfo=UTC),
    dt.datetime(2026, 1, 1, tzinfo=UTC), dt.datetime(2026, 5, 1, tzinfo=UTC),
]
BASELINE_KEY = "baseline_020_100"
MIN_TRAILING = 20


def norm_price(p):
    return round(round(p / TICK_GRID) * TICK_GRID, 2)


def load_bars():
    by_date = defaultdict(list)
    t0 = int(WINDOW_START.timestamp())
    with open(M1_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            t = int(row["time"])
            if t < t0:
                continue
            ny = dt.datetime.fromtimestamp(t, tz=UTC).astimezone(NY)
            wall = ny.hour * 60 + ny.minute
            if wall < SESSION_OPEN_M or wall > SESSION_LAST_M:
                continue
            by_date[ny.year * 10000 + ny.month * 100 + ny.day].append({
                "time": t, "open": float(row["open"]), "high": float(row["high"]),
                "low": float(row["low"]), "close": float(row["close"]),
                "vol": int(row["tick_volume"]), "wall": wall,
            })
    for d in by_date:
        by_date[d].sort(key=lambda b: b["time"])
    return by_date


def simulate(by_date, stop_pct, target_f):
    """Faithful parameterization of vwap_reference.simulate(). Returns
    (trades, session_days) with trades as dicts."""
    all_dates = sorted(by_date)
    trades = []
    session_days = []
    ticket = 0

    for di, ny_date in enumerate(all_dates):
        sess = by_date[ny_date]
        if not sess or sess[0]["wall"] != SESSION_OPEN_M:
            if sess:
                session_days.append(ny_date)   # skipped day, still a session day
            continue
        session_days.append(ny_date)
        session_open = sess[0]["open"]
        skipped = False

        cum_pv = cum_v = 0.0
        vwap_after = {}
        sess_high = sess_low = None
        pos = None
        pending = None

        def exit_during(bar, pos_):
            """Exit checks during `bar` for open pos_. Returns (price, reason)
            or None. Mirrors the reference's manage_exits_during with the
            parameterized stop/target."""
            if bar["wall"] >= FORCE_EXIT_M:
                return bar["open"], "time_exit"
            sign = 1 if pos_["dir"] == "BUY" else -1
            sl = pos_["sl"]
            vprev = vwap_after.get(bar["time"] - 60)
            target = None if vprev is None else pos_["entry"] + target_f * (vprev - pos_["entry"])
            if pos_["dir"] == "BUY":
                if bar["open"] <= sl:
                    return bar["open"], "stop_loss"
                if target is not None and bar["open"] >= target:
                    return bar["open"], "vwap_touch"
                lo = bar["low"] <= sl
                hi = target is not None and bar["high"] >= target
                if lo and hi:
                    return (sl, "stop_loss") if bar["close"] >= bar["open"] else (target, "vwap_touch")
                if lo:
                    return sl, "stop_loss"
                if hi:
                    return target, "vwap_touch"
            else:
                if bar["open"] >= sl:
                    return bar["open"], "stop_loss"
                if target is not None and bar["open"] <= target:
                    return bar["open"], "vwap_touch"
                hi = bar["high"] >= sl
                lo = target is not None and bar["low"] <= target
                if hi and lo:
                    return (target, "vwap_touch") if bar["close"] >= bar["open"] else (sl, "stop_loss")
                if hi:
                    return sl, "stop_loss"
                if lo:
                    return target, "vwap_touch"
            return None

        def close_trade(px, reason, bar, entry_bar):
            nonlocal pos, ticket
            sign = 1 if pos["dir"] == "BUY" else -1
            ticket += 1
            trades.append({
                "ticket": ticket,
                "ny_date": ny_date,
                "open_ts": entry_bar["time"],
                "close_ts": bar["time"],
                "dir": pos["dir"],
                "entry": pos["entry"],
                "sl": pos["sl"],
                "tp0": pos["tp0"],
                "exit_price": px,
                "reason": reason,
                "profit": round(sign * (px - pos["entry"]) * LOT * CONTRACT, 2),
            })
            pos = None

        for i, bar in enumerate(sess):
            # 1) exits
            if pos is not None:
                ev = exit_during(bar, pos)
                if ev is not None:
                    close_trade(ev[0], ev[1], bar, pos["entry_bar"])
                    pending = None

            # 2) accumulate completed-bar state
            typ = (bar["high"] + bar["low"] + bar["close"]) / 3.0
            if bar["vol"] > 0:
                cum_pv += typ * bar["vol"]
                cum_v += bar["vol"]
            vwap_now = cum_pv / cum_v if cum_v > 0 else None
            vwap_after[bar["time"]] = vwap_now
            sess_high = bar["high"] if sess_high is None else max(sess_high, bar["high"])
            sess_low = bar["low"] if sess_low is None else min(sess_low, bar["low"])
            net_pct = abs(bar["close"] / session_open - 1.0) * 100.0
            hl_pct = (sess_high - sess_low) / session_open * 100.0
            range_ok = net_pct <= RANGE_NET_PCT and hl_pct <= RANGE_HL_PCT

            # 3) pending entry at this bar's open
            if pending is not None and pos is None and bar["wall"] <= LAST_ENTRY_M:
                sign = 1 if pending["dir"] == "BUY" else -1
                entry = bar["open"]
                sl = norm_price(entry * (1.0 - sign * stop_pct / 100.0))
                pos = {"dir": pending["dir"], "entry": entry, "sl": sl,
                       "entry_bar": bar, "tp0": pending["vwap"]}
            pending = None

            # 4) new signal on this completed bar
            if pos is None and not skipped and vwap_now is not None:
                nxt = sess[i + 1] if i + 1 < len(sess) else None
                if nxt is not None and nxt["wall"] <= LAST_ENTRY_M:
                    v_ref = vwap_after.get(bar["time"] - SLOPE_WINDOW_MIN * 60)
                    if v_ref is not None and v_ref > 0:
                        slope_pct = (vwap_now / v_ref - 1.0) * 100.0
                        dev_pct = (bar["close"] / vwap_now - 1.0) * 100.0
                        move_abs = abs(bar["close"] - session_open)
                        maxmove_ok = move_abs <= session_open * MAXMOVE_PCT / 100.0
                        d = None
                        if range_ok and maxmove_ok:
                            if dev_pct <= -DEVIATION_PCT and slope_pct >= 0.0:
                                d = "BUY"
                            elif dev_pct >= DEVIATION_PCT and slope_pct <= 0.0:
                                d = "SELL"
                        if d:
                            pending = {"dir": d, "vwap": vwap_now}

        # end of session with position open -> carry to next date's first bar
        if pos is not None:
            carry = None
            for nd in all_dates[di + 1:]:
                if by_date[nd]:
                    carry = by_date[nd][0]
                    break
            if carry is None:
                close_trade(sess[-1]["close"], "time_exit", sess[-1], pos["entry_bar"])
            else:
                close_trade(carry["open"], "time_exit", carry, pos["entry_bar"])

    return trades, session_days


def daily_pnl_series(trades, session_days):
    """Zero-filled daily PnL over session days, keyed ny_date."""
    per_day = defaultdict(float)
    for t in trades:
        per_day[t["ny_date"]] += t["profit"]
    return {d: round(per_day.get(d, 0.0), 2) for d in session_days}


def sharpe_maxdd(daily_map, dates):
    vals = [daily_map[d] for d in dates]
    if not vals:
        return None, 0.0
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1) if len(vals) > 1 else 0.0
    std = math.sqrt(var)
    sharpe = (mean / std * math.sqrt(252)) if std > 0 else None
    cum = peak = maxdd = 0.0
    for x in vals:
        cum += x
        peak = max(peak, cum)
        maxdd = max(maxdd, peak - cum)
    return sharpe, maxdd


def fmt_ts(e):
    return dt.datetime.fromtimestamp(e, tz=UTC).strftime("%Y.%m.%d %H:%M")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading M1 session bars ...")
    by_date = load_bars()
    n_bars = sum(len(v) for v in by_date.values())
    print(f"session bar-days: {len(by_date)}  bars: {n_bars}")

    runs = {}
    for name, (sp, f) in CANDIDATES.items():
        trades, session_days = simulate(by_date, sp, f)
        runs[name] = {"cfg": (sp, f), "trades": trades,
                      "session_days": session_days,
                      "daily": daily_pnl_series(trades, session_days)}
        pnl = sum(t["profit"] for t in trades)
        wins = sum(1 for t in trades if t["profit"] > 0)
        print(f"{name}: stop={sp:.2f}% target={f:.2f}x  trades={len(trades)}  "
              f"win%={100.0 * wins / len(trades):.2f}  total={pnl:+.2f}  "
              f"expt={pnl / len(trades):+.3f}")

    # ---- engine equivalence check --------------------------------------
    b = runs[BASELINE_KEY]
    pnl = sum(t["profit"] for t in b["trades"])
    ok = (len(b["trades"]) == 281 and abs(pnl + 202.42) < 0.5)
    print(f"\nbaseline equivalence vs original (281 / -202.42): "
          f"{'OK' if ok else 'MISMATCH — STOP'}")
    if not ok:
        raise SystemExit("baseline run does not reproduce the original trades")

    all_days = sorted(b["session_days"])
    day2date = {d: dt.datetime.strptime(str(d), "%Y%m%d").replace(tzinfo=UTC)
                for d in all_days}

    # ---- in-sample full-range per candidate -----------------------------
    print("\n== in-sample full range (2021-01..2026-04) ==")
    insample = {}
    for name, r in runs.items():
        dates = [d for d in all_days]
        sh, dd = sharpe_maxdd(r["daily"], dates)
        pnl = sum(t["profit"] for t in r["trades"])
        insample[name] = {
            "trades": len(r["trades"]), "total_pnl": round(pnl, 2),
            "expectancy": round(pnl / len(r["trades"]), 3),
            "sharpe": None if sh is None else round(sh, 3),
            "max_dd": round(dd, 2),
        }
        print(f"  {name:>18}: n={len(r['trades']):>3} total={pnl:+9.2f} "
              f"expt={pnl / len(r['trades']):+.3f} sharpe={sh if sh is None else round(sh, 3)} dd={dd:.2f}")

    # ---- walk-forward ----------------------------------------------------
    print("\n== walk-forward (folds 2022H1..2026H1, selection on trailing expectancy) ==")
    wf_rows = []
    wf_trades = []
    for k in range(len(FOLD_BOUNDS) - 1):
        fs, fe = FOLD_BOUNDS[k], FOLD_BOUNDS[k + 1]
        # selection: trailing data strictly before fs
        best, best_exp = None, None
        for name, r in runs.items():
            trail = [t for t in r["trades"]
                     if day2date[t["ny_date"]] < fs]
            if len(trail) < MIN_TRAILING:
                continue
            exp = sum(t["profit"] for t in trail) / len(trail)
            if best_exp is None or exp > best_exp:
                best, best_exp = name, exp
        sel = best if best is not None else BASELINE_KEY
        fold_trades = [t for t in runs[sel]["trades"]
                       if fs <= day2date[t["ny_date"]] < fe]
        for t in fold_trades:
            t["cfg"] = sel
        wf_trades.extend(fold_trades)
        fp = sum(t["profit"] for t in fold_trades)
        wf_rows.append({
            "fold": f"{fs.strftime('%Y-%m')}..{fe.strftime('%Y-%m')}",
            "selected": sel, "trades": len(fold_trades),
            "pnl": round(fp, 2),
            "expectancy": round(fp / len(fold_trades), 3) if fold_trades else None,
        })
        print(f"  {fs.strftime('%Y-%m')}..{fe.strftime('%Y-%m')}: sel={sel:>18} "
              f"n={len(fold_trades):>3} pnl={fp:+9.2f}")

    # ---- WF metrics ------------------------------------------------------
    wf_daily = defaultdict(float)
    for t in wf_trades:
        wf_daily[t["ny_date"]] += t["profit"]
    wf_dates = [d for d in all_days
                if FOLD_BOUNDS[0] <= day2date[d] < FOLD_BOUNDS[-1]]
    wf_map = {d: round(wf_daily.get(d, 0.0), 2) for d in wf_dates}
    sh, dd = sharpe_maxdd(wf_map, wf_dates)
    wf_total = sum(t["profit"] for t in wf_trades)
    wf_expt = wf_total / len(wf_trades) if wf_trades else None

    # baseline 2022+ subset for context
    bl_dates = wf_dates
    bl_map = {d: runs[BASELINE_KEY]["daily"].get(d, 0.0) for d in bl_dates}
    bl_sh, _ = sharpe_maxdd(bl_map, bl_dates)
    bl_trades_22 = [t for t in runs[BASELINE_KEY]["trades"]
                    if FOLD_BOUNDS[0] <= day2date[t["ny_date"]]]
    bl_pnl_22 = sum(t["profit"] for t in bl_trades_22)

    per_year = defaultdict(lambda: [0, 0.0])
    for t in wf_trades:
        y = t["ny_date"] // 10000
        per_year[y][0] += 1
        per_year[y][1] += t["profit"]

    print("\n== WALK-FORWARD RESULT (the only number allowed to be called a result) ==")
    print(f"  trades={len(wf_trades)}  total={wf_total:+.2f}  "
          f"expectancy={wf_expt:+.3f}  sharpe={None if sh is None else round(sh, 3)}  maxDD={dd:.2f}")
    print(f"  baseline 2022+ subset: n={len(bl_trades_22)} total={bl_pnl_22:+.2f} "
          f"sharpe={round(bl_sh, 3)}   (baseline full-range anchor: -0.486)")
    print("  per-year:")
    for y in sorted(per_year):
        print(f"    {y}: n={per_year[y][0]:>3} pnl={per_year[y][1]:+9.2f}")

    # ---- gate evaluation --------------------------------------------------
    years_pos = sum(1 for y in per_year if per_year[y][1] > 0)
    best_year_share = (max(v[1] for v in per_year.values()) / wf_total * 100) \
        if wf_total > 0 else None
    g1 = (wf_expt is not None and wf_expt > 0.0
          and sh is not None and sh >= 0.0)
    g2 = len(wf_trades) >= 150
    g3 = (years_pos >= 4 and best_year_share is not None and best_year_share < 60.0)

    print("\n== GATES (pre-declared in GATE.md) ==")
    print(f"  G1 expectancy>0 & sharpe>=0 : expt={wf_expt:+.3f}, sharpe="
          f"{None if sh is None else round(sh, 3)} -> {'PASS' if g1 else 'FAIL'}")
    print(f"  G2 trades>=150             : {len(wf_trades)} -> {'PASS' if g2 else 'FAIL'}")
    print(f"  G3 >=4/5 years pos & best<60%: {years_pos}/5 pos, "
          f"best-year share="
          f"{'n/a' if best_year_share is None else f'{best_year_share:.1f}%'} "
          f"-> {'PASS' if g3 else 'FAIL'}")
    verdict = "CONFIRMED" if (g1 and g2 and g3) else "NOT CONFIRMED"
    print(f"  VERDICT: {verdict}")

    # ---- artifacts --------------------------------------------------------
    with open(OUT / "wf_folds.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(wf_rows[0].keys()))
        w.writeheader()
        w.writerows(wf_rows)
    with open(OUT / "wf_trades.csv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["ticket", "open_time", "close_time", "type", "entry", "sl",
                    "tp0", "exit_price", "exit_reason", "profit", "cfg"])
        for i, t in enumerate(wf_trades, 1):
            w.writerow([i, fmt_ts(t["open_ts"]), fmt_ts(t["close_ts"]), t["dir"],
                        f"{t['entry']:.4f}", f"{t['sl']:.2f}", f"{t['tp0']:.4f}",
                        f"{t['exit_price']:.4f}", t["reason"], f"{t['profit']:.2f}",
                        t.get("cfg", "")])
    summary = {
        "candidates": {k: {"stop_pct": v[0], "target_frac": v[1]}
                       for k, v in CANDIDATES.items()},
        "in_sample_full_range": insample,
        "walk_forward": {
            "folds": wf_rows,
            "trades": len(wf_trades),
            "total_pnl": round(wf_total, 2),
            "expectancy": None if wf_expt is None else round(wf_expt, 3),
            "sharpe": None if sh is None else round(sh, 3),
            "max_dd": round(dd, 2),
            "per_year": {str(y): {"trades": v[0], "pnl": round(v[1], 2)}
                         for y, v in sorted(per_year.items())},
            "baseline_2022_subset": {
                "trades": len(bl_trades_22), "total_pnl": round(bl_pnl_22, 2),
                "sharpe": None if bl_sh is None else round(bl_sh, 3)},
        },
        "gates": {"G1": g1, "G2": g2, "G3": g3, "verdict": verdict},
    }
    with open(OUT / "wf_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote: {OUT / 'wf_folds.csv'}, wf_trades.csv, wf_summary.json")


if __name__ == "__main__":
    main()
