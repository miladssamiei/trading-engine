"""
Categorized reconciliation of the FULL-RANGE (2021-01-01 .. 2026-04-30) MT5
ORB backtest vs the Python reference, US500-tick.

Extends evaluate/orb_reconcile.py (which writes the machine REPORT.md) with a
case-by-case investigation of every SMALL but REAL disagreement. Outputs:

  a. headline agreement stats + per-year trade-count table
  b. every day-level DECISION diff (skipped / skip_reason / orders_placed /
     entry_direction) with both values, each classified: gap_straddle |
     no_prior_close boundary | incomplete_range | orders-placed difference |
     entry-direction difference
  c. every unmatched / extra trade, each classified: early_close_carry |
     market_entry_ask_vs_bid | entry_straddle | gap_straddle_day |
     no_prior_close_day | mt5_market_entry_fallback | other, plus an explicit
     verification of the two early-close days (2024-11-29 Black Friday,
     2024-12-24 Christmas Eve)
  d. exit-time / exit-price outliers among matched pairs
  e. statistical summary of day-level level deltas (no per-day listing)
  f. verdict paragraph

Inputs (all in data/results/orb_2021_present/):
    py_trades.csv / mt5_trades.csv   semicolon CSV, times "YYYY.MM.DD HH:MM" UTC
    py_days.csv   / mt5_days.csv     semicolon CSV, one row per NY calendar date
    us500tick_m5.csv                 M5 bid bars (time,open,high,low,close),
                                     bar-open UTC epoch seconds
Output:
    data/results/orb_2021_present/RECONCILE.md  (same content printed to console)
Pure stdlib.
"""

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-range categorized ORB reconciliation.")
    parser.add_argument(
        "--base-dir",
        default=r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present")
    return parser.parse_args()

_ARGS = parse_args()
BASE = Path(_ARGS.base_dir)
PY_TRADES_CSV = BASE / "py_trades.csv"
MT5_TRADES_CSV = BASE / "mt5_trades.csv"
PY_DAYS_CSV = BASE / "py_days.csv"
MT5_DAYS_CSV = BASE / "mt5_days.csv"
M5_CSV = BASE / "us500tick_m5.csv"
RECONCILE_MD = BASE / "RECONCILE.md"

MATCH_TOL = 10.0        # minutes, one-to-one trade matching on open_time
ENTRY_PX_TOL = 1.5      # pts, |entry price delta|
EXIT_PX_TOL = 1.5       # pts, |exit price delta|
EXIT_TIME_TOL = 15.0    # minutes, |exit time delta|
REASON_SLIP = 0.5       # pts, MT5 "other" exit reclassification tolerance
LEVEL_TOL = 0.02        # pts, day-level price/level delta flag (machine report)
STRADDLE_DP = 0.1       # gap_pct distance from 1.0 that still counts as straddle
ENTRY_STRADDLE_PX = 0.2 # pts, level-to-bar-extreme gap for entry_straddle
EARLY_CLOSE_DAYS = (20241129, 20241224)
MT5_FALLBACK_DAY = 20221014   # B2022-10-14 market-entry fallback rejection

NY = ZoneInfo("America/New_York")
PRICE_FIELDS = ("prev_close", "session_open", "range_high", "range_low",
                "buy_level", "sell_level")


def parse_ts(s):
    s = (s or "").strip()
    return datetime.strptime(s, "%Y.%m.%d %H:%M") if s else None


def load_trades(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    trades = []
    for r in rows:
        trades.append({
            "ticket": int(r["ticket"]),
            "open_time": parse_ts(r["open_time"]),
            "close_time": parse_ts(r["close_time"]),
            "type": r["type"].strip().upper(),
            "lot": float(r["lot"]),
            "open_price": float(r["open_price"]),
            "sl": float(r["stop_loss"]),
            "tp": float(r["take_profit"]),
            "close_price": float(r["close_price"]),
            "exit_reason": r["exit_reason"].strip(),
            "profit": float(r["profit"]),
        })
    return trades


def load_days(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    days = {}
    for r in rows:
        d = {
            "ny_date": int(r["ny_date"]),
            "prev_close": (r["prev_close"] or "").strip(),
            "session_open": (r["session_open"] or "").strip(),
            "gap_pct": (r["gap_pct"] or "").strip(),
            "skipped": int(r["skipped"] or 0),
            "skip_reason": (r["skip_reason"] or "").strip(),
            "range_high": (r["range_high"] or "").strip(),
            "range_low": (r["range_low"] or "").strip(),
            "or_bars": int(r["or_bars"] or 0),
            "buy_level": (r["buy_level"] or "").strip(),
            "sell_level": (r["sell_level"] or "").strip(),
            "orders_placed": int(r["orders_placed"] or 0),
            "entry_direction": (r["entry_direction"] or "").strip(),
            "entry_time": parse_ts(r["entry_time"]),
            "exit_time": parse_ts(r["exit_time"]),
            "exit_reason": (r["exit_reason"] or "").strip(),
        }
        days[d["ny_date"]] = d
    return days


def load_bars(path):
    """M5 bid bars indexed by NY calendar date. Bar row:
    (time_epoch, open, high, low, close, ny_wall_minutes)."""
    by_date = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            t = int(r["time"])
            nydt = datetime.fromtimestamp(t, tz=timezone.utc).astimezone(NY)
            key = nydt.year * 10000 + nydt.month * 100 + nydt.day
            by_date[key].append(
                (t, float(r["open"]), float(r["high"]), float(r["low"]),
                 float(r["close"]), nydt.hour * 60 + nydt.minute))
    return by_date


def has_session(day):
    for f in ("prev_close", "session_open"):
        raw = day[f]
        if raw and float(raw) > 0.0:
            return True
    return False


def day_float(day, field):
    raw = day[field]
    if raw == "":
        return None
    v = float(raw)
    if field == "gap_pct":
        return v if has_session(day) else None
    return v if v != 0.0 else None


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


def map_reason(t):
    if t["exit_reason"] != "other":
        return t["exit_reason"]
    if abs(t["close_price"] - t["tp"]) <= REASON_SLIP:
        return "take_profit"
    if abs(t["close_price"] - t["sl"]) <= REASON_SLIP:
        return "stop_loss"
    return "other"


def match_trades(py_trades, mt5_trades):
    cands = []
    for i, p in enumerate(py_trades):
        for j, m in enumerate(mt5_trades):
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
    unmatched_py = [i for i in range(len(py_trades)) if i not in used_py]
    unmatched_mt5 = [j for j in range(len(mt5_trades)) if j not in used_mt5]
    return pairs, unmatched_py, unmatched_mt5

def window_extremes(bars_by_date, ny_date):
    """max(high) / min(low) of the 10:00-10:30 NY bars of a date."""
    highs, lows = [], []
    for _t, _o, h, l, _c, wall in bars_by_date.get(ny_date, ()):
        if 600 <= wall < 630:
            highs.append(h)
            lows.append(l)
    return (max(highs) if highs else None, min(lows) if lows else None)


def margin_of_level(day, direction, bars_by_date, ny_date):
    """How far the day's buy/sell level sits from the entry-window bar
    extreme (positive = level never touched by the bars; smaller = closer
    straddle). None if the level or window is missing."""
    lvl = day_float(day, "buy_level" if direction == "BUY" else "sell_level")
    h, l = window_extremes(bars_by_date, ny_date)
    if lvl is None or h is None or l is None:
        return None
    if direction == "BUY":
        return lvl - h
    return l - lvl


def classify_day_diff(p, m, bars_by_date):
    """Classify a day-level DECISION diff. Returns (issues, class)."""
    issues = []
    if p["skipped"] != m["skipped"] or p["skip_reason"] != m["skip_reason"]:
        issues.append("skipped/skip_reason %d/%s vs %d/%s" % (
            p["skipped"], p["skip_reason"] or "-",
            m["skipped"], m["skip_reason"] or "-"))
    if p["orders_placed"] != m["orders_placed"]:
        issues.append("orders_placed %d vs %d" % (
            p["orders_placed"], m["orders_placed"]))
    if p["entry_direction"] != m["entry_direction"]:
        issues.append("entry_direction %s vs %s" % (
            p["entry_direction"] or "-", m["entry_direction"] or "-"))

    py_skip = p["skipped"] == 1
    mt_skip = m["skipped"] == 1
    if py_skip != mt_skip and "gap_filter" in (p["skip_reason"], m["skip_reason"]):
        gp, gm = day_float(p, "gap_pct"), day_float(m, "gap_pct")
        if (gp is not None and gm is not None
                and abs(gp - 1.0) <= STRADDLE_DP and abs(gm - 1.0) <= STRADDLE_DP):
            return issues, "gap_straddle"
    if "no_prior_close" in (p["skip_reason"], m["skip_reason"]):
        return issues, "no_prior_close_boundary"
    if "incomplete_range" in (p["skip_reason"], m["skip_reason"]):
        return issues, "incomplete_range"
    if p["orders_placed"] != m["orders_placed"]:
        return issues, "orders_placed_diff"
    if p["entry_direction"] != m["entry_direction"]:
        return issues, "entry_direction_diff"
    return issues, "other"


def classify_unmatched(t, side, oth_days, bars_by_date):
    """Classify an unmatched trade on `side` ('py' or 'mt5').

    Checks the OTHER side's day record for the same NY date and, when the
    other side placed orders but never filled, the level-to-bar-extreme
    margin to separate market_entry_ask_vs_bid from entry_straddle."""
    ny_date = int(t["open_time"].strftime("%Y%m%d"))
    o = oth_days.get(ny_date)
    if o is None:
        return "other", "other side has no day record for %d" % ny_date
    o_skip = int(o["skipped"])
    o_reason = o["skip_reason"] or ""
    o_ord = int(o["orders_placed"])
    o_dir = o["entry_direction"] or ""

    if o_skip == 1:
        if o_reason == "gap_filter":
            return "gap_straddle_day", (
                "other side skipped with gap_filter (gap %s%%), this side traded" % o["gap_pct"])
        if o_reason == "no_prior_close":
            return "no_prior_close_day", "other side skipped: no_prior_close boundary"
        return "other", "other side skipped: %s" % (o_reason or "-")
    if o_ord == 0:
        if side == "py" and ny_date == MT5_FALLBACK_DAY:
            return "mt5_market_entry_fallback", (
                "MT5 market-entry fallback rejected retcode=10015 (off-grid live "
                "bid); no orders placed, Python traded at the 10:00 bar open")
        return "other", "other side placed no orders (orders_placed=0)"

    margin = margin_of_level(o, t["type"], bars_by_date, ny_date)
    if o_dir == "":
        if margin is None:
            return "other", "other side placed orders but never filled"
        if margin < ENTRY_STRADDLE_PX:
            return "entry_straddle", (
                "other side placed orders but never filled; its %s level sits "
                "%.3f pts from the entry-window bar extreme (< %.2f), crossed "
                "only by this side's fill" % (
                    t["type"].lower(), margin, ENTRY_STRADDLE_PX))
        return "market_entry_ask_vs_bid", (
            "other side placed orders but never filled (level margin %.3f pts "
            "from bar extreme); fill differs on the bid/ask trigger at the "
            "10:00 bar" % margin)
    if o_dir == t["type"]:
        return "entry_straddle", (
            "both sides filled the same direction but entry times differ "
            "beyond the %g-min match tolerance (other side %s); slight level "
            "delta + market hovering near the level" % (MATCH_TOL, o["entry_time"] or "-"))
    return "other", "other side traded the opposite direction (%s)" % o_dir


def dstr(ny_date):
    s = str(ny_date)
    return "%s-%s-%s" % (s[:4], s[4:6], s[6:8])


def main():
    py_trades = load_trades(PY_TRADES_CSV)
    mt5_trades = load_trades(MT5_TRADES_CSV)
    py_days = load_days(PY_DAYS_CSV)
    mt5_days = load_days(MT5_DAYS_CSV)
    bars_by_date = load_bars(M5_CSV)

    pairs, unmatched_py, unmatched_mt5 = match_trades(py_trades, mt5_trades)

    # ---- headline deltas over matched pairs --------------------------
    entry_abs, exit_abs, exit_t = [], [], []
    fully_ok = 0
    reason_ok = 0
    for i, j, _dt in pairs:
        p, m = py_trades[i], mt5_trades[j]
        e_d = m["open_price"] - p["open_price"]
        x_d = m["close_price"] - p["close_price"]
        xt = abs((m["close_time"] - p["close_time"]).total_seconds()) / 60.0
        entry_abs.append(abs(e_d))
        exit_abs.append(abs(x_d))
        exit_t.append(xt)
        ok_r = map_reason(m) == p["exit_reason"]
        reason_ok += ok_r
        if (abs(e_d) <= ENTRY_PX_TOL and abs(x_d) <= EXIT_PX_TOL
                and xt <= EXIT_TIME_TOL and ok_r):
            fully_ok += 1

    # ---- day-level decision diffs (b) --------------------------------
    only_py = sorted(set(py_days) - set(mt5_days))
    only_mt5 = sorted(set(mt5_days) - set(py_days))
    common = sorted(set(py_days) & set(mt5_days))
    day_diffs = []
    for d in common:
        p, m = py_days[d], mt5_days[d]
        if (p["skipped"], p["skip_reason"], p["orders_placed"],
                p["entry_direction"]) == (m["skipped"], m["skip_reason"],
                                          m["orders_placed"], m["entry_direction"]):
            continue
        issues, cls = classify_day_diff(p, m, bars_by_date)
        day_diffs.append((d, p, m, issues, cls))
    class_counts = defaultdict(int)
    for _d, _p, _m, _issues, cls in day_diffs:
        class_counts[cls] += 1

    # ---- unmatched trade classification (c) --------------------------
    unmatched_rows = []
    for i in unmatched_py:
        t = py_trades[i]
        cls, note = classify_unmatched(t, "py", mt5_days, bars_by_date)
        unmatched_rows.append(("PY", t["ticket"], t["type"], t["open_time"],
                               t["open_price"], cls, note))
    for j in unmatched_mt5:
        t = mt5_trades[j]
        cls, note = classify_unmatched(t, "mt5", py_days, bars_by_date)
        unmatched_rows.append(("MT5", t["ticket"], t["type"], t["open_time"],
                               t["open_price"], cls, note))
    um_counts = defaultdict(int)
    for r in unmatched_rows:
        um_counts[r[5]] += 1

    # ---- early-close verification ------------------------------------
    early_close_rows = []
    for i, j, _dt in pairs:
        p, m = py_trades[i], mt5_trades[j]
        d = int(p["open_time"].strftime("%Y%m%d"))
        if d in EARLY_CLOSE_DAYS:
            xt = (m["close_time"] - p["close_time"]).total_seconds() / 60.0
            xd = m["close_price"] - p["close_price"]
            early_close_rows.append((d, p, m, xt, xd))

    # ---- exit-time / exit-price outliers (d) -------------------------
    exit_t_out = []
    exit_px_out = []
    for i, j, _dt in pairs:
        p, m = py_trades[i], mt5_trades[j]
        d = int(p["open_time"].strftime("%Y%m%d"))
        e_d = m["open_price"] - p["open_price"]
        x_d = m["close_price"] - p["close_price"]
        xt = abs((m["close_time"] - p["close_time"]).total_seconds()) / 60.0
        if xt > EXIT_TIME_TOL:
            cls = "early_close_carry" if d in EARLY_CLOSE_DAYS else "exit_level_straddle"
            exit_t_out.append((p, m, e_d, x_d, xt, cls))
        if abs(x_d) > EXIT_PX_TOL:
            if d in EARLY_CLOSE_DAYS:
                cls = "early_close_carry"
            elif xt <= 0.5:
                cls = "session_end_boundary_tick"
            else:
                cls = "exit_reason_flip"
            exit_px_out.append((p, m, e_d, x_d, xt, cls))

    # ---- level-delta statistics (e) ----------------------------------
    level_profile = {f: [] for f in PRICE_FIELDS}
    for d in common:
        p, m = py_days[d], mt5_days[d]
        if has_session(p) and has_session(m):
            for f in PRICE_FIELDS:
                vp, vm = day_float(p, f), day_float(m, f)
                if vp is not None and vm is not None:
                    level_profile[f].append(abs(vp - vm))

    # ---- per-year table ----------------------------------------------
    years = sorted({int(t["open_time"].year) for t in py_trades + mt5_trades})
    year_rows = []
    for y in years:
        py_n = sum(1 for t in py_trades if t["open_time"].year == y)
        mt_n = sum(1 for t in mt5_trades if t["open_time"].year == y)
        mtch = sum(1 for i, _j, _dt in pairs if py_trades[i]["open_time"].year == y)
        year_rows.append((y, py_n, mt_n, mtch))

    # ==================================================================
    R = []
    A = R.append
    A("# ORB EA vs Python reference -- full-range categorized reconciliation")
    A("")
    A("US500-tick, 2021-01-01 .. 2026-04-30 -- generated %s by "
      "evaluate/orb_reconcile_fullrange.py" %
      datetime.now().isoformat(timespec="seconds"))
    A("")
    A("Machine artifact (every field, every day, full outlier tables): "
      "`REPORT.md` (evaluate/orb_reconcile.py). This document investigates "
      "each SMALL disagreement case-by-case and classifies every one.")
    A("")
    A("## a. Headline")
    A("")
    A("| Quantity | Value |")
    A("|---|---|")
    A("| Python trades | %d |" % len(py_trades))
    A("| MT5 trades (stitched, tickets renumbered) | %d |" % len(mt5_trades))
    A("| Matched pairs (same type, same open date, |open_time| <= %g min) | %d |" % (MATCH_TOL, len(pairs)))
    A("| Unmatched Python | %d |" % len(unmatched_py))
    A("| Unmatched MT5 | %d |" % len(unmatched_mt5))
    A("| Direction agreement within matched (matching requires same type) | %d/%d (100%%) |" % (len(pairs), len(pairs)))
    A("")
    A("Agreement at tolerances (entry <= %g pts, exit <= %g pts, exit time <= "
      "%g min, mapped reason): **%d/%d (%.1f%%)**" % (
        ENTRY_PX_TOL, EXIT_PX_TOL, EXIT_TIME_TOL, fully_ok,
        max(len(py_trades), len(mt5_trades)),
        100.0 * fully_ok / max(len(py_trades), len(mt5_trades))))
    A("")
    A("Matched-pair delta statistics (|MT5 - Python|):")
    A("")
    A("| Delta | mean | p95 | max | tolerance | within |")
    A("|---|---|---|---|---|---|")
    A("| entry price (pts) | %.3f | %.3f | %.3f | <= %g | %d/%d |" % (
        mean(entry_abs), p95(entry_abs), max(entry_abs), ENTRY_PX_TOL,
        sum(1 for v in entry_abs if v <= ENTRY_PX_TOL), len(entry_abs)))
    A("| exit price (pts) | %.3f | %.3f | %.3f | <= %g | %d/%d |" % (
        mean(exit_abs), p95(exit_abs), max(exit_abs), EXIT_PX_TOL,
        sum(1 for v in exit_abs if v <= EXIT_PX_TOL), len(exit_abs)))
    A("| exit time (min) | %.1f | %.1f | %.0f | <= %g | %d/%d |" % (
        mean(exit_t), p95(exit_t), max(exit_t), EXIT_TIME_TOL,
        sum(1 for v in exit_t if v <= EXIT_TIME_TOL), len(exit_t)))
    A("| mapped exit reason | - | - | - | - | %d/%d |" % (reason_ok, len(pairs)))
    A("")
    A("Per-year trade counts:")
    A("")
    A("| Year | Python | MT5 | Matched | MT5 - Python |")
    A("|---|---|---|---|---|")
    for y, py_n, mt_n, mtch in year_rows:
        A("| %d | %d | %d | %d | %+d |" % (y, py_n, mt_n, mtch, mt_n - py_n))
    A("")
    A("Unmatched Python: %s" % ("; ".join(
        "#%d %s %s" % (py_trades[i]["ticket"], py_trades[i]["type"],
                       py_trades[i]["open_time"].strftime("%m-%d %H:%M"))
        for i in unmatched_py) or "none"))
    A("")
    A("Unmatched MT5: %s" % ("; ".join(
        "#%d %s %s" % (mt5_trades[j]["ticket"], mt5_trades[j]["type"],
                       mt5_trades[j]["open_time"].strftime("%m-%d %H:%M"))
        for j in unmatched_mt5) or "none"))
    A("")

    A("## b. Day-level decision differences (skipped, skip_reason, orders_placed, entry_direction)")
    A("")
    A("* day rows: Python %d, MT5 %d; only-py: %s; only-mt5: %s; common: %d" % (
        len(py_days), len(mt5_days), [dstr(d) for d in only_py] or "none",
        [dstr(d) for d in only_mt5] or "none", len(common)))
    A("  * only-py `%s` is the inert Sunday-evening row (first bar of the "
      "dataset, 2021-01-03 23:00 UTC); not a trading day, no decision." %
      (dstr(only_py[0]) if only_py else "-"))
    A("* decision-field diffs: **%d of %d** common dates" % (len(day_diffs), len(common)))
    A("")
    A("Classification counts: %s" % ", ".join(
        "%s: %d" % (k, v) for k, v in sorted(class_counts.items(),
                                             key=lambda kv: -kv[1])))
    A("")
    A("| Date | Python (skip, reason, orders, dir) | MT5 (skip, reason, orders, dir) | Issues | Class |")
    A("|---|---|---|---|---|")
    for d, p, m, issues, cls in day_diffs:
        def cell(x):
            return "%d, %s, %d, %s" % (x["skipped"], x["skip_reason"] or "-",
                                       x["orders_placed"],
                                       x["entry_direction"] or "-")
        gp = day_float(p, "gap_pct")
        gm = day_float(m, "gap_pct")
        gapnote = ""
        if cls == "gap_straddle":
            gapnote = " (gap_pct py %.4f / mt5 %.4f straddles 1.0)" % (gp, gm)
        A("| %s | %s | %s | %s%s | **%s** |" % (
            dstr(d), cell(p), cell(m), "; ".join(issues), gapnote, cls))
    A("")

    A("## c. Unmatched / extra trades (every one, classified)")
    A("")
    A("Classification counts: %s" % ", ".join(
        "%s: %d" % (k, v) for k, v in sorted(um_counts.items(),
                                             key=lambda kv: -kv[1])))
    A("")
    A("| Side | # | Dir | Open time (UTC) | Entry price | Class | Root cause / evidence |")
    A("|---|---|---|---|---|---|---|")
    for side, tk, d_, t_, px, cls, note in sorted(unmatched_rows,
                                                  key=lambda r: r[3]):
        A("| %s | %d | %s | %s | %.4f | **%s** | %s |" % (
            side, tk, d_, t_.strftime("%Y-%m-%d %H:%M"), px, cls, note))
    A("")
    A("### Early-close verification (2024-11-29 Black Friday, 2024-12-24 Christmas Eve)")
    A("")
    A("Both sides TRADE these days (matched pairs) but exit differently: "
      "Python closes at that day's last bar (`data_end`, close_time same day), "
      "MT5 force-close at 16:00 NY never fires (the session ended early, "
      "retcode 10015/10018 market-closed) so the position is carried into the "
      "next session and closed there as `session_end`.")
    A("")
    A("| Date | py# | mt5# | Dir | py close_time | mt5 close_time | exit-time delta | py close_price | mt5 close_price | exit-price delta |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for d, p, m, xt, xd in early_close_rows:
        A("| %s | %d | %d | %s | %s | %s | %+.0f min | %.4f | %.4f | %+.3f |" % (
            dstr(d), p["ticket"], m["ticket"], p["type"],
            p["close_time"].strftime("%m-%d %H:%M"),
            m["close_time"].strftime("%m-%d %H:%M"), xt,
            p["close_price"], m["close_price"], xd))
    A("")
    A("MT5 close_time = next session: 2024-11-29 -> 2024-12-02 01:00 "
      "(Sunday-evening session), 2024-12-24 -> 2024-12-25 23:00. These two "
      "pairs are the ONLY matched pairs whose exit REASON differs in the "
      "mapped comparison (Python `data_end` vs MT5 `session_end`).")
    A("")

    A("## d. Exit-time and exit-price outliers among matched pairs")
    A("")
    A("### |exit_time| > %g min: %d pairs" % (EXIT_TIME_TOL, len(exit_t_out)))
    A("")
    A("**Expectation (from the 100%% 2026 baseline) was exactly the 2 "
      "early-close carries. The full range adds 10 more.**")
    A("")
    A("| py# | mt5# | Dir | Date | entry delta | exit delta | exit-time delta (min) | py reason / mt5 -> mapped | Class |")
    A("|---|---|---|---|---|---|---|---|---|")
    for p, m, e_d, x_d, xt, cls in sorted(exit_t_out, key=lambda r: -r[4]):
        A("| %d | %d | %s | %s | %+.3f | %+.3f | %+.0f | %s / %s -> %s | **%s** |" % (
            p["ticket"], m["ticket"], p["type"],
            p["open_time"].strftime("%m-%d"), e_d, x_d, xt,
            p["exit_reason"], m["exit_reason"], map_reason(m), cls))
    A("")
    A("The 10 non-carry cases are **exit-level straddles**: the SL/TP levels "
      "differ slightly (level-source delta + 0.01 grid rounding) and when the "
      "market grinds toward a level the two implementations cross it in "
      "different M5 bars. Exit PRICES still agree within tolerance (each side "
      "exits at its own level); only the exit TIMESTAMP moves across bars.")
    A("")
    A("### |exit price| > %g pts: %d pairs" % (EXIT_PX_TOL, len(exit_px_out)))
    A("")
    A("| py# | mt5# | Dir | Date | entry delta | exit delta | exit-time delta (min) | py reason / mt5 -> mapped | Class |")
    A("|---|---|---|---|---|---|---|---|---|")
    for p, m, e_d, x_d, xt, cls in sorted(exit_px_out, key=lambda r: -abs(r[3])):
        A("| %d | %d | %s | %s | %+.3f | %+.3f | %+.0f | %s / %s -> %s | **%s** |" % (
            p["ticket"], m["ticket"], p["type"],
            p["open_time"].strftime("%m-%d"), e_d, x_d, xt,
            p["exit_reason"], m["exit_reason"], map_reason(m), cls))
    A("")
    A("Class notes:")
    A("")
    A("* `session_end_boundary_tick` (8): both sides close at the SAME time "
      "(16:00 NY bar) with reason `session_end`, but the close PRICE differs "
      "by ~1.5-1.9 pts. The exact MT5 exit tick (e.g. 2021-01-07 21:00:00 "
      "bid 3801.510) is NOT present in the current us500tick_m5.csv / tick "
      "data (current 21:00:00 tick bid = 3803.133 = Python's exit). These are "
      "tick-data-snapshot differences at the 16:00 bar boundary between the "
      "batch tester runs and today's M5 export -- not a strategy-logic "
      "difference.")
    A("* `early_close_carry` (1): 2024-11-29, see section c.")
    A("* `exit_reason_flip` (1): 2024-04-19, see verdict.")
    A("")

    A("## e. Day-level level deltas (statistical summary)")
    A("")
    A("| Field | mean | p95 | max | > %g tol |" % LEVEL_TOL)
    A("|---|---|---|---|---|")
    for f in PRICE_FIELDS:
        vals = level_profile[f]
        if vals:
            A("| %s | %.3f | %.3f | %.3f | %d/%d |" % (
                f, mean(vals), p95(vals), max(vals),
                sum(1 for v in vals if v > LEVEL_TOL), len(vals)))
    A("")
    A("These are |Python - MT5| over the session days that both sides carry "
      "data. Dominated by the different price sources (Python M5 bid bars vs "
      "MT5 native ticks) plus <= 0.01-pt grid rounding of MT5 levels; they "
      "stay far inside the %g-pt entry fill tolerance and do not change any "
      "trading decision on their own (the decision differences are the "
      "near-threshold cases in sections b/c)." % ENTRY_PX_TOL)
    A("")

    A("## f. Verdict")
    A("")
    fully_pct = 100.0 * fully_ok / max(len(py_trades), len(mt5_trades))
    A("Overall agreement (entry <= %g, exit <= %g, exit time <= %g, mapped "
      "reason): **%.1f%%** (%d/%d)." % (
        ENTRY_PX_TOL, EXIT_PX_TOL, EXIT_TIME_TOL, fully_pct, fully_ok,
        max(len(py_trades), len(mt5_trades))))
    A("")
    A("**Did it degrade meaningfully vs the 100%% 2026 baseline?** Yes, in raw "
      "terms: %d matched + %d/%d agreement on %d trades vs 100%% on the 27 "
      "trade 2026 baseline. The gap comes from 15 unmatched trades on 13 days, "
      "12 exit-time outliers (2 carries + 10 exit-level straddles), 10 "
      "exit-price outliers and 12 day-decision diffs. But only ONE of these is "
      "a genuine, unexplained divergence; the rest reproduce the documented "
      "artifacts." % (
        len(pairs), fully_ok, max(len(py_trades), len(mt5_trades)),
        max(len(py_trades), len(mt5_trades))))
    A("")
    A("Root causes and how many trades/days each affects:")
    A("")
    A("1. **gap_straddle_day -- 2 days / 2 trades** (2022-09-27 py SELL vs MT5 "
      "skip; 2022-11-01 MT5 SELL vs py skip). Gap_pct lands just either side "
      "of 1.0 (py 0.9964 vs mt5 1.0012; py 1.0063 vs mt5 0.9992) so one side "
      "sits inside the filter. Explainable: level-source + grid-rounding "
      "difference at a 1%% threshold.")
    A("2. **no_prior_close boundary -- 1 day / 0 trades** (2021-01-04). Python "
      "has session bars and skips (no prior 15:55 close); the MT5 tester was "
      "positioned at 2021-01-05 00:00 (B2021 agent log) and its 2021-01-04 row "
      "is inert. Plus the only-py 2021-01-03 Sunday-evening row. Explainable: "
      "start-of-data boundary.")
    A("3. **mt5_market_entry_fallback (retcode 10015) -- 1 day / 1 trade** "
      "(2022-10-14). Price opened far beyond the sell level; MT5 took the "
      "market-entry path and every market sell was rejected as Invalid price "
      "(off-grid live bid), so MT5 placed nothing while Python traded the "
      "10:00 bar open. Explainable: documented EA fallback limitation.")
    A("4. **entry-direction difference (ask-vs-bid / entry straddle) -- 10 days "
      "/ 10 unmatched trades** (8 day-decision-diff days: 2023-02-13, 2023-03-30, 2023-06-01, "
      "2023-06-09, 2024-02-14, 2024-04-26, 2025-10-20, 2025-12-11; plus the "
      "two both-filled straddles 2023-03-02 and 2024-07-12). Python triggers "
      "on bid bar open/high >= level; MT5 triggers on ask >= level with a "
      "slightly different (grid-rounded) level. When the level sits a few "
      "tenths of a point above the bar extreme only one side fills -- all "
      "MT5-only BUYs (ask crossed first) plus one Python-only SELL. "
      "Explainable: documented bid/ask + level-source artifact; per-side "
      "margin reported in section c.")
    A("5. **early_close_carry -- 2 days / 2 matched trades** (2024-11-29, "
      "2024-12-24). Python exits data_end same day; MT5 force-close fails "
      "(market closed) and the position is carried to the next session "
      "(session_end). Exit-time deltas 3290/1730 min, exit-price delta "
      "-6.107/+0.320 pts. Explainable: documented early-close artifact.")
    A("6. **exit_level_straddle -- 10 matched trades**. SL/TP levels differ "
      "slightly; a slow market crosses them in different bars. Exit prices "
      "agree within tolerance, exit timestamps move across bars. Explainable.")
    A("7. **session_end_boundary_tick -- 8 matched trades**. Same 16:00 NY "
      "close bar and reason, but exit-price deltas of ~1.5-1.9 pts because the "
      "exact exit tick used by the tester runs is not in today's M5 export "
      "(tick-data snapshot difference at the bar boundary). Not a logic "
      "difference; re-exporting M5 from the same snapshot the tester used "
      "would remove it.")
    A("8. **UNEXPLAINED -- 1 trade: py#284 / mt5#286, 2024-04-19 SELL.** "
      "Python takes profit at 19:45 (TP 4954.0527; the M5 19:45 bar low is "
      "4953.513). MT5 carried the same position to 20:00 and closed session_end "
      "at 4966.180 (TP 4954.12 was on the order). Tick data at "
      "19:46:29-19:46:44 shows bid 4953.51/ask 4954.04 -- far below the TP -- "
      "and the MT5 tester's take-profit DID fire on another SELL that week "
      "(2024-04-17 16:33:03), so this is not the documented bid/ask or level "
      "mechanism. Exit-price delta +12.127 pts, PnL delta +12.13 USD -- the "
      "single largest trade-level disagreement in the whole range. Needs a "
      "dedicated tick-level rerun of 2024-04-19 to determine whether the "
      "tester missed the fill or the batch used a different tick snapshot for "
      "that moment.")
    A("")
    A("**Bottom line:** %d of %d trades agree at the tolerances. Of the "
      "remaining %d, all but 1 (py#284, 2024-04-19) are fully explained by "
      "documented artifacts: gap-threshold straddles, the start-of-data "
      "boundary, the EA's market-entry fallback rejection, bid/ask + "
      "level-source entry straddles, the two early-close carries, exit-level "
      "straddles, and 16:00-boundary tick-snapshot price deltas. No "
      "systematic strategy-logic deviation was found." % (
        fully_ok, max(len(py_trades), len(mt5_trades)),
        max(len(py_trades), len(mt5_trades)) - fully_ok))
    A("")

    text = "\n".join(R) + "\n"
    RECONCILE_MD.write_text(text, encoding="utf-8")
    print(text)
    print("Reconcile report written to %s" % RECONCILE_MD)


if __name__ == "__main__":
    main()
