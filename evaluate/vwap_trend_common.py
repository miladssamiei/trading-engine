# -*- coding: utf-8 -*-
"""
vwap_trend_common.py — shared engine for the VWAP trend-pullback Step 3
work: session loading, session VWAP, drift state machine, episode/signal
extraction. Signal semantics are IDENTICAL to Phase 1
(evaluate/vwap_trend_signal_quality.py); see that header + REPORT.md for
the documented ambiguity resolutions.
"""

import csv
import datetime as dt
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\vwap_trend_pullback_signal_quality")
M5_CSV = BASE / "us500tick_m5_vol.csv"

NY = ZoneInfo("America/New_York")
UTC = dt.timezone.utc

SESSION_OPEN_M = 570
SESSION_LAST_M = 959
FORCE_EXIT_M = 955
LAST_ENTRY_M = 930
SLOPE_REF_S = 900
CHANGE_REF_S = 3600
CHANGE_PCT = 0.1
TICK_GRID = 0.01


def norm_price(p):
    return round(round(p / TICK_GRID) * TICK_GRID, 2)


def load_sessions():
    by_date = defaultdict(list)
    with open(M5_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            t = int(row["time"])
            ny = dt.datetime.fromtimestamp(t, tz=UTC).astimezone(NY)
            wall = ny.hour * 60 + ny.minute
            if wall < SESSION_OPEN_M or wall > SESSION_LAST_M:
                continue
            by_date[ny.year * 10000 + ny.month * 100 + ny.day].append({
                "time": t, "open": float(row["open"]),
                "high": float(row["high"]), "low": float(row["low"]),
                "close": float(row["close"]), "vol": int(row["tick_volume"]),
                "wall": wall,
            })
    for d in by_date:
        by_date[d].sort(key=lambda b: b["time"])
    return dict(by_date)


def day_signal_indexes(bars):
    """Signal extraction identical to Phase 1. Returns a list of
    {"dir", "trig_idx", "entry_idx"} in time order, plus walk_end index
    (last bar opening before 15:55). Days must have the 09:30 bar."""
    cum_pv = cum_v = 0.0
    vwap_at = {}
    close_at = {}
    drifts = []
    for i, b in enumerate(bars):
        typ = (b["high"] + b["low"] + b["close"]) / 3.0
        if b["vol"] > 0:
            cum_pv += typ * b["vol"]
            cum_v += b["vol"]
        vwap_at[b["time"]] = (cum_pv / cum_v) if cum_v > 0 else None
        close_at[b["time"]] = b["close"]
        # drift conditions (a),(b),(c) at bar i
        D = None
        t_slope = b["time"] - SLOPE_REF_S
        t_chg = b["time"] - CHANGE_REF_S
        if t_slope in vwap_at and t_chg in close_at \
                and vwap_at[b["time"]] is not None:
            chg = b["close"] / close_at[t_chg] - 1.0
            if chg >= CHANGE_PCT / 100.0:
                if b["close"] > vwap_at[b["time"]] \
                        and vwap_at[b["time"]] > vwap_at[t_slope]:
                    D = "up"
            elif chg <= -CHANGE_PCT / 100.0:
                if b["close"] < vwap_at[b["time"]] \
                        and vwap_at[b["time"]] < vwap_at[t_slope]:
                    D = "down"
        drifts.append(D)

    walk_end = max(i for i, b in enumerate(bars) if b["wall"] < FORCE_EXIT_M)
    signals = []
    ep_dir = None
    ep_entered = False
    for i, b in enumerate(bars):
        if ep_dir is not None and not ep_entered and i + 1 < len(bars):
            nxt = bars[i + 1]
            if SESSION_OPEN_M + 60 <= nxt["wall"] <= LAST_ENTRY_M:
                against = (b["close"] < b["open"]) if ep_dir == "up" \
                    else (b["close"] > b["open"])
                if against:
                    signals.append({"dir": ep_dir, "trig_idx": i,
                                    "entry_idx": i + 1})
                    ep_entered = True
        D = drifts[i]
        if D is None:
            ep_dir, ep_entered = None, False
        elif D != ep_dir:
            ep_dir, ep_entered = D, False
    return signals, walk_end


def prepare_days():
    """All session days with signals + walk context. Skips days without the
    09:30 bar (consistent with Phase 1: 0 such days existed)."""
    by_date = load_sessions()
    days = []
    for ny_date in sorted(by_date):
        bars = by_date[ny_date]
        if not bars or bars[0]["wall"] != SESSION_OPEN_M:
            continue
        signals, walk_end = day_signal_indexes(bars)
        idx955 = next((i for i, b in enumerate(bars) if b["wall"] == FORCE_EXIT_M),
                      None)
        days.append({"ny_date": ny_date, "bars": bars, "signals": signals,
                     "walk_end": walk_end, "idx955": idx955})
    return days
