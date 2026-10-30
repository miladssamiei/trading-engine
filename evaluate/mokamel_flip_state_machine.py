# -*- coding: utf-8 -*-
"""
mokamel_flip_state_machine.py — Hypothesis #16 MOKAMEL_FLIP.

Isolated, independently testable implementation of the base-candle state
machine declared in SPEC_MOKAMEL_FLIP.txt section 3 (T1-T6 candle-close
transitions + P1-P7 M1 flip phase), frozen 2026-09-03 BEFORE this code was
written. The SPEC and GATE files are historical records and are never
modified by this code.

Conventions (all pre-declared in the SPEC):
- M5 signal tape: data/results/orb_2021_present/us500tick_m5.csv
- M1 flip tape:   data/results/vwap_2021_2026/us500tick_m1.csv
- Window: NY wall minutes [570, 870) on cash days (a cash day is an NY date
  whose wall-570 M5 bar exists). Per-day reset. Strict inequalities on all
  level breaches. I-7 / I-8 / I-9 exactly as frozen in the GATE.
- Entries are AGGRESSIVE fills (SPEC P4): level fill at R.low (short) /
  R.high (long), or the M1 open when the bar opens beyond the level.

Usage:
  python mokamel_flip_state_machine.py --days 20230306 20230310 --verbose
  python mokamel_flip_state_machine.py --run-full
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\Milad\Desktop\RR")
M5 = ROOT / "data/results/orb_2021_present/us500tick_m5.csv"
M1 = ROOT / "data/results/vwap_2021_2026/us500tick_m1.csv"

WIN_LO, WIN_HI = 570, 870


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


def atr14(h, l, c):
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy(float)


def load_tapes():
    m5, m1 = Tape(M5, 300), Tape(M1, 60)
    m5.atr = atr14(m5.h, m5.l, m5.c)
    return m5, m1


def day_bounds(tape):
    """dict nd -> (start, end) index slice of the tape's bars for that date."""
    out = {}
    b = np.flatnonzero(np.r_[True, np.diff(tape.nd) != 0, True])
    for k in range(len(b) - 1):
        out[int(tape.nd[b[k]])] = (int(b[k]), int(b[k + 1]))
    return out


def cash_day_windows(m5, m5_days):
    """[(nd, ws, we)] index slices of in-window [570,870) M5 candles on cash
    days (wall-570 bar exists)."""
    out = []
    for nd, (s, e) in sorted(m5_days.items()):
        w = m5.wall[s:e]
        if not (w == 570).any():
            continue
        m = (w >= WIN_LO) & (w < WIN_HI)
        if m.any():
            idx = np.flatnonzero(m)
            out.append((nd, s + int(idx[0]), s + int(idx[-1]) + 1))
    return out


def m1_day_slice(m1, m1_days, nd):
    s, e = m1_days.get(nd, (0, 0))
    if e <= s:
        return 0, 0
    w = m1.wall[s:e]
    m = (w >= WIN_LO) & (w < WIN_HI)
    if not m.any():
        return s, s
    idx = np.flatnonzero(m)
    return s + int(idx[0]), s + int(idx[-1]) + 1


def run_day(m5, m1, nd, ws, we, j_lo, j_hi):
    """One cash day of the state machine. Returns (rows, signals, tallies).
    rows: one per in-window M5 candle (audit trace, SPEC 3.6);
    signals: one per AGGRESSIVE-fill flip entry (SPEC P3/P4)."""
    tal = {"P2_same_bar_guard": 0, "P7_breach_no_probe": 0,
           "orphan_entry_bar": 0, "pending_at_window_end": 0}
    rows, sigs = [], []
    B = None          # base candle index (m5)
    R = None          # pending reference candle index (m5)
    R_h = R_l = 0.0
    r_close = -1
    pend = 0          # 0 none, -1 DOWN (short), +1 UP (long)
    probe = False
    day_times = m5.t[ws:we]
    jm = j_lo

    for k in range(ws, we):
        ev = []
        kc = m5.t[k] + 300
        flip_conf = False
        while jm < j_hi and m1.t[jm] < kc:
            if pend != 0 and m1.t[jm] >= r_close:
                mh, ml, mo = float(m1.h[jm]), float(m1.l[jm]), float(m1.o[jm])
                if pend == -1:
                    prb, brk = mh > R_h, ml < R_l
                else:
                    prb, brk = ml < R_l, mh > R_h
                if prb and brk:
                    probe = True
                    tal["P2_same_bar_guard"] += 1
                    ev.append("P2 same-bar probe+breach -> NO entry (I-9)")
                elif prb:
                    if not probe:
                        ev.append("P1 probe set")
                    probe = True
                elif brk:
                    if probe:
                        if pend == -1:
                            fill = R_l if mo >= R_l else mo
                        else:
                            fill = R_h if mo <= R_h else mo
                        ti = int(np.searchsorted(day_times, m1.t[jm],
                                                 side="right")) - 1
                        if ti >= 0 and day_times[ti] <= m1.t[jm] < day_times[ti] + 300:
                            tk = ws + ti
                            sigs.append(dict(
                                ny_date=nd, entry_ts=int(m1.t[jm]),
                                entry_wall=int(m1.wall[jm]), dir=int(pend),
                                fill=fill, r_id=int(m5.t[R]),
                                r_high=R_h, r_low=R_l,
                                base_id=int(m5.t[B]) if B is not None else -1,
                                trigger_ts=int(m5.t[tk]), m5_idx_trigger=tk,
                                m5_idx_r=R))
                            ev.append("P3/P4 flip %s entry @ %.2f"
                                      % ("SHORT" if pend == -1 else "LONG", fill))
                            flip_conf = True
                            pend = 0
                            R = None
                            probe = False
                        else:
                            tal["orphan_entry_bar"] += 1
                            ev.append("entry bar outside M5 grid -> skipped, tallied")
                            pend = 0
                            R = None
                            probe = False
                    else:
                        tal["P7_breach_no_probe"] += 1
                        ev.append("P7 breach without prior probe -> no entry")
            jm += 1

        o_, h_, l_, c_ = (float(m5.o[k]), float(m5.h[k]),
                          float(m5.l[k]), float(m5.c[k]))
        if B is None:
            B = k
            reason = "INIT new base - no prior base this day"
        elif pend != 0:
            bh, bl = float(m5.h[B]), float(m5.l[B])
            if c_ > bh or c_ < bl:
                reason = "T1 cancelled - close beyond base without flip; base reset to this candle"
                B = k
                pend = 0
                R = None
                probe = False
            else:
                reason = "HOLD pending continues (base frozen, I-8)"
        else:
            bh, bl, bc = float(m5.h[B]), float(m5.l[B]), float(m5.c[B])
            inside = (h_ <= bh) and (l_ >= bl)
            if c_ > bh or c_ < bl:
                B = k
                reason = "T2 reset - close beyond base without flip; new base"
            elif h_ <= bh and c_ < bc:
                R, pend, probe = k, -1, False
                R_h, R_l = float(m5.h[k]), float(m5.l[k])
                r_close = m5.t[k] + 300
                if inside:
                    B = k
                    reason = "T3 pending DOWN set (close<base close, base high not breached); base tightened to this candle"
                else:
                    reason = "T3 pending DOWN set (close<base close, base high not breached); base retained (low wick-breach, I-7)"
            elif l_ >= bl and c_ > bc:
                R, pend, probe = k, 1, False
                R_h, R_l = float(m5.h[k]), float(m5.l[k])
                r_close = m5.t[k] + 300
                if inside:
                    B = k
                    reason = "T4 pending UP set (close>base close, base low not breached); base tightened to this candle"
                else:
                    reason = "T4 pending UP set (close>base close, base low not breached); base retained (high wick-breach, I-7)"
            elif inside and c_ == bc:
                B = k
                reason = "T5 flat inside close - rolling base update, no arm"
            else:
                reason = "T6 no state change (close inside, wick breached an extreme, I-7)"

        rows.append(dict(
            ny_date=nd, time=int(m5.t[k]), wall=int(m5.wall[k]),
            open=o_, high=h_, low=l_, close=c_,
            current_base_id=int(m5.t[B]) if B is not None else -1,
            pending_direction={0: "", -1: "DOWN", 1: "UP"}[pend],
            flip_confirmed=bool(flip_conf),
            state_transition_reason="; ".join(ev + [reason]) if ev else reason))

    if pend != 0:
        tal["pending_at_window_end"] += 1
    return rows, sigs, tal


def run_all(m5, m1, day_list=None):
    """Run the machine over every cash day (or a subset). Returns
    (rows_df, sigs_df, tallies_dict)."""
    m5_days = day_bounds(m5)
    m1_days = day_bounds(m1)
    windows = cash_day_windows(m5, m5_days)
    if day_list is not None:
        want = set(int(d) for d in day_list)
        windows = [w for w in windows if w[0] in want]
    all_rows, all_sigs = [], []
    tal = {"cash_days_run": len(windows)}
    for nd, ws, we in windows:
        j_lo, j_hi = m1_day_slice(m1, m1_days, nd)
        rows, sigs, t = run_day(m5, m1, nd, ws, we, j_lo, j_hi)
        all_rows.extend(rows)
        all_sigs.extend(sigs)
        for kk, vv in t.items():
            tal[kk] = tal.get(kk, 0) + vv
    return pd.DataFrame(all_rows), pd.DataFrame(all_sigs), tal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs=2, type=int, metavar=("FIRST", "LAST"),
                    help="yyyymmdd range dumped verbosely (manual verification)")
    args = ap.parse_args()
    m5, m1 = load_tapes()
    if args.days:
        d0, d1 = args.days
        m5_days = day_bounds(m5)
        m1_days = day_bounds(m1)
        windows = [w for w in cash_day_windows(m5, m5_days) if d0 <= w[0] <= d1]
        print(f"manual-verification dump: {len(windows)} cash days in "
              f"[{d0}, {d1}] (pre-selected BEFORE any results were seen)")
        for nd, ws, we in windows:
            j_lo, j_hi = m1_day_slice(m1, m1_days, nd)
            rows, sigs, tal = run_day(m5, m1, nd, ws, we, j_lo, j_hi)
            print(f"\n===== {nd}  (M5 window candles {ws}:{we}, M1 bars {j_lo}:{j_hi}) =====")
            print("wall     open      high      low       close     | base      pend   flip reason")
            for r in rows:
                print(f"{r['wall']:04d} {r['open']:9.2f} {r['high']:9.2f} "
                      f"{r['low']:9.2f} {r['close']:9.2f} | {r['current_base_id']} "
                      f"{r['pending_direction']:>4}  {str(r['flip_confirmed']):>5}  "
                      f"{r['state_transition_reason']}")
            for s in sigs:
                print(f"  SIGNAL {s['ny_date']} wall {s['entry_wall']:04d} "
                      f"dir {s['dir']:+d} fill {s['fill']:.2f} "
                      f"R(hi/lo) {s['r_high']:.2f}/{s['r_low']:.2f} "
                      f"trigger_ts {s['trigger_ts']}")
            print(f"  tallies: {tal}")


if __name__ == "__main__":
    main()
