# -*- coding: utf-8 -*-
"""
vwap_mfe_mae.py — Step 1 + Step 2 (Phase 1) of the VWAP stop/target
exploration.  EVERYTHING IN PHASE 1 IS EXPLORATORY / IN-SAMPLE.

Step 1: for each of the 281 Python-reference VWAP trades, walk the M1 bar
        path from the entry bar to the trade's ACTUAL exit bar and compute
        MFE / MAE in % of entry price (the strategy's own terms):
          long : MFE = max(high)/entry-1, MAE = 1-min(low)/entry
          short: MFE = 1-max(high)/entry, MAE = min(low)/entry-1
        (both recorded as non-negative magnitudes.)
        -> data/results/vwap_2021_2026/mfe_mae.csv
        A second pair (mfe_eod/mae_eod) extends the walk to the 15:55
        forced-exit bar — how far the trade COULD have travelled under a
        different exit rule.

Step 2: reconstruct, per trade, what each (stop%, target-fraction) cell of
        stop grid {0.10, 0.15, 0.20, 0.25}% x target {50, 75, 100, 125}% of
        the way back to the MOVING VWAP would have produced, using the
        bar-by-bar path order (which level crossed first matters), with the
        SAME exit conventions as evaluate/vwap_reference.py:
          * exits checked from the bar AFTER the entry bar (the reference
            engine's convention — the MT5 EA differs here, see RECONCILE);
          * target during bar T = entry + f*(vwap_through[T-60] - entry);
          * stop = norm_price(entry*(1 -+ stop%)) on the 0.01 trade grid;
          * first bar with NY wall >= 15:55 -> time exit at its open;
            if the session ends earlier (early close) -> exit at the last
            session bar's close (time_eod);
          * same-bar both-level ambiguity resolved by the reference's
            close>=open path heuristic.
        -> data/results/vwap_rr_exploration/phase1_grid.csv

        VALIDATION: cell (0.20%, 100%) must reproduce the original 281
        trades' outcome table (win rate 41.64%, total PnL -202.42) — it
        replays the identical rule on the identical paths.

        DOCUMENTED APPROXIMATION (Phase 1): entries are held fixed at the
        original rule's entry points.  A different exit rule changes when
        the strategy is next flat and therefore which LATER entries would
        have existed; later trades may also overlap in book terms.  This is
        a counterfactual per-trade table, NOT a re-backtest.  Path also
        truncates at the 15:55 bar of the entry day (max life any config
        could have), NOT at the original exit — a looser stop must be
        allowed to survive past the original 0.2% stop hit.
"""

import csv
import datetime as dt
import math
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\vwap_2021_2026")
OUT = Path(r"C:\Users\Milad\Desktop\RR\data\results\vwap_rr_exploration")
M1_CSV = BASE / "us500tick_m1.csv"
TRADES_CSV = BASE / "py_trades.csv"

NY = ZoneInfo("America/New_York")
UTC = dt.timezone.utc
SESSION_OPEN_M = 570
SESSION_LAST_M = 959
FORCE_EXIT_M = 955
TICK_GRID = 0.01

STOPS = [0.10, 0.15, 0.20, 0.25]
FRACS = [0.50, 0.75, 1.00, 1.25]


def norm_price(p):
    return round(round(p / TICK_GRID) * TICK_GRID, 2)


def parse_ts(s):
    return int(dt.datetime.strptime(s, "%Y.%m.%d %H:%M")
               .replace(tzinfo=UTC).timestamp())


def fmt_ts(e):
    return dt.datetime.fromtimestamp(e, tz=UTC).strftime("%Y.%m.%d %H:%M")


def load_trades():
    trades = []
    with open(TRADES_CSV, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            e = parse_ts(r["open_time"])
            ny = dt.datetime.fromtimestamp(e, tz=UTC).astimezone(NY)
            trades.append({
                "ticket": int(r["ticket"]),
                "entry_ts": e,
                "exit_ts": parse_ts(r["close_time"]),
                "type": r["type"].strip(),
                "entry": float(r["open_price"]),
                "orig_reason": r["exit_reason"].strip(),
                "orig_pnl": float(r["profit"]),
                "ny_date": ny.year * 10000 + ny.month * 100 + ny.day,
            })
    trades.sort(key=lambda t: t["entry_ts"])
    return trades


def load_session_bars(want_dates):
    """Session M1 bars (wall 570..959) for the NY dates that carry trades."""
    by_date = defaultdict(list)
    with open(M1_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            t = int(row["time"])
            ny = dt.datetime.fromtimestamp(t, tz=UTC).astimezone(NY)
            d = ny.year * 10000 + ny.month * 100 + ny.day
            if d not in want_dates:
                continue
            wall = ny.hour * 60 + ny.minute
            if SESSION_OPEN_M <= wall <= SESSION_LAST_M:
                by_date[d].append({
                    "time": t, "open": float(row["open"]),
                    "high": float(row["high"]), "low": float(row["low"]),
                    "close": float(row["close"]), "vol": int(row["tick_volume"]),
                    "wall": wall,
                })
    for d in by_date:
        by_date[d].sort(key=lambda b: b["time"])
    return by_date


def vwap_series(bars):
    cpv = cv = 0.0
    after = {}
    for b in bars:
        typ = (b["high"] + b["low"] + b["close"]) / 3.0
        cpv += typ * b["vol"]
        cv += b["vol"]
        after[b["time"]] = (cpv / cv) if cv > 0 else None
    return after


def reconstruct(bars, ei, cutoff_i, entry, is_buy, stop_pct, f, vwap_after):
    """Exit reconstruction for one cell on one trade's path.
    Returns (exit_price, reason, pnl)."""
    sign = 1 if is_buy else -1
    sl = norm_price(entry * (1.0 - sign * stop_pct / 100.0))
    for k in range(ei + 1, cutoff_i + 1):
        b = bars[k]
        if b["wall"] >= FORCE_EXIT_M:
            px = b["open"]
            return px, "time", round(sign * (px - entry), 2)
        vprev = vwap_after.get(b["time"] - 60)
        target = None if vprev is None else entry + f * (vprev - entry)
        if is_buy:
            if b["open"] <= sl:
                return b["open"], "stop", round(b["open"] - entry, 2)
            if target is not None and b["open"] >= target:
                return b["open"], "touch", round(b["open"] - entry, 2)
            lo = b["low"] <= sl
            hi = target is not None and b["high"] >= target
            if lo and hi:
                if b["close"] >= b["open"]:
                    return sl, "stop", round(sl - entry, 2)
                return target, "touch", round(target - entry, 2)
            if lo:
                return sl, "stop", round(sl - entry, 2)
            if hi:
                return target, "touch", round(target - entry, 2)
        else:
            if b["open"] >= sl:
                return b["open"], "stop", round(entry - b["open"], 2)
            if target is not None and b["open"] <= target:
                return b["open"], "touch", round(entry - b["open"], 2)
            hi = b["high"] >= sl
            lo = target is not None and b["low"] <= target
            if hi and lo:
                if b["close"] >= b["open"]:
                    return target, "touch", round(entry - target, 2)
                return sl, "stop", round(entry - sl, 2)
            if hi:
                return sl, "stop", round(entry - sl, 2)
            if lo:
                return target, "touch", round(entry - target, 2)
    # session ended before 15:55 (early close) with the position open
    px = bars[cutoff_i]["close"]
    return px, "time_eod", round(sign * (px - entry), 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    trades = load_trades()
    print(f"trades loaded: {len(trades)}")
    by_date = load_session_bars({t["ny_date"] for t in trades})
    print(f"trade dates with session bars: {len(by_date)}")

    vwap_cache = {}

    mfe_rows = []
    # grid[cell] = list of (pnl, reason)
    grid = defaultdict(list)

    for t in trades:
        bars = by_date.get(t["ny_date"])
        if not bars:
            print(f"!! trade #{t['ticket']}: no session bars for {t['ny_date']}")
            continue
        if t["ny_date"] not in vwap_cache:
            vwap_cache[t["ny_date"]] = vwap_series(bars)
        vwap_after = vwap_cache[t["ny_date"]]
        idx = {b["time"]: i for i, b in enumerate(bars)}
        ei = idx.get(t["entry_ts"])
        xi = idx.get(t["exit_ts"])
        cutoff_i = next((i for i, b in enumerate(bars)
                         if b["wall"] >= FORCE_EXIT_M), len(bars) - 1)
        if ei is None or xi is None:
            print(f"!! trade #{t['ticket']}: entry/exit bar not in session "
                  f"(ei={ei} xi={xi})")
            continue

        is_buy = t["type"] == "BUY"
        highs = [b["high"] for b in bars[ei:xi + 1]]
        lows = [b["low"] for b in bars[ei:xi + 1]]
        if is_buy:
            mfe = (max(highs) / t["entry"] - 1.0) * 100.0
            mae = (1.0 - min(lows) / t["entry"]) * 100.0
            mfe_b = bars[ei + max(range(len(highs)), key=lambda i: highs[i])]["time"]
            mae_b = bars[ei + min(range(len(lows)), key=lambda i: lows[i])]["time"]
        else:
            mfe = (1.0 - min(lows) / t["entry"]) * 100.0
            mae = (max(highs) / t["entry"] - 1.0) * 100.0
            mfe_b = bars[ei + min(range(len(lows)), key=lambda i: lows[i])]["time"]
            mae_b = bars[ei + max(range(len(highs)), key=lambda i: highs[i])]["time"]

        highs2 = [b["high"] for b in bars[ei:cutoff_i + 1]]
        lows2 = [b["low"] for b in bars[ei:cutoff_i + 1]]
        if is_buy:
            mfe_eod = (max(highs2) / t["entry"] - 1.0) * 100.0
            mae_eod = (1.0 - min(lows2) / t["entry"]) * 100.0
        else:
            mfe_eod = (1.0 - min(lows2) / t["entry"]) * 100.0
            mae_eod = (max(highs2) / t["entry"] - 1.0) * 100.0

        mfe_rows.append({
            "ticket": t["ticket"], "ny_date": t["ny_date"], "type": t["type"],
            "open_time": fmt_ts(t["entry_ts"]), "entry_price": f"{t['entry']:.4f}",
            "orig_exit_reason": t["orig_reason"],
            "orig_exit_time": fmt_ts(t["exit_ts"]),
            "orig_pnl": f"{t['orig_pnl']:.2f}",
            "mfe_pct": f"{mfe:.4f}", "mae_pct": f"{mae:.4f}",
            "mfe_bar": fmt_ts(mfe_b), "mae_bar": fmt_ts(mae_b),
            "bars_to_exit": xi - ei + 1,
            "mfe_eod_pct": f"{mfe_eod:.4f}", "mae_eod_pct": f"{mae_eod:.4f}",
            "bars_eod": cutoff_i - ei + 1,
        })

        for sp in STOPS:
            for f in FRACS:
                px, reason, pnl = reconstruct(bars, ei, cutoff_i, t["entry"],
                                              is_buy, sp, f, vwap_after)
                grid[(sp, f)].append((pnl, reason))

    # ---- write mfe_mae.csv --------------------------------------------
    cols = ["ticket", "ny_date", "type", "open_time", "entry_price",
            "orig_exit_reason", "orig_exit_time", "orig_pnl",
            "mfe_pct", "mae_pct", "mfe_bar", "mae_bar", "bars_to_exit",
            "mfe_eod_pct", "mae_eod_pct", "bars_eod"]
    with open(BASE / "mfe_mae.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter=";")
        w.writeheader()
        w.writerows(mfe_rows)
    print(f"mfe_mae.csv written: {len(mfe_rows)} rows")

    # ---- grid table ----------------------------------------------------
    lines = ["stop_pct;target_frac;n;wins;winrate_pct;avg_win;avg_loss;"
             "expectancy;total_pnl;exits_stop;exits_touch;exits_time;"
             "exits_time_eod;avg_hold_bars_to_exit"]
    cell_stats = {}
    for sp in STOPS:
        for f in FRACS:
            res = grid[(sp, f)]
            pnls = [p for p, _ in res]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            n = len(res)
            wr = 100.0 * len(wins) / n
            aw = (sum(wins) / len(wins)) if wins else 0.0
            al = (sum(losses) / len(losses)) if losses else 0.0
            exp = sum(pnls) / n
            reasons = defaultdict(int)
            for _, r in res:
                reasons[r] += 1
            cell_stats[(sp, f)] = {
                "n": n, "winrate_pct": wr, "avg_win": aw, "avg_loss": al,
                "expectancy": exp, "total": sum(pnls), "reasons": dict(reasons),
            }
            lines.append(";".join([
                f"{sp:.2f}", f"{f:.2f}", str(n), str(len(wins)),
                f"{wr:.2f}", f"{aw:.2f}", f"{al:.2f}", f"{exp:.3f}",
                f"{sum(pnls):.2f}", str(reasons["stop"]), str(reasons["touch"]),
                str(reasons["time"]), str(reasons["time_eod"]), "-"]))
    with open(OUT / "phase1_grid.csv", "w", newline="") as fh:
        fh.write("\r\n".join(lines) + "\r\n")
    print(f"phase1_grid.csv written -> {OUT / 'phase1_grid.csv'}")

    # ---- console table + validation ------------------------------------
    print()
    print("=" * 96)
    print("Phase 1 — EXPLORATORY reconstruction on existing trade paths, NOT a validated backtest")
    print("=" * 96)
    print(f"{'stop%':>6} {'target':>7} {'n':>4} {'win%':>7} {'avgW':>7} {'avgL':>8}"
          f" {'EXPT':>8} {'total':>9}  {'stop/touch/time/eod':>22}")
    for sp in STOPS:
        for f in FRACS:
            c = cell_stats[(sp, f)]
            r = c["reasons"]
            print(f"{sp:>6.2f} {f:>6.0%} {c['n']:>4} {c['winrate_pct']:>7.2f}"
                  f" {c['avg_win']:>7.2f} {c['avg_loss']:>8.2f}"
                  f" {c['expectancy']:>8.3f} {c['total']:>9.2f}"
                  f"  {r.get('stop',0):>4}/{r.get('touch',0):>4}/"
                  f"{r.get('time',0):>2}/{r.get('time_eod',0):>2}")
    print()

    v = cell_stats[(0.20, 1.00)]
    print("VALIDATION cell (stop 0.20%, target 100% = original rule):")
    print(f"  n={v['n']} (original 281)  winrate={v['winrate_pct']:.2f}% "
          f"(original 41.64%)  total={v['total']:.2f} (original -202.42)  "
          f"expectancy={v['expectancy']:.3f} (original -0.720)")
    match = (v["n"] == 281 and abs(v["winrate_pct"] - 41.64) < 0.05
             and abs(v["total"] + 202.42) < 0.5)
    print(f"  reproduction: {'EXACT/OK' if match else 'MISMATCH — investigate'}")

    # ---- MFE/MAE summary stats -----------------------------------------
    import statistics as st
    mfeas = [float(r["mfe_pct"]) for r in mfe_rows]
    maes = [float(r["mae_pct"]) for r in mfe_rows]
    print()
    print("MFE/MAE (to original exit) summary, % of entry:")
    print(f"  MFE: mean {st.mean(mfeas):.3f}  median {st.median(mfeas):.3f}  "
          f"p90 {sorted(mfeas)[int(0.9*len(mfeas))]:.3f}")
    print(f"  MAE: mean {st.mean(maes):.3f}  median {st.median(maes):.3f}  "
          f"p90 {sorted(maes)[int(0.9*len(maes))]:.3f}")
    for thr in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        n_mfe = sum(1 for x in mfeas if x >= thr)
        n_mae = sum(1 for x in maes if x >= thr)
        print(f"  reached +/-{thr:.2%}: MFE {n_mfe:>3}/{len(mfeas)}   "
              f"MAE {n_mae:>3}/{len(maes)}")


if __name__ == "__main__":
    main()
