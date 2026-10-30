"""
Standalone reference implementation of the VWAP mean-reversion strategy.
Mirrors ea/VWAP_MeanReversion.mq5 exactly so the two CSV outputs can be
diffed line by line.

Input : us500tick_m1.csv   1-minute BID bars of "US500-tick"
                             columns: time,open,high,low,close,tick_volume
                             time = bar-open UTC epoch seconds (int)
Output: py_trades.csv       one row per simulated trade
        py_days.csv         one row per NY calendar date in the data

Pure pandas + stdlib; independent of all bridge/ensemble code.
Bar timestamps are UTC; a "day" is a New York calendar date (zoneinfo
handles the DST switches generically).

Timing conventions (identical to the EA):
  * Session = NY wall minutes [09:30, 16:00) = [570, 960).
  * VWAP accumulates over COMPLETED session bars only (no look-ahead):
    vwap_after(bar_i) = sum(typical*v) / sum(v) through bar i,
    typical = (H+L+C)/3, v = tick_volume.
  * Signal evaluated on completed bar S (while flat); market entry at
    the OPEN of the next bar E = S+1min. Entries stop when the entry
    bar opens after 15:30 NY (wall(E) <= 930).
  * 30-min VWAP slope: exact-timestamp reference S.time-1800s must
    exist IN THE SAME SESSION (missing -> no signal, no slope count).
  * Stop loss 0.2% beyond entry away from VWAP (rounded to the 0.01
    trade-tick grid like the EA's NormPrice).
  * Take profit = touch of the moving VWAP: while the position is open
    during bar T, the target is vwap_after(T-1min) (the value of the
    last completed bar at that time — this is what the tick-monitored
    EA uses). BUY closes when price rises to target, SELL when it
    falls to target.
  * Forced exit at 15:55 NY: first bar with wall >= 955 -> exit at its
    open, reason time_exit. If the session data ends earlier (early
    close), the position is carried to the first bar of the NEXT NY
    date (what the EA's next-day force-close does) and exited there
    (reason time_exit; flagged separately in the summary as carries).
"""

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

# ---------------------------------------------------------------------------
# Strategy parameters (MUST equal the MT5 EA input defaults)
# ---------------------------------------------------------------------------
DEVIATION_PCT = 0.25          # entry deviation from VWAP (%)
STOP_PCT = 0.2                # stop distance beyond entry (%)
RANGE_NET_PCT = 0.6           # range-day max |net move| from open (%)
RANGE_HL_PCT = 1.0            # range-day max high-low (%)
MAXMOVE_MODE = 0              # 0=percent, 1=points
MAXMOVE_PCT = 0.42            # percent mode (%)
MAXMOVE_POINTS = 30.0         # points mode
SLOPE_WINDOW_MIN = 30         # VWAP slope window (minutes)
SLOPE_MIN_PCT = 0.0           # long: slope >= x; short: slope <= -x
LOT = 1.0
CONTRACT_SIZE = 1.0
SYMBOL = "US500-tick"
TICK_GRID = 0.01              # symbol trade tick grid (NormPrice)

SESSION_OPEN_M = 9 * 60 + 30        # 570
SESSION_LAST_M = 16 * 60 - 1        # 959 (last session bar wall)
FORCE_EXIT_M = 15 * 60 + 55         # 955
LAST_ENTRY_M = 15 * 60 + 30         # 930 (entry-bar open cutoff)

NY = ZoneInfo("America/New_York")
UTC = timezone.utc

TRADES_HEADER = ("ticket;open_time;close_time;symbol;type;lot;open_price;"
                 "stop_loss;take_profit;close_price;exit_reason;swap;"
                 "commission;profit;dev_pct;slope_pct")
DAYS_HEADER = ("ny_date;session_open;session_high;session_low;net_pct_final;"
               "hl_pct_final;range_ok_final;skipped;skip_reason;slope_ok_bars;"
               "entry_signals;entries;exits_tp;exits_sl;exits_time;total_pnl")

SKIP_REASONS = ["no_session_open"]


def parse_args():
    p = argparse.ArgumentParser(description="Standalone VWAP MR reference.")
    p.add_argument("--data-csv",
                   default=r"C:\Users\Milad\Desktop\RR\data\results\vwap_2021_2026\us500tick_m1.csv")
    p.add_argument("--out-dir",
                   default=r"C:\Users\Milad\Desktop\RR\data\results\vwap_2021_2026")
    p.add_argument("--window-start", default="2021-01-01",
                   help="window start date, YYYY-MM-DD (UTC)")
    return p.parse_args()


def fmt_utc(ts):
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y.%m.%d %H:%M")


def fmt4(x):
    return "" if x is None else f"{x:.4f}"


def dstr(ny_date):
    return "{:04d}.{:02d}.{:02d}".format(ny_date // 10000, (ny_date // 100) % 100, ny_date % 100)


def norm_price(p):
    return round(round(p / TICK_GRID) * TICK_GRID, 2)


def load_bars(csv_path, window_start):
    df = pd.read_csv(csv_path)
    t0 = int(pd.Timestamp(window_start, tz=UTC).timestamp())
    bars = []
    for t, o, h, l, c, v in zip(df["time"], df["open"], df["high"], df["low"],
                                df["close"], df["tick_volume"]):
        t = int(t)
        ny_dt = datetime.fromtimestamp(t, tz=UTC).astimezone(NY)
        if t < t0:
            continue
        bars.append({
            "time": t,
            "open": float(o), "high": float(h), "low": float(l),
            "close": float(c), "vol": int(v),
            "ny_date": ny_dt.year * 10000 + ny_dt.month * 100 + ny_dt.day,
            "wall": ny_dt.hour * 60 + ny_dt.minute,
        })
    bars.sort(key=lambda b: b["time"])
    return bars


def simulate():
    args = parse_args()
    bars = load_bars(args.data_csv, args.window_start)
    print("input bars: {}".format(len(bars)))
    print("input range: {} -> {}".format(
        datetime.fromtimestamp(bars[0]["time"], tz=UTC).strftime("%Y-%m-%d %H:%M UTC"),
        datetime.fromtimestamp(bars[-1]["time"], tz=UTC).strftime("%Y-%m-%d %H:%M UTC")))

    by_date = defaultdict(list)
    for b in bars:
        by_date[b["ny_date"]].append(b)
    all_dates = sorted(by_date)

    trades = []
    day_rows = []
    stats = defaultdict(int)
    ticket = 0

    for di, ny_date in enumerate(all_dates):
        day_bars = by_date[ny_date]
        sess = [b for b in day_bars if SESSION_OPEN_M <= b["wall"] <= SESSION_LAST_M]

        row = {
            "ny_date": ny_date,
            "session_open": None, "session_high": None, "session_low": None,
            "net_pct_final": None, "hl_pct_final": None, "range_ok_final": 0,
            "skipped": 0, "skip_reason": "",
            "slope_ok_bars": 0, "signals": 0, "entries": 0,
            "exits_tp": 0, "exits_sl": 0, "exits_time": 0, "total_pnl": 0.0,
        }

        if not sess:
            day_rows.append(row)            # evening-only day (EA: zeros)
            continue

        if sess[0]["wall"] != SESSION_OPEN_M:
            row["skipped"], row["skip_reason"] = 1, "no_session_open"
            # EA skips the whole day (no VWAP accumulation? no: the EA sets
            # skipped and still accumulates session bars, but never trades;
            # net/hl finals still computed. Mirror that.)
            row["session_open"] = sess[0]["open"]

        session_open = sess[0]["open"]
        row["session_open"] = session_open

        # ---- walk session bars, mirroring the EA state machine ----------
        cum_pv = 0.0
        cum_v = 0.0
        vwap_after = {}          # bar time -> vwap through that bar
        sess_high = sess_low = None
        pos = None               # open position dict
        pending = None           # signal from the last completed bar

        def manage_exits_during(bar, target_vwap):
            """Exits checked DURING `bar` for an open position. Target is the
            vwap through the previous completed session bar (EA convention).
            Returns exit dict or None. Time exit has priority (EA checks
            force-close first on each tick)."""
            nonlocal pos
            if pos is None:
                return None
            w = bar["wall"]
            if w >= FORCE_EXIT_M:
                return {"price": bar["open"], "reason": "time_exit", "bar": bar}
            if pos["dir"] == "BUY":
                sl = pos["sl"]
                if bar["open"] <= sl:
                    return {"price": bar["open"], "reason": "stop_loss", "bar": bar}
                if target_vwap is not None and bar["open"] >= target_vwap:
                    return {"price": bar["open"], "reason": "vwap_touch", "bar": bar}
                low_hits = bar["low"] <= sl
                high_hits = (target_vwap is not None
                             and bar["high"] >= target_vwap)
                if low_hits and high_hits:
                    # path ambiguity: O->L->H (close>=open) => SL first
                    if bar["close"] >= bar["open"]:
                        return {"price": sl, "reason": "stop_loss", "bar": bar}
                    return {"price": target_vwap, "reason": "vwap_touch", "bar": bar}
                if low_hits:
                    return {"price": sl, "reason": "stop_loss", "bar": bar}
                if high_hits:
                    return {"price": target_vwap, "reason": "vwap_touch", "bar": bar}
            else:
                sl = pos["sl"]
                if bar["open"] >= sl:
                    return {"price": bar["open"], "reason": "stop_loss", "bar": bar}
                if target_vwap is not None and bar["open"] <= target_vwap:
                    return {"price": bar["open"], "reason": "vwap_touch", "bar": bar}
                high_hits = bar["high"] >= sl
                low_hits = (target_vwap is not None
                            and bar["low"] <= target_vwap)
                if high_hits and low_hits:
                    if bar["close"] >= bar["open"]:
                        return {"price": target_vwap, "reason": "vwap_touch", "bar": bar}
                    return {"price": sl, "reason": "stop_loss", "bar": bar}
                if high_hits:
                    return {"price": sl, "reason": "stop_loss", "bar": bar}
                if low_hits:
                    return {"price": target_vwap, "reason": "vwap_touch", "bar": bar}
            return None

        def close_position(exit_ev, exit_bar):
            nonlocal pos, ticket
            sign = 1 if pos["dir"] == "BUY" else -1
            profit = round(sign * (exit_ev["price"] - pos["entry"]) * LOT * CONTRACT_SIZE, 2)
            ticket += 1
            trades.append({
                "ticket": ticket,
                "ny_date": ny_date,
                "open_time": pos["entry_bar"]["time"],
                "close_time": exit_bar["time"],
                "type": pos["dir"],
                "entry": pos["entry"],
                "sl": pos["sl"],
                "tp0": pos["tp0"],
                "exit_price": exit_ev["price"],
                "exit_reason": exit_ev["reason"],
                "profit": profit,
                "dev_pct": pos["dev_pct"],
                "slope_pct": pos["slope_pct"],
            })
            row["total_pnl"] = round(row["total_pnl"] + profit, 2)
            if exit_ev["reason"] == "vwap_touch":
                row["exits_tp"] += 1
            elif exit_ev["reason"] == "stop_loss":
                row["exits_sl"] += 1
            else:
                row["exits_time"] += 1
            if exit_ev["reason"] == "time_exit" and exit_bar["ny_date"] != ny_date:
                stats["time_exit_carries"] += 1
            pos = None

        for i, bar in enumerate(sess):
            # -- 1) manage an open position during this bar -----------------
            if pos is not None:
                target = vwap_after.get(bar["time"] - 60)   # vwap through prev bar
                ev = manage_exits_during(bar, target)
                if ev is not None:
                    close_position(ev, bar)
                    pending = None

            # -- 2) accumulate the completed-bar state (this bar) -----------
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
            range_ok = (net_pct <= RANGE_NET_PCT and hl_pct <= RANGE_HL_PCT)

            # -- 3) pending entry executes at THIS bar's open ----------------
            if pending is not None and pos is None:
                entry_bar = bar
                if entry_bar["wall"] <= LAST_ENTRY_M:
                    sign = 1 if pending["dir"] == "BUY" else -1
                    entry = entry_bar["open"]        # bid open (EA fills market)
                    sl = norm_price(entry * (1.0 - sign * STOP_PCT / 100.0))
                    pos = {
                        "dir": pending["dir"], "entry": entry, "sl": sl,
                        "entry_bar": entry_bar, "tp0": pending["vwap"],
                        "dev_pct": pending["dev_pct"], "slope_pct": pending["slope_pct"],
                    }
                    row["entries"] += 1
                pending = None

            # -- 4) evaluate a NEW signal on this completed bar -------------
            if pos is None and not row["skipped"] and vwap_now is not None:
                nxt = sess[i + 1] if i + 1 < len(sess) else None
                if nxt is not None and nxt["wall"] <= LAST_ENTRY_M:
                    ref_t = bar["time"] - SLOPE_WINDOW_MIN * 60
                    v_ref = vwap_after.get(ref_t)
                    if v_ref is not None and v_ref > 0:
                        row["slope_ok_bars"] += 1
                        slope_pct = (vwap_now / v_ref - 1.0) * 100.0
                        dev_pct = (bar["close"] / vwap_now - 1.0) * 100.0
                        move_abs = abs(bar["close"] - session_open)
                        max_move_ok = ((move_abs <= MAXMOVE_POINTS) if MAXMOVE_MODE == 1
                                       else (move_abs <= session_open * MAXMOVE_PCT / 100.0))
                        d = 0
                        if range_ok and max_move_ok:
                            if dev_pct <= -DEVIATION_PCT and slope_pct >= SLOPE_MIN_PCT:
                                d = "BUY"
                            elif dev_pct >= DEVIATION_PCT and slope_pct <= -SLOPE_MIN_PCT:
                                d = "SELL"
                        if d:
                            row["signals"] += 1
                            pending = {"dir": d, "vwap": vwap_now,
                                       "dev_pct": dev_pct, "slope_pct": slope_pct}

            # bookkeeping finals after each bar
            row["session_high"], row["session_low"] = sess_high, sess_low
            row["net_pct_final"], row["hl_pct_final"] = net_pct, hl_pct
            row["range_ok_final"] = 1 if range_ok else 0

        # -- end of session bars: position still open? --------------------
        if pos is not None:
            stats["data_end_open"] += 1
            # EA force-close fires on the first tick of the NEXT NY date
            # (market halted before 15:55). Mirror: first bar after today.
            carry_bar = None
            for nd in all_dates[di + 1:]:
                if by_date[nd]:
                    carry_bar = by_date[nd][0]
                    break
            if carry_bar is None:
                # absolute data end: close at the day's last bar close
                lb = sess[-1]
                close_position({"price": lb["close"], "reason": "time_exit"}, lb)
                stats["data_end_closes"] += 1
            else:
                close_position({"price": carry_bar["open"], "reason": "time_exit"},
                               carry_bar)
                stats["time_exit_carries"] += 1

        row["skipped"] = 1 if row["skipped"] else 0
        day_rows.append(row)

    write_outputs(trades, day_rows, args.out_dir)
    print_summary(trades, day_rows, stats)


def write_outputs(trades, day_rows, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    lines = [TRADES_HEADER]
    for t in trades:
        lines.append(";".join([
            str(t["ticket"]),
            fmt_utc(t["open_time"]),
            fmt_utc(t["close_time"]),
            SYMBOL,
            t["type"],
            f"{LOT:.2f}",
            fmt4(t["entry"]),
            fmt4(t["sl"]),
            fmt4(t["tp0"]),
            fmt4(t["exit_price"]),
            t["exit_reason"],
            "0.00",
            "0.00",
            f"{t['profit']:.2f}",
            f"{t['dev_pct']:.4f}",
            f"{t['slope_pct']:.4f}",
        ]))
    (out / "py_trades.csv").write_text("\r\n".join(lines) + "\r\n",
                                       encoding="ascii", newline="")

    lines = [DAYS_HEADER]
    for r in day_rows:
        lines.append(";".join([
            str(r["ny_date"]),
            fmt4(r["session_open"]),
            fmt4(r["session_high"]),
            fmt4(r["session_low"]),
            fmt4(r["net_pct_final"]),
            fmt4(r["hl_pct_final"]),
            str(r["range_ok_final"]),
            str(r["skipped"]),
            r["skip_reason"],
            str(r["slope_ok_bars"]),
            str(r["signals"]),
            str(r["entries"]),
            str(r["exits_tp"]),
            str(r["exits_sl"]),
            str(r["exits_time"]),
            f"{r['total_pnl']:.2f}",
        ]))
    (out / "py_days.csv").write_text("\r\n".join(lines) + "\r\n",
                                     encoding="ascii", newline="")
    print(f"\noutputs written: {out / 'py_trades.csv'}")
    print(f"                  {out / 'py_days.csv'}")


def print_summary(trades, day_rows, stats):
    session_rows = [r for r in day_rows if r["session_open"]]
    buys = sum(1 for t in trades if t["type"] == "BUY")
    sells = len(trades) - buys
    wins = sum(1 for t in trades if t["profit"] > 0)
    total_pnl = round(sum(t["profit"] for t in trades), 2)
    exits = defaultdict(int)
    for t in trades:
        exits[t["exit_reason"]] += 1

    pnl_by_date = {r["ny_date"]: 0.0 for r in day_rows}
    for t in trades:
        pnl_by_date[t["ny_date"]] += t["profit"]
    daily = [pnl_by_date[r["ny_date"]] for r in day_rows if r["session_open"]]
    mean = sum(daily) / len(daily) if daily else 0.0
    std = (sum((x - mean) ** 2 for x in daily) / (len(daily) - 1)) ** 0.5 if len(daily) > 1 else 0.0
    sharpe = mean / std * (252 ** 0.5) if std > 0 else 0.0
    cum = peak = max_dd = 0.0
    for x in daily:
        cum += x
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    print()
    print("=" * 68)
    print("VWAP mean-reversion reference - run summary")
    print("=" * 68)
    print(f" session days                          : {len(session_rows)}")
    skipped = defaultdict(int)
    for r in day_rows:
        if r["skipped"]:
            skipped[r["skip_reason"]] += 1
    skip_txt = ", ".join(f"{k}: {skipped[k]}" for k in SKIP_REASONS if k in skipped)
    print(f" skipped days by reason                : {skip_txt if skip_txt else 'none'}")
    print(f" entries triggered (buy/sell)          : {buys} / {sells}")
    exit_txt = ", ".join(f"{k}: {v}" for k, v in sorted(exits.items()))
    print(f" exits by reason                       : {exit_txt if exit_txt else 'none'}")
    print(f" total trades                          : {len(trades)}")
    if trades:
        print(f" win rate                              : {100.0 * wins / len(trades):.2f}% ({wins}/{len(trades)})")
    print(f" total PnL                             : {total_pnl:.2f}")
    print(f" per-day PnL mean / std                : {mean:.4f} / {std:.4f}")
    print(f" daily-return Sharpe (x sqrt(252))     : {sharpe:.4f}")
    print(f" max drawdown (cumulative PnL, $)      : {max_dd:.2f}")
    print(f" time-exit carries to next date        : {stats['time_exit_carries']}")
    print(f" data-end open positions               : {stats['data_end_open']}")


if __name__ == "__main__":
    simulate()
