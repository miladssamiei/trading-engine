"""
Standalone reference implementation of the Opening Range Breakout (ORB)
strategy. Mirrors the MT5 EA logic exactly so the two CSV outputs can be
diffed line by line.

Input : us500tick_m5.csv   5-minute BID bars of "US500-tick"
                            columns: time,open,high,low,close
                            time = bar-open UTC epoch seconds (int)
Output: py_trades.csv       one row per simulated trade
        py_days.csv         one row per NY calendar date in the test window

Pure pandas + stdlib; independent of all bridge/ensemble code.
Bar timestamps are UTC; a "day" is a New York calendar date (zoneinfo
handles the 2026-03-08 DST switch).
"""

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

# ---------------------------------------------------------------------------
# Strategy parameters (MT5 EA defaults)
# ---------------------------------------------------------------------------
RANGE_OFFSET_PCT = 0.15       # stop-order offset beyond the opening range (%)
TAKE_PROFIT_PCT = 0.8         # take-profit distance from the entry level (%)
GAP_FILTER_PCT = 1.0          # max allowed |gap| vs prior session close (%)
LOT = 1.0
CONTRACT_SIZE = 1.0
SYMBOL = "US500-tick"

SESSION_OPEN = (9, 30)        # opening range start, NY wall time
RANGE_END = (10, 0)           # range end / stop-order placement start
ORDER_CANCEL = (10, 30)       # pending orders cancelled, NY wall time
SESSION_CLOSE = (16, 0)       # session end, NY wall time
MIN_OR_BARS = 5               # session bars needed for a complete range
PREV_CLOSE_HM = (15, 55)      # prior-session-close bar, NY wall time
PREV_CLOSE_LOOKBACK_DAYS = 10

# ---------------------------------------------------------------------------
# Paths and test window (CLI-overridable; defaults match the 2026 run)
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone ORB reference implementation.")
    parser.add_argument(
        "--data-csv",
        default=r"C:\Users\Milad\Desktop\RR\data\results\orb_2026\us500tick_m5.csv")
    parser.add_argument(
        "--out-dir",
        default=r"C:\Users\Milad\Desktop\RR\data\results\orb_2026")
    parser.add_argument("--window-start", default="2026-01-01",
                        help="window start date, YYYY-MM-DD (UTC)")
    return parser.parse_args()


_ARGS = parse_args()
DATA_CSV = Path(_ARGS.data_csv)
OUT_DIR = Path(_ARGS.out_dir)
TRADES_CSV = OUT_DIR / "py_trades.csv"
DAYS_CSV = OUT_DIR / "py_days.csv"
WINDOW_START_UTC = (datetime.fromisoformat(_ARGS.window_start)
                    .replace(tzinfo=timezone.utc))

NY = ZoneInfo("America/New_York")
UTC = timezone.utc

SESSION_OPEN_M = SESSION_OPEN[0] * 60 + SESSION_OPEN[1]        # 09:30 -> 570
RANGE_END_M = RANGE_END[0] * 60 + RANGE_END[1]                 # 10:00 -> 600
ORDER_CANCEL_M = ORDER_CANCEL[0] * 60 + ORDER_CANCEL[1]        # 10:30 -> 630
SESSION_CLOSE_M = SESSION_CLOSE[0] * 60 + SESSION_CLOSE[1]     # 16:00 -> 960
PREV_CLOSE_M = PREV_CLOSE_HM[0] * 60 + PREV_CLOSE_HM[1]        # 15:55 -> 955

TRADES_HEADER = ("ticket;open_time;close_time;symbol;type;lot;open_price;"
                 "stop_loss;take_profit;close_price;exit_reason;"
                 "swap;commission;profit")
DAYS_HEADER = ("ny_date;prev_close;session_open;gap_pct;skipped;skip_reason;"
               "range_high;range_low;or_bars;buy_level;sell_level;orders_placed;"
               "entry_direction;entry_time;exit_time;exit_reason")

SKIP_REASONS = ["no_prior_close", "gap_filter", "no_session_open",
                "incomplete_range"]


def fmt_utc(ts):
    """UTC epoch seconds -> 'YYYY.MM.DD HH:MM' (MT5 EA time format)."""
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y.%m.%d %H:%M")


def fmt4(x):
    """Price/percent with 4 decimals; empty string when missing."""
    return "" if x is None else f"{x:.4f}"


def dstr(ny_date):
    return "{:04d}.{:02d}.{:02d}".format(*ny_date)


def load_bars():
    """Read the full CSV (2025-12-01 .. 2026-04-30) and tag each bar with its
    NY calendar date and NY wall-clock minute-of-day."""
    df = pd.read_csv(DATA_CSV)
    bars = []
    for t, o, h, l, c in zip(df["time"], df["open"], df["high"],
                             df["low"], df["close"]):
        t = int(t)
        ny_dt = datetime.fromtimestamp(t, tz=UTC).astimezone(NY)
        bars.append({
            "time": t,
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "ny_date": (ny_dt.year, ny_dt.month, ny_dt.day),
            "wall": ny_dt.hour * 60 + ny_dt.minute,
        })
    bars.sort(key=lambda b: b["time"])
    return bars


def find_prev_close(ny_date, close_at_1555):
    """Close of the 15:55 NY bar on the most recent prior NY date that has
    such a bar (search back up to 10 calendar days through the FULL csv)."""
    base = datetime(*ny_date)
    for back in range(1, PREV_CLOSE_LOOKBACK_DAYS + 1):
        d = base - timedelta(days=back)
        key = (d.year, d.month, d.day)
        if key in close_at_1555:
            return close_at_1555[key]
    return None


def simulate_entry(place_bars, buy_level, sell_level, stats):
    """Scan the placement bars (NY 10:00 .. 10:30) in time order and trigger
    the first stop-order fill (OCO: at most one entry per day).

    The first bar's open beyond a level is the market-entry analog of the EA;
    level-touch entries fill at the stop level. When one bar spans both
    levels the fill is resolved by the O->L->H / O->H->L path heuristic.
    """
    for b in place_bars:
        if b["open"] >= buy_level:
            return {"dir": "BUY", "price": b["open"], "bar": b, "via_open": True}
        if b["open"] <= sell_level:
            return {"dir": "SELL", "price": b["open"], "bar": b, "via_open": True}
        if b["high"] >= buy_level and b["low"] <= sell_level:
            stats["both_levels_in_bar"] += 1
            if b["close"] >= b["open"]:
                return {"dir": "SELL", "price": sell_level, "bar": b,
                        "via_open": False}
            return {"dir": "BUY", "price": buy_level, "bar": b,
                    "via_open": False}
        if b["high"] >= buy_level:
            return {"dir": "BUY", "price": buy_level, "bar": b,
                    "via_open": False}
        if b["low"] <= sell_level:
            return {"dir": "SELL", "price": sell_level, "bar": b,
                    "via_open": False}
    return None


def simulate_exit(day_bars, entry, stats):
    """Manage the open position from the entry bar until TP / SL / session
    end / data end. Returns (exit_bar, exit_price, exit_reason)."""
    b0 = entry["bar"]
    sl, tp = entry["sl"], entry["tp"]
    is_buy = entry["dir"] == "BUY"

    # Same-bar exit on the entry bar (documented convention): the TP check
    # always applies; the same-bar SL only applies to level-touch entries
    # (long: path O->H->L, entry via the high then reversal;
    #  short: path O->L->H, entry via the low then reversal).
    if is_buy:
        if b0["high"] >= tp:
            stats["same_bar_take_profit"] += 1
            return b0, tp, "take_profit"
        if (not entry["via_open"] and b0["close"] < b0["open"]
                and b0["low"] <= sl):
            stats["same_bar_stop_loss"] += 1
            return b0, sl, "stop_loss"
    else:
        if b0["low"] <= tp:
            stats["same_bar_take_profit"] += 1
            return b0, tp, "take_profit"
        if (not entry["via_open"] and b0["close"] > b0["open"]
                and b0["high"] >= sl):
            stats["same_bar_stop_loss"] += 1
            return b0, sl, "stop_loss"

    # Subsequent bars of the same NY date before 16:00.
    for b in day_bars:
        if b["time"] <= b0["time"] or b["wall"] >= SESSION_CLOSE_M:
            continue
        if is_buy:
            if b["open"] <= sl:
                return b, b["open"], "stop_loss"        # gapped through stop
            if b["open"] >= tp:
                return b, b["open"], "take_profit"      # gapped through TP
            if b["low"] <= sl and b["high"] >= tp:      # both inside the bar
                if b["close"] >= b["open"]:
                    return b, sl, "stop_loss"           # path O->L->H
                return b, tp, "take_profit"             # path O->H->L
            if b["low"] <= sl:
                return b, sl, "stop_loss"
            if b["high"] >= tp:
                return b, tp, "take_profit"
        else:
            if b["open"] >= sl:
                return b, b["open"], "stop_loss"
            if b["open"] <= tp:
                return b, b["open"], "take_profit"
            if b["high"] >= sl and b["low"] <= tp:
                if b["close"] >= b["open"]:
                    return b, tp, "take_profit"         # path O->L->H
                return b, sl, "stop_loss"               # path O->H->L
            if b["high"] >= sl:
                return b, sl, "stop_loss"
            if b["low"] <= tp:
                return b, tp, "take_profit"

    # Session end: first bar at/after 16:00 NY, exit at that bar's open.
    for b in day_bars:
        if b["time"] > b0["time"] and b["wall"] >= SESSION_CLOSE_M:
            return b, b["open"], "session_end"

    # Data ended before 16:00 with the position still open (rare):
    # exit at the close of the last bar of that date.
    last = day_bars[-1]
    return last, last["close"], "data_end"


def run():
    bars = load_bars()
    print("input bars: {}".format(len(bars)))
    print("input range: {} -> {}".format(
        datetime.fromtimestamp(bars[0]["time"], tz=UTC)
        .strftime("%Y-%m-%d %H:%M UTC"),
        datetime.fromtimestamp(bars[-1]["time"], tz=UTC)
        .strftime("%Y-%m-%d %H:%M UTC")))

    # Prior-close lookup may reach into December 2025: use the FULL csv.
    close_at_1555 = {b["ny_date"]: b["close"] for b in bars
                     if b["wall"] == PREV_CLOSE_M}

    # Strategy processing is restricted to the test window.
    window_start_ts = int(WINDOW_START_UTC.timestamp())
    by_date = defaultdict(list)
    for b in bars:
        if b["time"] >= window_start_ts:
            by_date[b["ny_date"]].append(b)

    stats = defaultdict(int)
    trades = []
    day_rows = []

    for ny_date in sorted(by_date):
        day_bars = by_date[ny_date]
        session_bars = [b for b in day_bars
                        if SESSION_OPEN_M <= b["wall"] < RANGE_END_M]
        place_bars = [b for b in day_bars
                      if RANGE_END_M <= b["wall"] < ORDER_CANCEL_M]
        has_0930 = any(b["wall"] == SESSION_OPEN_M for b in session_bars)

        row = {
            "ny_date": ny_date,
            "prev_close": None,
            "session_open": None,
            "gap_pct": None,
            "skipped": 0,
            "skip_reason": "",
            "range_high": None,
            "range_low": None,
            "or_bars": len(session_bars),
            "buy_level": None,
            "sell_level": None,
            "orders_placed": 0,
            "entry_direction": "",
            "entry_time": None,
            "exit_time": None,
            "exit_reason": "",
            "has_0930": has_0930,
        }

        # -- gap filter (needs session bars) --------------------------------
        if session_bars:
            row["session_open"] = session_bars[0]["open"]
            prev_close = find_prev_close(ny_date, close_at_1555)
            if prev_close is None:
                row["skipped"], row["skip_reason"] = 1, "no_prior_close"
            else:
                row["prev_close"] = prev_close
                row["gap_pct"] = (abs(row["session_open"] / prev_close - 1.0)
                                  * 100.0)
                if row["gap_pct"] > GAP_FILTER_PCT:
                    row["skipped"], row["skip_reason"] = 1, "gap_filter"

        # -- opening range ---------------------------------------------------
        if session_bars:
            row["range_high"] = max(b["high"] for b in session_bars)
            row["range_low"] = min(b["low"] for b in session_bars)

        # -- order placement (needs place bars and a non-skipped day) --------
        entry = None
        if place_bars and not row["skipped"]:
            if not has_0930:
                row["skipped"], row["skip_reason"] = 1, "no_session_open"
            elif row["or_bars"] < MIN_OR_BARS:
                row["skipped"], row["skip_reason"] = 1, "incomplete_range"
            else:
                row["buy_level"] = row["range_high"] * (1.0 + RANGE_OFFSET_PCT / 100.0)
                row["sell_level"] = row["range_low"] * (1.0 - RANGE_OFFSET_PCT / 100.0)
                row["orders_placed"] = 1
                entry = simulate_entry(place_bars, row["buy_level"],
                                       row["sell_level"], stats)

        # -- position management ---------------------------------------------
        if entry is not None:
            if entry["dir"] == "BUY":
                entry["sl"] = row["range_low"]
                entry["tp"] = row["buy_level"] * (1.0 + TAKE_PROFIT_PCT / 100.0)
            else:
                entry["sl"] = row["range_high"]
                entry["tp"] = row["sell_level"] * (1.0 - TAKE_PROFIT_PCT / 100.0)

            if not (RANGE_END_M <= entry["bar"]["wall"] < ORDER_CANCEL_M):
                stats["entries_outside_window"] += 1

            exit_bar, exit_price, exit_reason = simulate_exit(day_bars, entry,
                                                              stats)
            sign = 1 if entry["dir"] == "BUY" else -1
            profit = round(sign * (exit_price - entry["price"])
                           * LOT * CONTRACT_SIZE, 2) + 0.0

            row["entry_direction"] = entry["dir"]
            row["entry_time"] = entry["bar"]["time"]
            row["exit_time"] = exit_bar["time"]
            row["exit_reason"] = exit_reason

            trades.append({
                "ny_date": ny_date,
                "open_time": entry["bar"]["time"],
                "close_time": exit_bar["time"],
                "type": entry["dir"],
                "open_price": entry["price"],
                "stop_loss": entry["sl"],
                "take_profit": entry["tp"],
                "close_price": exit_price,
                "exit_reason": exit_reason,
                "profit": profit,
            })

            if exit_reason == "data_end":
                stats["data_end_exits"] += 1
                print("!! DATA_END EXIT on {}: position still open at data "
                      "end, closed at last bar close {:.4f}".format(
                          dstr(ny_date), exit_price))

        day_rows.append(row)

    write_outputs(trades, day_rows)
    print_summary(trades, day_rows, stats)


def write_outputs(trades, day_rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = [TRADES_HEADER]
    for i, t in enumerate(trades, start=1):
        lines.append(";".join([
            str(i),
            fmt_utc(t["open_time"]),
            fmt_utc(t["close_time"]),
            SYMBOL,
            t["type"],
            f"{LOT:.2f}",
            fmt4(t["open_price"]),
            fmt4(t["stop_loss"]),
            fmt4(t["take_profit"]),
            fmt4(t["close_price"]),
            t["exit_reason"],
            "0.00",
            "0.00",
            f"{t['profit']:.2f}",
        ]))
    TRADES_CSV.write_text("\r\n".join(lines) + "\r\n",
                          encoding="ascii", newline="")

    lines = [DAYS_HEADER]
    for r in day_rows:
        y, m, d = r["ny_date"]
        lines.append(";".join([
            f"{y:04d}{m:02d}{d:02d}",
            fmt4(r["prev_close"]),
            fmt4(r["session_open"]),
            fmt4(r["gap_pct"]),
            str(r["skipped"]),
            r["skip_reason"],
            fmt4(r["range_high"]),
            fmt4(r["range_low"]),
            str(r["or_bars"]),
            fmt4(r["buy_level"]),
            fmt4(r["sell_level"]),
            str(r["orders_placed"]),
            r["entry_direction"],
            fmt_utc(r["entry_time"]) if r["entry_time"] is not None else "",
            fmt_utc(r["exit_time"]) if r["exit_time"] is not None else "",
            r["exit_reason"],
        ]))
    DAYS_CSV.write_text("\r\n".join(lines) + "\r\n",
                        encoding="ascii", newline="")


def print_summary(trades, day_rows, stats):
    session_rows = [r for r in day_rows if r["or_bars"] > 0]
    inert_rows = [r for r in day_rows if r["or_bars"] == 0]
    skipped = defaultdict(int)
    for r in day_rows:
        if r["skipped"]:
            skipped[r["skip_reason"]] += 1
    orders_rows = [r for r in day_rows if r["orders_placed"]]
    exits = defaultdict(int)
    for t in trades:
        exits[t["exit_reason"]] += 1
    buys = sum(1 for t in trades if t["type"] == "BUY")
    sells = len(trades) - buys
    wins = sum(1 for t in trades if t["profit"] > 0)
    total_pnl = round(sum(t["profit"] for t in trades), 2)

    # Per-day PnL over ALL day records (zero days included), date order.
    pnl_by_date = {r["ny_date"]: 0.0 for r in day_rows}
    for t in trades:
        pnl_by_date[t["ny_date"]] += t["profit"]
    daily = [pnl_by_date[r["ny_date"]] for r in day_rows]
    mean = sum(daily) / len(daily)
    std = (sum((x - mean) ** 2 for x in daily) / (len(daily) - 1)) ** 0.5
    sharpe = mean / std * (252 ** 0.5) if std > 0 else 0.0
    cum = peak = max_dd = 0.0
    for x in daily:
        cum += x
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    print()
    print("=" * 68)
    print("ORB reference implementation - run summary")
    print("=" * 68)
    print(f" trading days in window                : {len(session_rows)}")
    print(f" day records emitted                   : {len(day_rows)} "
          f"({len(session_rows)} session days + {len(inert_rows)} evening-only)")
    skip_txt = ", ".join(f"{k}: {skipped[k]}" for k in SKIP_REASONS
                         if k in skipped)
    print(f" skipped days by reason                : {skip_txt if skip_txt else 'none'}")
    print(f" days with orders                      : {len(orders_rows)}")
    print(f" entries triggered (buy/sell)          : {buys} / {sells}")
    exit_txt = ", ".join(f"{k}: {v}" for k, v in sorted(exits.items()))
    print(f" exits by reason                       : {exit_txt if exit_txt else 'none'}")
    print(f" total trades                          : {len(trades)}")
    if trades:
        print(f" win rate                              : "
              f"{100.0 * wins / len(trades):.2f}% ({wins}/{len(trades)})")
    else:
        print(" win rate                              : n/a")
    print(f" total PnL                             : {total_pnl:.2f}")
    print(f" per-day PnL mean / std                : {mean:.4f} / {std:.4f}")
    print(f" daily-return Sharpe (x sqrt(252))     : {sharpe:.4f}")
    print(f" max drawdown (cumulative PnL, $)      : {max_dd:.2f}")

    print("-" * 68)
    print(" anomalies / sanity checks")
    miss = [r for r in session_rows if not r["has_0930"]]
    extra = (" -> " + ", ".join(dstr(r["ny_date"]) for r in miss)) if miss else ""
    print(f" session days missing the 09:30 bar    : {len(miss)}{extra}")
    short = [r for r in session_rows if r["or_bars"] < MIN_OR_BARS]
    extra = (" -> " + ", ".join(dstr(r["ny_date"]) for r in short)) if short else ""
    print(f" session days with < {MIN_OR_BARS} range bars      : {len(short)}{extra}")
    gap_days = [r for r in day_rows if r["skip_reason"] == "gap_filter"]
    extra = (" -> " + ", ".join(dstr(r["ny_date"]) for r in gap_days)) if gap_days else ""
    print(f" gap-filter skips                      : {len(gap_days)}{extra}")
    print(f" entries outside 10:00-10:30 NY        : {stats['entries_outside_window']}")
    print(f" both-levels-in-one-bar entries        : {stats['both_levels_in_bar']}")
    print(f" same-bar exits (tp/sl on entry bar)   : "
          f"{stats['same_bar_take_profit'] + stats['same_bar_stop_loss']} "
          f"(tp {stats['same_bar_take_profit']}, "
          f"sl {stats['same_bar_stop_loss']})")
    print(f" data_end exits                        : {stats['data_end_exits']}")
    extra = (", ".join(dstr(r["ny_date"]) for r in inert_rows)) if inert_rows else ""
    print(f" inert evening-only rows               : {len(inert_rows)} -> {extra}")
    print()
    print(f" outputs written: {TRADES_CSV}")
    print(f"                  {DAYS_CSV}")


if __name__ == "__main__":
    run()
