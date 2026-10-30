"""Read-only first prospective H105 quote-path observation. No orders."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/results/us100_native_full_source_h94/us100cash_ftmo_native_m1_2023_2026.csv"
EXPECTED = "f9552440b4af1e6d20fd9934d0c54b89a21131ccf9948d2b2b1f875dad90b9a5"
OUT = ROOT / "data/results/us100_native_weekday_h105/forward_observations/2026-09-28"
TERMINAL = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
SYMBOL = "US100.cash"
NY = ZoneInfo("America/New_York")
ATHENS = ZoneInfo("Europe/Athens")
ENTRY_START = datetime(2026, 9, 28, 9, 30, tzinfo=NY)
ENTRY_END = ENTRY_START + timedelta(minutes=5)
EXIT_START = datetime(2026, 10, 1, 11, 0, tzinfo=NY)
EXIT_END = EXIT_START + timedelta(minutes=15)
SCHEDULED_RUN = datetime(2026, 10, 1, 11, 20, tzinfo=NY)
SWAP_NORMAL = -6.7512
EXTRA_STRESS = 2.50


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def source_feature() -> dict:
    if sha(SOURCE) != EXPECTED:
        raise ValueError("Frozen H105 native M1 source hash changed")
    frame = pd.read_csv(SOURCE, usecols=["minute_epoch_s", "bid_high", "bid_low", "bid_close"])
    t = pd.to_datetime(frame.minute_epoch_s, unit="s", utc=True).dt.tz_convert(NY)
    day = t.dt.strftime("%Y-%m-%d")
    wall = t.dt.hour * 60 + t.dt.minute
    cash = frame[(wall >= 570) & (wall < 960) & (t.dt.weekday < 5)].copy()
    cash["date"] = day[(wall >= 570) & (wall < 960) & (t.dt.weekday < 5)].to_numpy()
    cash["minute"] = wall[(wall >= 570) & (wall < 960) & (t.dt.weekday < 5)].to_numpy()
    grouped = cash.groupby("date", sort=True).agg(h=("bid_high", "max"), l=("bid_low", "min"),
                                                   c=("bid_close", "last"),
                                                   first_minute=("minute", "first"))
    grouped = grouped[grouped.first_minute == 570]
    # Import the original ATR function without changing frozen H105 code.
    import sys
    sys.path.insert(0, str(ROOT / "evaluate"))
    from xauusd_kc_breakout_oos_phase1 import wilder_atr
    atr = wilder_atr(grouped.h.to_numpy(float), grouped.l.to_numpy(float),
                     grouped.c.to_numpy(float), 20)
    prior = grouped.index[grouped.index < ENTRY_START.date().isoformat()]
    if len(prior) == 0:
        raise ValueError("No completed cash date before forward Monday")
    prior_date = prior[-1]
    j = grouped.index.get_loc(prior_date)
    age = (ENTRY_START.date() - datetime.fromisoformat(prior_date).date()).days
    if age > 4 or not np.isfinite(atr[j]):
        raise ValueError("H105 prior ATR or freshness unavailable")
    return {"prior_cash_date": prior_date, "prior_atr20": float(atr[j]),
            "prior_date_age_days": age, "source_sha256": EXPECTED}


def connect() -> tuple[bool, object]:
    if not mt5.initialize(TERMINAL):
        raise RuntimeError(f"FTMO MT5 initialize: {mt5.last_error()}")
    account = mt5.account_info()
    if account is None or account.server != "FTMO-Demo" or account.trade_mode != 0:
        mt5.shutdown()
        raise RuntimeError("Wrong FTMO server or demo mode")
    info = mt5.symbol_info(SYMBOL)
    if info is None or info.custom:
        mt5.shutdown()
        raise RuntimeError("Native US100.cash unavailable")
    was_visible = bool(info.visible)
    if not mt5.symbol_select(SYMBOL, True):
        mt5.shutdown()
        raise RuntimeError("Could not select native US100.cash")
    return was_visible, info


def rollovers(start_msc: int, end_msc: int) -> tuple[int, int]:
    a = pd.Timestamp(start_msc, unit="ms", tz="UTC")
    b = pd.Timestamp(end_msc, unit="ms", tz="UTC")
    days = weight = 0
    for day in pd.date_range(a.tz_convert(ATHENS).date(), b.tz_convert(ATHENS).date(), freq="D"):
        midnight = day.tz_localize(ATHENS).tz_convert("UTC")
        if a < midnight < b:
            days += 1
            weight += 3 if day.weekday() == 4 else 1
    return days, weight


def result(entry_msc: int, entry_ask: float, exit_msc: int, exit_bid: float,
           reason: str, stop: float) -> dict:
    days, weight = rollovers(entry_msc, exit_msc)
    swap = weight * SWAP_NORMAL
    net = exit_bid - entry_ask + swap
    return {"status": "RESOLVED", "reason": reason, "entry_time_msc": entry_msc,
            "entry_ask": entry_ask, "stop": stop, "exit_time_msc": exit_msc,
            "exit_bid": exit_bid, "rollovers": days, "weighted_swap_days": weight,
            "snapshot_swap_units": swap, "net_units": net,
            "stress_net_units": net - EXTRA_STRESS}


def path_outcome(ms: np.ndarray, bid: np.ndarray, ask: np.ndarray,
                 atr: float, label: str) -> dict:
    entry_lo = int(ENTRY_START.timestamp() * 1000)
    entry_hi = int(ENTRY_END.timestamp() * 1000)
    exit_lo = int(EXIT_START.timestamp() * 1000)
    exit_hi = int(EXIT_END.timestamp() * 1000)
    entry = np.flatnonzero((ms >= entry_lo) & (ms < entry_hi))
    if len(entry) == 0:
        return {"status": "NO_ENTRY", "fill_model": label}
    k0 = int(entry[0])
    stop = float(ask[k0]) - 2 * atr
    if stop <= 0:
        raise ValueError("Nonpositive paper stop")
    until_exit = np.flatnonzero((np.arange(len(ms)) >= k0) & (ms < exit_lo) & (bid <= stop))
    scheduled = np.flatnonzero((ms >= exit_lo) & (ms < exit_hi))
    if len(until_exit):
        kx = int(until_exit[0])
        exit_bid = float(bid[kx]) if label == "TICK" else min(float(bid[kx]), stop)
        reason = "BID_STOP_QUOTE" if label == "TICK" else "M1_STOP_PROXY"
    elif len(scheduled):
        kx = int(scheduled[0])
        exit_bid = float(bid[kx])
        reason = "THURSDAY_FLAT"
    else:
        return {"status": "UNRESOLVED_EXIT", "fill_model": label,
                "entry_time_msc": int(ms[k0]), "entry_ask": float(ask[k0]), "stop": stop}
    answer = result(int(ms[k0]), float(ask[k0]), int(ms[kx]), exit_bid, reason, stop)
    answer["fill_model"] = label
    return answer


def m1_path_outcome(ms: np.ndarray, bid_open: np.ndarray, ask_open: np.ndarray,
                    bid_low: np.ndarray, atr: float) -> dict:
    entry_lo = int(ENTRY_START.timestamp() * 1000)
    entry_hi = int(ENTRY_END.timestamp() * 1000)
    exit_lo = int(EXIT_START.timestamp() * 1000)
    exit_hi = int(EXIT_END.timestamp() * 1000)
    entry = np.flatnonzero((ms >= entry_lo) & (ms < entry_hi))
    if len(entry) == 0:
        return {"status": "NO_ENTRY", "fill_model": "M1"}
    k0 = int(entry[0])
    stop = float(ask_open[k0]) - 2 * atr
    for k in range(k0, len(ms)):
        if ms[k] >= exit_lo:
            if ms[k] >= exit_hi:
                break
            answer = result(int(ms[k0]), float(ask_open[k0]), int(ms[k]),
                            float(bid_open[k]), "THURSDAY_FLAT", stop)
            answer["fill_model"] = "M1"
            return answer
        if bid_open[k] <= stop:
            answer = result(int(ms[k0]), float(ask_open[k0]), int(ms[k]),
                            float(bid_open[k]), "M1_STOP_GAP_PROXY", stop)
            answer["fill_model"] = "M1"
            return answer
        if bid_low[k] <= stop:
            answer = result(int(ms[k0]), float(ask_open[k0]), int(ms[k]),
                            stop, "M1_STOP_PROXY", stop)
            answer["fill_model"] = "M1"
            return answer
    return {"status": "UNRESOLVED_EXIT", "fill_model": "M1",
            "entry_time_msc": int(ms[k0]), "entry_ask": float(ask_open[k0]), "stop": stop}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    feature = source_feature()
    prior_visible, info = connect()
    try:
        current_swap = {"swap_mode": int(info.swap_mode), "swap_long": float(info.swap_long),
                        "point": float(info.point), "rollover3days": int(info.swap_rollover3days)}
        if args.preflight:
            if sys.stdout is not None:
                print(json.dumps({"status": "PREFLIGHT_PASS", "feature": feature,
                                  "current_symbol_swap": current_swap,
                                  "scheduled_run_ny": SCHEDULED_RUN.isoformat(),
                                  "orders": "NONE"}, indent=2))
            return
        if datetime.now(timezone.utc) < EXIT_END.astimezone(timezone.utc):
            raise RuntimeError("Observation attempted before the Thursday exit window ended")
        if OUT.exists():
            raise FileExistsError("Forward observation directory already exists")
        OUT.mkdir(parents=True)
        raw = mt5.copy_ticks_range(SYMBOL, ENTRY_START,
                                   EXIT_END - timedelta(milliseconds=1), mt5.COPY_TICKS_ALL)
        retrieved = datetime.now(timezone.utc)
        if raw is None:
            raise RuntimeError(f"Native tick query failed: {mt5.last_error()}")
        if len(raw) == 0:
            raise RuntimeError("Native tick query returned zero ticks")
        tick_path = OUT / "native_ticks.npz"
        np.savez_compressed(tick_path, ticks=raw)
        ms = raw["time_msc"]
        inside = ((ms >= int(ENTRY_START.timestamp() * 1000)) &
                  (ms < int(EXIT_END.timestamp() * 1000)))
        valid = np.isfinite(raw["bid"]) & np.isfinite(raw["ask"]) & (raw["bid"] > 0) & (raw["ask"] >= raw["bid"])
        quality = {"count": int(len(raw)), "out_of_range": int((~inside).sum()),
                   "invalid_bid_ask": int((~valid).sum()),
                   "unordered_adjacent": int((np.diff(ms) < 0).sum()),
                   "first_msc": int(ms[0]), "last_msc": int(ms[-1]),
                   "raw_array_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                   "archive_sha256": sha(tick_path)}
        filtered = raw[valid & inside]
        if quality["unordered_adjacent"] or len(filtered) == 0:
            raise ValueError("Unordered or empty valid native tick path")
        tick = path_outcome(filtered["time_msc"].astype(np.int64),
                            filtered["bid"].astype(float), filtered["ask"].astype(float),
                            feature["prior_atr20"], "TICK")
        minute = filtered["time_msc"] // 60000
        _, first_index = np.unique(minute, return_index=True)
        min_ms = filtered["time_msc"][first_index].astype(np.int64)
        min_bid = filtered["bid"][first_index].astype(float)
        min_ask = filtered["ask"][first_index].astype(float)
        # To reproduce the historical M1 stop proxy, inspect every valid
        # quote's BID low within each minute after its first quote.
        min_low = np.minimum.reduceat(filtered["bid"].astype(float), first_index)
        m1 = m1_path_outcome(min_ms, min_bid, min_ask, min_low,
                             feature["prior_atr20"])
        lag_min = (retrieved - SCHEDULED_RUN.astimezone(timezone.utc)).total_seconds() / 60
        report = {"observation_week_monday_ny": "2026-09-28", "retrieved_utc": retrieved.isoformat(),
                  "scheduled_run_ny": SCHEDULED_RUN.isoformat(), "capture_lag_minutes": lag_min,
                  "capture_label": "ON_SCHEDULE" if abs(lag_min) <= 60 else "LATE_CAPTURE",
                  "account_server": "FTMO-Demo", "symbol": SYMBOL,
                  "query_start_ny": ENTRY_START.isoformat(), "query_end_ny": EXIT_END.isoformat(),
                  "feature": feature, "current_symbol_swap": current_swap,
                  "frozen_swap_normal_units": SWAP_NORMAL, "extra_stress_units": EXTRA_STRESS,
                  "quality": quality, "tick_quote_path": tick, "m1_proxy_path": m1,
                  "interpretation": "Read-only paper quotes; no real order, guaranteed stop fill or validated edge."}
        (OUT / "OBSERVATION.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                               encoding="utf-8")
        if sys.stdout is not None:
            print(json.dumps({"capture_label": report["capture_label"], "quality": quality,
                              "tick_quote_path": tick, "m1_proxy_path": m1}, indent=2))
    finally:
        if not prior_visible:
            mt5.symbol_select(SYMBOL, False)
        mt5.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        failure = OUT.parent / "2026-09-28_TASK_FAILURE.txt"
        if not failure.exists():
            failure.write_text(datetime.now(timezone.utc).isoformat() + "\n" +
                               traceback.format_exc(), encoding="utf-8")
        raise
