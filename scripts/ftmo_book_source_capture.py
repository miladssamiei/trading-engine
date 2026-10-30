"""Bounded, read-only FTMO market-depth source capture; never sends orders."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def valid_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def capture(args: argparse.Namespace) -> dict:
    import MetaTrader5 as mt5

    result: dict = {"source": "FTMO-Demo MT5 live Market Depth",
                    "started_utc": iso_now(), "duration_seconds": args.seconds,
                    "poll_seconds": args.poll_seconds, "symbols": {}, "status": "RUNNING"}
    if not mt5.initialize(args.terminal):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    original_visibility: dict[str, bool] = {}
    subscribed: list[str] = []
    try:
        account = mt5.account_info()
        if account is None or account.server != args.expected_server:
            raise RuntimeError("Connected account/server mismatch")
        if account.trade_mode != 0:
            raise RuntimeError("Connected account is not a demo account")
        for symbol in args.symbols:
            info = mt5.symbol_info(symbol)
            if info is None or info.custom:
                raise RuntimeError(f"Missing or custom symbol: {symbol}")
            original_visibility[symbol] = bool(info.visible)
            if not mt5.symbol_select(symbol, True):
                raise RuntimeError(f"Could not select symbol: {symbol}")
            if not mt5.market_book_add(symbol):
                raise RuntimeError(f"Could not subscribe to depth: {symbol}")
            subscribed.append(symbol)
            result["symbols"][symbol] = {"declared_book_depth": int(info.ticks_bookdepth),
                                          "samples": 0, "nonempty": 0,
                                          "distinct_states": 0, "invalid_levels": 0,
                                          "first_nonempty_utc": None,
                                          "last_nonempty_utc": None}
        previous: dict[str, str] = {}
        started = time.monotonic()
        with args.output.joinpath("snapshots.jsonl").open("x", encoding="utf-8") as stream:
            while time.monotonic() - started < args.seconds:
                received = iso_now()
                for symbol in args.symbols:
                    tick = mt5.symbol_info_tick(symbol)
                    book = mt5.market_book_get(symbol)
                    levels = []
                    for item in book or ():
                        price = valid_float(item.price)
                        volume = valid_float(item.volume_dbl)
                        if price is None or price <= 0 or volume is None or volume <= 0:
                            result["symbols"][symbol]["invalid_levels"] += 1
                            continue
                        levels.append({"type": int(item.type), "price": price,
                                       "volume": volume})
                    bid = valid_float(tick.bid) if tick else None
                    ask = valid_float(tick.ask) if tick else None
                    raw_time_msc = int(tick.time_msc) if tick else None
                    row = {"received_at_utc": received, "symbol": symbol,
                           "bid": bid, "ask": ask,
                           "raw_broker_tick_time_msc": raw_time_msc,
                           "levels": levels}
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                    stats = result["symbols"][symbol]
                    stats["samples"] += 1
                    if levels:
                        stats["nonempty"] += 1
                        stats["first_nonempty_utc"] = stats["first_nonempty_utc"] or received
                        stats["last_nonempty_utc"] = received
                        key = json.dumps(levels, separators=(",", ":"))
                        if previous.get(symbol) != key:
                            stats["distinct_states"] += 1
                            previous[symbol] = key
                stream.flush()
                time.sleep(args.poll_seconds)
        result["status"] = "COMPLETE"
    finally:
        for symbol in subscribed:
            mt5.market_book_release(symbol)
        for symbol, was_visible in original_visibility.items():
            if not was_visible:
                mt5.symbol_select(symbol, False)
        mt5.shutdown()
    result["ended_utc"] = iso_now()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal", required=True)
    parser.add_argument("--expected-server", default="FTMO-Demo")
    parser.add_argument("--symbols", nargs="+", default=["US500.cash", "XAUUSD"])
    parser.add_argument("--seconds", type=int, default=5400)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 7200 or not 0.2 <= args.poll_seconds <= 10:
        raise ValueError("Capture duration or poll interval outside bounded range")
    if not Path(args.terminal).is_file():
        raise FileNotFoundError(args.terminal)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        summary = capture(args)
    except Exception as exc:
        summary = {"source": "FTMO-Demo MT5 live Market Depth", "status": "ERROR",
                   "ended_utc": iso_now(), "error_type": type(exc).__name__,
                   "error": str(exc)}
    file = args.output / "snapshots.jsonl"
    if file.exists():
        summary["snapshot_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
        summary["snapshot_bytes"] = file.stat().st_size
    (args.output / "capture_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output": str(args.output)}))
    if summary["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
