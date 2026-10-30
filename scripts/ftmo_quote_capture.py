"""Read-only, forward MT5 quote capture for a connected FTMO terminal.

No account number or credential is written. This produces partial evidence;
sessions, financing and independent account review must be supplied separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


QUOTE_COLUMNS = ("timestamp_utc", "received_at_utc", "bid", "ask",
                 "symbol", "contract_id", "source_kind")


def utc_iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def safe_number(value, *, positive: bool = False, nonnegative: bool = False):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or (positive and number <= 0) or (nonnegative and number < 0):
        return None
    return number


def build_snapshot(account, symbol, alias: str, captured_at: datetime) -> dict:
    """Explicit allowlist: account.login and similar fields never serialize."""
    return {
        "schema_version": 1,
        "provenance": {"source_kind": "ftmo_export",
                       "source": "read-only MT5 Python symbol_info/account_info capture",
                       "received_at": utc_iso(captured_at)},
        "account": {
            "account_fingerprint": alias,
            "server": str(account.server),
            "broker": str(account.company),
            "platform": "MT5",
            "currency": str(account.currency),
            "initial_balance": None,
            "program": None,
            "account_type": None,
            "phase": None,
            "captured_at": utc_iso(captured_at),
            "balance": safe_number(account.balance),
            "equity": safe_number(account.equity),
        },
        "symbol": {
            "name": str(symbol.name),
            "contract_id": str(symbol.name),
            "quote_currency": None,
            "profit_currency": str(symbol.currency_profit) if symbol.currency_profit else None,
            "tick_size": safe_number(symbol.trade_tick_size, positive=True),
            "tick_value_profit": safe_number(symbol.trade_tick_value_profit, positive=True),
            "tick_value_loss": safe_number(symbol.trade_tick_value_loss, positive=True),
            "contract_size": safe_number(symbol.trade_contract_size, positive=True),
            "volume_min": safe_number(symbol.volume_min, positive=True),
            "volume_max": safe_number(symbol.volume_max, positive=True),
            "volume_step": safe_number(symbol.volume_step, positive=True),
            "margin_initial": safe_number(symbol.margin_initial, nonnegative=True),
            "stops_level": safe_number(symbol.trade_stops_level, nonnegative=True),
            "freeze_level": safe_number(symbol.trade_freeze_level, nonnegative=True),
            "filling_modes": None,
            "trading_sessions": None,
            "swap_mode": None,
            "swap_long": safe_number(symbol.swap_long),
            "swap_short": safe_number(symbol.swap_short),
            "rollover_multipliers": None,
        },
    }


def quote_row(tick, received_at: datetime, symbol: str, previous_ms: int,
              server_utc_offset_hours: float = 0):
    """Return a new as-received quote, or None for stale/invalid data."""
    try:
        observed_ms = int(tick.time_msc)
        bid = float(tick.bid)
        ask = float(tick.ask)
    except (AttributeError, TypeError, ValueError):
        return None
    if observed_ms <= previous_ms or observed_ms <= 0:
        return None
    if not (math.isfinite(bid) and math.isfinite(ask) and 0 < bid <= ask):
        return None
    try:
        observed_at = (datetime.fromtimestamp(observed_ms / 1000, tz=timezone.utc)
                       - timedelta(hours=server_utc_offset_hours))
    except (OverflowError, OSError, ValueError):
        return None
    if observed_at > received_at or received_at - observed_at > timedelta(seconds=5):
        return None
    return {"timestamp_utc": utc_iso(observed_at),
            "received_at_utc": utc_iso(received_at),
            "bid": f"{bid:.10f}", "ask": f"{ask:.10f}",
            "symbol": symbol, "contract_id": symbol,
            "source_kind": "ftmo_export"}


def write_json_new(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def capture(args) -> None:
    import MetaTrader5 as mt5

    terminal = Path(args.terminal).resolve()
    if not terminal.is_file():
        raise ValueError("MT5 terminal executable was not found")
    if not mt5.initialize(str(terminal)):
        raise RuntimeError("Could not connect to the specified MT5 terminal")
    try:
        account = mt5.account_info()
        if account is None:
            raise RuntimeError("The terminal has no connected account")
        # Check in memory only. Never write account.login or the account tuple.
        active_login = account.login
        if account.server != args.expected_server:
            raise ValueError("Connected MT5 server differs from --expected-server")
        if "ftmo" not in (str(account.server) + " " + str(account.company)).lower():
            raise ValueError("Connected terminal is not identified as FTMO")
        symbol = mt5.symbol_info(args.symbol)
        if symbol is None or bool(symbol.custom):
            raise ValueError("Symbol is missing or is a custom/imported symbol")
        was_visible = bool(symbol.visible)
        if not mt5.symbol_select(args.symbol, True):
            raise RuntimeError("Could not select the requested broker symbol")
        try:
            if args.output_dir.exists():
                raise FileExistsError("Output directory already exists; choose a new capture directory")
            args.output_dir.mkdir(parents=True)
            now = datetime.now(timezone.utc)
            write_json_new(args.output_dir / "snapshot_partial.json",
                           build_snapshot(account, symbol, args.alias, now))
            count = 0
            ignored = 0
            last_ms = 0
            last_received = None
            first_raw_ms = None
            last_raw_ms = None
            start = time.monotonic()
            with (args.output_dir / "quotes.csv").open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=QUOTE_COLUMNS)
                writer.writeheader()
                try:
                    while time.monotonic() - start < args.minutes * 60:
                        current = mt5.account_info()
                        if current is None or current.login != active_login or current.server != args.expected_server:
                            raise RuntimeError("Connected account changed; capture stopped")
                        tick = mt5.symbol_info_tick(args.symbol)
                        received = datetime.now(timezone.utc)
                        row = quote_row(tick, received, args.symbol, last_ms,
                                        args.server_utc_offset_hours)
                        if row is not None and (last_received is None or received >= last_received):
                            writer.writerow(row)
                            stream.flush()
                            last_ms = int(tick.time_msc)
                            last_raw_ms = last_ms
                            if first_raw_ms is None:
                                first_raw_ms = last_ms
                            last_received = received
                            count += 1
                        else:
                            ignored += 1
                        time.sleep(args.poll_ms / 1000)
                except KeyboardInterrupt:
                    pass
            summary = {"symbol": args.symbol, "server": args.expected_server,
                       "captured_rows": count, "ignored_polls": ignored,
                       "server_utc_offset_hours": args.server_utc_offset_hours,
                       "raw_server_timestamp_first_ms": first_raw_ms,
                       "raw_server_timestamp_last_ms": last_raw_ms,
                       "clock_rule": "timestamp_utc = raw MT5 tick.time_msc minus explicit server UTC offset; rows over five seconds old or in the future are rejected",
                       "clock_source": "https://ftmo.com/en/faq/what-are-the-account-specifications/",
                       "snapshot_sha256": digest(args.output_dir / "snapshot_partial.json"),
                       "quotes_sha256": digest(args.output_dir / "quotes.csv"),
                       "notes": "Read-only sampled latest ticks, not every market tick. Partial FTMO evidence only; no execution or full qualification claim."}
            write_json_new(args.output_dir / "capture_summary.json", summary)
            print(f"Captured {count} new quotes for {args.symbol} in {args.output_dir}")
        finally:
            if not was_visible and mt5.account_info() is not None and mt5.account_info().login == active_login:
                mt5.symbol_select(args.symbol, False)
    finally:
        mt5.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal", required=True, help="Path to the intended FTMO terminal64.exe")
    parser.add_argument("--expected-server", required=True, help="Exact server shown in that terminal")
    parser.add_argument("--symbol", required=True, help="Exact FTMO broker symbol")
    parser.add_argument("--alias", required=True, help="Non-secret account alias, never a login number")
    parser.add_argument("--server-utc-offset-hours", type=float, required=True,
                        help="FTMO server clock offset from UTC for this short capture, e.g. 3 during summer")
    parser.add_argument("--output-dir", required=True, type=Path, help="New output directory")
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--poll-ms", type=int, default=200)
    args = parser.parse_args()
    if not math.isfinite(args.minutes) or not 0 < args.minutes <= 60 * 24:
        parser.error("--minutes must be in (0, 1440]")
    if not 50 <= args.poll_ms <= 60_000:
        parser.error("--poll-ms must be in [50, 60000]")
    if not math.isfinite(args.server_utc_offset_hours) or not -12 <= args.server_utc_offset_hours <= 14:
        parser.error("--server-utc-offset-hours must be in [-12, 14]")
    if not args.alias.strip() or args.alias.isdigit():
        parser.error("--alias must be a nonnumeric, non-secret label")
    capture(args)


if __name__ == "__main__":
    main()
