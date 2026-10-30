"""
orchestrator.py — Master controller for fold testing pipeline (plan §2.3).

Ties together terminal_manager, config_generator, job_queue, bridge_manager,
and result_collector into the real job lifecycle:

  1. Pull next pending job from job_queue (queued → running)
  2. Start bridge server for that terminal on assigned port
  3. Generate .ini configs for the (symbol, fold) pair
  4. Launch MT5 terminal with /config:<path> (Strategy Tester)
  5. Wait for completion (terminal exits when ShutdownTerminal=1)
  6. Collect results via result_collector (parse XML/agent log)
  7. Mark job complete in job_queue (running → completed)
  8. Stop bridge, free terminal for next job
"""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from evaluate.terminal_manager import TerminalManager
from evaluate.config_generator import ConfigGenerator
from evaluate.job_queue import JobQueue
from evaluate.bridge_manager import BridgeManager
from evaluate.result_collector import parse_report, apply_guards, write_results
from config import BASE_LOT_MAP


def _log(msg: str) -> None:
    print(f"[Orch] {msg}", flush=True)


# MT5 terminal symbol → internal symbol (for SYMBOL_PARAMS / BASE_LOT_MAP lookups).
# The tester trades carry the terminal symbol (e.g. "US500-tick" for the custom
# Dukascopy import, or "US500.c" for the broker feed), but economics live under
# the internal "USA500" key in config.py.
_MT5_TO_INTERNAL_SYMBOL = {
    "US500.C": "USA500",
    "US500-TICK": "USA500",
}


def _normalize_symbol(symbol: str) -> str:
    if not symbol:
        return symbol
    return _MT5_TO_INTERNAL_SYMBOL.get(symbol.strip().upper(), symbol.strip().upper())


# ── Constants ──────────────────────────────────────────────────────────────

DEFAULT_TERMINAL_EXE  = r"C:\Program Files\MetaTrader 5\terminal64.exe"
DEFAULT_TERMINAL_DATA = (
    r"C:\Users\Milad\AppData\Roaming\MetaQuotes\Terminal"
    r"\D0E8209F77C8CF37AD8BF550E51FF075"
)
DEFAULT_AGENT_LOG_DIR = (
    r"C:\Users\Milad\AppData\Roaming\MetaQuotes\Tester"
    r"\D0E8209F77C8CF37AD8BF550E51FF075"
    r"\Agent-127.0.0.1-3000\logs"
)

DEFAULT_FROM_DATE = "2026-05-26"  # 7 calendar days, 5 biz days ~480 M15 bars, enough for 300+180
DEFAULT_TO_DATE   = "2026-06-02"


def _find_agent_log(log_dir: str | Path) -> Path | None:
    """Find the most recent UTF-16 LE agent log file."""
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return None
    logs = sorted(
        log_dir.glob("*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return logs[0] if logs else None


def _agent_log_bytes_from(log_path: str | Path, start_offset: int = 0) -> bytes:
    """Read agent log bytes starting from the given byte offset.

    Returns the raw bytes from start_offset to end-of-file.
    If start_offset is 0, returns the entire file.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return b""
    with open(log_path, "rb") as f:
        if start_offset > 0:
            f.seek(start_offset)
        return f.read()


def _decode_agent_log(raw: bytes) -> str:
    """Decode agent log bytes from UTF-16 LE/BE or UTF-8."""
    if raw[:2] == b"\xff\xfe":
        return raw.decode("utf-16-le", errors="replace")
    elif raw[:2] == b"\xfe\xff":
        return raw.decode("utf-16-be", errors="replace")
    return raw.decode("utf-8", errors="replace")


def _parse_last_test_segment(log_path: str | Path) -> tuple[bytes, bytes]:
    """Read the whole agent log and return the segment of the LAST test run.

    Returns (raw_segment, raw_full). The segment starts at the last
    'testing of Experts' line, which makes parsing immune to log rotation
    and offset drift (the previous byte-offset approach leaked metrics from
    earlier tests when a run aborted).
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return b"", b""
    with open(log_path, "rb") as f:
        raw_full = f.read()
    text = _decode_agent_log(raw_full)
    idx = text.rfind("testing of Experts")
    if idx < 0:
        return raw_full, raw_full
    return text[idx:].encode("utf-8", errors="replace"), raw_full


def _parse_agent_log(raw: bytes) -> dict:
    """Extract key metrics from MT5 agent log bytes (UTF-16 LE/BE or UTF-8).

    Accepts raw bytes so callers can pass a tail slice for multi-test logs.
    Returns dict with trade-level metrics reconstructed from log entries.
    """
    result = {
        "final_balance": 10000.0,
        "total_trades": 0,
        "profit_trades": 0,
        "loss_trades": 0,
        "test_passed": False,
        "ticks": 0,
        "bars": 0,
        "sharpe_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "profit_factor": 0.0,
        "expected_payoff": 0.0,
        "win_rate_pct": 0.0,
        "consecutive_wins": 0,
        "consecutive_losses": 0,
        "total_net_profit": 0.0,
    }
    if not raw:
        return result

    text = _decode_agent_log(raw)

    # Final balance — take LAST match (multi-test log file)
    m = re.findall(r"final balance\s+([\d.]+)\s*USD", text)
    if m:
        result["final_balance"] = float(m[-1])

    # EA trade count from Deinit log — take LAST match
    m = re.findall(r"Total trades placed:\s*(\d+)", text)
    if m:
        result["total_trades"] = int(m[-1])

    # Ticks/bars — take LAST match (multi-test log file)
    m = re.findall(r"(\d+)\s+ticks,\s*(\d+)\s+bars", text)
    if m:
        result["ticks"] = int(m[-1][0])
        result["bars"] = int(m[-1][1])

    # Test passed
    if re.search(r"Test passed in", text):
        result["test_passed"] = True

    # ── Trade-level reconstruction (replaces XML report gap) ──────────
    _parse_trades_from_log(text, result)

    return result


def _parse_trades_from_log(text: str, result: dict) -> None:
    """Parse individual trade entries from the agent log to compute
    per-trade PnL, Sharpe, win rate, drawdown, etc.

    Trade open format:
      Trade #N | DIRECTION lot SYMBOL @ PRICE | ticket=TICKET
    Trade close format:
      market close_dir lot SYMBOL, close #TICKET (bid / ask / executed_price)

    Sharpe matches CAIRO's canonical formula (parallel/common.py:1104):
      sharpe = mean(pnl) / std(pnl) * sqrt(BARS_PER_YEAR)
    where pnl is per-bar fractional returns.
    BARS_PER_YEAR = 6048 for M15 (252 days * 24 bars/day).
    """
    import math
    from datetime import datetime
    from config import SYMBOL_PARAMS, BASE_LOT_MAP

    # ── CAIRO-style constants (matches config.py:62-63, common.py:115-121) ──
    R_MAX = 0.005          # max fractional risk per bar
    COMMISSION_PRICE = 0.00005

    # Bars-per-year for M15 (252 trading days * 24 bars/day)
    BARS_PER_YEAR = 252 * 24  # 6048

    # Parse trade opens: {ticket: {direction, entry, symbol, lot, bar_ts}}
    opens = {}
    for m in re.finditer(
        r"(\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2})\s+"
        r"Trade #\d+\s*\|\s*(BUY|SELL)\s+([\d.]+)\s+([\w.\-]+)\s*@\s*([\d.]+)\s*\|\s*ticket=(\d+)",
        text,
    ):
        bar_ts = m.group(1)
        direction = m.group(2)
        lot = float(m.group(3))
        symbol = _normalize_symbol(m.group(4))
        entry_price = float(m.group(5))
        ticket = int(m.group(6))
        opens[ticket] = {
            "direction": direction,
            "lot": lot,
            "symbol": symbol,
            "entry": entry_price,
            "bar_ts": bar_ts,
        }

    # Parse trade closes: {ticket: {close_price, bar_ts}}
    closes = {}
    for m in re.finditer(
        r"(\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2})\s+"
        r"[^,]*close #(\d+)\s*\(([\d.]+)\s*/\s*[\d.]+\s*/\s*([\d.]+)\)",
        text,
    ):
        bar_ts_close = m.group(1)
        ticket = int(m.group(2))
        close_price = float(m.group(4))
        closes[ticket] = {"close_price": close_price, "bar_ts": bar_ts_close}

    if not opens:
        return

    # ── Build bar index map covering FULL test range (matching compute_pnl) ─
    all_ts = set()
    for op in opens.values():
        all_ts.add(op["bar_ts"])
    for cl in closes.values():
        all_ts.add(cl["bar_ts"])
    if not all_ts:
        return

    # Convert "YYYY.MM.DD HH:MM:SS" to epoch seconds, then bar index (900s bars)
    bar_epochs = []
    for ts_str in all_ts:
        try:
            dt = datetime.strptime(ts_str, "%Y.%m.%d %H:%M:%S")
            bar_epochs.append(int(dt.timestamp()) // 900)
        except ValueError:
            continue
    if not bar_epochs:
        return

    min_bar = min(bar_epochs)
    max_bar = max(bar_epochs)
    bar_count = max(1, max_bar - min_bar + 1)

    # ── Match opens to closes, compute per-trade PnL and bar-level returns ──
    per_bar_returns = [0.0] * bar_count
    trade_pnls = []
    profit_trades = 0
    loss_trades = 0
    gross_profit = 0.0
    gross_loss = 0.0
    consec_wins = 0
    consec_losses = 0
    max_consec_wins = 0
    max_consec_losses = 0
    equity_curve = [10000.0]

    for ticket, op in sorted(opens.items()):
        cl = closes.get(ticket)
        if cl is None:
            continue

        symbol = op["symbol"]
        entry = op["entry"]
        close_price = cl["close_price"]
        lot = op["lot"]
        direction = op["direction"]

        sym_params = SYMBOL_PARAMS.get(
            symbol, {"contract_size": 100_000, "pip_size": 0.0001}
        )
        contract_size = sym_params.get("contract_size", 100_000)
        spread_fb = sym_params.get("spread_fallback", 0.00015)
        commission_p = sym_params.get("commission", COMMISSION_PRICE)
        base_lot = BASE_LOT_MAP.get(symbol, 0.1)

        # ── Dollar PnL (for win_rate, profit_factor, drawdown) ──────
        if direction == "BUY":
            pnl = (close_price - entry) * lot * contract_size
        else:
            pnl = (entry - close_price) * lot * contract_size

        trade_pnls.append(pnl)

        # ── Per-bar fractional return (matches compute_pnl logic) ──────
        entry_adj = max(entry, 1e-6)
        sz = min(R_MAX / max(abs(entry - close_price) / entry_adj, 1e-5), 2.0)
        if direction == "BUY":
            raw = (close_price - entry - spread_fb - commission_p) / entry_adj
        else:
            raw = (entry - close_price - spread_fb - commission_p) / entry_adj
        bar_return = raw * sz * (lot / base_lot if base_lot > 0 else 1.0)

        try:
            dt = datetime.strptime(op["bar_ts"], "%Y.%m.%d %H:%M:%S")
            bar_idx = int(dt.timestamp()) // 900 - min_bar
        except ValueError:
            bar_idx = 0
        if 0 <= bar_idx < bar_count:
            per_bar_returns[bar_idx] = bar_return

        # ── Trade stats ───────────────────────────────────────────────
        if pnl > 0:
            profit_trades += 1
            gross_profit += pnl
            consec_wins += 1
            consec_losses = 0
            if consec_wins > max_consec_wins:
                max_consec_wins = consec_wins
        else:
            loss_trades += 1
            gross_loss += abs(pnl)
            consec_losses += 1
            consec_wins = 0
            if consec_losses > max_consec_losses:
                max_consec_losses = consec_losses

        equity_curve.append(equity_curve[-1] + pnl)

    total_trades = len(trade_pnls)
    if total_trades == 0:
        return

    total_net_profit = sum(trade_pnls)

    # ── Sharpe: CAIRO canonical formula (common.py:1104) ──────────────
    # mean(pnl) / std(pnl) * sqrt(BARS_PER_YEAR) on per-bar returns
    n = len(per_bar_returns)
    mean_bar = sum(per_bar_returns) / n if n > 0 else 0.0
    variance = sum((r - mean_bar) ** 2 for r in per_bar_returns) / n if n > 0 else 0.0
    std_bar = math.sqrt(variance) if variance > 1e-9 else 0.0
    sharpe = (mean_bar / std_bar) * math.sqrt(BARS_PER_YEAR) if std_bar > 1e-9 else 0.0

    # ── Max drawdown from equity curve ────────────────────────────────
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100.0 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # ── Additional metrics ────────────────────────────────────────────
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
    expected_payoff = total_net_profit / total_trades
    win_rate_pct = (profit_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    result["total_net_profit"] = round(total_net_profit, 6)
    result["gross_profit"] = round(gross_profit, 6)
    result["gross_loss"] = round(gross_loss, 6)
    result["profit_factor"] = round(profit_factor, 6)
    result["expected_payoff"] = round(expected_payoff, 6)
    result["total_trades"] = total_trades
    result["profit_trades"] = profit_trades
    result["loss_trades"] = loss_trades
    result["win_rate_pct"] = round(win_rate_pct, 2)
    result["sharpe_ratio"] = round(sharpe, 6)
    result["max_drawdown_pct"] = round(max_dd, 2)
    result["consecutive_wins"] = max_consec_wins
    result["consecutive_losses"] = max_consec_losses
    result["max_drawdown"] = round(peak * max_dd / 100.0 if peak > 0 else 0.0, 6)


# ═══════════════════════════════════════════════════════════════════════════
# EA trade CSV parsing (plan §3.3 — preferred over log regex)
# ═══════════════════════════════════════════════════════════════════════════

def _parse_trades_csv(csv_path: str | Path, bar_count: int = 0) -> dict | None:
    """Parse the EA's structured trade CSV (MULTI_CAIRO_EA writes this
    via FileWriteString on OnDeinit). Preferred over log regex.

    CSV columns (semicolon-delimited):
      ticket;open_time;close_time;type;lot;open_price;close_price;swap;commission;profit

    Returns a metrics dict matching _parse_trades_from_log() output,
    or None if the CSV is missing/empty.
    """
    import csv
    import math

    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None

    # ── CAIRO constants ──────────────────────────────────────────────
    R_MAX = 0.005
    COMMISSION_PRICE = 0.00005
    BARS_PER_YEAR = 252 * 24  # 6048 for M15

    # Read CSV (semicolon delimited)
    rows = []
    raw_text = ""
    try:
        raw_text = csv_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw_text = csv_path.read_text(encoding="latin-1")
    except Exception as e:
        print(f"[CSV] Read error: {e}")
        return None

    if not raw_text.strip():
        return None

    # Parse semicolon-delimited CSV using csv module
    try:
        reader = csv.DictReader(
            raw_text.splitlines(),
            delimiter=";",
            skipinitialspace=True,
        )
        rows = list(reader)
    except Exception as e:
        print(f"[CSV] Parse error: {e}")
        return None

    if not rows:
        return None

    # ── Extract trades ───────────────────────────────────────────────
    from config import SYMBOL_PARAMS, BASE_LOT_MAP

    trades = []
    for row in rows:
        try:
            ticket = int(row.get("ticket", "0"))
            open_time_str = row.get("open_time", "")
            close_time_str = row.get("close_time", "")
            row_symbol = row.get("symbol", "").strip()
            if not row_symbol:
                row_symbol = "EURUSD"
            direction = row.get("type", "").strip()
            lot = float(row.get("lot", "0.01"))
            open_price = float(row.get("open_price", "0"))
            close_price = float(row.get("close_price", "0"))
            swap = float(row.get("swap", "0"))
            commission = float(row.get("commission", "0"))
            profit = float(row.get("profit", "0"))

            # Parse timestamps: "2026.05.29 05:00" or "2026.05.29 05:00:00"
            open_dt_str = open_time_str.strip()
            close_dt_str = close_time_str.strip()
            if " " not in open_dt_str:
                continue

            date_part, time_part = open_dt_str.rsplit(" ", 1)
            time_parts = time_part.split(":")
            if len(time_parts) == 2:
                open_dt_str = f"{date_part} {time_parts[0]}:{time_parts[1]}:00"
            close_date, close_time = close_dt_str.rsplit(" ", 1)
            close_parts = close_time.split(":")
            if len(close_parts) == 2:
                close_dt_str = f"{close_date} {close_parts[0]}:{close_parts[1]}:00"

            from datetime import datetime
            try:
                open_dt = datetime.strptime(open_dt_str, "%Y.%m.%d %H:%M:%S")
                close_dt = datetime.strptime(close_dt_str, "%Y.%m.%d %H:%M:%S")
            except ValueError:
                continue

            trades.append({
                "ticket": ticket,
                "symbol": row_symbol,
                "direction": direction,
                "lot": lot,
                "open_price": open_price,
                "close_price": close_price,
                "swap": swap,
                "commission": commission,
                "profit": profit,
                "open_dt": open_dt,
                "close_dt": close_dt,
            })
        except (ValueError, KeyError, AttributeError):
            continue

    if not trades:
        return None

    # ── Build bar timeline from trade open/close times ──────────────
    # M15 bars = 900 seconds, compute bar index epoch
    min_bar = None
    max_bar = None
    bar_indices = {}
    for t in trades:
        for dt in (t["open_dt"], t["close_dt"]):
            epoch = int(dt.timestamp())
            bar = epoch // 900
            if min_bar is None or bar < min_bar:
                min_bar = bar
            if max_bar is None or bar > max_bar:
                max_bar = bar
            bar_indices[bar] = True

    if min_bar is None:
        return None

    # Per-bar returns MUST span the full test bar count (matching
    # compute_pnl's output length in common.py:1077). Use the provided
    # bar_count from the agent log when available, otherwise fall back
    # to the trade timestamps range.
    if bar_count and bar_count > 0:
        num_bars = bar_count
    else:
        num_bars = max(1, max_bar - min_bar + 1)
    per_bar_returns = [0.0] * num_bars

    # ── Compute per-bar returns (CAIRO canonical formula) ───────────
    trade_pnls = []
    profit_trades = 0
    loss_trades = 0
    gross_profit = 0.0
    gross_loss = 0.0
    consec_wins = 0
    consec_losses = 0
    max_consec_wins = 0
    max_consec_losses = 0
    equity_curve = [10000.0]

    for t in trades:
        pnl = t["profit"]
        trade_pnls.append(pnl)

        if pnl > 0:
            profit_trades += 1
            gross_profit += pnl
            consec_wins += 1
            consec_losses = 0
            if consec_wins > max_consec_wins:
                max_consec_wins = consec_wins
        else:
            loss_trades += 1
            gross_loss += abs(pnl)
            consec_losses += 1
            consec_wins = 0
            if consec_losses > max_consec_losses:
                max_consec_losses = consec_losses

        equity_curve.append(equity_curve[-1] + pnl)

        # Fractional bar return (matches compute_pnl)
        t_symbol = _normalize_symbol(t.get("symbol", "EURUSD"))
        sym_params = SYMBOL_PARAMS.get(
            t_symbol, {"contract_size": 100_000, "pip_size": 0.0001}
        )
        spread_fb = sym_params.get("spread_fallback", 0.00015)
        commission_p = sym_params.get("commission", 0.00005)
        base_lot = BASE_LOT_MAP.get(t_symbol, 0.1)

        entry = max(t["open_price"], 1e-6)
        entry_adj = entry
        sz = min(R_MAX / max(abs(t["close_price"] - entry) / entry_adj, 1e-5), 2.0)
        lot = t["lot"]

        if t["direction"] == "BUY":
            raw = (t["close_price"] - entry - spread_fb - commission_p) / entry_adj
        else:
            raw = (entry - t["close_price"] - spread_fb - commission_p) / entry_adj
        bar_return = raw * sz * (lot / base_lot if base_lot > 0 else 1.0)

        bar_idx_val = int(t["open_dt"].timestamp()) // 900 - min_bar
        if 0 <= bar_idx_val < num_bars:
            per_bar_returns[bar_idx_val] = bar_return

    # ── Compute metrics ─────────────────────────────────────────────
    total_trades = len(trade_pnls)
    total_net_profit = sum(trade_pnls)

    n = len(per_bar_returns)
    mean_bar = sum(per_bar_returns) / n if n > 0 else 0.0
    variance = sum((r - mean_bar) ** 2 for r in per_bar_returns) / n if n > 0 else 0.0
    std_bar = math.sqrt(variance) if variance > 1e-9 else 0.0
    sharpe = (mean_bar / std_bar) * math.sqrt(BARS_PER_YEAR) if std_bar > 1e-9 else 0.0

    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100.0 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
    expected_payoff = total_net_profit / total_trades
    win_rate_pct = (profit_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    return {
        "total_net_profit": round(total_net_profit, 6),
        "gross_profit": round(gross_profit, 6),
        "gross_loss": round(gross_loss, 6),
        "profit_factor": round(profit_factor, 6),
        "expected_payoff": round(expected_payoff, 6),
        "total_trades": total_trades,
        "profit_trades": profit_trades,
        "loss_trades": loss_trades,
        "win_rate_pct": round(win_rate_pct, 2),
        "sharpe_ratio": round(sharpe, 6),
        "max_drawdown_pct": round(max_dd, 2),
        "consecutive_wins": max_consec_wins,
        "consecutive_losses": max_consec_losses,
        "max_drawdown": round(peak * max_dd / 100.0 if peak > 0 else 0.0, 6),
        "final_balance": 10000.0 + total_net_profit,
    }


# ═══════════════════════════════════════════════════════════════════════════

def fold_test_window(fold_id: str) -> tuple[str, str]:
    """Derive the MT5 test window for a fold from its YYYYMM id.

    The Python evaluator computes fold metrics over the FULL test month
    (test_start = train_end = YYYY-MM-01, test_end = next month's 01).
    The MT5 job must span the same range for a valid reconciliation.
    """
    from datetime import timedelta
    try:
        y, m = int(str(fold_id)[:4]), int(str(fold_id)[4:6])
    except (TypeError, ValueError):
        return DEFAULT_FROM_DATE, DEFAULT_TO_DATE
    start = datetime(y, m, 1)
    end = (start + timedelta(days=31)).replace(day=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _job_dates(job: dict, from_date: str, to_date: str) -> tuple[str, str]:
    """Job-level dates take priority; otherwise derive from fold_id."""
    jf = (job.get("from_date") or "").strip()
    jt = (job.get("to_date") or "").strip()
    if jf and jt:
        return jf, jt
    return fold_test_window(job.get("fold_id", ""))


class Orchestrator:
    """Master controller for the fold testing pipeline."""

    def __init__(
        self,
        terminal_exe: str = "",
        terminal_data: str = "",
        agent_log_dir: str = "",
        from_date: str = DEFAULT_FROM_DATE,
        to_date: str = DEFAULT_TO_DATE,
    ):
        self.terminal_exe  = terminal_exe or DEFAULT_TERMINAL_EXE
        self.terminal_data = terminal_data or DEFAULT_TERMINAL_DATA
        self.agent_log_dir = agent_log_dir or DEFAULT_AGENT_LOG_DIR
        self.from_date     = from_date
        self.to_date       = to_date

        self.jq = JobQueue()
        self.jq.connect()

    def close(self):
        self.jq.close()

    def _prewarm_bridge(self, port: int, symbol: str, n_bars: int = 300) -> None:
        """Send N real EURUSD bars to the bridge to fill the warmup buffer.

        Uses bars from the training parquet. Resamples to the active timeframe
        from config (RR_MT5_PERIOD) to match what the EA will send.
        """
        try:
            import pandas as pd
            from config import REINFORCE_TIMEFRAME
            parquet = (
                r"C:\Users\Milad\Desktop\Reinforce\Reinforce_Windows_20260613"
                r"\multi_live\data\EURUSD\15m\live_fold\parquets"
                r"\fold_202606_train.parquet"
            )
            df_m15 = pd.read_parquet(parquet)

            # Resample to active timeframe if needed
            if REINFORCE_TIMEFRAME == "15m":
                df = df_m15
                bar_seconds = 900
            else:
                rule = {"1h": "1h", "5m": "5min", "1m": "1min"}.get(REINFORCE_TIMEFRAME, "1h")
                df_m15["_time"] = pd.date_range("2015-01-01", periods=len(df_m15), freq="15min")
                df_m15 = df_m15.set_index("_time")
                df = df_m15[["open","high","low","close","avg_spread"]].resample(rule).agg({
                    "open": "first", "high": "max", "low": "min",
                    "close": "last", "avg_spread": "mean",
                }).dropna()
                bar_seconds = {"1h": 3600, "5m": 300, "1m": 60}.get(REINFORCE_TIMEFRAME, 3600)

            n = min(n_bars, len(df))
            start_idx = max(0, len(df) - n)
            for i in range(start_idx, start_idx + n):
                row = df.iloc[i]
                bar = json.dumps({
                    "symbol": symbol,
                    "open":   float(row["open"]),
                    "high":   float(row["high"]),
                    "low":    float(row["low"]),
                    "close":  float(row["close"]),
                    "spread": float(row.get("avg_spread", 0.0001)),
                    "timestamp": int(time.time()) + (i - start_idx) * bar_seconds,
                }) + "\n"
                try:
                    sock = socket.create_connection(("127.0.0.1", port), timeout=2)
                    sock.sendall(bar.encode())
                    raw = b""
                    while b"\n" not in raw:
                        chunk = sock.recv(4096)
                        if not chunk:
                            break
                        raw += chunk
                    sock.close()
                except Exception:
                    pass
            _log(f"[Prewarm] Sent {n} {REINFORCE_TIMEFRAME} bars to port {port}"
                 f" (last bar close={df.close.iloc[-1]:.5f})")
        except Exception as e:
            _log(f"[Prewarm] Failed: {e}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Main job lifecycle ──────────────────────────────────────────────

    def run_one_job(
        self,
        job: dict,
        terminal_id: str = "terminal_0",
        terminal_index: int = 0,
    ) -> dict:
        """Execute a single job end-to-end.

        Returns result dict with: success, job_id, symbol, fold_id,
        final_balance, total_trades, sharpe, max_dd, win_rate, pnl,
        checkpoint_path, elapsed_s, errors.
        """
        job_id   = job["job_id"]
        symbol   = job["symbol"]
        fold_id  = job["fold_id"]
        ckpt     = job["checkpoint_path"]
        port     = 19998  # match MULTI_CAIRO_EA.ex5 compiled default

        # Per-job test window: job columns win, else derive from fold_id
        job_from, job_to = _job_dates(job, self.from_date, self.to_date)

        model = job.get("tester_model") or 0
        fidelity_mode = "every_tick" if model == 0 else (
            "m1_ohlc" if model == 1 else f"model_{model}"
        )
        data_source = job.get("data_source") or "broker_live"

        result = {
            "success": False,
            "job_id": job_id,
            "symbol": symbol,
            "fold_id": fold_id,
            "fidelity_mode": fidelity_mode,
            "data_source": data_source,
            "checkpoint_path": ckpt,
            "final_balance": 0.0,
            "total_trades": 0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.0,
            "pnl": 0.0,
            "elapsed_s": 0.0,
            "errors": [],
        }
        t0 = time.time()

        # 1. Mark job as running
        if not self.jq.start_job(job_id, terminal_id, port):
            result["errors"].append("start_job_failed")
            return result
        _log(f"Job {job_id}: {symbol} fold_{fold_id} - RUNNING (port {port})")

        # 2. Start bridge server
        bridge = BridgeManager(port=port, checkpoint_path=ckpt, symbol=symbol)
        bridge.free_port(port)
        if not bridge.start():
            self.jq.fail_job(job_id, "bridge_start_failed")
            result["errors"].append("bridge_start_failed")
            return result

        if not bridge.wait_ready(timeout=60.0):
            bridge.stop()
            self.jq.fail_job(job_id, "bridge_not_ready")
            result["errors"].append("bridge_not_ready")
            return result
        _log(f"Job {job_id}: bridge ready on port {port}")

        # Pre-warm skipped — using longer date range so terminal provides
        # natural warmup bars. Pre-warm bars from training parquet are NOT
        # continuous with terminal bars, causing feature distribution shift
        # and all-HOLD predictions.
        # self._prewarm_bridge(port, symbol, 300)

        # 3. Generate configs
        gen = ConfigGenerator(output_dir=str(
            Path(self.terminal_data) / "tester"
        ))
        configs = gen.generate(
            symbol=symbol,
            fold_id=fold_id,
            checkpoint_path=ckpt,
            bridge_port=port,
            from_date=job_from,
            to_date=job_to,
            ea_name="MULTI_CAIRO_EA",
            ea_params={"BridgePort": str(port),
                        "LotSize": str(BASE_LOT_MAP.get(symbol, 0.03))},
            model=job.get("tester_model") or 0,
            data_source=data_source,
        )
        tester_ini = configs["tester_config"]
        _log(f"Job {job_id}: config written to {tester_ini} ({job_from} -> {job_to})")

        # 4. Launch terminal
        terminal_proc = None
        try:
            # Remove any stale trades.csv so an aborted test cannot leak the
            # previous job's trade file into the result parser.
            for _sd in [Path(self.agent_log_dir).parent,
                        Path(self.terminal_data) / "Tester"]:
                for _sub in [_sd / "MQL5" / "Files" / "trades.csv",
                             _sd / "trades.csv"]:
                    try:
                        if _sub.exists():
                            _sub.unlink()
                    except Exception:
                        pass

            # Find the agent log we will parse afterwards. The whole file is
            # read at collection time and sliced to the LAST test segment,
            # so no byte-offset snapshot is needed here.
            log_path = _find_agent_log(self.agent_log_dir)
            cmd = [self.terminal_exe, "/config:" + str(tester_ini.resolve())]
            _log(f"Job {job_id}: launching terminal...")
            terminal_proc = subprocess.Popen(
                cmd,
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    if sys.platform == "win32" else 0
                ),
            )
        except Exception as e:
            bridge.stop()
            self.jq.fail_job(job_id, f"terminal_launch_failed: {e}")
            result["errors"].append(f"terminal_launch_failed: {e}")
            return result

        # 5. Wait for terminal to exit (full-month every-tick tests can take
        #    several minutes per job — 30 min cap)
        _log(f"Job {job_id}: waiting for terminal (PID {terminal_proc.pid})...")
        try:
            terminal_proc.wait(timeout=1800.0)
        except subprocess.TimeoutExpired:
            _log(f"Job {job_id}: terminal timed out - killing")
            try:
                terminal_proc.kill()
                terminal_proc.wait(timeout=5)
            except Exception:
                pass
            bridge.stop()
            self.jq.fail_job(job_id, "terminal_timeout")
            result["errors"].append("terminal_timeout")
            return result

        _log(f"Job {job_id}: terminal exited, collecting results...")

        # 5b. The tracked terminal64 process can exit BEFORE the agent
        #     finishes the test (LiveUpdate self-restart etc.). Poll the
        #     agent log for the completion summary before judging the run.
        _poll_deadline = time.time() + 300.0
        while time.time() < _poll_deadline:
            if log_path and log_path.exists():
                _raw_seg, _ = _parse_last_test_segment(log_path)
                _txt = _decode_agent_log(_raw_seg)
                if "bars generated" in _txt or "Test passed in" in _txt:
                    break
            time.sleep(5.0)

        # 6. Collect results
        # Priority: EA trade CSV (plan §3.3) > log regex > agent log balance only
        agent_metrics = {}
        csv_used = False

        # 6a. Always parse agent log first — needed for bar count,
        #     final_balance, and test_passed. The CSV supplies trade-level
        #     metrics (Sharpe, win_rate) with the correct bar count.
        log_metrics = {}
        log_bar_count = 0
        if log_path and log_path.exists():
            raw_seg, raw_full = _parse_last_test_segment(log_path)
            log_metrics = _parse_agent_log(raw_seg)
            log_bar_count = log_metrics.get("bars", 0)
            seg_text = _decode_agent_log(raw_seg)
            _log(f"Job {job_id}: log={log_path.name}, "
                 f"balance={log_metrics['final_balance']:.2f}, "
                 f"bars={log_bar_count}, "
                 f"passed={log_metrics['test_passed']}")

            # ── Completeness gate ─────────────────────────────────────────
            # An aborted test (terminal self-exit, tick-download stall) has
            # NO completion summary in the agent log. Only the CURRENT test
            # segment may be consulted — full-file parses leak stale metrics
            # from earlier tests (this masked aborted runs as completed).
            if not log_metrics["test_passed"] and "bars generated" not in seg_text:
                bridge.stop()
                self.jq.fail_job(job_id, "test_incomplete_no_summary")
                result["errors"].append("test_incomplete_no_summary")
                result["elapsed_s"] = time.time() - t0
                _log(f"Job {job_id}: FAILED - test incomplete (no summary). Will retry.")
                return result

        # 6b. Try EA trade CSV — preferred source for trade-level metrics
        csv_path = None
        for search_dir in [
            Path(self.agent_log_dir).parent,   # Agent-127.0.0.1-3000/
            Path(self.terminal_data) / "Tester",
        ]:
            for sub in [
                search_dir / "MQL5" / "Files" / "trades.csv",
                search_dir / "trades.csv",
            ]:
                if sub.exists():
                    csv_path = sub
                    break
            if csv_path:
                break

        if csv_path:
            # Prefer the EA's structured trade CSV (real per-trade profit from
            # DEAL_PROFIT) over the lossy agent-log regex. A stale CSV cannot
            # leak in because it is deleted before each terminal launch, and the
            # completeness gate above only lets completed tests reach here.
            csv_metrics = _parse_trades_csv(csv_path, bar_count=log_bar_count)
            if csv_metrics:
                agent_metrics = csv_metrics
                csv_used = True
                _log(f"Job {job_id}: parsed trades.csv "
                     f"({agent_metrics['total_trades']} trades)")

        if not csv_used:
            agent_metrics = log_metrics
            if log_metrics.get("total_trades", 0) > 0:
                _log(f"Job {job_id}: using agent-log trade reconstruction "
                     f"({log_metrics['total_trades']} trades)")

        # Check for XML report (MT5 may generate both HTML + optional XML)
        # For now, use agent log metrics — in production this will parse XML
        xml_dir = Path(self.terminal_data) / "tester"
        xml_candidates = list(xml_dir.glob(f"*{symbol}*{fold_id}*.xml")) + \
                         list(xml_dir.glob("*.xml"))
        report_metrics = None
        for xp in sorted(xml_candidates,
                         key=lambda p: p.stat().st_mtime, reverse=True):
            report_metrics = parse_report(xp)
            if report_metrics:
                _log(f"Job {job_id}: parsed XML report: {xp.name}")
                break

        # Merge: XML report takes priority, agent log fills gaps
        # Agent log now includes trade-level reconstruction:
        #   sharpe_ratio, win_rate_pct, max_drawdown_pct, gross_profit,
        #   gross_loss, profit_factor, profit_trades, loss_trades, etc.
        if report_metrics:
            final_trades = report_metrics.get("total_trades", 0)
            final_sharpe = report_metrics.get("sharpe_ratio", 0.0)
            final_max_dd = report_metrics.get("max_drawdown_pct", 0.0)
            final_winr   = report_metrics.get("win_rate_pct", 0.0)
            final_pnl    = report_metrics.get("total_net_profit", 0.0)
            final_gross_profit  = report_metrics.get("gross_profit", 0.0)
            final_gross_loss    = report_metrics.get("gross_loss", 0.0)
            final_profit_factor = report_metrics.get("profit_factor", 0.0)
            final_expected_payoff = report_metrics.get("expected_payoff", 0.0)
            final_profit_trades = report_metrics.get("profit_trades", 0)
            final_loss_trades   = report_metrics.get("loss_trades", 0)
            final_consec_wins   = report_metrics.get("consecutive_wins", 0)
            final_consec_losses = report_metrics.get("consecutive_losses", 0)
            final_max_dd_money  = report_metrics.get("max_drawdown", 0.0)
        else:
            balance = agent_metrics.get("final_balance", 10000.0)
            final_trades = agent_metrics.get("total_trades", 0)
            final_sharpe = agent_metrics.get("sharpe_ratio", 0.0)
            final_max_dd = agent_metrics.get("max_drawdown_pct", 0.0)
            final_winr   = agent_metrics.get("win_rate_pct", 0.0)
            final_pnl    = balance - 10000.0  # account truth (incl. swap/comm)
            final_gross_profit  = agent_metrics.get("gross_profit", 0.0)
            final_gross_loss    = agent_metrics.get("gross_loss", 0.0)
            final_profit_factor = agent_metrics.get("profit_factor", 0.0)
            final_expected_payoff = agent_metrics.get("expected_payoff", 0.0)
            final_profit_trades = agent_metrics.get("profit_trades", 0)
            final_loss_trades   = agent_metrics.get("loss_trades", 0)
            final_consec_wins   = agent_metrics.get("consecutive_wins", 0)
            final_consec_losses = agent_metrics.get("consecutive_losses", 0)
            final_max_dd_money  = agent_metrics.get("max_drawdown", 0.0)

        result.update({
            "success": True,
            "final_balance": agent_metrics.get("final_balance", 10000.0),
            "total_trades": final_trades,
            "sharpe": final_sharpe,
            "max_dd": final_max_dd,
            "win_rate": final_winr,
            "pnl": final_pnl,
            "gross_profit": final_gross_profit,
            "gross_loss": final_gross_loss,
            "profit_factor": final_profit_factor,
            "expected_payoff": final_expected_payoff,
            "profit_trades": final_profit_trades,
            "loss_trades": final_loss_trades,
            "consecutive_wins": final_consec_wins,
            "consecutive_losses": final_consec_losses,
            "max_drawdown_money": final_max_dd_money,
            "elapsed_s": time.time() - t0,
        })

        # 7. Mark job complete
        self.jq.complete_job(
            job_id,
            sharpe=final_sharpe,
            max_dd=final_max_dd,
            win_rate=final_winr,
            trade_count=final_trades,
            pnl=final_pnl,
        )

        # 8. Stop bridge
        bridge.stop()

        _log(f"Job {job_id}: COMPLETE - elapsed {result['elapsed_s']:.1f}s, "
             f"trades={final_trades}, pnl={final_pnl:.2f}")
        return result

    def status(self) -> dict:
        """Return pipeline status summary."""
        return {
            "jobs": self.jq.status_summary(),
            "pending": self.jq.pending_count(),
            "running": self.jq.running_count(),
        }


# ── Main / self-test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(
        description="RR Orchestrator — master controller for fold testing"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=(
            r"C:\Users\Milad\Desktop\Reinforce\Reinforce_Windows_20260613"
            r"\multi_live\data\EURUSD\15m\live_fold"
        ),
        help="Checkpoint directory for the test job",
    )
    parser.add_argument(
        "--from-date",
        type=str,
        default="2026-06-01",
        help="Test start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--to-date",
        type=str,
        default="2026-06-02",
        help="Test end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--symbol", type=str, default="EURUSD",
        help="Trading symbol",
    )
    parser.add_argument(
        "--fold-id", type=str, default="202606",
        help="Fold ID string",
    )
    parser.add_argument(
        "--terminal-exe", type=str, default="",
        help="Override terminal EXE path",
    )
    parser.add_argument(
        "--terminal-data", type=str, default="",
        help="Override terminal data folder",
    )
    parser.add_argument(
        "--skip-terminal", action="store_true",
        help="Skip terminal launch (test job_queue + config generation only)",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run ALL queued jobs sequentially (ignore --symbol/--fold-id)",
    )
    args = parser.parse_args()

    print("=" * 70)
    if args.batch:
        print("  ORCHESTRATOR BATCH RUN")
    else:
        print("  ORCHESTRATOR INTEGRATION TEST")
    print("=" * 70)

    # ── Setup ───────────────────────────────────────────────────────────
    print("\n[1] Setting up job queue...")
    orch = Orchestrator(
        terminal_exe=args.terminal_exe,
        terminal_data=args.terminal_data,
        from_date=args.from_date,
        to_date=args.to_date,
    )

    if args.batch:
        pass  # jobs pre-queued externally
    else:
        db = orch.jq._conn
        db.execute(
            """INSERT INTO jobs (symbol, fold_id, checkpoint_path, status, tick_coverage_pct)
               VALUES (?, ?, ?, 'queued', 100.0)""",
            (args.symbol, args.fold_id, args.checkpoint_dir),
        )
        db.commit()
    summary = orch.status()
    print(f"    Queue: {summary['jobs']}")
    print(f"    Pending: {summary['pending']}")

    # ── Run jobs ────────────────────────────────────────────────────────
    all_results = []
    if args.batch:
        print(f"\n[2] Running batch - processing all queued jobs...")
        for attempt in range(100):  # safety limit
            job = orch.jq.next_queued()
            if not job:
                break
            print(f"\n  --- Batch job {attempt + 1}: {job['symbol']} fold_{job['fold_id']} ---")
            result = orch.run_one_job(
                job,
                terminal_id=f"orch_batch_{attempt}",
                terminal_index=attempt,
            )
            all_results.append(result)
            print(f"  Done: {'OK' if result['success'] else 'FAIL'}")
        print(f"\n[2] Batch complete: {len(all_results)} jobs processed")
    else:
        # ── Pull and run the job ────────────────────────────────────────
        job = orch.jq.next_queued()
        if not job:
            print("\n[FAIL] No queued jobs found")
            orch.close()
            sys.exit(1)

        print(f"\n[2] Next job: ID={job['job_id']}  "
              f"{job['symbol']} fold_{job['fold_id']}")
        print(f"    Checkpoint: {job['checkpoint_path']}")

        if args.skip_terminal:
            print("\n[3] --skip-terminal: stopping after config generation")
            result = {
                "success": True, "job_id": job["job_id"],
                "symbol": args.symbol, "fold_id": args.fold_id,
                "final_balance": 10000.0, "total_trades": 0,
                "sharpe": 0.0, "max_dd": 0.0, "win_rate": 0.0,
                "pnl": 0.0, "elapsed_s": 0.0, "errors": [],
            }
        else:
            print(f"\n[3] Running job through full pipeline...")
            result = orch.run_one_job(
                job,
                terminal_id="orch_test",
                terminal_index=0,
            )
        all_results = [result]

    # ── Report ──────────────────────────────────────────────────────────
    for res in all_results:
        print(f"\n{'=' * 70}")
        print(f"  JOB RESULT: {res['symbol']} fold_{res['fold_id']}")
        print(f"{'=' * 70}")
        print(f"  Job ID:        {res['job_id']}")
        print(f"  Symbol:        {res['symbol']}")
        print(f"  Fold:          {res['fold_id']}")
        print(f"  Success:       {res['success']}")
        print(f"  Final balance: {res['final_balance']:.2f}")
        print(f"  Total trades:  {res['total_trades']}")
        print(f"  PnL:           {res['pnl']:.2f}")
        print(f"  Elapsed:       {res['elapsed_s']:.1f}s")
        if res["errors"]:
            print(f"  Errors:        {res['errors']}")

    # ── Job queue final status ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  JOB QUEUE FINAL STATUS")
    print(f"{'=' * 70}")
    if all_results:
        for res in all_results:
            final_job = orch.jq.get_job(res["job_id"])
            if final_job:
                print(f"  [{final_job['symbol']} fold_{final_job['fold_id']}] "
                      f"status={final_job['status']} sharpe={final_job['sharpe']:.4f}")
    print(f"  Queue summary: {orch.jq.status_summary()}")

    # ── Write results CSV ───────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  RESULTS CSV")
    print(f"{'=' * 70}")
    results = []
    for res in all_results:
        results.append({
            "fold_id": res["fold_id"],
            "symbol": res["symbol"],
            "fidelity_mode": res.get("fidelity_mode", ""),
            "data_source": res.get("data_source", ""),
            "total_net_profit": res["pnl"],
            "gross_profit": res.get("gross_profit", 0.0),
            "gross_loss": res.get("gross_loss", 0.0),
            "profit_factor": res.get("profit_factor", 0.0),
            "expected_payoff": res.get("expected_payoff", 0.0),
            "max_drawdown": res.get("max_drawdown_money", 0.0),
            "max_drawdown_pct": res["max_dd"],
            "total_trades": res["total_trades"],
            "win_rate_pct": res["win_rate"],
            "profit_trades": res.get("profit_trades", 0),
            "loss_trades": res.get("loss_trades", 0),
            "consecutive_wins": res.get("consecutive_wins", 0),
            "consecutive_losses": res.get("consecutive_losses", 0),
            "sharpe_ratio": res["sharpe"],
            "recovery_factor": 0.0,
            "total_swap": 0.0,
            "total_commission": 0.0,
            "flagged": not res["success"],
            "flag_reasons": "; ".join(res["errors"]),
        })
    written_pass, written_excl = write_results(
        results,
        ROOT_DIR / "data" / "results" / "results.csv",
        ROOT_DIR / "data" / "results" / "results_excluded.csv",
    )
    print(f"    Passed rows:  {written_pass}")
    print(f"    Excluded rows:{written_excl}")

    # Show the actual CSV row
    csv_path = ROOT_DIR / "data" / "results" / "results.csv"
    if csv_path.exists():
        print(f"\n    {csv_path.name} content:")
        for line in csv_path.read_text().splitlines():
            print(f"    {line}")

    orch.close()
    print(f"\n{'=' * 70}")
    print(f"  INTEGRATION TEST COMPLETE")
    print(f"{'=' * 70}")
