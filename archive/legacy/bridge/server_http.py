"""
server_http.py — Thin HTTP wrapper around bridge_core.

Production bridge server: listens for bar POSTs from MT5 EA via HTTP WebRequest.
Uses the same bridge_core module as server_socket.py (Strategy Tester transport).

Extracted + refactored from CAIRO's multi_live/04_live_engine.py (Phase 1).
"""

from __future__ import annotations

import gc
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ── Path setup: import CAIRO modules ─────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "train"))            # common.py, regime_detector, train_dt
sys.path.insert(0, str(ROOT_DIR / "live"))             # equity_curve_trader, circuit_breaker, etc.

# ── Bridge core imports ──────────────────────────────────────────────────────
from bridge.bridge_core import (
    ACTIVE_SYMBOLS,
    ALGOS,
    ALGOS_DT,
    ALGOS_GBM,
    ALGOS_SB3,
    BarBuffer,
    BarValidator,
    LiveModels,
    _log,
    compute_features_df,
    gc_col,
    process_bar_core,
)
from config import (
    ETA,
    HOLD_MIN_BARS,
    LOT_SCALE_MAX,
    MAX_LOOKBACK,
    PROP_CHALLENGE_BALANCE,
    PROP_DAILY_LOSS_LIMIT,
    PROP_ENABLED,
    PROP_LOT_SOFT_THRESH,
    PROP_PROFIT_TARGET,
    PROP_TOTAL_DD_LIMIT,
    R_MAX,
    WARMUP_BARS,
    get_base_lot,
)

# ── Optional CAIRO risk modules ──────────────────────────────────────────────
try:
    from prop_firm_guard import PropFirmGuard
    _PROP_GUARD_AVAILABLE = True
except ImportError:
    PropFirmGuard = None
    _PROP_GUARD_AVAILABLE = False

try:
    from position_sizer import PositionSizer
    _SIZER_AVAILABLE = True
except ImportError:
    PositionSizer = None
    _SIZER_AVAILABLE = False

try:
    from telegram_notifier import TelegramNotifier, DailyPnLTracker
    _NOTIFIER_AVAILABLE = True
except ImportError:
    TelegramNotifier = None
    DailyPnLTracker = None
    _NOTIFIER_AVAILABLE = False

try:
    from trade_logger import TradeLogger
    _TRADE_LOGGER_AVAILABLE = True
except ImportError:
    TradeLogger = None
    _TRADE_LOGGER_AVAILABLE = False

try:
    from equity_curve_trader import EquityCurveTrend as _ECT
    _ect = _ECT()
    _ECT_AVAILABLE = True
except ImportError:
    _ect = None
    _ECT_AVAILABLE = False

try:
    from circuit_breaker import CircuitBreaker
    _CB_AVAILABLE = True
except ImportError:
    CircuitBreaker = None
    _CB_AVAILABLE = False

# ── Module-level state ───────────────────────────────────────────────────────
HTTP_HOST = "0.0.0.0"
HTTP_PORT = int(os.environ.get("RR_BRIDGE_PORT", "9999"))
_CHECKPOINT_PATH: str | None = os.environ.get("RR_CHECKPOINT_PATH", None)

_active_models:   dict[str, LiveModels]    = {}
_active_buffers:  dict[str, BarBuffer]     = {}
_active_sizers:   dict                     = {}
_active_base_lots: dict[str, float]        = {}
_bar_validators:  dict[str, BarValidator]  = {}
_bars_since_flip: dict[str, int]           = {}
_last_equity_data: dict[str, float]        = {"balance": 0.0, "equity": 0.0}

_pnl_tracker:    DailyPnLTracker | None = None
_pnl_trackers:   dict[str, DailyPnLTracker] = {}
_notifier:       TelegramNotifier | None = None
_trade_logger:   TradeLogger | None = None
_prop_guard:     PropFirmGuard | None = None
_cb:             CircuitBreaker | None = None

SHUTDOWN_FLAG = False


# ── Terminate any existing Python processes on the same port ────────────────
def _free_port(port: int):
    import subprocess
    out = subprocess.run(
        ["netstat", "-ano"], capture_output=True, text=True,
    ).stdout
    for line in out.splitlines():
        if f":{port}" in line and "LISTENING" in line:
            pid = line.strip().split()[-1]
            try:
                os.kill(int(pid), signal.SIGTERM)
                _log(f"[INIT] Killed PID {pid} holding port {port}")
            except Exception:
                pass


# ── HTTP handler ─────────────────────────────────────────────────────────────


class BridgeHTTPHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler wrapping bridge_core's process_bar_core."""

    def log_message(self, fmt, *args):
        pass  # suppress default stderr logging

    def _send_json(self, data: dict, code: int = 200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok"})
        elif self.path == "/status":
            status = {}
            for sym in ACTIVE_SYMBOLS:
                buf = _active_buffers.get(sym)
                status[sym] = {
                    "buffer_bars": len(buf) if buf else 0,
                    "buffer_ready": buf.ready() if buf else False,
                    "models_loaded": len(_active_models.get(sym, LiveModels(sym, Path("."))).algos) if sym in _active_models else 0,
                }
            self._send_json({"status": status})
        else:
            self._send_json({"error": "not_found"}, code=404)

    def do_POST(self):
        if self.path == "/bar":
            self._handle_bar()
        elif self.path == "/equity":
            self._handle_equity()
        elif self.path == "/history":
            self._handle_history()
        else:
            self._send_json({"error": "not_found"}, code=404)

    def _handle_bar(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode() if content_len > 0 else "{}"

        try:
            bar = json.loads(body)
        except json.JSONDecodeError:
            self._send_json({"action": "HOLD", "lot": 0.0, "reason": "bad_json"})
            return

        symbol = bar.get("symbol", "").upper()
        if not symbol:
            self._send_json({"action": "HOLD", "lot": 0.0, "reason": "missing_symbol"})
            return

        models = _active_models.get(symbol)
        buffer = _active_buffers.get(symbol)
        sizer  = _active_sizers.get(symbol)
        base_lot = _active_base_lots.get(symbol, 0.1)

        if models is None or buffer is None:
            self._send_json({"action": "HOLD", "lot": 0.0,
                             "reason": f"unknown_symbol:{symbol}"})
            return

        # Per-symbol validator (lazy init)
        if symbol not in _bar_validators:
            _bar_validators[symbol] = BarValidator(symbol=symbol)

        # Track previous action for hold-guard
        sym_tracker = _pnl_trackers.get(symbol)
        prev_action = sym_tracker.current_action if sym_tracker else "HOLD"
        prev_bars_since = _bars_since_flip.get(symbol, 0)

        # Core pipeline
        action_str, details, elapsed_ms, new_bars_since = process_bar_core(
            bar=bar,
            models=models,
            buffer=buffer,
            validator=_bar_validators[symbol],
            feature_cols=models.feature_cols,
            prev_action=prev_action,
            bars_since_flip=prev_bars_since,
        )
        _bars_since_flip[symbol] = new_bars_since

        # Position sizing
        lot_scale = 1.0
        if sizer is not None:
            import math as _math
            close_price = float(bar.get("close", 0))
            prev_close = float(buffer.bars["close"].iloc[-2]) if len(buffer.bars) >= 2 else close_price
            if prev_close > 0 and close_price > 0:
                sizer.update_bar(_math.log(close_price / prev_close))
            atr14 = 0.0
            guard_factor = (0.0 if (_prop_guard and _prop_guard.is_hard_stop) else
                            0.5 if (_prop_guard and _prop_guard.is_soft_warn) else 1.0)
            lot_scale = sizer.compute_scale(atr14, close_price, guard_factor)

        # ECT
        if _ECT_AVAILABLE and _ect is not None and action_str != "HOLD":
            eq_now = float(_last_equity_data.get("equity") or _last_equity_data.get("balance") or 0.0)
            ect_factor = _ect.update_and_scale(eq_now)
            if ect_factor == 0.0:
                action_str = "HOLD"
                lot_scale = 0.0
            elif ect_factor < 1.0:
                lot_scale = round(lot_scale * ect_factor, 4)

        # Circuit Breaker
        guard_reason = "ok"
        if _CB_AVAILABLE and _cb is not None and action_str != "HOLD":
            bar_ts = datetime.fromtimestamp(bar.get("timestamp", time.time()), tz=timezone.utc)
            action_str, lot_scale, cb_reason = _cb.gate_action(action_str, lot_scale, symbol, bar, bar_ts)

        # PropFirmGuard
        if _prop_guard:
            action_str, lot_scale, guard_reason = _prop_guard.check_signal(action_str, lot_scale)

        # Response
        self._send_json({
            "action":       action_str,
            "lot":          round(base_lot * lot_scale, 4),
            "weight":       details.get("cairo_weight", 1.0),
            "lot_scale":    round(lot_scale, 4),
            "base_lot":     base_lot,
            "symbol":       symbol,
            "winner_votes": details.get("winner_votes", 0),
            "n_models":     details.get("n_models", 0),
            "votes":        details.get("model_votes", {}),
            "latency_ms":   round(elapsed_ms, 1),
            "guard_reason": guard_reason,
        })

        # PnL tracking
        sym_tracker = _pnl_trackers.get(symbol, _pnl_tracker)
        if sym_tracker:
            sym_tracker.record_bar(
                action_str, bar.get("close", 0),
                datetime.fromtimestamp(bar.get("timestamp", time.time()), tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                details.get("model_votes", {}), details.get("cairo_weight", 1.0),
                lot_scale, bar.get("spread", bar.get("avg_spread", 0.0001)),
            )

        # Balance/equity tracking from bar payload
        if "balance" in bar:
            _last_equity_data["balance"] = float(bar["balance"])
        if "equity" in bar:
            _last_equity_data["equity"] = float(bar["equity"])

    def _handle_equity(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode() if content_len > 0 else "{}"
        try:
            data = json.loads(body)
            _last_equity_data["balance"] = float(data.get("balance", _last_equity_data["balance"]))
            _last_equity_data["equity"] = float(data.get("equity", _last_equity_data["equity"]))
            self._send_json({"status": "ok"})
        except Exception as e:
            self._send_json({"status": "error", "reason": str(e)}, code=400)

    def _handle_history(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode() if content_len > 0 else "{}"
        try:
            bars = json.loads(body)
            if not isinstance(bars, list):
                self._send_json({"status": "error", "reason": "expected list"}, code=400)
                return
            symbol = None
            for b in bars:
                sym = b.get("symbol", "").upper()
                if sym and symbol is None:
                    symbol = sym
                buf = _active_buffers.get(sym) if sym else None
                if buf:
                    buf.append(b)
            self._send_json({"status": "ok", "bars_ingested": len(bars)})
        except Exception as e:
            self._send_json({"status": "error", "reason": str(e)}, code=400)


# ── Main ─────────────────────────────────────────────────────────────────────


def load_models_from_checkpoint(checkpoint_path: str | None = None) -> dict[str, LiveModels]:
    """Load LiveModels for all active symbols from checkpoint directories."""
    checkpoint_base = Path(checkpoint_path) if checkpoint_path else None
    models_map: dict[str, LiveModels] = {}

    for symbol in ACTIVE_SYMBOLS:
        if checkpoint_base and checkpoint_base.exists():
            fold_dir = checkpoint_base
        else:
            fold_dir = ROOT_DIR / "multi_live" / "data" / symbol / "15m" / "live_fold"

        _log(f"[INIT] Loading models for {symbol} from {fold_dir}")
        lm = LiveModels(symbol, fold_dir)
        lm.load()
        models_map[symbol] = lm

    return models_map


def main():
    global _active_models, _active_buffers, _active_sizers, _active_base_lots
    global _notifier, _pnl_tracker, _pnl_trackers, _trade_logger, _prop_guard, _cb

    checkpoint_path = _CHECKPOINT_PATH or os.environ.get("RR_CHECKPOINT_PATH")
    port = int(os.environ.get("RR_BRIDGE_PORT", str(HTTP_PORT)))

    _free_port(port)

    _log(f"[INIT] RR Bridge Server starting...")
    _log(f"  Active symbols: {ACTIVE_SYMBOLS}")
    _log(f"  HTTP port: {port}")
    if checkpoint_path:
        _log(f"  Checkpoint path: {checkpoint_path}")

    # Telegram notifier
    if _NOTIFIER_AVAILABLE and TelegramNotifier:
        _notifier = TelegramNotifier()
        _pnl_tracker = DailyPnLTracker(_notifier)
        _log("[INIT] Telegram notifier initialized")

    # Trade logger
    if _TRADE_LOGGER_AVAILABLE and TradeLogger:
        _trade_logger = TradeLogger()
        _log("[INIT] Trade logger initialized")

    # Circuit breaker
    if _CB_AVAILABLE and CircuitBreaker:
        _cb = CircuitBreaker()
        _log("[INIT] Circuit breaker initialized")

    # PropFirmGuard
    if _PROP_GUARD_AVAILABLE and PropFirmGuard and PROP_ENABLED:
        _prop_guard = PropFirmGuard(
            challenge_balance=PROP_CHALLENGE_BALANCE or 10_000.0,
            daily_loss_limit=PROP_DAILY_LOSS_LIMIT,
            total_dd_limit=PROP_TOTAL_DD_LIMIT,
            profit_target=PROP_PROFIT_TARGET,
            soft_warn_threshold=PROP_LOT_SOFT_THRESH,
        )
        _log("[INIT] PropFirmGuard initialized")

    # Load models
    models_map = load_models_from_checkpoint(checkpoint_path)
    _active_models = models_map

    # Per-symbol setup
    for symbol, lm in models_map.items():
        _active_buffers[symbol] = BarBuffer()
        if _SIZER_AVAILABLE and PositionSizer:
            _active_sizers[symbol] = PositionSizer()
        _active_base_lots[symbol] = get_base_lot(symbol)
        _bar_validators[symbol] = BarValidator(symbol=symbol)
        _bars_since_flip[symbol] = 0
        if _pnl_tracker:
            _pnl_trackers[symbol] = DailyPnLTracker(_notifier)

    # HTTP server
    server = ThreadingHTTPServer((HTTP_HOST, port), BridgeHTTPHandler)

    def _shutdown(signum=None, frame=None):
        global SHUTDOWN_FLAG
        SHUTDOWN_FLAG = True
        _log("[SHUTDOWN] Signal received, stopping server...")
        server.shutdown()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    _log(f"[HTTP] Server listening on {HTTP_HOST}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _log("[SHUTDOWN] Server stopped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RR Bridge Server — HTTP wrapper for bridge_core")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=os.environ.get("RR_CHECKPOINT_PATH"),
        help="Path to a specific fold's checkpoint directory "
             "(e.g. data/checkpoints/EURUSD/fold_202301/)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("RR_BRIDGE_PORT", "9999")),
        help="TCP port to listen on (default: 9999, or RR_BRIDGE_PORT env var)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=os.environ.get("REINFORCE_ACTIVE_SYMBOLS", "EURUSD,USA500,XAUUSD"),
        help="Comma-separated list of active symbols (overrides env var)",
    )

    args = parser.parse_args()

    # Override module-level defaults from CLI via env vars (source of truth for main())
    if args.checkpoint_path:
        os.environ["RR_CHECKPOINT_PATH"] = args.checkpoint_path
    if args.port:
        os.environ["RR_BRIDGE_PORT"] = str(args.port)
    if args.symbols:
        os.environ["REINFORCE_ACTIVE_SYMBOLS"] = args.symbols
        # Reload ACTIVE_SYMBOLS (update the module-level list in bridge_core)
        import bridge.bridge_core as _bc
        _bc.ACTIVE_SYMBOLS[:] = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    main()
