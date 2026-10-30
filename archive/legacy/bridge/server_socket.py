"""
server_socket.py — TCP socket server wrapping bridge_core for MT5 Strategy Tester.

Listens on 127.0.0.1:<port> for newline-delimited JSON bar payloads from
the bridge_client.dll, returns newline-delimited JSON signal responses.

Each connection handles one bar request (stateless transport, stateful bridge).
State (buffer, models, prev_action) is maintained across connections per symbol.

Usage:
    python bridge/server_socket.py --port 9998 --checkpoint-path data/checkpoints/EURUSD/fold_202606/ --symbols EURUSD
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "train"))
sys.path.insert(0, str(ROOT_DIR / "live"))

from bridge.bridge_core import (
    ACTIVE_SYMBOLS,
    BarBuffer,
    BarValidator,
    LiveModels,
    _log,
    process_bar_core,
)

# ── Server state ────────────────────────────────────────────────────────────
_active_models:    dict[str, LiveModels]   = {}
_active_buffers:   dict[str, BarBuffer]    = {}
_active_validators: dict[str, BarValidator] = {}
_prev_actions:     dict[str, str]          = {}
_bars_since_flip:  dict[str, int]          = {}

SHUTDOWN_FLAG = False

# ── Per-bar decision logging (diagnostics) ──────────────────────────────────
# Enabled by setting RR_BRIDGE_DECISION_LOG=<csv path> in the bridge env.
# One row per processed bar: signal-bar timestamp, per-model votes, pooled
# ensemble probs, raw CAIRO action (pre-HoldGuard) and final action.
_DECISION_LOG = os.environ.get(
    "RR_BRIDGE_DECISION_LOG",
    r"C:\Users\Milad\Desktop\RR\data\results\diag_202603\bridge_decisions_v2.csv",
).strip()
_DECISION_HEADER = ("bar_ts;bar_open;bar_high;bar_low;bar_close;bar_spread;n_models;"
                    "gbm_vote;tqc_vote;dt_vote;"
                    "ens_p0;ens_p1;ens_p2;"
                    "raw_action;final_action;bars_since_flip;reason;elapsed_ms")


def _write_decision_row(bar: dict, details: dict, action_str: str,
                        new_flip: int, elapsed_ms: float) -> None:
    if not _DECISION_LOG:
        return
    try:
        import csv as _csv
        votes = details.get("model_votes") or {}
        probs = details.get("ens_probs") or [None, None, None]
        need_hdr = not os.path.exists(_DECISION_LOG)
        with open(_DECISION_LOG, "a", newline="", encoding="utf-8") as f:
            w = _csv.writer(f, delimiter=";")
            if need_hdr:
                w.writerow(_DECISION_HEADER.split(";"))
            w.writerow([
                bar.get("timestamp", ""),
                bar.get("open", ""),
                bar.get("high", ""),
                bar.get("low", ""),
                bar.get("close", ""),
                bar.get("spread", bar.get("avg_spread", "")),
                details.get("n_models", 0),
                votes.get("GBM", ""), votes.get("TQC", ""), votes.get("DT", ""),
                probs[0] if probs[0] is not None else "",
                probs[1] if probs[1] is not None else "",
                probs[2] if probs[2] is not None else "",
                details.get("cairo_action", ""),
                action_str,
                new_flip,
                details.get("reason", "") if details.get("reason") else "",
                round(elapsed_ms, 2),
            ])
    except Exception as e:
        _log(f"[DECISION_LOG] write failed: {e}")


def _load_symbol_models(symbol: str, checkpoint_dir: str | Path) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    models = LiveModels(symbol, checkpoint_dir)
    models.load()
    _active_models[symbol] = models
    _active_buffers[symbol] = BarBuffer()
    _active_validators[symbol] = BarValidator(symbol=symbol)
    _prev_actions[symbol] = "HOLD"
    _bars_since_flip[symbol] = 0
    _log(f"[INIT] [{symbol}] Loaded {len(models.algos)} models from {checkpoint_dir}")


def _handle_connection(conn: socket.socket, addr: tuple) -> None:
    try:
        conn.settimeout(5.0)
        raw = b""
        while b"\n" not in raw:
            chunk = conn.recv(4096)
            if not chunk:
                return
            raw += chunk
        line = raw.decode("utf-8", errors="replace").strip()
    except socket.timeout:
        return
    except Exception as e:
        _log(f"[SOCKET] Recv error from {addr}: {e}")
        return

    if not line:
        return

    # ── Ping ──────────────────────────────────────────────────────────────
    if '"ping"' in line.lower():
        try:
            base_lot = 0.01
            if ACTIVE_SYMBOLS and ACTIVE_SYMBOLS[0] in _active_models:
                base_lot = _active_models[ACTIVE_SYMBOLS[0]].base_lot
            resp = json.dumps({"pong": 1, "base_lot": base_lot})
            conn.sendall(resp.encode() + b"\n")
        except Exception:
            pass
        return

    # ── Bar ────────────────────────────────────────────────────────────────
    try:
        bar = json.loads(line)
    except json.JSONDecodeError as e:
        _log(f"[SOCKET] Invalid JSON from {addr}: {e}")
        try:
            error_resp = json.dumps({"error": "invalid_json", "detail": str(e)}) + "\n"
            conn.sendall(error_resp.encode())
        except Exception:
            pass
        return

    symbol = bar.get("symbol", "").upper()
    if not symbol:
        symbol = ACTIVE_SYMBOLS[0]

    # Normalize MT5 terminal symbol names to internal names
    # (e.g. "US500.C" -> "USA500", used for model lookup)
    MT5_TO_INTERNAL = {"US500.C": "USA500", "US500-TICK": "USA500"}
    symbol = MT5_TO_INTERNAL.get(symbol, symbol)

    if symbol not in _active_models:
        try:
            error_resp = json.dumps({"error": "symbol_not_loaded", "symbol": symbol}) + "\n"
            conn.sendall(error_resp.encode())
        except Exception:
            pass
        return

    action_str, details, elapsed_ms, new_flip = process_bar_core(
        bar=bar,
        models=_active_models[symbol],
        buffer=_active_buffers[symbol],
        validator=_active_validators[symbol],
        feature_cols=_active_models[symbol].feature_cols,
        prev_action=_prev_actions[symbol],
        bars_since_flip=_bars_since_flip[symbol],
    )

    _write_decision_row(bar, details, action_str, new_flip, elapsed_ms)

    _prev_actions[symbol] = action_str
    _bars_since_flip[symbol] = new_flip

    response = json.dumps({
        "action": action_str,
        "cairo_weight": details.get("cairo_weight", 0),
        "winner_votes": details.get("winner_votes", 0),
        "n_models": details.get("n_models", 0),
        "elapsed_ms": round(elapsed_ms, 2),
    }) + "\n"

    try:
        conn.sendall(response.encode())
    except Exception as e:
        _log(f"[SOCKET] Send error to {addr}: {e}")


def _server_loop(port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(5)
    sock.settimeout(1.0)

    _log(f"[INIT] TCP server listening on 127.0.0.1:{port}")

    while not SHUTDOWN_FLAG:
        try:
            conn, addr = sock.accept()
            t = threading.Thread(target=_handle_connection, args=(conn, addr), daemon=True)
            t.start()
        except socket.timeout:
            continue
        except Exception as e:
            if not SHUTDOWN_FLAG:
                _log(f"[SOCKET] Accept error: {e}")

    sock.close()
    _log("[INIT] TCP server stopped")


def main():
    global SHUTDOWN_FLAG

    import argparse
    parser = argparse.ArgumentParser(
        description="RR Bridge Server — TCP socket wrapper for bridge_core")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Checkpoint directory for a specific fold",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("RR_BRIDGE_PORT", "9999")),
        help="TCP port to listen on (default: 9999)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="EURUSD",
        help="Comma-separated list of active symbols (default: EURUSD)",
    )
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    ACTIVE_SYMBOLS[:] = symbols

    # ── Load models for all active symbols ─────────────────────────────────
    checkpoint_dir = Path(args.checkpoint_path)
    if not checkpoint_dir.exists():
        _log(f"[INIT] ERROR: checkpoint path not found: {checkpoint_dir}")
        sys.exit(1)

    for sym in symbols:
        _load_symbol_models(sym, checkpoint_dir)

    # ── Signal handler for graceful shutdown ───────────────────────────────
    def _on_signal(sig, frame):
        global SHUTDOWN_FLAG
        _log("[INIT] Shutting down...")
        SHUTDOWN_FLAG = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    _server_loop(args.port)


if __name__ == "__main__":
    main()
