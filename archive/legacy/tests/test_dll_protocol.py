"""test_dll_protocol.py — Integration test for bridge_client.dll protocol.

Starts server_socket.py with a real checkpoint, then exercises the wire
protocol using Python sockets (mimicking what the C DLL does).

If bridge_client.dll is compiled and available, also tests via ctypes.

Usage:
    python tests/test_dll_protocol.py
    python tests/test_dll_protocol.py --dll bridge/bridge_client.dll
"""
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
PYTHON_EXE = r"C:\Users\Milad\miniconda3\envs\reinforce_rl\python.exe"

CAIRO_FOLD = Path(
    r"C:\Users\Milad\Desktop\Reinforce\Reinforce_Windows_20260613"
    r"\multi_live\data\EURUSD\15m\live_fold"
)
PARQUET = CAIRO_FOLD / "parquets" / "fold_202606_train.parquet"

TEST_PORT = 19998
N_BARS = 500


def wait_for_port(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=1.0)
            sock.close()
            return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False


def send_bar(port: int, bar: dict) -> dict:
    sock = socket.create_connection(("127.0.0.1", port), timeout=5.0)
    payload = json.dumps(bar) + "\n"
    sock.sendall(payload.encode())
    raw = b""
    while b"\n" not in raw:
        chunk = sock.recv(4096)
        if not chunk:
            break
        raw += chunk
    sock.close()
    return json.loads(raw.decode().strip())


def send_ping(port: int) -> bool:
    sock = socket.create_connection(("127.0.0.1", port), timeout=5.0)
    sock.sendall(b'{"ping":1}\n')
    raw = b""
    while b"\n" not in raw:
        chunk = sock.recv(4096)
        if not chunk:
            break
        raw += chunk
    sock.close()
    resp = json.loads(raw.decode().strip())
    return resp.get("pong") == 1


def test_python_socket():
    print("=" * 70)
    print("  DLL PROTOCOL TEST — Python socket (simulating bridge_client.dll)")
    print("=" * 70)

    # ── Start server ─────────────────────────────────────────────────────
    print(f"\n[1] Starting server_socket.py on port {TEST_PORT}...")
    server_cmd = [
        PYTHON_EXE, "-u", "-W", "ignore",
        str(PROJECT_ROOT / "bridge" / "server_socket.py"),
        "--port", str(TEST_PORT),
        "--checkpoint-path", str(CAIRO_FOLD),
        "--symbols", "EURUSD",
    ]
    env = os.environ.copy()
    env["USE_REGIME_FEATURES"] = "0"
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(PROJECT_ROOT),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    try:
        if not wait_for_port(TEST_PORT, timeout=60.0):
            print("[FAIL] Server did not start within 60s")
            return False

        print("[OK] Server is listening")

        # ── Ping test ────────────────────────────────────────────────────
        print(f"\n[2] Ping test...")
        if send_ping(TEST_PORT):
            print("[OK] Ping -> pong")
        else:
            print("[FAIL] Ping failed")
            return False

        # ── Load bar data ─────────────────────────────────────────────────
        print(f"\n[3] Loading bar data: {PARQUET.name}")
        df = pd.read_parquet(PARQUET)
        print(f"    Rows: {len(df)}, using first {N_BARS} bars")

        # ── Send bars ─────────────────────────────────────────────────────
        print(f"\n[4] Sending {N_BARS} bars via TCP (simulating DLL)...")
        responses = []
        non_hold_count = 0
        errors = 0

        t_start = time.time()
        for idx in range(N_BARS):
            row = df.iloc[idx]
            bar = {
                "symbol": "EURUSD",
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "spread": float(row.get("avg_spread", 0.0001)),
                "timestamp": int(time.time()) + idx * 900,
            }

            try:
                resp = send_bar(TEST_PORT, bar)
                if "error" in resp:
                    errors += 1
                else:
                    responses.append(resp)
                    if resp.get("action") not in ("HOLD", ""):
                        non_hold_count += 1
            except Exception as e:
                errors += 1

            if (idx + 1) % 100 == 0:
                elapsed = time.time() - t_start
                print(f"    ... {idx+1}/{N_BARS} bars ({elapsed:.1f}s)  "
                      f"| errors={errors}  signals={non_hold_count}")

        t_elapsed = time.time() - t_start

        # ── Results ───────────────────────────────────────────────────────
        print(f"\n[5] RESULTS")
        print(f"    Bars sent:              {N_BARS}")
        print(f"    Responses:              {len(responses)}")
        print(f"    Errors:                 {errors}")
        print(f"    Non-HOLD signals:       {non_hold_count}")
        print(f"    Total time:             {t_elapsed:.2f}s ({N_BARS/t_elapsed:.0f} bars/s)")

        from collections import Counter
        action_counts = Counter(r.get("action") for r in responses)
        print(f"    Action distribution:    {dict(action_counts)}")

        # Check key properties
        first_non_hold = next((r for r in responses if r.get("action") not in ("HOLD", "")), None)
        print(f"    First non-HOLD:         {json.dumps(first_non_hold) if first_non_hold else 'NONE'}")

        # Latency stats
        latencies = [r.get("elapsed_ms", 0) for r in responses]
        if latencies:
            print(f"    Predict latency:")
            print(f"      min:  {min(latencies):.2f}ms")
            print(f"      p50:  {np.percentile(latencies, 50):.2f}ms")
            print(f"      p95:  {np.percentile(latencies, 95):.2f}ms")
            print(f"      max:  {max(latencies):.2f}ms")

        passed = (
            errors == 0
            and len(responses) > 0
            and non_hold_count > 0
        )

        print(f"\n    Protocol test:          {'PASS' if passed else 'FAIL'}")

        return passed

    finally:
        print(f"\n[6] Stopping server...")
        if sys.platform == "win32":
            server_proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("[OK] Server stopped")


def test_dll_ctypes(dll_path: str):
    """Optional: test the actual compiled DLL via ctypes."""
    import ctypes

    print(f"\n[CTYPES] Testing bridge_client.dll at: {dll_path}")
    dll = ctypes.CDLL(dll_path)

    # BridgePing
    dll.BridgePing.argtypes = [ctypes.c_int]
    dll.BridgePing.restype = ctypes.c_int

    # BridgeSendReceive
    dll.BridgeSendReceive.argtypes = [
        ctypes.c_int,          # port
        ctypes.c_char_p,       # request
        ctypes.c_char_p,       # response buffer
        ctypes.c_int,          # response_size
    ]
    dll.BridgeSendReceive.restype = ctypes.c_int

    # Start the server (reuse test_python_socket's start/stop pattern)
    print("[CTYPES] Starting server for DLL test...")
    server_cmd = [
        PYTHON_EXE, "-u", "-W", "ignore",
        str(PROJECT_ROOT / "bridge" / "server_socket.py"),
        "--port", str(TEST_PORT),
        "--checkpoint-path", str(CAIRO_FOLD),
        "--symbols", "EURUSD",
    ]
    env = os.environ.copy()
    env["USE_REGIME_FEATURES"] = "0"
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(PROJECT_ROOT),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    try:
        if not wait_for_port(TEST_PORT, timeout=60.0):
            print("[CTYPES] FAIL — server did not start")
            return False

        # Ping via DLL
        result = dll.BridgePing(TEST_PORT)
        print(f"[CTYPES] BridgePing({TEST_PORT}) = {result}  {'OK' if result == 1 else 'FAIL'}")
        if result != 1:
            return False

        # SendReceive via DLL
        request = json.dumps({
            "symbol": "EURUSD",
            "open": 1.1234,
            "high": 1.1240,
            "low": 1.1230,
            "close": 1.1238,
            "spread": 0.00015,
            "timestamp": 1234567890,
        }).encode()
        response = ctypes.create_string_buffer(4096)
        n = dll.BridgeSendReceive(TEST_PORT, request, response, 4096)
        if n > 0:
            resp_str = response.value.decode()
            print(f"[CTYPES] BridgeSendReceive returned {n} bytes: {resp_str}")
            try:
                resp_json = json.loads(resp_str)
                print(f"[CTYPES] Action: {resp_json.get('action', '?')}  OK")
            except json.JSONDecodeError:
                print(f"[CTYPES] FAIL — response not valid JSON")
                return False
        else:
            print(f"[CTYPES] FAIL — BridgeSendReceive returned {n}")
            return False

        return True

    finally:
        if sys.platform == "win32":
            server_proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Integration test for bridge_client.dll protocol")
    parser.add_argument("--dll", type=str, default=None,
                        help="Path to bridge_client.dll for ctypes testing (optional)")
    args = parser.parse_args()

    # Always run the Python socket simulation test
    ok_socket = test_python_socket()

    # Optionally test the compiled DLL
    ok_dll = True
    if args.dll:
        dll_abs = Path(args.dll).resolve()
        if dll_abs.exists():
            ok_dll = test_dll_ctypes(str(dll_abs))
        else:
            print(f"[SKIP] DLL not found at: {dll_abs}")

    print()
    print("=" * 70)
    print(f"  Python socket: {'PASS' if ok_socket else 'FAIL'}")
    if args.dll:
        print(f"  DLL ctypes:    {'PASS' if ok_dll else 'FAIL'}")
    print("=" * 70)

    sys.exit(0 if ok_socket else 1)
