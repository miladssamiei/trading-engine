"""
dll_preflight.py — Verify bridge_client.dll loads in Strategy Tester sandbox.

Pre-flight check before a terminal accepts any fold-testing jobs.
Spins up a bridge server, runs a minimal EA (test_bridge_client.ex5) in
Strategy Tester, and checks the agent log for:
  1. DLL loaded without errors
  2. BridgePing returned 1 (or at least 0 without crashing)
  3. BridgeSendReceive returned valid JSON (or handled gracefully)

Fails fast if the DLL can't load in the sandbox — prevents a terminal
from accepting jobs with broken DLL deployment.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from evaluate.terminal_manager import TerminalManager
from evaluate.config_generator import ConfigGenerator
from evaluate.bridge_manager import BridgeManager

# ── Constants ──────────────────────────────────────────────────────────────

PREFLIGHT_EA   = "test_bridge_client.ex5"
PREFLIGHT_PORT = 19998  # default bridge port used by the preflight EA
PREFLIGHT_SYMBOL = "EURUSD"
PREFLIGHT_PERIOD = "H1"

# Real MT5 terminal and agent paths (the default installation we validated)
DEFAULT_TERMINAL_EXE  = r"C:\Program Files\MetaTrader 5\terminal64.exe"
DEFAULT_TERMINAL_DATA = (
    r"C:\Users\Milad\AppData\Roaming\MetaQuotes\Terminal"
    r"\D0E8209F77C8CF37AD8BF550E51FF075"
)
DEFAULT_AGENT_LOG = (
    r"C:\Users\Milad\AppData\Roaming\MetaQuotes\Tester"
    r"\D0E8209F77C8CF37AD8BF550E51FF075"
    r"\Agent-127.0.0.1-3000\logs"
)


def _log(msg: str) -> None:
    print(f"[Preflight] {msg}", flush=True)


# ── EA source verification ─────────────────────────────────────────────────

def verify_preflight_ea_compiled(terminal_data: str | Path) -> bool:
    """Check that preflight EA .ex5 exists in terminal's Experts folder."""
    path = Path(terminal_data) / "MQL5" / "Experts" / PREFLIGHT_EA
    ok = path.exists()
    if ok:
        _log(f"Preflight EA: {path} ({path.stat().st_size} bytes)")
    else:
        _log(f"Preflight EA NOT FOUND: {path}")
    return ok


# ── Log parsing ────────────────────────────────────────────────────────────

def _find_agent_log(log_dir: str | Path) -> Path | None:
    """Find the most recent agent log file."""
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return None
    logs = sorted(
        log_dir.glob("*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return logs[0] if logs else None


def parse_preflight_log(log_path: str | Path) -> dict:
    """Parse agent log for preflight test results.

    Returns dict with keys: dll_loaded, ping_result, sendreceive_bytes,
    sendreceive_action, errors, test_passed, raw_lines.
    """
    result = {
        "dll_loaded": False,
        "ping_result": None,
        "sendreceive_bytes": None,
        "sendreceive_action": None,
        "sendreceive_raw": "",
        "errors": [],
        "test_passed": False,
        "success": False,
    }
    if not Path(log_path).exists():
        result["errors"].append("log_not_found")
        return result

    # MT5 logs are UTF-16 LE; handle gracefully
    raw = Path(log_path).read_bytes()
    if raw[:2] == b"\xff\xfe":
        text = raw.decode("utf-16-le", errors="replace")
    elif raw[:2] == b"\xfe\xff":
        text = raw.decode("utf-16-be", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")

    # DLL loaded
    if re.search(r"bridge_client\.dll", text):
        result["dll_loaded"] = True

    # BridgePing result
    m = re.search(r"BridgePing\(\d+\)\s*=\s*(\d+)", text)
    if m:
        result["ping_result"] = int(m.group(1))

    # BridgeSendReceive
    m = re.search(r"BridgeSendReceive returned (\d+) bytes", text)
    if m:
        result["sendreceive_bytes"] = int(m.group(1))

    m = re.search(r'RAW RESPONSE:\s*(\{.*"action".*\})', text)
    if m:
        raw = m.group(1)
        result["sendreceive_raw"] = raw
        try:
            resp = json.loads(raw)
            result["sendreceive_action"] = resp.get("action", "?")
        except json.JSONDecodeError:
            result["errors"].append("response_not_json")

    # PASS / FAIL
    if re.search(r"PASS.*Valid JSON response", text):
        result["test_passed"] = True
    elif re.search(r"FAIL", text):
        result["errors"].append("ea_reported_fail")

    if re.search(r"error|ERROR|failed|FAILED", text, re.IGNORECASE) \
            and not result["test_passed"]:
        for line in text.splitlines():
            if "error" in line.lower():
                result["errors"].append(line.strip()[:120])

    # Overall success computation
    result["success"] = (
        result["dll_loaded"]
        and result["ping_result"] is not None
        and result["sendreceive_bytes"] is not None
        and result["sendreceive_bytes"] > 0
        and result["test_passed"]
    )

    return result


# ── Preflight runner ───────────────────────────────────────────────────────

def run_preflight(
    checkpoint_path: str,
    terminal_id: str = "preflight_test",
    terminal_data: str = "",
    terminal_exe: str = "",
    symbol: str = PREFLIGHT_SYMBOL,
    period: str = PREFLIGHT_PERIOD,
    from_date: str = "2026-06-01",
    to_date: str = "2026-06-02",
) -> dict:
    """Run a full preflight: start bridge, run EA in tester, check result.

    Returns a result dict with: success, dll_loaded, ping_result,
    sendreceive_action, errors, log_summary.
    """
    terminal_data = terminal_data or DEFAULT_TERMINAL_DATA
    terminal_exe   = terminal_exe or DEFAULT_TERMINAL_EXE
    agent_log_dir  = DEFAULT_AGENT_LOG

    # 1. Verify EA is compiled
    if not verify_preflight_ea_compiled(terminal_data):
        return {"success": False, "errors": ["ea_not_compiled"]}

    # 2. Start bridge server
    _log(f"Starting bridge on port {PREFLIGHT_PORT}...")
    bridge = BridgeManager(port=PREFLIGHT_PORT, checkpoint_path=checkpoint_path,
                           symbol=symbol)
    bridge.free_port(PREFLIGHT_PORT)

    if not bridge.start():
        return {"success": False, "errors": ["bridge_start_failed"]}

    if not bridge.wait_ready(timeout=60.0):
        bridge.stop()
        return {"success": False, "errors": ["bridge_not_ready"]}

    _log("Bridge ready")

    # 3. Generate tester config and run terminal
    try:
        gen = ConfigGenerator(output_dir=str(
            Path(terminal_data) / "tester"
        ))
        configs = gen.generate(
            symbol=symbol,
            fold_id="preflight",
            checkpoint_path=checkpoint_path,
            bridge_port=PREFLIGHT_PORT,
            from_date=from_date,
            to_date=to_date,
            period=period,
        )

        tester_ini = configs["tester_config"]
        _log(f"Running terminal with config: {tester_ini}")

        # Note: terminal manager is used here for the real data folder
        subprocess.run(
            [terminal_exe, "/config:" + str(tester_ini.resolve())],
            timeout=60,
        )
        time.sleep(15)  # wait for test to complete and log to flush

    except Exception as e:
        bridge.stop()
        return {"success": False, "errors": [f"terminal_run_failed: {e}"]}

    # 4. Parse agent log
    log_file = _find_agent_log(agent_log_dir)
    if log_file is None:
        bridge.stop()
        return {"success": False, "errors": ["no_agent_log_found"]}

    _log(f"Agent log: {log_file}")
    result = parse_preflight_log(log_file)

    # 5. Evaluate
    result["success"] = (
        result["dll_loaded"]
        and result["ping_result"] is not None
        and result["sendreceive_bytes"] is not None
        and result["sendreceive_bytes"] > 0
        and len(result["errors"]) == 0
    )

    if result["success"]:
        _log(f"PASS — DLL loaded, ping={result['ping_result']}, "
             f"action={result['sendreceive_action']}")
    else:
        _log(f"FAIL — errors: {result['errors']}")

    bridge.stop()
    return result


# ── Self-test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preflight check — verify DLL loads in MT5 Strategy Tester sandbox"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="",
        help="Path to a checkpoint directory for the bridge server",
    )
    parser.add_argument(
        "--skip-terminal",
        action="store_true",
        help="Skip terminal tester run, only check EA exists + log parsing",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint_path or (
        r"C:\Users\Milad\Desktop\Reinforce\Reinforce_Windows_20260613"
        r"\multi_live\data\EURUSD\15m\live_fold"
    )

    print("=== DLL Preflight Self-test ===")

    # 1. Verify EA compiled
    print(f"\n[1] EA compiled: {verify_preflight_ea_compiled(DEFAULT_TERMINAL_DATA)}")

    # 2. Parse existing agent log from previous test (no terminal run needed)
    log_file = _find_agent_log(DEFAULT_AGENT_LOG)
    if log_file:
        print(f"\n[2] Found existing agent log: {log_file}")
        result = parse_preflight_log(log_file)
        print(f"    DLL loaded:       {result['dll_loaded']}")
        print(f"    Ping result:      {result['ping_result']}")
        print(f"    SendRecv bytes:   {result['sendreceive_bytes']}")
        print(f"    Action:           {result['sendreceive_action']}")
        print(f"    Test passed:      {result['test_passed']}")
        if result.get("sendreceive_raw"):
            print(f"    Raw response:     {result['sendreceive_raw'][:120]}")
        if result["errors"]:
            print(f"    Errors ({len(result['errors'])}):")
            for e in result.get("errors", [])[:5]:
                print(f"      - {e[:100]}")
        print(f"    OVERALL:          {'PASS' if result['success'] else 'FAIL'}")
    else:
        print(f"\n[2] No agent log found at {DEFAULT_AGENT_LOG}")

    # 3. Full preflight (optional — requires terminal)
    if not args.skip_terminal:
        print(f"\n[3] Running full preflight (terminal + bridge)...")
        full_result = run_preflight(
            checkpoint_path=checkpoint,
            terminal_data=DEFAULT_TERMINAL_DATA,
        )
        print(f"\n    Full preflight: {'PASS' if full_result['success'] else 'FAIL'}")
        if full_result.get("errors"):
            print(f"    Errors: {full_result['errors']}")

    print("\n=== Done ===")
