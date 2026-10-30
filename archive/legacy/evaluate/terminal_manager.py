"""
terminal_manager.py — Manage MT5 terminal instances for fold testing.

Each terminal instance is identified by a terminal_id and has:
  - A unique bridge port (RR_BRIDGE_PORT_START + index)
  - A dedicated data folder (data/terminals/<terminal_id>/)
  - A subprocess handle for the terminal64.exe process
  - Health monitoring via MetaTrader5 Python package

Provides:
  - TerminalManager:  spawn / wait_ready / health_check / stop / force_stop
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    RR_BRIDGE_PORT_START,
    RR_DLL_NAME,
)


def _log(msg: str) -> None:
    print(f"[TerminalMgr] {msg}", flush=True)


class TerminalManager:
    """Manages a single MT5 terminal instance for fold testing."""

    # Default terminal EXE path — configurable per instance
    DEFAULT_TERMINAL_EXE = r"C:\Program Files\MetaTrader 5\terminal64.exe"

    def __init__(
        self,
        terminal_id: str,
        terminal_index: int,
        terminal_exe: str = "",
        base_data_dir: str = "",
    ):
        self.terminal_id    = terminal_id
        self.terminal_index = terminal_index
        self.bridge_port    = RR_BRIDGE_PORT_START + terminal_index

        self._terminal_exe  = terminal_exe or self.DEFAULT_TERMINAL_EXE
        self._data_dir      = Path(base_data_dir) if base_data_dir else (
            ROOT_DIR / "data" / "terminals" / terminal_id
        )
        self._proc: subprocess.Popen | None = None
        self._ready        = False

    # ── Data directory setup ──────────────────────────────────────────────

    def ensure_data_dir(self) -> None:
        """Create the terminal data directory and required subdirectories."""
        dirs = [
            self._data_dir / "MQL5" / "Libraries",
            self._data_dir / "MQL5" / "Experts",
            self._data_dir / "MQL5" / "Include",
            self._data_dir / "MQL5" / "Files",
            self._data_dir / "MQL5" / "Logs",
            self._data_dir / "tester",
            self._data_dir / "tester" / "logs",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def deploy_dll(self) -> bool:
        """Copy bridge_client.dll to the terminal's MQL5/Libraries/."""
        src = ROOT_DIR / "bridge" / RR_DLL_NAME
        dst = self._data_dir / "MQL5" / "Libraries" / RR_DLL_NAME
        if not src.exists():
            _log(f"[{self.terminal_id}] DLL not found at {src}")
            return False
        dst.write_bytes(src.read_bytes())
        _log(f"[{self.terminal_id}] DLL deployed: {dst} ({src.stat().st_size} bytes)")
        return True

    # ── Process lifecycle ─────────────────────────────────────────────────

    def spawn(self, timeout: float = 120.0) -> bool:
        """Launch the terminal process and wait for it to initialize.

        Returns True if terminal initializes within timeout.
        """
        if self._proc is not None:
            _log(f"[{self.terminal_id}] Already running (PID {self._proc.pid})")
            return self._ready

        self.ensure_data_dir()
        self.deploy_dll()

        cmd = [
            self._terminal_exe,
            "/portable",
        ]
        env = os.environ.copy()
        env["MT5_TERMINAL_ID"] = self.terminal_id

        _log(f"[{self.terminal_id}] Launching: {' '.join(cmd)}")
        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(self._data_dir),
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                if sys.platform == "win32" else 0,
            )
        except FileNotFoundError:
            _log(f"[{self.terminal_id}] Terminal EXE not found: {self._terminal_exe}")
            return False
        except Exception as e:
            _log(f"[{self.terminal_id}] Failed to launch: {e}")
            return False

        _log(f"[{self.terminal_id}] Process started (PID {self._proc.pid})")
        return self.wait_ready(timeout)

    def wait_ready(self, timeout: float = 120.0) -> bool:
        """Poll for terminal readiness via MetaTrader5 Python package.

        `mt5.initialize()` connects to a local terminal process.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc and self._proc.poll() is not None:
                _log(f"[{self.terminal_id}] Process exited early (code {self._proc.returncode})")
                self._ready = False
                return False

            try:
                import MetaTrader5 as mt5
                path = str(self._data_dir)
                if mt5.initialize(path=path, portable=True):
                    mt5.shutdown()
                    self._ready = True
                    elapsed = time.time() - (deadline - timeout)
                    _log(f"[{self.terminal_id}] Ready after {elapsed:.1f}s")
                    return True
                mt5.shutdown()
            except Exception as e:
                pass

            time.sleep(2.0)

        _log(f"[{self.terminal_id}] Timed out after {timeout}s")
        self._ready = False
        return False

    def health_check(self) -> dict:
        """Return terminal health status dict.

        Keys: running, ready, cpu_percent, memory_mb
        """
        result = {
            "running": False,
            "ready": self._ready,
            "cpu_percent": None,
            "memory_mb": None,
        }
        if self._proc is None:
            return result

        poll = self._proc.poll()
        result["running"] = (poll is None)

        if result["running"]:
            try:
                import psutil
                p = psutil.Process(self._proc.pid)
                result["cpu_percent"] = round(p.cpu_percent(interval=0.1), 1)
                mem = p.memory_info()
                result["memory_mb"] = round(mem.rss / (1024 * 1024), 1)
            except ImportError:
                result["cpu_percent"] = -1
                result["memory_mb"] = -1
            except Exception:
                pass

        return result

    def verify_dll(self) -> bool:
        """Check that bridge_client.dll is present and has correct size."""
        dll_path = self._data_dir / "MQL5" / "Libraries" / RR_DLL_NAME
        if not dll_path.exists():
            _log(f"[{self.terminal_id}] DLL missing from {dll_path}")
            return False
        return dll_path.stat().st_size > 0

    def verify_historical_data(self, symbol: str = "EURUSD") -> bool:
        """Verify historical data is queryable for the given symbol."""
        try:
            import MetaTrader5 as mt5
            path = str(self._data_dir)
            if not mt5.initialize(path=path, portable=True):
                _log(f"[{self.terminal_id}] Cannot initialize for data check")
                return False

            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 1)
            mt5.shutdown()

            if rates is None or len(rates) == 0:
                _log(f"[{self.terminal_id}] No historical data for {symbol}")
                return False

            _log(f"[{self.terminal_id}] Historical data OK: {symbol} ({len(rates)} bar(s))")
            return True

        except Exception as e:
            _log(f"[{self.terminal_id}] Data check failed: {e}")
            return False

    # ── Shutdown ──────────────────────────────────────────────────────────

    def stop(self, timeout: float = 30.0) -> bool:
        """Gracefully stop the terminal."""
        if self._proc is None:
            return True

        pid = self._proc.pid
        _log(f"[{self.terminal_id}] Stopping (PID {pid})...")

        # Send graceful close via WM_CLOSE or Ctrl+C equivalent
        if sys.platform == "win32":
            try:
                self._proc.send_signal(signal.CTRL_BREAK_EVENT)
            except Exception:
                pass

        try:
            self._proc.wait(timeout=timeout)
            _log(f"[{self.terminal_id}] Stopped gracefully")
            self._proc = None
            self._ready = False
            return True
        except subprocess.TimeoutExpired:
            _log(f"[{self.terminal_id}] Graceful stop timed out — forcing")
            return self.force_stop()

    def force_stop(self) -> bool:
        """Forcefully terminate the terminal process."""
        if self._proc is None:
            return True

        pid = self._proc.pid
        _log(f"[{self.terminal_id}] Force-killing PID {pid}")
        try:
            self._proc.kill()
            self._proc.wait(timeout=10)
        except Exception as e:
            import psutil
            try:
                parent = psutil.Process(pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            except Exception:
                pass

        self._proc = None
        self._ready = False
        return True

    # ── Port calculation ──────────────────────────────────────────────────

    @staticmethod
    def port_for_index(index: int) -> int:
        """Return bridge port for a terminal index."""
        return RR_BRIDGE_PORT_START + index

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


# ── Self-test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== TerminalManager self-test ===")
    mgr = TerminalManager(
        terminal_id="test_terminal",
        terminal_index=0,
    )
    print(f"  ID:          {mgr.terminal_id}")
    print(f"  Port:        {mgr.bridge_port}")
    print(f"  Data dir:    {mgr._data_dir}")
    print(f"  Terminal EXE:{mgr._terminal_exe}")
    print()

    print("[1] ensure_data_dir()...")
    mgr.ensure_data_dir()
    print(f"    Data dir exists: {mgr._data_dir.exists()}")
    print(f"    Libraries dir:   {(mgr._data_dir / 'MQL5' / 'Libraries').exists()}")
    print()

    print("[2] deploy_dll()...")
    if mgr.deploy_dll():
        print(f"    DLL deployed OK")
    else:
        print("    DLL deploy FAILED")
    print()

    print("[3] spawn() + wait_ready()...")
    if mgr.spawn(timeout=60.0):
        print(f"    Terminal ready (PID {mgr.pid})")
    else:
        print("    Terminal did not become ready within 60s")

    if mgr.is_running:
        print(f"\n[4] health_check(): {mgr.health_check()}")
        print(f"\n[5] verify_dll(): {mgr.verify_dll()}")
        print(f"\n[6] verify_historical_data('EURUSD'): {mgr.verify_historical_data('EURUSD')}")
        print(f"\n[7] stop()...")
        mgr.stop()
        print(f"    Terminated. Running: {mgr.is_running}")
    else:
        print("\n[?] Terminal not running — check logs in data/terminals/test_terminal/")

    print("\n=== Done ===")
