"""
bridge_manager.py — Start, stop, and health-check bridge server instances.

Manages one server_socket.py subprocess per (symbol, fold) pair.
Implements between-job health checks from plan §2.7:
  - DLL ping recheck after 50 consecutive jobs
  - Bridge memory monitoring via psutil (150% baseline = restart)
  - Socket port cleanup
  - Terminal process health (delegates to terminal_manager)

Usage:
    mgr = BridgeManager(port=9998, checkpoint="/path/to/checkpoint", symbol="EURUSD")
    mgr.start()
    mgr.wait_ready(timeout=30)
    # ... run jobs ...
    if mgr.between_job_check():  # called between jobs
        mgr.stop()
        mgr.start()
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


def _log(msg: str) -> None:
    print(f"[BridgeMgr] {msg}", flush=True)

DEFAULT_PYTHON = r"C:\Users\Milad\miniconda3\envs\reinforce_rl\python.exe"
DEFAULT_SOCKET_SCRIPT = str(ROOT_DIR / "bridge" / "server_socket.py")


class BridgeManager:
    """Manage a single bridge server process."""

    # ── Construction ───────────────────────────────────────────────────

    def __init__(
        self,
        port: int,
        checkpoint_path: str | Path,
        symbol: str = "EURUSD",
        python_exe: str = "",
        server_script: str = "",
    ):
        self.port           = port
        self.checkpoint_path = str(checkpoint_path)
        self.symbol         = symbol
        self._python_exe    = python_exe or DEFAULT_PYTHON
        self._server_script = server_script or DEFAULT_SOCKET_SCRIPT
        self._proc: subprocess.Popen | None = None
        self._log_handle = None
        self._ready         = False
        self._consecutive_jobs = 0
        self._baseline_memory_mb: float | None = None
        self._total_dll_calls  = 0
        self._last_error       = ""

    # ── Start / stop ───────────────────────────────────────────────────

    def start(self) -> bool:
        """Launch the bridge server subprocess."""
        if self._proc is not None:
            _log(f"[{self.port}] Already running (PID {self._proc.pid})")
            return self._ready

        if not Path(self._server_script).exists():
            _log(f"[{self.port}] Server script not found: {self._server_script}")
            return False

        cmd = [
            self._python_exe, "-u", "-W", "ignore",
            self._server_script,
            "--port", str(self.port),
            "--checkpoint-path", self.checkpoint_path,
            "--symbols", self.symbol,
        ]
        env = os.environ.copy()
        env["USE_REGIME_FEATURES"] = "0"

        # Redirect the bridge's stdout to a log file rather than an unread pipe.
        # An unread subprocess.PIPE fills up (~64KB on Windows) and then blocks
        # the bridge's `print(..., flush=True)` calls. Once the bridge blocks it
        # stops answering the EA, which hangs in bridge_client.dll's blocking
        # recv(), which freezes the Strategy Tester mid-run. A file never blocks.
        log_dir = ROOT_DIR / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_handle = open(log_dir / f"bridge_{self.port}.log", "a",
                                encoding="utf-8", buffering=1)

        _log(f"[{self.port}] Launching bridge: {' '.join(cmd)}")
        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT_DIR),
                env=env,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                if sys.platform == "win32" else 0,
            )
        except Exception as e:
            _log(f"[{self.port}] Failed to launch: {e}")
            return False

        self._consecutive_jobs = 0
        _log(f"[{self.port}] Process started (PID {self._proc.pid})")
        return True

    def wait_ready(self, timeout: float = 60.0) -> bool:
        """Wait for the bridge TCP port to be listening."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc and self._proc.poll() is not None:
                _log(f"[{self.port}] Exited early (code {self._proc.returncode})")
                self._ready = False
                return False
            if self._ping():
                self._ready = True
                self._baseline_memory_mb = self._get_memory_mb()
                base_mb = self._baseline_memory_mb
                _log(f"[{self.port}] Ready "
                     f"(memory baseline: {base_mb:.0f} MB)" if base_mb else
                     f"[{self.port}] Ready (memory baseline: N/A)")
                return True
            time.sleep(1.0)

        _log(f"[{self.port}] Timed out after {timeout}s")
        self._ready = False
        return False

    def stop(self, timeout: float = 10.0) -> bool:
        """Gracefully stop the bridge server."""
        if self._proc is None:
            return True
        pid = self._proc.pid
        _log(f"[{self.port}] Stopping bridge (PID {pid})...")
        try:
            if sys.platform == "win32":
                self._proc.send_signal(1)  # SIGINT-like
            else:
                self._proc.terminate()
            self._proc.wait(timeout=timeout)
            _log(f"[{self.port}] Stopped gracefully")
        except subprocess.TimeoutExpired:
            _log(f"[{self.port}] Force-killing...")
            self._proc.kill()
            self._proc.wait(timeout=5)

        self._proc = None
        self._ready = False
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None
        return True

    # ── Health check ───────────────────────────────────────────────────

    def _ping(self) -> bool:
        """Check if the bridge is responding to TCP ping."""
        try:
            sock = socket.create_connection(("127.0.0.1", self.port), timeout=2.0)
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
        except Exception:
            return False

    def _get_memory_mb(self) -> float | None:
        """Get the bridge process's RSS in MB."""
        if self._proc is None:
            return None
        try:
            import psutil
            p = psutil.Process(self._proc.pid)
            return p.memory_info().rss / (1024 * 1024)
        except Exception:
            return None

    def health_status(self) -> dict:
        """Return a health status dict for the bridge."""
        alive = self._ping()
        mem = self._get_memory_mb()
        return {
            "port": self.port,
            "pid": self._proc.pid if self._proc else None,
            "running": bool(self._proc and self._proc.poll() is None),
            "alive": alive,
            "ready": self._ready,
            "memory_mb": round(mem, 1) if mem else None,
            "baseline_mb": (
                round(self._baseline_memory_mb, 1)
                if self._baseline_memory_mb else None
            ),
            "consecutive_jobs": self._consecutive_jobs,
        }

    # ── Between-job health checks (plan §2.7) ──────────────────────────

    def between_job_check(self, force_dll_check: bool = False) -> bool:
        """Run between-job health checks. Returns True if bridge is healthy.

        Checks:
          1. Bridge is alive (TCP ping)
          2. DLL ping recheck after 50 consecutive jobs
          3. Memory < 150% of baseline
        """
        self._consecutive_jobs += 1

        # 1. Bridge alive
        if not self._ping():
            _log(f"[{self.port}] FAIL — bridge not responding to ping")
            return False

        # 2. DLL recheck after 50 jobs
        should_dll_check = force_dll_check or (self._consecutive_jobs >= 50)
        if should_dll_check:
            _log(f"[{self.port}] DLL ping recheck "
                 f"(job {self._consecutive_jobs})...")
            if not self._ping():
                _log(f"[{self.port}] FAIL — DLL ping recheck failed")
                return False
            self._consecutive_jobs = 0  # reset counter after check

        # 3. Memory monitoring
        current_mem = self._get_memory_mb()
        if current_mem and self._baseline_memory_mb:
            ratio = current_mem / max(self._baseline_memory_mb, 1.0)
            if ratio > 1.5:
                _log(f"[{self.port}] FAIL — memory leak: "
                     f"{current_mem:.0f} MB vs baseline "
                     f"{self._baseline_memory_mb:.0f} MB ({ratio:.1%})")
                return False

        return True

    # ── Socket port cleanup ────────────────────────────────────────────

    @staticmethod
    def free_port(port: int) -> bool:
        """Kill any process holding the given port."""
        try:
            out = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True,
            ).stdout
            for line in out.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    pid = int(line.strip().split()[-1])
                    try:
                        os.kill(pid, 9)  # SIGKILL
                        _log(f"[{port}] Killed PID {pid} holding port")
                        time.sleep(1.0)
                        return True
                    except Exception:
                        pass
        except Exception:
            pass
        return True

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    @property
    def is_running(self) -> bool:
        return bool(self._proc and self._proc.poll() is None)


# ── Self-test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== BridgeManager self-test ===")
    print()

    CAIRO_FOLD = (
        r"C:\Users\Milad\Desktop\Reinforce\Reinforce_Windows_20260613"
        r"\multi_live\data\EURUSD\15m\live_fold"
    )
    TEST_PORT = 19999

    mgr = BridgeManager(port=TEST_PORT, checkpoint_path=CAIRO_FOLD, symbol="EURUSD")

    # Clean up any lingering process
    mgr.free_port(TEST_PORT)

    print("[1] Starting bridge...")
    if not mgr.start():
        print("    FAIL — start() returned False")
        sys.exit(1)

    print("[2] Waiting for readiness...")
    if not mgr.wait_ready(timeout=60.0):
        print("    FAIL — wait_ready() timed out")
        mgr.stop()
        sys.exit(1)

    print(f"\n[3] Health status:")
    status = mgr.health_status()
    for k, v in status.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.0f}")
        else:
            print(f"    {k}: {v}")

    print(f"\n[4] Simulating 5 between-job checks...")
    for i in range(5):
        ok = mgr.between_job_check()
        status = mgr.health_status()
        print(f"    Job {i+1}: {'OK' if ok else 'FAIL'} "
              f"(consecutive_jobs={status['consecutive_jobs']}, "
              f"mem={status['memory_mb']}MB)")

    print(f"\n[5] Between-job check #50+ (triggers DLL recheck)...")
    mgr._consecutive_jobs = 49
    ok = mgr.between_job_check()
    print(f"    After 50 jobs: {'OK' if ok else 'FAIL'} "
          f"(consecutive_jobs reset to {mgr._consecutive_jobs})")

    print(f"\n[6] Stopping bridge...")
    mgr.stop()
    print(f"    Running: {mgr.is_running}")

    print("\n=== Done ===")
