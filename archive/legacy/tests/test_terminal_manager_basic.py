"""Quick dry-run test for terminal_manager.py — no terminal spawning."""
import sys
sys.path.insert(0, ".")
from evaluate.terminal_manager import TerminalManager

mgr = TerminalManager(terminal_id="test_dry", terminal_index=0)
print(f"ID: {mgr.terminal_id}")
print(f"Port: {mgr.bridge_port}")
print(f"Data dir: {mgr._data_dir}")

mgr.ensure_data_dir()
libs = mgr._data_dir / "MQL5" / "Libraries"
print(f"Libraries dir exists: {libs.exists()}")

ok = mgr.deploy_dll()
dll_path = libs / "bridge_client.dll"
print(f"DLL deployed: {ok}, size: {dll_path.stat().st_size} bytes")
print(f"verify_dll(): {mgr.verify_dll()}")
print(f"Port for index 2: {TerminalManager.port_for_index(2)}")
print("OK")
