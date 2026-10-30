"""Quick single-job test with fixed port + last-300-bar pre-warm."""
import sys, time, subprocess, json, socket
from pathlib import Path

sys.path.insert(0, ".")
from evaluate.bridge_manager import BridgeManager
from evaluate.config_generator import ConfigGenerator

PORT = 19998
CAIRO_FOLD = (r"C:\Users\Milad\Desktop\Reinforce\Reinforce_Windows_20260613"
              r"\multi_live\data\EURUSD\15m\live_fold")
TERMINAL_DATA = (r"C:\Users\Milad\AppData\Roaming\MetaQuotes\Terminal"
                 r"\D0E8209F77C8CF37AD8BF550E51FF075")
AGENT_LOG_DIR = (r"C:\Users\Milad\AppData\Roaming\MetaQuotes\Tester"
                 r"\D0E8209F77C8CF37AD8BF550E51FF075"
                 r"\Agent-127.0.0.1-3000\logs")

# Start bridge
bridge = BridgeManager(port=PORT, checkpoint_path=CAIRO_FOLD, symbol="EURUSD")
bridge.free_port(PORT)
bridge.start()
bridge.wait_ready(timeout=60)
print(f"Bridge ready on port {PORT}")

# Pre-warm with last 300 bars (price ~1.166 matching test period)
print("Pre-warming with last 300 training bars...")
import pandas as pd
parquet = CAIRO_FOLD + "/parquets/fold_202606_train.parquet"
df = pd.read_parquet(parquet)
start = len(df) - 300
for i in range(start, len(df)):
    row = df.iloc[i]
    bar = json.dumps({
        "symbol": "EURUSD",
        "open": float(row["open"]), "high": float(row["high"]),
        "low": float(row["low"]), "close": float(row["close"]),
        "spread": float(row.get("avg_spread", 0.0001)),
        "timestamp": int(time.time()) + (i - start) * 900,
    }) + "\n"
    try:
        sock = socket.create_connection(("127.0.0.1", PORT), timeout=2)
        sock.sendall(bar.encode())
        raw = b""
        while b"\n" not in raw:
            c = sock.recv(4096)
            if not c: break
            raw += c
        sock.close()
    except: pass
print("Pre-warm done")

# Generate config
gen = ConfigGenerator(output_dir=f"{TERMINAL_DATA}/tester")
configs = gen.generate(
    symbol="EURUSD", fold_id="202606", checkpoint_path=CAIRO_FOLD,
    bridge_port=PORT, from_date="2026-06-01", to_date="2026-06-02",
    period="H1", ea_name="MULTI_CAIRO_EA",
)

# Launch terminal
print("Launching terminal...")
term = subprocess.Popen(
    [r"C:\Program Files\MetaTrader 5\terminal64.exe",
     "/config:" + str(configs["tester_config"].resolve())],
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
)
term.wait(timeout=300)
print(f"Terminal exit: {term.returncode}")

# Collect results
time.sleep(5)
log_files = sorted(Path(AGENT_LOG_DIR).glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
log = log_files[0] if log_files else None
if log:
    raw = log.read_bytes()
    text = raw.decode("utf-16-le" if raw[:2] == b"\xff\xfe" else "utf-8")
    lines = text.split("\n")[-60:]
    for l in lines:
        if any(w in l for w in ["MULTI_CAIRO","Bridge","Trade","Deinit","final","Bar ","action","balance"]):
            print(l[:200])

bridge.stop()
