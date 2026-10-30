"""Test if bridge produces signals near 1.166 price (last 500 training bars)."""
import json, socket, time, subprocess, sys
import pandas as pd

bridge = subprocess.Popen(
    [r'C:\Users\Milad\miniconda3\envs\reinforce_rl\python.exe','-u','-W','ignore',
     r'C:\Users\Milad\Desktop\RR\bridge\server_socket.py',
     '--port','19997',
     '--checkpoint-path',r'C:\Users\Milad\Desktop\Reinforce\Reinforce_Windows_20260613\multi_live\data\EURUSD\15m\live_fold',
     '--symbols','EURUSD'],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    cwd=r'C:\Users\Milad\Desktop\RR')
time.sleep(30)

df = pd.read_parquet(
    r'C:\Users\Milad\Desktop\Reinforce\Reinforce_Windows_20260613'
    r'\multi_live\data\EURUSD\15m\live_fold\parquets\fold_202606_train.parquet')

# Use bars near end of training (last 500 bars, price ~1.166)
start = len(df) - 500
print(f'Training bars {start} to {len(df)-1}, close range: {df.close.iloc[start]:.5f} - {df.close.iloc[-1]:.5f}')

# Pre-warm with first 300 of those 500
for i in range(start, start + 300):
    row = df.iloc[i]
    bar = json.dumps({'symbol':'EURUSD','open':float(row.open),'high':float(row.high),
        'low':float(row.low),'close':float(row.close),
        'spread':float(row.get('avg_spread',0.0001)),
        'timestamp':int(time.time())+(i-start)*900})+'\n'
    try:
        sock=socket.create_connection(('127.0.0.1',19997),timeout=2)
        sock.sendall(bar.encode())
        raw=b''
        while b'\n' not in raw:
            c=sock.recv(4096)
            if not c: break
            raw+=c
        sock.close()
    except: pass

# Test bars 300-500 (post-warmup)
signals = []
for i in range(start + 300, start + 500):
    row = df.iloc[i]
    bar = json.dumps({'symbol':'EURUSD','open':float(row.open),'high':float(row.high),
        'low':float(row.low),'close':float(row.close),
        'spread':float(row.get('avg_spread',0.0001)),
        'timestamp':int(time.time())+(i-start)*900})+'\n'
    try:
        sock=socket.create_connection(('127.0.0.1',19997),timeout=2)
        sock.sendall(bar.encode())
        raw=b''
        while b'\n' not in raw:
            c=sock.recv(4096)
            if not c: break
            raw+=c
        sock.close()
        resp=json.loads(raw.decode().strip())
        a = resp.get('action','?')
        if a != 'HOLD':
            signals.append({'bar': i+1, 'action': a, 'close': float(row.close)})
    except Exception as e:
        pass

print(f'Signals in 200 post-warmup bars: {len(signals)}')
from collections import Counter
if signals:
    print(f'Distribution: {Counter(s["action"] for s in signals)}')
    print(f'First 10 signals:')
    for s in signals[:10]:
        print(f'  bar {s["bar"]}: {s["action"]} @ {s["close"]:.5f}')

bridge.kill()
bridge.wait(timeout=5)
