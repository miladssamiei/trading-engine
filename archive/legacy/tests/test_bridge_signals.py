"""Quick bridge response test — do post-warmup bars return BUY/SELL?"""
import json, socket, time, subprocess, sys

bridge = subprocess.Popen(
    [r'C:\Users\Milad\miniconda3\envs\reinforce_rl\python.exe','-u','-W','ignore',
     r'C:\Users\Milad\Desktop\RR\bridge\server_socket.py',
     '--port','19999',
     '--checkpoint-path',r'C:\Users\Milad\Desktop\Reinforce\Reinforce_Windows_20260613\multi_live\data\EURUSD\15m\live_fold',
     '--symbols','EURUSD'],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    cwd=r'C:\Users\Milad\Desktop\RR')
time.sleep(30)

base = 1.1650
for i in range(300):
    price = base + i * 0.00002
    bar = json.dumps({'symbol':'EURUSD','open':round(price-0.0001,5),'high':round(price+0.0003,5),'low':round(price-0.0003,5),'close':round(price+0.0001,5),'spread':0.00006,'timestamp':int(time.time())+i*900})+'\n'
    try:
        sock=socket.create_connection(('127.0.0.1',19999),timeout=2)
        sock.sendall(bar.encode())
        raw=b''
        while b'\n' not in raw:
            c=sock.recv(4096)
            if not c: break
            raw+=c
        sock.close()
    except: pass

print('Warmup done. 10 test bars:')
for i in range(10):
    price = base + 0.005 + i * 0.00002
    bar = json.dumps({'symbol':'EURUSD','open':round(price-0.0001,5),'high':round(price+0.0003,5),'low':round(price-0.0003,5),'close':round(price+0.0001,5),'spread':0.00006,'timestamp':int(time.time())+310*900+i*900})+'\n'
    try:
        sock=socket.create_connection(('127.0.0.1',19999),timeout=2)
        sock.sendall(bar.encode())
        raw=b''
        while b'\n' not in raw:
            c=sock.recv(4096)
            if not c: break
            raw+=c
        sock.close()
        resp=json.loads(raw.decode().strip())
        print(f'Bar {301+i}: action={resp.get("action")} weight={resp.get("cairo_weight")} votes={resp.get("winner_votes")}')
    except Exception as e:
        print(f'Bar {301+i}: ERROR {e}')

bridge.kill()
bridge.wait(timeout=5)
