# -*- coding: utf-8 -*-
"""
children_reval_run_batches.py - standalone Model=4 revalidation batch
runner for ORB and SP2L (SPEC_TICK_MODEL_REVALIDATION_CHILDREN).

Runs the EXACT same 6 yearly batches as each strategy's original
Model=0 validation with the ONLY change Model=0 -> Model=4:

  orb : Expert=ORB_Strategy, Period=M5, sandbox trades.csv/orb_days.csv
  sp2l: Expert=SP2L_SpikeGap, Period=M1, sandbox trades.csv/sp2l_days.csv

Usage:
  python children_reval_run_batches.py --strategy orb 2021 ... 2026
  python children_reval_run_batches.py --strategy sp2l 2021 ... 2026
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

TERM = r"C:\Program Files\MetaTrader 5\terminal64.exe"
AGENTS = [Path(r"C:\Users\Milad\AppData\Roaming\MetaQuotes\Tester")
          / "D0E8209F77C8CF37AD8BF550E51FF075" / f"Agent-127.0.0.1-{p}"
          for p in (3000, 3001)]
OUT = Path(r"C:\Users\Milad\Desktop\RR\data\results"
           r"\tick_model_revalidation_children")

WINDOWS = {
    "2021": ("2021-01-01", "2022-01-01"),
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-01-01"),
    "2026": ("2026-01-01", "2026-05-01"),
}

STRATEGIES = {
    "orb": {"expert": "ORB_Strategy", "period": "M5",
            "files": ("trades.csv", "orb_days.csv")},
    "sp2l": {"expert": "SP2L_SpikeGap", "period": "M1",
             "files": ("trades.csv", "sp2l_days.csv")},
}

INI_TMPL = """[Common]
Login=
Password=
Server=
EnableDDE=0
EnableNews=0

[Tester]
Expert={expert}
Symbol=US500-tick
Period={period}
Model=4
Optimization=0
FromDate={from_date}
ToDate={to_date}
ForwardMode=0
Deposit=10000
Currency=USD
Leverage=100
ExecutionMode=0
OptimizationCriterion=0
VisualMode=0
FrameMode=0
ReplaceReport=1
ShutdownTerminal=1
"""


def log_sizes(agent):
    d = agent / "logs"
    out = {}
    if d.exists():
        for f in sorted(d.glob("*.log")):
            try:
                out[f.name] = f.stat().st_size
            except OSError:
                pass
    return out


def read_new_log_text(agent, before):
    chunks = []
    d = agent / "logs"
    if not d.exists():
        return ""
    for f in sorted(d.glob("*.log")):
        b0 = before.get(f.name, 0)
        try:
            size = f.stat().st_size
        except OSError:
            continue
        if size <= b0:
            continue
        with open(f, "rb") as fh:
            fh.seek(b0)
            raw = fh.read()
        for enc in ("utf-16-le", "utf-8", "cp1252"):
            try:
                chunks.append(raw.decode(enc))
                break
            except UnicodeDecodeError:
                continue
    return "\n".join(chunks)


def find_sandbox_agent(files):
    for agent in AGENTS:
        if all((agent / "MQL5" / "Files" / n).exists() for n in files[:1]):
            return agent
    for agent in AGENTS:
        if any((agent / "MQL5" / "Files" / n).exists() for n in files):
            return agent
    return None


def run_one(strategy, label):
    cfg = STRATEGIES[strategy]
    from_d, to_d = WINDOWS[label]
    ini = OUT / strategy / "ini" / f"tester_{strategy}_m4_{label}.ini"
    ini.parent.mkdir(parents=True, exist_ok=True)
    ini.write_text(INI_TMPL.format(expert=cfg["expert"],
                                   period=cfg["period"],
                                   from_date=from_d, to_date=to_d),
                   encoding="ascii")
    bdir = OUT / strategy / "batches" / f"year_{label}"
    bdir.mkdir(parents=True, exist_ok=True)

    before = {a: log_sizes(a) for a in AGENTS}
    t0 = time.time()
    for agent in AGENTS:
        for name in cfg["files"]:
            f = agent / "MQL5" / "Files" / name
            if f.exists():
                f.unlink()

    proc = subprocess.Popen([TERM, f"/config:{ini}"])
    timed_out = False
    try:
        proc.wait(timeout=7200)
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/F", "/IM", "terminal64.exe"],
                       capture_output=True, text=True)
        proc.kill()
        timed_out = True
    wall = time.time() - t0

    agent = find_sandbox_agent(cfg["files"])
    got = {}
    if agent:
        for name in cfg["files"]:
            src = agent / "MQL5" / "Files" / name
            if src.exists():
                shutil.copy2(src, bdir / name)
                got[name] = True
            else:
                got[name] = False

    texts = [read_new_log_text(a, before[a]) for a in AGENTS]
    text = "\n".join(t for t in texts if t)
    (bdir / "agent_utf8.log").write_text(text, encoding="utf-8", newline="")

    ticks = bars = None
    balance = None
    passed = "Test passed" in text
    m = re.search(r"(\d+) ticks, (\d+) bars generated", text)
    if m:
        ticks, bars = int(m.group(1)), int(m.group(2))
    m = re.search(r"final balance ([\d.]+) USD", text)
    if m:
        balance = float(m.group(1))
    real_begin = re.findall(r"real ticks begin from .*", text)
    gen_real = "generating based on real ticks" in text

    n_trades = None
    if got.get("trades.csv"):
        with open(bdir / "trades.csv", newline="") as fh:
            n_trades = sum(1 for _ in fh) - 1

    meta = {"strategy": strategy, "label": label,
            "window": [from_d, to_d], "ini": str(ini),
            "agent": str(agent) if agent else None,
            "wall_time_s": round(wall, 1), "timed_out": timed_out,
            "returncode": proc.returncode, "files": got,
            "n_trades": n_trades, "ticks": ticks, "bars": bars,
            "final_balance": balance, "test_passed": passed,
            "real_ticks_begin": real_begin[:1],
            "gen_real_ticks": gen_real,
            "log_bytes": len(text.encode("utf-8"))}
    (bdir / "meta.json").write_text(json.dumps(meta, indent=2),
                                    encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=list(STRATEGIES))
    ap.add_argument("labels", nargs="*")
    a = ap.parse_args()
    labels = a.labels or list(WINDOWS)
    (OUT / a.strategy).mkdir(parents=True, exist_ok=True)

    running = subprocess.run(["tasklist", "/FI", "IMAGENAME eq terminal64.exe"],
                             capture_output=True, text=True).stdout
    if "terminal64.exe" in running:
        subprocess.run(["taskkill", "/F", "/IM", "terminal64.exe"],
                       capture_output=True, text=True)
        time.sleep(3)

    rows = []
    for label in labels:
        print(f"\n===== {a.strategy} / {label} =====", flush=True)
        rows.append(run_one(a.strategy, label))
    with open(OUT / a.strategy / "batches" / "summary.csv", "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    bad = [r["label"] for r in rows
           if not (r["test_passed"] and all(r["files"].values()))]
    if bad:
        print(f"FAILED BATCHES: {bad}")
        sys.exit(1)
    print(f"\nsummary -> {OUT / a.strategy / 'batches' / 'summary.csv'}")


if __name__ == "__main__":
    main()
