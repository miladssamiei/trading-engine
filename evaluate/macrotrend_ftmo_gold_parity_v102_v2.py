"""Correct v1.02 Gold EA parity for the tester's 2023 H1 warmup history."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

import macrotrend_ftmo_gold_parity_2024_v2 as independent
import macrotrend_ftmo_gold_parity_v102 as audit

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/results/macrotrend_ftmo_gold_parity_v102_20240929"
OUT = ROOT / "data/results/macrotrend_ftmo_gold_parity_v102_20240929_v2"
TERMINAL = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rates_and_probe_m1() -> tuple[pd.DataFrame, dict]:
    if not mt5.initialize(path=TERMINAL):
        raise RuntimeError(f"FTMO MT5 initialize failed: {mt5.last_error()}")
    try:
        raw=mt5.copy_rates_range("XAUUSD",mt5.TIMEFRAME_H1,
                                 datetime(2023,1,1,tzinfo=timezone.utc),
                                 datetime(2026,9,25,tzinfo=timezone.utc))
        if raw is None:
            raise RuntimeError(f"XAUUSD H1 fetch failed: {mt5.last_error()}")
        samples={}
        for name,year,month in (("early_2024",2024,2),("early_2025",2025,2),("late_2026",2026,9)):
            start=datetime(year,month,2,tzinfo=timezone.utc)
            end=datetime(year,month,3,tzinfo=timezone.utc)
            chunk=mt5.copy_rates_range("XAUUSD",mt5.TIMEFRAME_M1,start,end)
            if chunk is None or len(chunk)==0:
                samples[name]=dict(rows=0,in_requested_window=False)
            else:
                first=int(chunk[0]["time"])
                last=int(chunk[-1]["time"])
                samples[name]=dict(rows=len(chunk),first_utc=datetime.fromtimestamp(first,timezone.utc).isoformat(),
                                   last_utc=datetime.fromtimestamp(last,timezone.utc).isoformat(),
                                   in_requested_window=int(start.timestamp())<=first<=last<=int(end.timestamp()))
    finally:
        mt5.shutdown()
    rates=pd.DataFrame(raw).sort_values("time").drop_duplicates("time")
    rates.to_csv(OUT/"ftmo_xauusd_h1_2023_warmup_snapshot.csv",index=False)
    return rates,samples


def main() -> None:
    for name in ("SPEC_V102_PARITY_V2.txt","GATE_V102_PARITY_V2.txt"):
        if not (OUT/name).is_file():
            raise RuntimeError(f"Frozen document missing: {name}")
    old_meta=json.loads((OLD/"audit_metadata.json").read_text(encoding="utf-8"))
    runs={}
    hashes={}
    for engine in ("A","B"):
        name=f"journal_v102_{engine}.txt"
        source=OLD/name
        if sha(source)!=old_meta["hashes"][name]:
            raise RuntimeError(f"Frozen {engine} log changed")
        destination=OUT/name
        shutil.copyfile(source,destination)
        hashes[name]=sha(destination)
        runs[engine]=destination.read_text(encoding="utf-8").splitlines()
    audit.OUT=OUT
    independent.OUT=OUT
    rates,m1_samples=load_rates_and_probe_m1()
    events=audit.parse_events(runs)
    full_signals,contexts=independent.signal_engine(rates)
    signals=full_signals[full_signals.time>="2024-01-01"].copy()
    signals.to_csv(OUT/"python_signals_scored_2024_2026.csv",index=False)
    compared,summary=audit.compare(events,signals,contexts)
    coverage=audit.signal_coverage(events,signals)
    first=datetime.fromtimestamp(int(rates.time.iloc[0]),timezone.utc)
    last=datetime.fromtimestamp(int(rates.time.iloc[-1]),timezone.utc)
    g1=(len(rates)==22074 and first==datetime(2023,1,3,1,tzinfo=timezone.utc) and
        last==datetime(2026,9,24,23,tzinfo=timezone.utc) and rates.time.is_monotonic_increasing and
        all("MTE v1.02: initialized XAUUSD "+engine in "\n".join(runs[engine]) for engine in ("A","B")))
    action=compared[compared.kind.isin(["OPEN","MIN_OVERRIDE","MIN_BLOCK"])]
    stops=compared[compared.kind.isin(["OPEN","RATCHET"])]
    g2=bool((action.verdict=="PASS").all())
    g3=bool((stops.verdict=="PASS").all())
    g4=all(s["overlapping_opens"]==0 and s["orphan_exits"]==0 and s["open_at_test_end"]<=1 for s in summary.values())
    verdict="PASS" if all((g1,g2,g3,g4)) else "FAIL"
    failures=compared[compared.verdict=="FAIL"]
    for engine in ("A","B"):
        x=coverage[coverage.engine==engine]
        summary[engine]["coverage"]={k:int(v) for k,v in x.status.value_counts().to_dict().items()}
        finals=[float(m[1]) for line in runs[engine] if (m:=audit.FINAL.search(line))]
        summary[engine]["final_balance_eur"]=finals[-1] if finals else None
    hashes["ftmo_xauusd_h1_2023_warmup_snapshot.csv"]=sha(OUT/"ftmo_xauusd_h1_2023_warmup_snapshot.csv")
    meta=dict(verdict=verdict,gates=dict(G1=g1,G2=g2,G3=g3,G4=g4,G5="UNTESTED"),
              source_ex5_sha256=json.loads((OLD/"run_inputs.json").read_text())["ex5_sha256"],
              first_h1_utc=first.isoformat(),last_h1_utc=last.isoformat(),h1_rows=len(rates),
              m1_samples=m1_samples,hashes=hashes,summary=summary)
    (OUT/"audit_metadata.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    lines=["FTMO GOLD EA V1.02 VS PYTHON - CORRECTED 2023 WARMUP",f"Verdict: {verdict}",
           f"G1 frozen tester and 2023-2026 H1 source: {'PASS' if g1 else 'FAIL'}; {len(rates)} H1 rows, {first} through {last}",
           f"G2 all logged entries and minimum-lot decisions match Python: {'PASS' if g2 else 'FAIL'}",
           f"G3 initial and trailing SL within 0.01 Gold unit: {'PASS' if g3 else 'FAIL'}",
           f"G4 chronological tester position accounting: {'PASS' if g4 else 'FAIL'}",
           "G5 same-feed tick fill/intrabar exit parity: UNTESTED",""]
    for engine in ("A","B"):
        s=summary[engine]
        c=s["coverage"]
        lines.extend([f"Engine {engine}: Python raw 2024-26 signals {s['python_raw_signals']}; MT5 opens {s['OPEN']} ({s['min_lot_open']} minimum-lot); override attempts {s['MIN_OVERRIDE']}; cap blocks {s['MIN_BLOCK']}; ratchets {s['RATCHET']}; stops {s['STOP_EXIT']}; market closes {s['MARKET_CLOSE']}; open at test end {s['open_at_test_end']}; panic {s['PANIC']}; post-fill risk closures {s['POST_FILL_RISK']}; final tester balance EUR {s['final_balance_eur']}; mismatched events {s['failures']}.",
                      f"  Signal coverage: {c.get('LOGGED_ACTION',0)} logged actions; {c.get('POSITION_OCCUPIED',0)} signals while occupied; {c.get('UNEXPLAINED_FROM_H1_AND_JOURNAL',0)} flat signals without logged action."])
    lines.extend(["",f"All divergent events: {len(failures)}"])
    for _, row in failures.iterrows():
        lines.append(f"  {row.engine} {row.time} {row.kind}: MT5 SL {row.get('sl')}, Python SL {row.predicted_sl}, difference {row.sl_error}; {row.line}")
    lines.extend(["","M1 availability probes (small date windows; completeness across the tape remains unproven):"])
    for name,sample in m1_samples.items():
        lines.append(f"  {name}: {sample}")
    lines.extend(["","The tester reports many discarded real ticks caused by M1 price mismatch. A matched signal/stop is not evidence of clean fill or exit execution.",
                  "The minimum-lot policy can take more than 0.5% initial-stop risk, with a configured 3.5% ceiling. Tester balances are observed, not independently replicated.",
                  "The original V1.02 audit and its failed early-2024 comparison are preserved unchanged."])
    report="\n".join(lines)+"\n"
    (OUT/"REPORT_V102_PARITY_V2.txt").write_text(report,encoding="utf-8")
    print(report)


if __name__=="__main__":
    main()
