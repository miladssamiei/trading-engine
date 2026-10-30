"""Audit fresh v1.02 FTMO Gold tester runs against independent H1 Python signals."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

import macrotrend_ftmo_gold_parity_2024_v2 as independent

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/macrotrend_ftmo_gold_parity_v102_20240929"
JOURNAL = Path(r"C:\Users\Milad\AppData\Roaming\MetaQuotes\Tester\81A933A9AFC5DE3C23B15CAB19C63850\Agent-127.0.0.1-3000\logs\20260929.log")
EVENT_TIME = re.compile(r"(20\d\d\.\d\d\.\d\d \d\d:\d\d:\d\d)\s+")
OPEN = re.compile(r"MTE: OPEN (BUY|SELL) engine=([AB]) lots=([\d.]+) entry=([\d.]+) SL=([\d.]+) spread=([\d.]+) planned_stop_risk=([\d.]+) min_lot_override=(yes|no)")
OVERRIDE = re.compile(r"MTE: MIN_LOT_OVERRIDE requested ([\d.]+) -> ([\d.]+) lots; initial-stop risk ([\d.]+) \(([\d.]+)%")
BLOCK = re.compile(r"MTE: minimum lot blocked: requested ([\d.]+), broker minimum ([\d.]+), minimum-stop risk ([\d.]+) \(([\d.]+)%")
RATCHET = re.compile(r"MTE: ratcheted ticket=(\d+) SL ([\d.]+) -> ([\d.]+)")
STOP = re.compile(r"stop loss triggered #(\d+) (buy|sell) ([\d.]+) XAUUSD ([\d.]+) sl: ([\d.]+) \[#(\d+) (?:buy|sell) [\d.]+ XAUUSD at ([\d.]+)\]")
FINAL = re.compile(r"final balance ([\d.]+) EUR")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def segments() -> dict[str, list[str]]:
    lines = JOURNAL.read_text(encoding="utf-16", errors="replace").splitlines()
    result = {}
    for engine, chart in (("A", "H4"), ("B", "H1")):
        starts = [i for i, line in enumerate(lines) if
                  f"XAUUSD,{chart}: testing of Experts\\MacroTrendEnsemble_FTMO.ex5 from 2024.01.01 00:00 to 2026.09.25 00:00" in line]
        candidates = []
        for start in starts:
            ends = [i for i in range(start+1, len(lines)) if
                    f"test Experts\\MacroTrendEnsemble_FTMO.ex5 on XAUUSD,{chart} thread finished" in lines[i]]
            if ends and any("MTE v1.02: initialized XAUUSD " + engine in line for line in lines[start:ends[0]+1]):
                candidates.append(lines[start:ends[0]+1])
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one complete v1.02 {engine} segment, found {len(candidates)}")
        segment = candidates[0]
        (OUT/f"journal_v102_{engine}.txt").write_text("\n".join(segment)+"\n", encoding="utf-8")
        result[engine] = segment
    return result


def parse_events(runs: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for engine, lines in runs.items():
        for line in lines:
            match = EVENT_TIME.search(line)
            if not match:
                continue
            stamp = match[1].replace(".", "-")
            item = dict(engine=engine, time=stamp, hour=stamp[:14]+"00:00", line=line)
            o, v, b, r, s = (pattern.search(line) for pattern in (OPEN, OVERRIDE, BLOCK, RATCHET, STOP))
            if o:
                item.update(kind="OPEN", direction=1 if o[1]=="BUY" else -1, lots=float(o[3]),
                            entry=float(o[4]), sl=float(o[5]), spread=float(o[6]),
                            planned_risk=float(o[7]), min_override=o[8])
            elif v:
                item.update(kind="MIN_OVERRIDE", requested_lots=float(v[1]), lots=float(v[2]),
                            planned_risk=float(v[3]), risk_pct=float(v[4]))
            elif b:
                item.update(kind="MIN_BLOCK", requested_lots=float(b[1]), lots=float(b[2]),
                            planned_risk=float(b[3]), risk_pct=float(b[4]))
            elif r:
                item.update(kind="RATCHET", ticket=int(r[1]), old_sl=float(r[2]), sl=float(r[3]))
            elif s:
                item.update(kind="STOP_EXIT", ticket=int(s[1]), direction=1 if s[2]=="buy" else -1,
                            lots=float(s[3]), entry=float(s[4]), sl=float(s[5]), exit_deal=int(s[6]))
            elif "MTE: PANIC_FLATTEN" in line:
                item.update(kind="PANIC")
            elif "MTE: close ticket=" in line:
                item.update(kind="MARKET_CLOSE")
            elif "POST_FILL_RISK" in line:
                item.update(kind="POST_FILL_RISK")
            elif "MTE: REJECT" in line:
                item.update(kind="REJECT")
            else:
                continue
            rows.append(item)
    result = pd.DataFrame(rows)
    result.to_csv(OUT/"mt5_v102_events.csv", index=False)
    return result


def compare(events: pd.DataFrame, signals: pd.DataFrame, contexts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    signal_map = signals.set_index(["engine", "time"])
    context_map = contexts.set_index("time")
    rows = []
    for _, row in events.iterrows():
        item = row.to_dict()
        engine, hour, kind = item["engine"], item["hour"], item["kind"]
        item.update(signal_match=False, predicted_sl=float("nan"), sl_error=float("nan"), verdict="OBSERVED")
        if kind in ("OPEN", "MIN_OVERRIDE", "MIN_BLOCK"):
            key=(engine,hour)
            if key in signal_map.index:
                ref=signal_map.loc[key]
                item["signal_match"] = True
                item["python_direction"] = int(ref.direction)
                if kind=="OPEN":
                    side=int(item["direction"])
                    distance=(max(abs(item["entry"]-ref.anchor),1.5*ref.atr) if engine=="A" else 4*ref.atr)
                    item["predicted_sl"]=independent.tick_round(item["entry"]-side*distance)
                    item["sl_error"]=abs(item["sl"]-item["predicted_sl"])
                    item["verdict"]="PASS" if side==int(ref.direction) and item["sl_error"]<=0.0100001 else "FAIL"
                elif kind=="MIN_OVERRIDE":
                    item["verdict"]="PASS" if abs(item["lots"]-0.01)<1e-9 and item["requested_lots"]<0.01 and item["risk_pct"]<=3.500001 else "FAIL"
                else:
                    item["verdict"]="PASS" if item["risk_pct"]>3.5 else "FAIL"
            else:
                item["verdict"]="FAIL"
        elif kind=="RATCHET":
            if hour in context_map.index:
                ref=context_map.loc[hour]
                if engine=="A" and ref.h4_bucket_changed:
                    item["predicted_sl"]=independent.tick_round(ref.h4_ema29)
                elif engine=="B":
                    side=1 if item["sl"]>item["old_sl"] else -1
                    item["predicted_sl"]=independent.tick_round(ref.h1_close-side*4*ref.h1_atr14)
            if pd.notna(item["predicted_sl"]):
                item["sl_error"]=abs(item["sl"]-item["predicted_sl"])
                item["verdict"]="PASS" if item["sl_error"]<=0.0100001 else "FAIL"
            else:
                item["verdict"]="FAIL"
        rows.append(item)
    result=pd.DataFrame(rows)
    result.to_csv(OUT/"event_comparison_v102.csv",index=False)
    summary={}
    for engine in ("A","B"):
        x=result[result.engine==engine]
        count={kind:int((x.kind==kind).sum()) for kind in ("OPEN","MIN_OVERRIDE","MIN_BLOCK","RATCHET","STOP_EXIT","PANIC","MARKET_CLOSE","POST_FILL_RISK","REJECT")}
        failures=x[x.verdict=="FAIL"]
        opens=x[x.kind=="OPEN"]
        summary[engine]={**count,"failures":len(failures),"final_balance":None,
                         "min_lot_open":int((opens.min_override=="yes").sum()),
                         "python_raw_signals":int((signals.engine==engine).sum())}
        outstanding=0
        overlaps=orphans=0
        for kind in x.kind:
            if kind=="OPEN":
                if outstanding:
                    overlaps+=1
                outstanding+=1
            elif kind in ("STOP_EXIT","MARKET_CLOSE"):
                if outstanding:
                    outstanding-=1
                else:
                    orphans+=1
        summary[engine].update(overlapping_opens=overlaps,orphan_exits=orphans,
                               open_at_test_end=outstanding)
    return result,summary


def signal_coverage(events: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Account for raw signals without treating occupancy as a missed order."""
    rows=[]
    for engine in ("A","B"):
        x=events[events.engine==engine]
        acted=set(x.loc[x.kind.isin(["OPEN","MIN_OVERRIDE","MIN_BLOCK","REJECT"]),"hour"])
        intervals=[]
        start=None
        for _, event in x.iterrows():
            if event.kind=="OPEN":
                if start is not None:
                    intervals.append((start,event.time,"OVERLAPPING_OPEN"))
                start=event.time
            elif event.kind in ("STOP_EXIT","MARKET_CLOSE") and start is not None:
                intervals.append((start,event.time,event.kind))
                start=None
        if start is not None:
            intervals.append((start,"9999-12-31 23:59:59","OPEN_AT_TEST_END"))
        for _, signal in signals[signals.engine==engine].iterrows():
            hour=signal.time
            if hour in acted:
                status="LOGGED_ACTION"
            elif any(a<=hour<b for a,b,_ in intervals):
                status="POSITION_OCCUPIED"
            else:
                status="UNEXPLAINED_FROM_H1_AND_JOURNAL"
            rows.append(dict(engine=engine,time=hour,direction=int(signal.direction),status=status))
    coverage=pd.DataFrame(rows)
    coverage.to_csv(OUT/"python_signal_coverage.csv",index=False)
    return coverage


def main() -> None:
    for name in ("SPEC_V102_PARITY.txt","GATE_V102_PARITY.txt"):
        if not (OUT/name).is_file():
            raise RuntimeError(f"Frozen document absent: {name}")
    runs=segments()
    events=parse_events(runs)
    independent.OUT=OUT
    rates=independent.load_rates()
    signals, contexts=independent.signal_engine(rates)
    compared,summary=compare(events,signals,contexts)
    coverage=signal_coverage(events,signals)
    dates=pd.to_datetime(rates.time,unit="s",utc=True)
    g1=(len(rates)>=16000 and dates.min().year==2024 and dates.max().year==2026 and
        rates.time.is_monotonic_increasing and all(any("MTE v1.02: initialized XAUUSD " + e in line for line in runs[e]) for e in ("A","B")))
    for engine in ("A","B"):
        finals=[float(m[1]) for line in runs[engine] if (m:=FINAL.search(line))]
        summary[engine]["final_balance"]=finals[-1] if finals else None
    g2=bool((compared.loc[compared.kind.isin(["OPEN","MIN_OVERRIDE","MIN_BLOCK"]),"verdict"]=="PASS").all())
    g3=bool((compared.loc[compared.kind.isin(["OPEN","RATCHET"]),"verdict"]=="PASS").all())
    g4=all(s["overlapping_opens"]==0 and s["orphan_exits"]==0 and
           s["open_at_test_end"]<=1 for s in summary.values())
    verdict="PASS" if all((g1,g2,g3,g4)) else "FAIL"
    failures=compared[compared.verdict=="FAIL"]
    hashes={p.name:sha(p) for p in [OUT/"ftmo_xauusd_h1_snapshot.csv",OUT/"journal_v102_A.txt",OUT/"journal_v102_B.txt",OUT/"MTE_v102_A.set",OUT/"MTE_v102_B.set",OUT/"tester_v102_A.ini",OUT/"tester_v102_B.ini"]}
    meta=dict(verdict=verdict,gates=dict(G1=g1,G2=g2,G3=g3,G4=g4,G5="UNTESTED"),summary=summary,hashes=hashes)
    (OUT/"audit_metadata.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    lines=["FTMO GOLD EA V1.02 VS INDEPENDENT PYTHON - 2024 TO 2026",f"Verdict: {verdict}",
           f"G1 source/run: {'PASS' if g1 else 'FAIL'}; broker H1 bars {len(rates)}; {dates.min()} to {dates.max()}",
           f"G2 signal/minimum-lot: {'PASS' if g2 else 'FAIL'}",
           f"G3 initial/ratchet stops: {'PASS' if g3 else 'FAIL'}",
           f"G4 tester exits accounted: {'PASS' if g4 else 'FAIL'}",
           "G5 independent same-feed tick fills and intrabar exits: UNTESTED",""]
    for engine in ("A","B"):
        s=summary[engine]
        classes=coverage[coverage.engine==engine].status.value_counts().to_dict()
        s["signal_coverage"]={key:int(value) for key,value in classes.items()}
        lines.append(f"Engine {engine}: Python raw signals {s['python_raw_signals']}; MT5 opens {s['OPEN']} (min-lot opens {s['min_lot_open']}); override attempts {s['MIN_OVERRIDE']}; risk-cap blocks {s['MIN_BLOCK']}; ratchets {s['RATCHET']}; stop exits {s['STOP_EXIT']}; market closes {s['MARKET_CLOSE']}; open at test end {s['open_at_test_end']}; overlapping opens {s['overlapping_opens']}; orphan exits {s['orphan_exits']}; panic {s['PANIC']}; post-fill risk closures {s['POST_FILL_RISK']}; rejects {s['REJECT']}; final tester balance EUR {s['final_balance']}; parity failures {s['failures']}.")
        lines.append(f"  Python signals: {classes.get('LOGGED_ACTION',0)} with logged action; {classes.get('POSITION_OCCUPIED',0)} while position open; {classes.get('UNEXPLAINED_FROM_H1_AND_JOURNAL',0)} without a logged action or known occupancy.")
    lines.extend(["",f"Divergent logged events: {len(failures)}"])
    for _, row in failures.head(30).iterrows():
        lines.append(f"  {row.engine} {row.time} {row.kind}: logged SL {row.get('sl')}, Python SL {row.predicted_sl}, error {row.sl_error}; {row.line}")
    lines.extend(["","Python raw signals include those shadowed by an open position and are not expected to equal executed trades.",
                  "Tester real ticks may be discarded due M1 price mismatch; MT5 cash PnL and exact fills/exits are observations, not independent Python confirmations.",
                  "Minimum-lot override can exceed 0.5% target risk, up to the configured 3.5% cap; concurrent risk and slippage are not controlled by this parity audit."])
    report="\n".join(lines)+"\n"
    (OUT/"REPORT_V102_PARITY.txt").write_text(report,encoding="utf-8")
    print(report)


if __name__=="__main__":
    main()
