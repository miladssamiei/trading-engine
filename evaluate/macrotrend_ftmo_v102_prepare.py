"""Prepare reproducible FTMO v1.02 Gold tester inputs; does not place trades."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/macrotrend_ftmo_gold_parity_v102_20240929"
PROFILE = Path(r"C:\Users\Milad\AppData\Roaming\MetaQuotes\Terminal\81A933A9AFC5DE3C23B15CAB19C63850\MQL5\Profiles\Tester")
EXPERT = Path(r"C:\Users\Milad\AppData\Roaming\MetaQuotes\Terminal\81A933A9AFC5DE3C23B15CAB19C63850\MQL5\Experts\MacroTrendEnsemble_FTMO.ex5")
SOURCE = EXPERT.with_suffix(".mq5")


def main() -> None:
    for name in ("SPEC_V102_PARITY.txt", "GATE_V102_PARITY.txt"):
        if not (OUT/name).is_file():
            raise RuntimeError(f"Missing frozen {name}")
    if not EXPERT.is_file():
        raise RuntimeError("FTMO EA binary missing")
    existing = (PROFILE/"MacroTrendEnsemble_FTMO.set").read_text(encoding="utf-16")
    lines = [line for line in existing.splitlines() if line and not line.startswith(";")]
    lines = [line for line in lines if not line.startswith(("InpUseBrokerMinimumLot=", "InpMaxMinimumLotRiskPct="))]
    binary_hash = hashlib.sha256(EXPERT.read_bytes()).hexdigest()
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if binary_hash.upper() != "D083B1FC58042D16CE38C6208FB69665B69ACE261176B2EAF254729D4BF23019":
        raise RuntimeError(f"Installed EX5 changed since v1.02 freeze: {binary_hash}")
    files = {}
    for engine, chart, value in (("A", "H4", "0"), ("B", "H1", "1")):
        set_name = f"MTE_v102_{engine}.set"
        params = [line.replace("InpEngineType=1||0||0||1||N", f"InpEngineType={value}||0||0||1||N") for line in lines]
        index = next(i for i, line in enumerate(params) if line.startswith("InpRiskPercent="))
        params[index+1:index+1] = ["InpUseBrokerMinimumLot=true||true||0||true||N",
                                   "InpMaxMinimumLotRiskPct=3.5||3.5||0.35||5.0||N"]
        payload = "; frozen v1.02 parity settings\n" + "\n".join(params) + "\n"
        (OUT/set_name).write_text(payload, encoding="utf-16")
        (PROFILE/set_name).write_text(payload, encoding="utf-16")
        ini_name = f"tester_v102_{engine}.ini"
        report = OUT/f"tester_v102_{engine}.xml"
        ini = ("[Common]\nLogin=\nPassword=\nServer=\nEnableDDE=0\nEnableNews=0\n\n"
               "[Tester]\nExpert=MacroTrendEnsemble_FTMO\n"
               f"ExpertParameters={set_name}\nSymbol=XAUUSD\nPeriod={chart}\nModel=4\n"
               "Optimization=0\nFromDate=2024-01-01\nToDate=2026-09-25\nForwardMode=0\n"
               "Deposit=10000\nCurrency=EUR\nLeverage=100\nExecutionMode=0\n"
               "OptimizationCriterion=0\nVisualMode=0\nFrameMode=0\nReplaceReport=1\n"
               f"Report={report}\nShutdownTerminal=1\n")
        (OUT/ini_name).write_text(ini, encoding="utf-8")
        files[engine] = {"set":set_name,"ini":ini_name,"chart":chart}
    (OUT/"run_inputs.json").write_text(json.dumps({"ex5_sha256":binary_hash,"mq5_sha256":source_hash,"files":files},indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"ex5_sha256":binary_hash,"mq5_sha256":source_hash,"files":files},indent=2))


if __name__ == "__main__":
    main()
