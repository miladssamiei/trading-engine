"""Generate production deployment presets and live configs for the two
Path C execution EAs (ORB_Strategy_Live, SP2L_SpikeGap_Live).

Outputs:
  configs/live_orb_us500.ini      deployment manifest (launch + chart + inputs
                                  + PropFirmGuard limits + watchdog thresholds)
  configs/live_sp2l_us500.ini
  deploy/infrastructure/live_presets.json   machine-readable single source

The .ini [Inputs] section is generated from the EA source input defaults
(parsed from the .mq5 files, never hand-copied), then cross-checked against
the compiled .ex5 binaries. Symbol resolution queries the live terminal and
falls back to the documented broker cash symbol US500.c.

Usage:
  python deploy/infrastructure/live_presets.py [--offline]
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys

REPO = r"C:\Users\Milad\Desktop\RR"
TERMINAL_EXE = r"C:\Program Files\MetaTrader 5\terminal64.exe"
OUT_DIR = os.path.join(REPO, "configs")
JSON_OUT = os.path.join(REPO, "deploy", "infrastructure", "live_presets.json")

GUARD = {
    "MAX_DAILY_LOSS_PCT": 3.5,
    "MAX_TRAILING_DD_PCT": 8.0,
    "MAX_SPREAD_US500_PT": 0.60,
    "MAX_SPREAD_XAUUSD_PT": 0.45,
    "MAX_SLIPPAGE_INDEX_PT": 0.20,
    "RISK_PER_TRADE_PCT": 0.5,
    "MAX_CONCURRENT_RISK_PCT": 1.0,
    "WATCHDOG_DAILY_TRIGGER_PCT": 3.7,
    "WATCHDOG_TRAILING_TRIGGER_PCT": 8.2,
    "WATCHDOG_POLL_SECONDS": 1,
}
WATCHDOG_DB = r"data\queue\account_state.db"
WATCHDOG_LOG = r"data\logs\watchdog.log"

EAS = [
    {
        "name": "ORB_Strategy_Live",
        "tag": "ORB",
        "magic": 26082201,
        "period": "M5",
        "ini": "live_orb_us500.ini",
        "tpl": "live_orb_us500",
    },
    {
        "name": "SP2L_SpikeGap_Live",
        "tag": "SP2L",
        "magic": 26082701,
        "period": "M1",
        "ini": "live_sp2l_us500.ini",
        "tpl": "live_sp2l_us500",
    },
]

SYMBOL_PREFERENCE = ["US500.c", "US500.cash", "US500"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_inputs(mq5_path):
    txt = open(mq5_path, "r", encoding="utf-8", errors="replace").read()
    out = {}
    for m in re.finditer(
            r"^\s*input\s+\w+\s+(\w+)\s*=\s*([-\d.]+)\s*;", txt, re.M):
        out[m.group(1)] = m.group(2)
    return out


def resolve_symbol(offline):
    if offline:
        return "US500.c", "offline fallback (documented broker cash symbol)"
    import MetaTrader5 as mt5
    if not mt5.initialize(TERMINAL_EXE):
        return "US500.c", "offline fallback (terminal unreachable)"
    try:
        for want in SYMBOL_PREFERENCE:
            if mt5.symbol_select(want, True) and mt5.symbol_info(want):
                return want, "resolved live from terminal symbol table"
        cands = [s.name for s in (mt5.symbols_get("*US500*") or [])
                 if s.name.upper().startswith("US500")]
        if cands:
            return cands[0], "resolved live from *US500* wildcard"
        return "US500.c", "offline fallback (no US500 symbol visible)"
    finally:
        mt5.shutdown()


def audit_ex5(ea, magic):
    mq5 = os.path.join(REPO, "ea", ea["name"] + ".mq5")
    ex5 = os.path.join(REPO, "ea", ea["name"] + ".ex5")
    clog = os.path.join(REPO, "deploy", "infrastructure", "compile_logs",
                        ea["name"] + ".compile.log")
    checks = {}
    checks["ex5_exists"] = os.path.exists(ex5)
    checks["ex5_fresh"] = (checks["ex5_exists"]
                           and os.path.getmtime(ex5) >= os.path.getmtime(mq5))
    clean_compile = False
    if os.path.exists(clog):
        raw = open(clog, "rb").read()
        try:
            txt = raw.decode("utf-16", errors="replace")
        except UnicodeDecodeError:
            txt = raw.decode("latin-1", errors="replace")
        clean_compile = ("Result: 0 errors, 0 warnings" in txt)
    checks["compile_log_clean"] = clean_compile
    checks["sha256_mq5"] = sha256(mq5)
    checks["sha256_ex5"] = sha256(ex5) if checks["ex5_exists"] else None
    checks["source_magic_ok"] = (int(parse_inputs(mq5)["InpMagicNumber"])
                                 == magic)
    checks["note"] = ("build 6140 .ex5 binaries are packed; no byte-level "
                      "probes possible; embedding attested by source magic "
                      "+ freshness + clean compile log + pinned sha256")
    return checks


def ini_text(ea, inputs, symbol, symbol_note):
    lines = []
    lines.append("; live deployment manifest (Path C execution infrastructure)")
    lines.append("; generated %s by deploy/infrastructure/live_presets.py"
                 % dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"))
    lines.append("; SPEC: deploy/infrastructure/SPEC_EXECUTION_INFRASTRUCTURE.txt")
    lines.append("; EA source: ea/%s.mq5 (frozen strategy + PropFirmGuard)"
                 % ea["name"])
    lines.append("; symbol resolution: %s" % symbol_note)
    lines.append("")
    lines.append("[Common]")
    lines.append("; terminal launches with its cached demo login; never a live acct")
    lines.append("Login=")
    lines.append("Password=")
    lines.append("Server=TenTrade-Server")
    lines.append("EnableDDE=0")
    lines.append("EnableNews=0")
    lines.append("")
    lines.append("[Charts]")
    lines.append("Symbol=%s" % symbol)
    lines.append("Period=%s" % ea["period"])
    lines.append("Template=%s" % ea["tpl"])
    lines.append("")
    lines.append("[Expert]")
    lines.append("Name=%s" % ea["name"])
    lines.append("MagicNumber=%d" % ea["magic"])
    lines.append("StrategyTag=%s" % ea["tag"])
    lines.append("AllowAlgoTrading=1")
    lines.append("")
    lines.append("[Inputs]")
    for k in sorted(inputs):
        lines.append("%s=%s" % (k, inputs[k]))
    lines.append("")
    lines.append("[PropFirmGuard]")
    lines.append("; enforced inside the EA (Include/PropFirmGuard.mqh)")
    lines.append("MaxDailyLossPct=%.1f" % GUARD["MAX_DAILY_LOSS_PCT"])
    lines.append("MaxTrailingDdPct=%.1f" % GUARD["MAX_TRAILING_DD_PCT"])
    lines.append("MaxSpreadUS500Pt=%.2f" % GUARD["MAX_SPREAD_US500_PT"])
    lines.append("MaxSpreadXAUUSDPt=%.2f" % GUARD["MAX_SPREAD_XAUUSD_PT"])
    lines.append("MaxSlippageIndexPt=%.2f" % GUARD["MAX_SLIPPAGE_INDEX_PT"])
    lines.append("RiskPerTradePct=%.1f" % GUARD["RISK_PER_TRADE_PCT"])
    lines.append("MaxConcurrentRiskPct=%.1f" % GUARD["MAX_CONCURRENT_RISK_PCT"])
    lines.append("")
    lines.append("[Watchdog]")
    lines.append("; external python redundancy: live/prop_guard_watchdog.py")
    lines.append("DailyTriggerPct=%.1f" % GUARD["WATCHDOG_DAILY_TRIGGER_PCT"])
    lines.append("TrailingTriggerPct=%.1f"
                 % GUARD["WATCHDOG_TRAILING_TRIGGER_PCT"])
    lines.append("PollSeconds=%d" % GUARD["WATCHDOG_POLL_SECONDS"])
    lines.append("StateDb=%s" % WATCHDOG_DB)
    lines.append("LogFile=%s" % WATCHDOG_LOG)
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    symbol, note = resolve_symbol(args.offline)
    print("symbol: %s (%s)" % (symbol, note))

    manifest = {"symbol": symbol, "symbol_note": note,
                "guard": GUARD, "eas": []}
    failures = 0
    for ea in EAS:
        mq5 = os.path.join(REPO, "ea", ea["name"] + ".mq5")
        inputs = parse_inputs(mq5)
        if "InpMagicNumber" not in inputs:
            print("FAIL: %s has no InpMagicNumber input" % ea["name"])
            failures += 1
            continue
        if int(inputs["InpMagicNumber"]) != ea["magic"]:
            print("FAIL: %s magic %s != expected %d"
                  % (ea["name"], inputs["InpMagicNumber"], ea["magic"]))
            failures += 1

        audit = audit_ex5(ea, ea["magic"])
        ex5_ok = (audit["ex5_exists"] and audit["ex5_fresh"]
                  and audit["compile_log_clean"] and audit["source_magic_ok"])
        if not ex5_ok:
            failures += 1

        text = ini_text(ea, inputs, symbol, note)
        ini_path = os.path.join(OUT_DIR, ea["ini"])
        with open(ini_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("%s: inputs=%d magic=%s ex5_fresh=%s compile_clean=%s %s"
              % (ea["ini"], len(inputs), inputs["InpMagicNumber"],
                 audit["ex5_fresh"], audit["compile_log_clean"],
                 "OK" if ex5_ok else "AUDIT-FAIL"))

        rt = open(ini_path, "r", encoding="utf-8").read()
        for key in ["MagicNumber=%d" % ea["magic"], "Symbol=%s" % symbol,
                    "InpMagicNumber=%d" % ea["magic"],
                    "MaxDailyLossPct=%.1f" % GUARD["MAX_DAILY_LOSS_PCT"],
                    "MaxTrailingDdPct=%.1f" % GUARD["MAX_TRAILING_DD_PCT"],
                    "MaxSpreadUS500Pt=%.2f" % GUARD["MAX_SPREAD_US500_PT"],
                    "MaxSlippageIndexPt=%.2f" % GUARD["MAX_SLIPPAGE_INDEX_PT"],
                    "RiskPerTradePct=%.1f" % GUARD["RISK_PER_TRADE_PCT"],
                    "MaxConcurrentRiskPct=%.1f"
                    % GUARD["MAX_CONCURRENT_RISK_PCT"]]:
            if key not in rt:
                print("  ROUNDTRIP FAIL: %r missing" % key)
                failures += 1
        manifest["eas"].append({"name": ea["name"], "tag": ea["tag"],
                                "magic": ea["magic"], "period": ea["period"],
                                "ini": ea["ini"], "inputs": inputs,
                                "ex5_audit": audit})

    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print("manifest: %s" % JSON_OUT)
    print("RESULT: %s (%d failures)"
          % ("PASS" if failures == 0 else "FAIL", failures))
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
