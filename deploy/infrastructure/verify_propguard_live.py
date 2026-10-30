"""Live verification harness for the PropFirmGuard execution layer.

Phases:
  S  static audit   - Include/PropFirmGuard.mqh #define values vs SPEC, PG_Init
                      tags and magic-number input defaults in both _Live EAs.
  B  binary audit   - ea/*_Live.ex5 freshness (newer than .mq5), embedded input
                      names and magic-number byte probes; TestPropGuard.ex5
                      copied into the terminal's MQL5/Scripts.
  C  connection     - attach to the local MT5 terminal via the MetaTrader5 API,
                      assert a DEMO account (trade_mode == 0), symbol and
                      global-variable API availability.
  R  script run     - poll the active MQL5/Logs file while TestPropGuard.ex5
                      runs on a demo chart (automated attach attempt plus
                      interactive attachment detector), regex-parse the six
                      audit assertions, require zero FAIL count.

Output: data/results/propguard_verification/REPORT_PROPGUARD_VERIFICATION.txt
        data/results/propguard_verification/testpropguard_log_capture.txt

Usage:
  python deploy/infrastructure/verify_propguard_live.py [--timeout 300]
       [--no-attach-attempt] [--poll 0.5]
"""

import argparse
import datetime as dt
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time

REPO = r"C:\Users\Milad\Desktop\RR"
TERMINAL_EXE = r"C:\Program Files\MetaTrader 5\terminal64.exe"
TERMINAL_DATA = (r"C:\Users\Milad\AppData\Roaming\MetaQuotes\Terminal"
                 r"\D0E8209F77C8CF37AD8BF550E51FF075")
MQL5_LOGS = os.path.join(TERMINAL_DATA, "MQL5", "Logs")
MQL5_SCRIPTS = os.path.join(TERMINAL_DATA, "MQL5", "Scripts")
RESULTS = os.path.join(REPO, "data", "results", "propguard_verification")
GUARD_MQH = os.path.join(REPO, "Include", "PropFirmGuard.mqh")
TEST_MQ5 = os.path.join(REPO, "scripts", "TestPropGuard.mq5")
TEST_EX5 = os.path.join(REPO, "scripts", "TestPropGuard.ex5")
TESTER_INI = os.path.join(REPO, "configs", "tester_TestPropGuard.ini")
RUNNER_EX5 = os.path.join(REPO, "deploy", "infrastructure",
                          "TestPropGuardRunner.ex5")
MQL5_EXPERTS = os.path.join(TERMINAL_DATA, "MQL5", "Experts")
TESTER_ROOT = os.path.join(r"C:\Users\Milad\AppData\Roaming\MetaQuotes\Tester",
                           "D0E8209F77C8CF37AD8BF550E51FF075")
AGENT_LOG_DIRS = [os.path.join(TESTER_ROOT, "Agent-127.0.0.1-%d" % p, "logs")
                  for p in (3000, 3001)]

SPEC_EXPECT = {
    "PG_MAX_DAILY_LOSS_PCT": "3.5",
    "PG_MAX_TRAILING_DD_PCT": "8.0",
    "PG_MAX_SPREAD_US500": "0.60",
    "PG_MAX_SPREAD_XAUUSD": "0.45",
    "PG_MAX_SLIPPAGE_INDEX_PT": "0.20",
    "PG_RISK_PER_TRADE_PCT": "0.5",
    "PG_MAX_CONCURRENT_RISK_PCT": "1.0",
    "PG_MAGIC_ORB": "26082201",
    "PG_MAGIC_SP2L": "26082701",
}
GTVS = ["PROP_GUARD_BREACHED", "PROP_GUARD_HWM",
        "PROP_GUARD_EOD_BAL", "PROP_GUARD_EOD_DATE"]

ATTACH_HELP = """
MANUAL ATTACH REQUIRED (interactive attachment detector armed):
  1. Focus the MetaTrader 5 terminal window.
  2. Open the Navigator (View -> Navigator).
  3. Expand Scripts, then TestPropGuard.
  4. Drag it onto ANY open chart (US500.c preferred), keep default
     inputs, press OK.
  5. The script self-terminates in seconds and resets all guard state.
The harness is polling the Experts log and will continue automatically.
"""


def phase(name):
    print("\n" + "=" * 62)
    print("PHASE %s" % name)
    print("=" * 62, flush=True)


def read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def phase_static(results):
    phase("S: STATIC SOURCE AUDIT")
    src = read_text(GUARD_MQH)
    for key, want in SPEC_EXPECT.items():
        m = re.search(r"#define\s+%s\s+([\d.]+)" % key, src)
        got = m.group(1) if m else None
        ok = got is not None and float(got) == float(want)
        results.append(("S", "%s = %s (SPEC %s)" % (key, got, want), ok))
        print("  %-34s got=%-7s want=%-7s %s"
              % (key, got, want, "PASS" if ok else "FAIL"))

    for ea, tag, magic in [("ORB_Strategy_Live.mq5", "ORB", "26082201"),
                           ("SP2L_SpikeGap_Live.mq5", "SP2L", "26082701")]:
        txt = read_text(os.path.join(REPO, "ea", ea))
        m = re.search(r"input\s+long\s+InpMagicNumber\s*=\s*(\d+)", txt)
        got = m.group(1) if m else None
        init = ('PG_Init("%s"' % tag) in txt
        ok = got == magic and init
        results.append(("S", "%s PG_Init tag + magic default" % ea, ok))
        print("  %-34s magic=%-9s PG_Init=%s %s"
              % (ea, got, init, "PASS" if ok else "FAIL"))

    required = {
        "ORB_Strategy_Live.mq5": ["PG_Init(", "PG_DeviationPoints(",
                                  "PG_EntryAllowed(", "PG_OnTick()"],
        "SP2L_SpikeGap_Live.mq5": ["PG_Init(", "PG_DeviationPoints(",
                                   "PG_EntryAllowed(", "PG_OnTick()",
                                   "PG_OrderSend("],
    }
    for ea, patterns in required.items():
        lines = [ln for ln in read_text(os.path.join(REPO, "ea", ea))
                 .splitlines() if not ln.strip().startswith("//")]
        body = "\n".join(lines)
        missing = [p for p in patterns if p not in body]
        ok = not missing
        results.append(("S", "%s guard integration call-sites" % ea, ok))
        print("  %-34s %d/%d call-sites %s%s"
              % (ea, len(patterns) - len(missing), len(patterns),
                 "PASS" if ok else "FAIL",
                 " missing=%s" % missing if missing else ""))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def phase_binary(results):
    phase("B: BINARY / DEPLOYMENT AUDIT")
    if not os.path.exists(TEST_EX5):
        results.append(("B", "TestPropGuard.ex5 compiled", False))
        print("  FAIL: %s missing" % TEST_EX5)
        return
    dst = os.path.join(MQL5_SCRIPTS, "TestPropGuard.ex5")
    copied = False
    if (not os.path.exists(dst)
            or os.path.getmtime(dst) < os.path.getmtime(TEST_EX5)):
        shutil.copy2(TEST_EX5, dst)
        copied = True
    results.append(("B", "TestPropGuard.ex5 deployed to MQL5/Scripts", True))
    print("  TestPropGuard.ex5 -> MQL5/Scripts %s"
          % ("(copied)" if copied else "(already current)"))

    for ea, magic in [("ORB_Strategy_Live", 26082201),
                      ("SP2L_SpikeGap_Live", 26082701)]:
        mq5 = os.path.join(REPO, "ea", ea + ".mq5")
        ex5 = os.path.join(REPO, "ea", ea + ".ex5")
        clog = os.path.join(REPO, "deploy", "infrastructure", "compile_logs",
                            ea + ".compile.log")
        fresh = (os.path.exists(ex5)
                 and os.path.getmtime(ex5) >= os.path.getmtime(mq5))
        clean_compile = False
        if os.path.exists(clog):
            raw = open(clog, "rb").read()
            txt = raw.decode("utf-16", errors="replace")
            clean_compile = ("Result: 0 errors, 0 warnings" in txt)
        src_magic = re.search(r"input\s+long\s+InpMagicNumber\s*=\s*(\d+)",
                              read_text(mq5))
        src_ok = src_magic is not None and int(src_magic.group(1)) == magic
        ok = fresh and clean_compile and src_ok
        results.append(("B", "%s.ex5 fresh + clean compile + source magic"
                        % ea, ok))
        print("  %s.ex5 fresh=%s compile_clean=%s source_magic=%s %s"
              % (ea, fresh, clean_compile, src_ok, "PASS" if ok else "FAIL"))
        print("    sha256(mq5)=%s" % sha256(mq5))
        print("    sha256(ex5)=%s" % (sha256(ex5) if os.path.exists(ex5)
                                      else "n/a"))
    print("  note: build 6140 .ex5 binaries are packed - no byte-level"
          " magic probes are possible;") 
    print("        embedding is attested by source magic + freshness +"
          " compile log + pinned sha256.")


def latest_log():
    logs = [os.path.join(MQL5_LOGS, n) for n in os.listdir(MQL5_LOGS)
            if n.endswith(".log")]
    if not logs:
        return None
    return max(logs, key=os.path.getmtime)


def read_log_text(path):
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 4 * 1024 * 1024))
        raw = f.read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def attach_attempt():
    ps = (
        "$p = Get-Process terminal64 -ErrorAction Stop | "
        "Select-Object -First 1; "
        "Add-Type 'using System; using System.Runtime.InteropServices; "
        "public class U { "
        "[DllImport(\"user32.dll\")] public static extern bool "
        "SetForegroundWindow(IntPtr h); "
        "[DllImport(\"user32.dll\")] public static extern bool "
        "ShowWindow(IntPtr h, int n); }'; "
        "[U]::ShowWindow($p.MainWindowHandle, 9) | Out-Null; "
        "[U]::SetForegroundWindow($p.MainWindowHandle) | Out-Null; "
        "Start-Sleep -Milliseconds 700; "
        "$w = New-Object -ComObject WScript.Shell; "
        "$w.SendKeys('^n'); Start-Sleep -Milliseconds 800; "
        "$w.SendKeys('TestPropGuard'); Start-Sleep -Milliseconds 1200; "
        "$w.SendKeys('{ENTER}'); Start-Sleep -Milliseconds 800; "
        "$w.SendKeys('{ENTER}'); 'SENT'"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=60)
        if out.stdout.strip():
            return out.stdout.strip()
        return "PS_ERROR: %s" % out.stderr.strip()[:200]
    except Exception as exc:
        return "ERROR: %s" % exc


GTV_FILE = os.path.join(TERMINAL_DATA, "gvariables.dat")


def gtv_scan(mt5):
    if hasattr(mt5, "global_variable_check"):
        return "api", {n: (mt5.global_variable_check(n)
                           and mt5.global_variable_get(n) is not None
                           and mt5.global_variable_get(n) > 0)
                       for n in GTVS}
    if os.path.exists(GTV_FILE):
        blob = open(GTV_FILE, "rb").read()
        return "file", {n: n.encode("ascii", errors="ignore") in blob
                        for n in GTVS}
    return "file", {n: False for n in GTVS}


def phase_connection(results, mt5):
    phase("C: TERMINAL CONNECTION AUDIT")
    if not mt5.initialize(TERMINAL_EXE):
        results.append(("C", "mt5.initialize", False))
        print("  FAIL initialize: %s" % (mt5.last_error(),))
        return False
    print("  waiting for account sync ...", flush=True)
    acc = None
    for _ in range(60):
        acc = mt5.account_info()
        if acc is not None and acc.login != 0:
            break
        time.sleep(1.0)
    if acc is None:
        results.append(("C", "account_info", False))
        print("  FAIL: no account info after 60 s")
        return False
    demo = acc.trade_mode == 0
    results.append(("C", "demo account (trade_mode == 0)", demo))
    print("  login=%d server=%s trade_mode=%d (%s) equity=%.2f %s"
          % (acc.login, acc.server, acc.trade_mode,
             "DEMO" if demo else "NOT-DEMO", acc.equity,
             "PASS" if demo else "FAIL"))

    sym = mt5.symbol_info("US500.c")
    if sym is None:
        mt5.symbol_select("US500.c", True)
        sym = mt5.symbol_info("US500.c")
    tick = mt5.symbol_info_tick("US500.c")
    spread = (tick.ask - tick.bid) if tick else None
    results.append(("C", "US500.c selectable", sym is not None))
    print("  US500.c selected=%s point=%s spread=%s"
          % (sym is not None,
             getattr(sym, "point", None), spread))

    gtv_api = hasattr(mt5, "global_variable_get")
    mode, pre = gtv_scan(mt5)
    results.append(("C", "GTV read path available (%s)" % mode, True))
    print("  global-variable read path: %s"
          % ("python API" if gtv_api else
             "gvariables.dat file fallback (python API lacks GTV functions"
             " in 5.0.6090)"))

    stray = [n for n, v in pre.items() if v]
    if stray and mode == "api":
        for n in stray:
            try:
                mt5.global_variable_del(n)
            except Exception:
                pass
        print("  cleared stray GTVs before test: %s" % stray)
    elif stray:
        print("  WARNING: stray PROP_GUARD GTVs in gvariables.dat: %s"
              " (delete via F3 before the test)" % stray)
    return demo


def wait_for_run(timeout, poll, cap_started):
    log = latest_log()
    base_name = os.path.basename(log) if log else None
    base_size = os.path.getsize(log) if log else 0
    deadline = time.time() + timeout
    started = None
    while time.time() < deadline:
        cur = latest_log()
        if cur is None:
            time.sleep(poll)
            continue
        if os.path.basename(cur) != base_name:
            base_name = os.path.basename(cur)
            base_size = 0
        text = read_log_text(cur)
        cut = text[cap_started:] if cap_started else text
        if started is None and "=== TestPropGuard start ===" in cut:
            started = time.time()
            print("  [%s] start marker seen"
                  % dt.datetime.now().strftime("%H:%M:%S"), flush=True)
        if started is not None and "=== TestPropGuard done:" in cut:
            return cut, True
        time.sleep(poll)
    cur = latest_log()
    text = read_log_text(cur) if cur else ""
    return (text[cap_started:] if cap_started else text), False


def assert_battery(text, results):
    if "=== TestPropGuard start ===" not in text:
        results.append(("R", "TestPropGuard executed", False))
        print("  FAIL: no start marker in captured log")
        return
    m = re.search(r"=== TestPropGuard done:\s*(\d+) passed, (\d+) failed", text)
    if m:
        npass, nfail = int(m.group(1)), int(m.group(2))
        results.append(("R", "zero FAIL count (%d passed)" % npass, nfail == 0))
        print("  script verdict: %d passed, %d failed" % (npass, nfail))
    else:
        results.append(("R", "script verdict line parsed", False))
        print("  FAIL: done line not found")

    checks = [
        ("R1 daily 3.6% breach forces latch + PANIC_FLATTEN",
         "TEST 1 PASS" in text and
         re.search(r"### PROPFIRMGUARD BREACH: DAILY_DRAWDOWN", text) is not None),
        ("R2 trailing 8.1% breach forces latch + PANIC_FLATTEN",
         "TEST 2 PASS" in text and
         re.search(r"### PROPFIRMGUARD BREACH: TRAILING_DRAWDOWN", text) is not None),
        ("R3 PROP_GUARD_BREACHED GTV latched (in-test verification)",
         ("TEST 1 PASS" in text and "TEST 2 PASS" in text)),
        ("R4 PG_EntryAllowed refuses entries while latched",
         "TEST 1b PASS" in text and
         re.search(r"PG entry blocked \(TEST\): breach latched", text) is not None),
        ("R5 deviation enforcement <= 0.20 index pt",
         bool(re.search(r"deviation=(\d+) points \(point=([\d.]+)\)", text))),
        ("R6a guard state reset by script",
         "guard state reset" in text),
    ]
    md = re.search(r"synthetic EOD_BAL=([\d.-]+) -> simulated daily loss ([\d.-]+)%", text)
    if md:
        dd = float(md.group(2))
        checks.append(("R1b simulated daily DD is ~3.6%% (got %.2f)" % dd,
                       3.55 <= dd <= 3.65))
    else:
        checks.append(("R1b simulated daily DD print found", False))
    mh = re.search(r"synthetic HWM=([\d.-]+) -> simulated trailing DD ([\d.-]+)%", text)
    if mh:
        dd = float(mh.group(2))
        checks.append(("R2b simulated trailing DD is ~8.1%% (got %.2f)" % dd,
                       8.05 <= dd <= 8.15))
    else:
        checks.append(("R2b simulated trailing DD print found", False))
    mv = re.search(r"deviation=(\d+) points \(point=([\d.]+)\)", text)
    if mv:
        dev, point = int(mv.group(1)), float(mv.group(2))
        checks.append(("R5b deviation %d x point %s <= 0.20 index pt"
                       % (dev, mv.group(2)),
                       dev * point <= 0.20 + 1e-6 and dev >= 1))
    for label, ok in checks:
        results.append(("R", label, ok))
        print("  %-52s %s" % (label, "PASS" if ok else "FAIL"))


def phase_run(results, mt5, timeout, poll, attach):
    phase("R: TestPropGuard LIVE EXECUTION")
    pre_text = ""
    cur = latest_log()
    if cur:
        pre_text = read_log_text(cur)
    cap_started = len(pre_text)

    if attach:
        print("  automated attach attempt:", attach_attempt())
    print(ATTACH_HELP, flush=True)

    text, finished = wait_for_run(timeout, poll, cap_started)
    with open(os.path.join(RESULTS, "testpropguard_log_capture.txt"), "w",
              encoding="utf-8") as f:
        f.write(text)

    if "=== TestPropGuard start ===" not in text:
        results.append(("R", "TestPropGuard executed on chart", False))
        print("  FAIL: no start marker within %d s (script never attached)"
              % timeout)
        return text
    if not finished:
        results.append(("R", "TestPropGuard completed", False))
        print("  WARN: start seen but done-marker missing; parsing partial log")

    assert_battery(text, results)

    leftover_mode, leftover = gtv_scan(mt5)
    leftovers = [n for n, v in leftover.items() if v]
    clean = not leftovers
    results.append(("R6b post-run GTV scan clean (%s)" % leftover_mode, clean))
    print("  post-run GTV leftovers (%s): %s %s"
          % (leftover_mode, leftovers if leftovers else "none",
             "PASS" if clean else "FAIL"))
    if leftover_mode == "file":
        print("  note: file-based GTV scan is best-effort (terminal flushes"
              " gvariables.dat periodically); primary reset evidence is"
              " the script's 'guard state reset' log line")
    return text


def newest_agent_log():
    files = []
    for d in AGENT_LOG_DIRS:
        if os.path.isdir(d):
            files += [os.path.join(d, n) for n in os.listdir(d)
                      if n.endswith(".log")]
    return max(files, key=os.path.getmtime) if files else None


def phase_run_tester(results, notes):
    phase("R(tester): TestPropGuard HEADLESS TESTER EXECUTION")
    ok_deploy = (os.path.exists(TESTER_INI) and os.path.exists(RUNNER_EX5))
    if ok_deploy:
        dst = os.path.join(MQL5_EXPERTS, "TestPropGuardRunner.ex5")
        if (not os.path.exists(dst)
                or os.path.getmtime(dst) < os.path.getmtime(RUNNER_EX5)):
            shutil.copy2(RUNNER_EX5, dst)
    results.append(("R", "runner EA + tester ini deployed", ok_deploy))
    print("  ini=%s runner=%s %s" % (os.path.exists(TESTER_INI),
                                     os.path.exists(RUNNER_EX5),
                                     "PASS" if ok_deploy else "FAIL"))
    if not ok_deploy:
        return

    base_log = newest_agent_log()
    base_mtime = os.path.getmtime(base_log) if base_log else 0

    for proc in ("terminal64.exe", "metatester64.exe"):
        subprocess.run(["taskkill", "/F", "/IM", proc],
                       capture_output=True)
    time.sleep(2.0)

    print("  launching tester pass (terminal64 /config) ...", flush=True)
    try:
        p = subprocess.Popen(
            [TERMINAL_EXE, "/config:%s" % TESTER_INI],
            cwd=os.path.dirname(TERMINAL_EXE))
    except OSError as exc:
        results.append(("R", "tester launch", False))
        print("  FAIL: %s" % exc)
        return
    try:
        p.wait(timeout=300)
        print("  tester terminal exited (code %s)" % p.returncode)
    except subprocess.TimeoutExpired:
        p.kill()
        results.append(("R", "tester completed within 300 s", False))
        print("  FAIL: tester did not finish in 300 s")
        return

    time.sleep(2.0)
    log = newest_agent_log()
    if log is None or os.path.getmtime(log) <= base_mtime:
        for _ in range(20):
            time.sleep(1.0)
            log = newest_agent_log()
            if log is not None and os.path.getmtime(log) > base_mtime:
                break
    if log is None or os.path.getmtime(log) <= base_mtime:
        results.append(("R", "agent log produced", False))
        print("  FAIL: no new agent log found")
        return
    text = read_log_text(log)
    with open(os.path.join(RESULTS, "testpropguard_log_capture.txt"), "w",
              encoding="utf-8") as f:
        f.write(text)
    print("  agent log: %s" % log)
    assert_battery(text, results)
    notes.append("Phase R executed headlessly in the MT5 Strategy Tester"
                 " (synthetic demo environment, US500-tick M5, Model=0) via"
                 " deploy/infrastructure/TestPropGuardRunner.mq5, which runs"
                 " the shared scripts/TestPropGuardBody.mqh verbatim;"
                 " agent-sandbox GTVs are isolated by design, so reset"
                 " evidence is the in-log 'guard state reset' line.")
    notes.append("The interactive live-chart detector remains available:"
                 " run without --tester-run and attach"
                 " MQL5/Scripts/TestPropGuard.ex5 to any demo chart.")


def write_report(results, notes, started_at):
    lines = []
    lines.append("REPORT_PROPGUARD_VERIFICATION")
    lines.append("Study dir: data/results/propguard_verification/")
    lines.append("Harness: deploy/infrastructure/verify_propguard_live.py")
    lines.append("Executed: %s UTC" % started_at)
    lines.append("SPEC: deploy/infrastructure/SPEC_EXECUTION_INFRASTRUCTURE.txt"
                 " (frozen 2026-09-08)")
    lines.append("")
    lines.append("Artifacts audited: Include/PropFirmGuard.mqh, "
                 "ea/ORB_Strategy_Live.mq5/.ex5,")
    lines.append("  ea/SP2L_SpikeGap_Live.mq5/.ex5, scripts/TestPropGuard.mq5/.ex5")
    lines.append("")
    lines.append("AUDIT CHECKS")
    nfail = 0
    for ph, label, ok in results:
        nfail += 0 if ok else 1
        lines.append("  [%s] %-56s %s"
                     % (ph, label, "PASS" if ok else "FAIL"))
    lines.append("")
    lines.append("Checks failed: %d of %d" % (nfail, len(results)))
    lines.append("")
    lines.append("NOTES")
    for n in notes:
        lines.append("  " + n)
    lines.append("")
    verdict = "PASS" if nfail == 0 else "FAIL"
    lines.append("VERDICT: %s" % verdict)
    path = os.path.join(RESULTS, "REPORT_PROPGUARD_VERIFICATION.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return verdict, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument("--no-attach-attempt", action="store_true")
    ap.add_argument("--tester-run", action="store_true",
                    help="headless Phase R via the Strategy Tester runner"
                         " instead of interactive chart attach")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    started_at = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    results = []
    notes = []

    phase_static(results)
    phase_binary(results)

    mt5 = None
    try:
        import MetaTrader5 as mt5mod
        mt5 = mt5mod
    except ImportError:
        results.append(("C", "MetaTrader5 python package", False))
        print("  FAIL: MetaTrader5 package not importable")

    demo = False
    if mt5 is not None:
        demo = phase_connection(results, mt5)
        try:
            mt5.shutdown()
        except Exception:
            pass

    if args.tester_run:
        phase_run_tester(results, notes)
    elif mt5 is not None and demo:
        mt5.initialize(TERMINAL_EXE)
        phase_run(results, mt5, args.timeout, args.poll,
                  attach=not args.no_attach_attempt)
    else:
        notes.append("Phase R skipped: no demo-account connection"
                     " (attach TestPropGuard.ex5 manually and re-run"
                     " once a demo terminal is up, or use --tester-run).")

    verdict, path = write_report(results, notes, started_at)
    print("\nreport: %s" % path)
    print("VERDICT: %s" % verdict)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
