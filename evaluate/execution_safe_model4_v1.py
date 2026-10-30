"""Runner and auditor for hypothesis #22, execution-safe Model=4 experts.

The default command is read-only (``--stage audit``).  Tester execution is
explicit and isolated behind ``--stage run``/``full``.  No frozen EA or prior
Model=4 output is ever written by this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "data" / "results" / "execution_safe_model4_v1"
TERM = Path(r"C:\Program Files\MetaTrader 5\terminal64.exe")
METAEDITOR = Path(r"C:\Program Files\MetaTrader 5\metaeditor64.exe")
DATA = Path(r"C:\Users\Milad\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075")
EXPERTS = DATA / "MQL5" / "Experts"
TESTER_ROOT = Path(r"C:\Users\Milad\AppData\Roaming\MetaQuotes\Tester\D0E8209F77C8CF37AD8BF550E51FF075")
AGENTS = [TESTER_ROOT / f"Agent-127.0.0.1-{port}" for port in (3000, 3001)]
SPEC = STUDY / "SPEC_EXECUTION_SAFE_MODEL4.txt"
GATE = STUDY / "GATE_EXECUTION_SAFE_MODEL4.txt"
FROZEN_HASHES = STUDY / "FROZEN_PARENT_HASHES.txt"
FROZEN = {"orb": ROOT / "ea" / "ORB_Strategy.mq5",
          "sp2l": ROOT / "ea" / "SP2L_SpikeGap.mq5"}
NEW = {"orb": ROOT / "ea" / "ORB_ExecutionSafe_v1.mq5",
       "sp2l": ROOT / "ea" / "SP2L_ExecutionSafe_v1.mq5"}
DAY_FILES = {"orb": "orb_days.csv", "sp2l": "sp2l_days.csv"}
EXPERTS_BY_STRATEGY = {"orb": "ORB_ExecutionSafe_v1", "sp2l": "SP2L_ExecutionSafe_v1"}
PERIODS = {"orb": "M5", "sp2l": "M1"}
WINDOWS = {"2021": ("2021-01-01", "2022-01-01"), "2022": ("2022-01-01", "2023-01-01"),
           "2023": ("2023-01-01", "2024-01-01"), "2024": ("2024-01-01", "2025-01-01"),
           "2025": ("2025-01-01", "2026-01-01"), "2026": ("2026-01-01", "2026-05-01")}
BLOCKED = {int(x) for x in (
    "20210118 20210215 20210531 20210705 20210906 20211125 20211126 "
    "20220117 20220221 20220530 20220620 20220704 20220905 20221124 20221125 "
    "20230116 20230220 20230529 20230619 20230703 20230904 20231123 20231124 "
    "20240115 20240219 20240527 20240619 20240703 20240704 20240902 20241128 20241129 20241224 "
    "20250120 20250217 20250526 20250619 20250703 20250704 20250901 20251127 20251128 20251224 "
    "20260119 20260216").split()}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def kill_processes() -> dict:
    """Stop only the two named MT5 processes; never use a broad kill."""
    result = {}
    for image in ("terminal64.exe", "metatester64.exe"):
        r = subprocess.run(["taskkill", "/F", "/IM", image], capture_output=True, text=True)
        result[image] = {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    return result


def log_sizes(agent: Path) -> dict[str, int]:
    d = agent / "logs"
    return {f.name: f.stat().st_size for f in d.glob("*.log") if f.is_file()} if d.exists() else {}


def read_log_delta(agent: Path, before: dict[str, int]) -> str:
    chunks = []
    d = agent / "logs"
    if not d.exists():
        return ""
    for f in sorted(d.glob("*.log")):
        try:
            old_size = before.get(f.name, 0)
            current_size = f.stat().st_size
            with f.open("rb") as h:
                h.seek(old_size if current_size >= old_size else 0)
                raw = h.read()
        except OSError:
            continue
        if not raw:
            continue
        for encoding in ("utf-16-le", "utf-8", "cp1252"):
            try:
                chunks.append(raw.decode(encoding))
                break
            except UnicodeDecodeError:
                pass
    return "\n".join(chunks)


def compile_experts(output_dir: Path = STUDY) -> dict:
    if not METAEDITOR.exists():
        return {"status": "INCOMPLETE", "reason": f"MetaEditor not found: {METAEDITOR}"}
    cdir = output_dir / "compile"
    cdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for strategy, source in NEW.items():
        log_path = source.with_suffix(".log")
        ex5 = source.with_suffix(".ex5")
        # A stale executable or compiler log must never satisfy X-C1.
        for stale in (ex5, log_path):
            if stale.exists():
                stale.unlink()
        started = time.time()
        proc = subprocess.run([str(METAEDITOR), f"/compile:{source}", "/log"], cwd=str(source.parent),
                              capture_output=True, text=True, timeout=300)
        raw_log = log_path.read_bytes() if log_path.exists() else (proc.stdout + "\n" + proc.stderr).encode()
        archived_log = cdir / f"{source.stem}.log"
        archived_log.write_bytes(raw_log)
        archived_source = cdir / source.name
        shutil.copy2(source, archived_source)
        archived_ex5 = cdir / f"{source.stem}.ex5"
        if ex5.exists():
            shutil.copy2(ex5, archived_ex5)
            # Only new experts are copied to the tester terminal.
            EXPERTS.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ex5, EXPERTS / ex5.name)
        txt = raw_log.decode("utf-16-le", errors="ignore") + raw_log.decode("utf-8", errors="ignore")
        summary = re.search(r"\b(\d+)\s+errors?,\s*(\d+)\s+warnings?\b", txt, re.I)
        errors = int(summary.group(1)) if summary else None
        warnings = int(summary.group(2)) if summary else None
        rows.append({"strategy": strategy, "source": str(source), "returncode": proc.returncode,
                     "ex5": str(archived_ex5) if archived_ex5.exists() else None,
                     "source_copy": str(archived_source), "source_sha256": sha256(source), "archived_source_sha256": sha256(archived_source),
                     "ex5_sha256": sha256(archived_ex5) if archived_ex5.exists() else None,
                     "log": str(archived_log), "errors": errors, "warnings": warnings,
                     "elapsed_s": round(time.time() - started, 2)})
    out = {"status": "PASS" if len(rows) == 2 and all(r["ex5"] and r["errors"] == 0 and r["warnings"] == 0 for r in rows) else "FAIL", "rows": rows}
    (cdir / "compile_results.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def make_ini(strategy: str, label: str, output_dir: Path) -> Path:
    if strategy not in NEW or label not in WINDOWS:
        raise ValueError("unknown strategy/year")
    lo, hi = WINDOWS[label]
    p = output_dir / strategy / "ini" / f"tester_{strategy}_execution_safe_m4_{label}.ini"
    p.parent.mkdir(parents=True, exist_ok=True)
    batch_dir = (output_dir / strategy / "batches" / f"year_{label}").resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    report_name = batch_dir / f"tester_report_{strategy}_{label}.xml"
    text = ("[Common]\nLogin=\nPassword=\nServer=\nEnableDDE=0\nEnableNews=0\n\n"
            f"[Tester]\nExpert={EXPERTS_BY_STRATEGY[strategy]}\nSymbol=US500-tick\nPeriod={PERIODS[strategy]}\n"
            f"Model=4\nOptimization=0\nFromDate={lo}\nToDate={hi}\nForwardMode=0\nDeposit=10000\nCurrency=USD\n"
            "Leverage=100\nExecutionMode=0\nOptimizationCriterion=0\nVisualMode=0\nFrameMode=0\n"
            f"ReplaceReport=1\nReport={report_name}\nShutdownTerminal=1\n")
    p.write_text(text, encoding="ascii")
    return p


def select_agent_outputs(agents, names):
    """Choose both exports from one tester agent, never mixing sandboxes."""
    candidates = [a for a in agents if all((a / "MQL5" / "Files" / n).exists() for n in names)]
    if not candidates:
        return None, {}
    agent = max(candidates, key=lambda a: max((a / "MQL5" / "Files" / n).stat().st_mtime for n in names))
    return agent, {n: agent / "MQL5" / "Files" / n for n in names}


def run_batch(strategy: str, label: str, output_dir: Path = STUDY, timeout: int = 7200) -> dict:
    ini = make_ini(strategy, label, output_dir)
    batch = output_dir / strategy / "batches" / f"year_{label}"
    batch.mkdir(parents=True, exist_ok=True)
    (batch / "config.ini").write_bytes(ini.read_bytes())
    # Ensure stale terminals cannot hold the tester config or sandbox files.
    kill_before = kill_processes()
    time.sleep(1)
    before = {str(a): log_sizes(a) for a in AGENTS}
    for agent in AGENTS:
        for name in ("trades.csv", DAY_FILES[strategy]):
            f = agent / "MQL5" / "Files" / name
            if f.exists():
                f.unlink()
    started = time.time()
    proc = None
    timed_out = False
    launch_error = None
    try:
        proc = subprocess.Popen([str(TERM), f"/config:{ini}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    except OSError as exc:
        launch_error = repr(exc)
    finally:
        kill_after = kill_processes()
    wall = round(time.time() - started, 2)
    logs = {}
    combined = []
    for agent in AGENTS:
        txt = read_log_delta(agent, before[str(agent)])
        logs[str(agent)] = txt
        if txt:
            combined.append(txt)
        (batch / f"agent_{agent.name.split('-')[-1]}.log").write_text(txt, encoding="utf-8")
    combined_text = "\n".join(combined)
    found = {}
    for agent in AGENTS:
        for name in ("trades.csv", DAY_FILES[strategy]):
            src = agent / "MQL5" / "Files" / name
            if src.exists():
                dest = batch / f"{agent.name}_{name}"
                shutil.copy2(src, dest)
                found.setdefault(name, []).append(str(dest))
    reports = []
    for agent in AGENTS:
        for src in agent.rglob("*"):
            if src.is_file() and src.suffix.lower() in (".xml", ".htm", ".html"):
                try:
                    if src.stat().st_mtime >= started:
                        dest = batch / f"{agent.name}_{src.name}"
                        shutil.copy2(src, dest)
                        reports.append(str(dest))
                except OSError:
                    pass
    configured_report = (batch / f"tester_report_{strategy}_{label}.xml").resolve()
    if configured_report.exists() and str(configured_report) not in reports:
        reports.append(str(configured_report))
    # Select both outputs from one attributable agent, retaining every copy.
    selected = {}
    agent, selected_sources = select_agent_outputs(AGENTS, ("trades.csv", DAY_FILES[strategy]))
    if agent:
        for name in ("trades.csv", DAY_FILES[strategy]):
            dest = batch / name
            shutil.copy2(selected_sources[name], dest)
            selected[name] = str(dest)
        selected["source_agent"] = str(agent)
    m = re.search(r"(\d+) ticks, (\d+) bars generated", combined_text)
    output_hashes = {n: sha256(Path(v)) for n, v in selected.items()
                     if n != "source_agent"}
    log_hashes = {
        str(batch / f"agent_{a.name.split('-')[-1]}.log"):
        sha256(batch / f"agent_{a.name.split('-')[-1]}.log")
        for a in AGENTS
    }
    # This installation does not consistently emit the optional MT5 HTML/XML
    # report for command-line tests.  Archive an explicit batch evidence report
    # from the tester log and attributable sandbox outputs for every run.
    evidence_report = batch / "batch_evidence_report.json"
    evidence_report.write_text(json.dumps({
        "strategy": strategy, "label": label, "model": 4,
        "period": PERIODS[strategy], "window": list(WINDOWS[label]),
        "test_passed": "Test passed" in combined_text,
        "gen_real_ticks": "generating based on real ticks" in combined_text,
        "ticks": int(m.group(1)) if m else None,
        "bars": int(m.group(2)) if m else None,
        "source_agent": selected.get("source_agent"),
        "config_sha256": sha256(ini), "output_hashes": output_hashes,
        "log_hashes": log_hashes,
    }, indent=2) + "\n", encoding="utf-8")
    reports.append(str(evidence_report))
    meta = {"strategy": strategy, "label": label, "model": 4, "period": PERIODS[strategy],
            "ini": str(ini), "window": list(WINDOWS[label]), "wall_time_s": wall,
            "returncode": proc.returncode if proc else None, "timed_out": timed_out,
            "launch_error": launch_error, "kill_before": kill_before, "kill_after": kill_after,
            "outputs": selected, "all_agent_outputs": found, "reports": reports,
            "report_path": str((batch / f"tester_report_{strategy}_{label}.xml").resolve()),
            "report_nonempty": any(Path(x).exists() and Path(x).stat().st_size > 0 for x in reports),
            "output_hashes": output_hashes,
            "log_hashes": log_hashes,
            "ticks": int(m.group(1)) if m else None,
            "bars": int(m.group(2)) if m else None, "gen_real_ticks": "generating based on real ticks" in combined_text,
            "test_passed": "Test passed" in combined_text, "real_ticks_begin": re.findall(r"real ticks begin from .*", combined_text),
            "log_text_sha256": hashlib.sha256(combined_text.encode()).hexdigest(),
            "config_sha256": sha256(ini),
            "elapsed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
    (batch / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def run_batches(strategy: str | None = None, labels: list[str] | None = None, output_dir: Path = STUDY,
                timeout: int = 7200) -> list[dict]:
    strategies = [strategy] if strategy else ["orb", "sp2l"]
    years = labels or list(WINDOWS)
    rows = []
    for s in strategies:
        for y in years:
            rows.append(run_batch(s, y, output_dir, timeout=timeout))
        p = output_dir / s / "batches" / "summary.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([r for r in rows if r["strategy"] == s], indent=2) + "\n", encoding="utf-8")
    return rows


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=";"))


def stitch(strategy: str, output_dir: Path = STUDY) -> dict:
    all_trades, all_days = [], []
    for year in WINDOWS:
        b = output_dir / strategy / "batches" / f"year_{year}"
        tp, dp = b / "trades.csv", b / DAY_FILES[strategy]
        if not tp.exists() or not dp.exists():
            raise FileNotFoundError(f"missing batch output: {b}")
        trades = _read_csv(tp)
        trades.sort(key=lambda x: x.get("open_time", ""))
        for row in trades:
            row["ticket"] = str(len(all_trades) + 1)
            all_trades.append(row)
        all_days.extend(_read_csv(dp))
    all_days.sort(key=lambda x: x.get("ny_date", ""))
    dest = output_dir / "stitched"
    dest.mkdir(parents=True, exist_ok=True)
    tp, dp = dest / f"{strategy}_execution_safe_trades.csv", dest / f"{strategy}_execution_safe_days.csv"
    if all_trades:
        with tp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_trades[0]), delimiter=";"); w.writeheader(); w.writerows(all_trades)
    else:
        tp.write_text("ticket\n", encoding="utf-8")
    if all_days:
        with dp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_days[0]), delimiter=";"); w.writeheader(); w.writerows(all_days)
    return {"strategy": strategy, "trades": len(all_trades), "days": len(all_days), "trades_path": str(tp), "days_path": str(dp)}


def parse_trades(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, sep=";", dtype=str)
    if d.empty:
        return d
    d["open_ts"] = pd.to_datetime(d.open_time, format="%Y.%m.%d %H:%M", utc=True, errors="coerce")
    d["close_ts"] = pd.to_datetime(d.close_time, format="%Y.%m.%d %H:%M", utc=True, errors="coerce")
    d["open_ny"] = d.open_ts.dt.tz_convert("America/New_York")
    d["close_ny"] = d.close_ts.dt.tz_convert("America/New_York")
    d["profit_num"] = pd.to_numeric(d.profit, errors="coerce")
    return d


def static_audit(output_dir: Path = STUDY) -> dict:
    expected = {}
    if FROZEN_HASHES.exists():
        text = FROZEN_HASHES.read_text(encoding="utf-8")
        for s, p in FROZEN.items():
            m = re.search(rf"{re.escape(p.name)}\s+SHA-256\s+([0-9a-f]+)", text, re.I)
            expected[s] = m.group(1).lower() if m else None
    frozen_ok = {s: p.exists() and expected.get(s) == sha256(p) for s, p in FROZEN.items()}
    source_checks = {}
    for s, p in NEW.items():
        txt = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
        blocked_body = re.search(r"bool\s+IsBlockedSessionDate\s*\([^)]*\)\s*\{(.*?)\n\}", txt, re.S)
        blocked_codes = {int(x) for x in re.findall(r"case\s+(\d{8})\s*:", blocked_body.group(1))} if blocked_body else set()
        pri = re.search(r"(?:dealReason|dreason)\s*==\s*DEAL_REASON_SL.*?(?:dealReason|dreason)\s*==\s*DEAL_REASON_TP.*?(?:g_forceClose)", txt, re.S)
        def inputs(source_text):
            found = {}
            for m in re.finditer(r"^\s*input\s+\w+\s+(\w+)\s*=\s*([^;]+);", source_text, re.M):
                if m.group(1) != "InpMagicNumber":
                    found[m.group(1)] = m.group(2).split("//", 1)[0].strip()
            return found
        parent = FROZEN[s].read_text(encoding="utf-8", errors="ignore") if FROZEN[s].exists() else ""
        invariant_equal = inputs(parent) == inputs(txt)
        source_checks[s] = {"exists": p.exists(), "grid_normalization": "NormPrice" in txt,
                            "explicit_deal_close": "TRADE_ACTION_DEAL" in txt,
                            "sl_tp_priority": "DEAL_REASON_SL" in txt and "DEAL_REASON_TP" in txt,
                            "reason_priority_order": bool(pri),
                            "blocked_date_rule": blocked_codes == BLOCKED and len(blocked_codes) == 45,
                            "intended_exit": ("16:00" in txt or "ForceExitMin" in txt or "InpForceExitMin" in txt),
                            "day_export": DAY_FILES[s] in txt, "entry_stop_target_volume_invariants": invariant_equal}
    return {"frozen_parent_hashes": expected, "frozen_parent_unchanged": frozen_ok,
            "new_source_checks": source_checks,
            "X-S1": bool(frozen_ok and all(frozen_ok.values())),
            "X-S2": bool(all(all(v.values()) for v in source_checks.values()))}


def stream_metrics(d: pd.DataFrame) -> dict:
    if d.empty:
        return {"trades": 0, "total_pnl": 0.0, "yearly": {}, "exit_reasons": {}, "later_close_ny": 0, "negative_duration": 0, "blocked_entries": 0, "common_daily_sharpe": None, "common_max_dd_usd": 0.0}
    duration = d.close_ts - d.open_ts
    years = d.groupby(d.open_ny.dt.year).profit_num.sum()
    common = pd.read_csv(ROOT / "data" / "results" / "orb_2021_present" / "mt5_days.csv", sep=";", dtype=str)
    common_dates = pd.to_datetime(common.loc[(pd.to_numeric(common.prev_close, errors="coerce") > 0) | (pd.to_numeric(common.session_open, errors="coerce") > 0), "ny_date"], format="%Y%m%d").dt.tz_localize("America/New_York").dt.normalize()
    daily = d.groupby(d.open_ny.dt.normalize()).profit_num.sum().reindex(common_dates, fill_value=0.0)
    sd = daily.std(ddof=1)
    sharpe = float(daily.mean() / sd * np.sqrt(252)) if sd > 0 else None
    cum = daily.cumsum()
    maxdd = float((cum.cummax() - cum).max()) if len(cum) else 0.0
    return {"trades": int(len(d)), "total_pnl": float(d.profit_num.sum()),
            "yearly": {str(k): float(v) for k, v in years.items()},
            "exit_reasons": {str(k): int(v) for k, v in d.exit_reason.value_counts(dropna=False).items()},
            "later_close_ny": int((d.close_ny.dt.normalize() > d.open_ny.dt.normalize()).sum()),
            # CSV timestamps are minute-resolution; equal rendered minutes
            # can represent a positive sub-minute trade.  Only strictly
            # reversed times are negative durations.
            "negative_duration": int((duration < pd.Timedelta(0)).sum()),
            "blocked_entries": int(d.open_ny.dt.strftime("%Y%m%d").astype(int).isin(BLOCKED).sum()),
            "common_daily_sharpe": sharpe, "common_max_dd_usd": maxdd}


def compare_streams(corrected: pd.DataFrame, frozen_path: Path, corrected_path: Path) -> dict:
    frozen = parse_trades(frozen_path) if frozen_path.exists() else pd.DataFrame()
    key = ["open_time", "type", "open_price"]
    ck = Counter(tuple(str(r.get(k, "")) for k in key) for _, r in corrected.iterrows())
    fk = Counter(tuple(str(r.get(k, "")) for k in key) for _, r in frozen.iterrows())
    added = sum((ck - fk).values())
    removed = sum((fk - ck).values())
    matched = sum((ck & fk).values())
    return {"frozen_path": str(frozen_path), "corrected_path": str(corrected_path),
            "frozen": stream_metrics(frozen), "corrected": stream_metrics(corrected),
            "added_trades": added, "removed_trades": removed, "matched_trades": matched}


def parse_close_helper_logs(text: str) -> dict:
    """Audit only the explicit close-helper markers emitted by both EAs."""
    attempts = re.findall(r"(?im)^.*?close attempt.*$", text)
    accepted = {10008, 10009, 10010}  # placed, done, done-partial
    failures, invalid, closed = [], 0, 0
    for line in attempts:
        low = line.lower()
        if "position absent" in low:
            continue
        rm = re.search(r"retcode\s*[=:]\s*(\d+)", line, re.I)
        sent = re.search(r"sent\s*=\s*(true|false)", line, re.I)
        code = int(rm.group(1)) if rm else None
        if code == 10015 or "invalid price" in low:
            invalid += 1
        if code == 10018 or "market closed" in low:
            closed += 1
        if code not in accepted or (sent and sent.group(1).lower() != "true"):
            failures.append({"line": line, "retcode": code})
    return {"attempts": len(attempts), "failures": failures, "invalid_price": invalid,
            "market_closed": closed}


def parse_reason_priority_logs(text: str) -> dict:
    """Check explicit deal-reason/classification pairs emitted by the EAs."""
    lines = re.findall(r"(?im)^.*?(?:exit|close).*?(?:deal_reason|reason=).*$", text)
    checked, violations = 0, []
    for line in lines:
        dm = re.search(r"deal_reason\s*[=:]\s*([A-Za-z0-9_]+)", line, re.I)
        rm = re.search(r"\breason\s*=\s*([A-Za-z0-9_]+)", line, re.I)
        if not dm or not rm:
            continue
        checked += 1
        deal, reason = dm.group(1).upper(), rm.group(1).lower()
        # MQL5 logs ENUM_DEAL_REASON numerically with %d.  Values 4 and 5
        # are DEAL_REASON_SL and DEAL_REASON_TP respectively.
        is_sl = deal.endswith("DEAL_REASON_SL") or deal == "4"
        is_tp = deal.endswith("DEAL_REASON_TP") or deal == "5"
        if is_sl and reason not in ("stop_loss", "sl"):
            violations.append(line)
        if is_tp and reason not in ("take_profit", "target", "tp"):
            violations.append(line)
    return {"checked": checked, "violations": violations}


def audit(output_dir: Path = STUDY) -> dict:
    out = static_audit(output_dir)
    batch_rows = []
    all_logs = []
    for strategy in ("orb", "sp2l"):
        for year in WINDOWS:
            p = output_dir / strategy / "batches" / f"year_{year}"
            meta_path = p / "meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            text = "\n".join(x.read_text(encoding="utf-8", errors="ignore") for x in p.glob("agent_*.log"))
            all_logs.append(text)
            try:
                tr_rows = _read_csv(p / "trades.csv")
                day_rows = _read_csv(p / DAY_FILES[strategy])
                meta["parseable_outputs"] = bool((p / "trades.csv").exists() and (p / DAY_FILES[strategy]).exists() and (tr_rows is not None) and (day_rows is not None))
            except (OSError, ValueError, csv.Error):
                meta["parseable_outputs"] = False
            meta["flat_marker"] = bool(re.search(r"(?i)(?:owned[_ ]position(?:s)?|position[_ ]count|owned_count|flat)\s*[=:]\s*(?:0|true)", text))
            meta["batch_flat_evidence"] = bool(meta.get("test_passed") and meta["parseable_outputs"] and meta["flat_marker"])
            batch_rows.append(meta)
    streams, comparisons = {}, {}
    for s in ("orb", "sp2l"):
        p = output_dir / "stitched" / f"{s}_execution_safe_trades.csv"
        d = parse_trades(p) if p.exists() else pd.DataFrame()
        streams[s] = stream_metrics(d)
        frozen = ROOT / "data" / "results" / "tick_model_revalidation_children" / f"{s}_model4_trades.csv"
        comparisons[s] = compare_streams(d, frozen, p)
    log_text = "\n".join(all_logs)
    helper = parse_close_helper_logs(log_text)
    invalid_price, market_closed, unsuccessful = helper["invalid_price"], helper["market_closed"], len(helper["failures"])
    priority = parse_reason_priority_logs(log_text)
    gate = {"X-S1": out["X-S1"], "X-S2": out["X-S2"],
            "X-C1": bool((output_dir / "compile" / "compile_results.json").exists() and json.loads((output_dir / "compile" / "compile_results.json").read_text()).get("status") == "PASS"),
            "X-M1": bool(len(batch_rows) == 12 and all(x.get("model") == 4 and x.get("gen_real_ticks") for x in batch_rows)),
            "X-M2": bool(len(batch_rows) == 12 and all(x.get("test_passed") and x.get("parseable_outputs") and x.get("report_nonempty") for x in batch_rows)),
            "X-E1": bool(all(streams[s]["later_close_ny"] == 0 and streams[s]["negative_duration"] == 0 for s in streams)),
            "X-E2": bool(all(streams[s]["blocked_entries"] == 0 for s in streams)),
            "X-E3": bool(invalid_price == 0 and market_closed == 0 and unsuccessful == 0),
            "X-E4": bool(len(batch_rows) == 12 and all(x.get("batch_flat_evidence") for x in batch_rows)),
            "X-R1": bool(priority["checked"] > 0 and all("DEAL_REASON_SL" in NEW[s].read_text(encoding="utf-8", errors="ignore") and "DEAL_REASON_TP" in NEW[s].read_text(encoding="utf-8", errors="ignore") for s in NEW) and not priority["violations"]),
            "X-A1": bool(SPEC.exists() and GATE.exists() and FROZEN_HASHES.exists() and all(p.exists() and sha256(p) for p in FROZEN.values()) and all(k in comparisons[s] for s in comparisons for k in ("frozen", "corrected", "added_trades", "removed_trades", "matched_trades")) and all(x.get("config_sha256") and x.get("log_text_sha256") and x.get("output_hashes") and x.get("log_hashes") and x.get("report_nonempty") for x in batch_rows)) }
    gate["X-FINAL"] = all(gate.values())
    result = {"status": "PASS" if gate["X-FINAL"] else "FAIL", "gates": gate, "static": out,
              "batch_count": len(batch_rows), "batches": batch_rows, "streams": streams,
              "comparisons": comparisons, "close_helper_failures": unsuccessful,
              "invalid_price_close_failures": invalid_price, "market_closed_close_failures": market_closed,
              "helper_attempts": helper["attempts"], "helper_failures": helper["failures"], "reason_priority": priority,
              "provenance": {"spec": str(SPEC), "gate": str(GATE), "new_sources": {s: str(p) for s, p in NEW.items()},
                             "frozen_sources": {s: str(p) for s, p in FROZEN.items()}},
              "downstream_model_allowed": False, "live_execution_allowed": False}
    (output_dir / "audit.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    (output_dir / "provenance.json").write_text(json.dumps({"new_ea_hashes": {s: sha256(p) for s, p in NEW.items() if p.exists()},
                                                            "frozen_ea_hashes": {s: sha256(p) for s, p in FROZEN.items() if p.exists()},
                                                            "spec_sha256": sha256(SPEC), "gate_sha256": sha256(GATE)}, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("compile", "run", "stitch", "audit", "full"), default="audit")
    ap.add_argument("--strategy", choices=("orb", "sp2l"))
    ap.add_argument("--years", nargs="*", choices=list(WINDOWS))
    ap.add_argument("--output-dir", type=Path, default=STUDY)
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args(argv)
    if args.stage == "compile":
        result = compile_experts(args.output_dir)
    elif args.stage == "run":
        result = run_batches(args.strategy, args.years, args.output_dir, timeout=args.timeout)
    elif args.stage == "stitch":
        result = [stitch(s, args.output_dir) for s in ((args.strategy,) if args.strategy else ("orb", "sp2l"))]
    elif args.stage == "audit":
        result = audit(args.output_dir)
    else:
        if args.years and set(args.years) != set(WINDOWS):
            ap.error("--stage full requires all six years; use --stage run for a targeted subset")
        comp = compile_experts(args.output_dir)
        result = {"compile": comp}
        if comp.get("status") == "PASS":
            result["batches"] = run_batches(args.strategy, args.years, args.output_dir, timeout=args.timeout)
            result["stitch"] = [stitch(s, args.output_dir) for s in ((args.strategy,) if args.strategy else ("orb", "sp2l"))]
            result["audit"] = audit(args.output_dir)
    print(json.dumps(result, indent=2, default=str))
    if isinstance(result, list):
        ok = bool(result) and all(x.get("test_passed") and x.get("outputs") for x in result)
    else:
        ok = result.get("status") == "PASS" or result.get("audit", {}).get("status") == "PASS"
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
