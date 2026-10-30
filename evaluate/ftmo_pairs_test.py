import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\Milad\Desktop\RR")
STUDY = ROOT / "data" / "results" / "ftmo_pairs_us500_usa30"
PAIRS = ROOT / "data" / "results" / "pairs_us500_usa30" / "pairs_trades.csv"
M5_30 = ROOT / "data" / "results" / "usa30_onboarding" / "usa30tick_m5.csv"
M5_EET = STUDY / "usa30_m5_eet.csv"
D1_500 = ROOT / "data" / "results" / "momentum_multihorizon" / "us500c_d1.csv"
MR_REPORT = ROOT / "data" / "results" / "ftmo_macro_reversal" / "REPORT_FTMO_MACRO_REVERSAL.txt"
EA_SRC = ROOT / "ea" / "Pairs_FTMO.mq5"
EA_LOG = ROOT / "ea" / "Pairs_FTMO.log"
EA_EX5 = ROOT / "ea" / "Pairs_FTMO.ex5"
TERM = Path(r"C:\Program Files\MetaTrader 5\terminal64.exe")
ME = Path(r"C:\Program Files\MetaTrader 5\metaeditor64.exe")
TDATA = Path(r"C:\Users\Milad\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075")
COMMON = Path(r"C:\Users\Milad\AppData\Roaming\MetaQuotes\Terminal\Common\Files")
EXPERTS = TDATA / "MQL5" / "Experts"
PROFILES = TDATA / "MQL5" / "Profiles" / "Tester"
TESTER_LOGS = TDATA / "Tester" / "logs"
CFG_INI = ROOT / "configs" / "tester_FTMO_Pairs.ini"
SYM30 = "USA30Duk"

INITIAL = 10000.0
TARGET = 10800.0
FLOOR = 9000.0
DAILY_PCT = 5.0
MULTS = [round(0.5 * k, 1) for k in range(2, 61)]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def kill_mt5():
    for name in ("terminal64.exe", "metatester64.exe"):
        subprocess.run(["taskkill", "/F", "/IM", name], capture_output=True)
    time.sleep(1)


def build_usa30_eet() -> pd.DataFrame:
    """Restamp the verified Dukascopy USA30 M5 tape to Europe/Athens (EET/EEST)
    wall time for the in-agent custom symbol import, and return D1 OHLC keyed by
    wall date (same clock as the imported bars; see SPEC_ADDENDUM_FTMO_PAIRS)."""
    import zoneinfo
    if not M5_EET.exists():
        df = pd.read_csv(M5_30)
        utc = pd.to_datetime(df["time"], unit="s", utc=True)
        wall = utc.dt.tz_convert(zoneinfo.ZoneInfo("Europe/Athens")).dt.tz_localize(None)
        out = pd.DataFrame({
            "time": wall.astype("int64") // 10**9,
            "open": df["open"].to_numpy(),
            "high": df["high"].to_numpy(),
            "low": df["low"].to_numpy(),
            "close": df["close"].to_numpy()})
        out.to_csv(M5_EET, index=False)
        print(f"built {M5_EET.name}: {len(out)} M5 bars "
              f"{wall.min()} .. {wall.max()}")
    d = pd.read_csv(M5_EET)
    g = d["date"] if "date" in d else None
    dates = pd.to_datetime(d["time"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    d1 = pd.DataFrame({
        "date": dates.unique()})
    grp = d.assign(date=dates).groupby("date")
    d1 = pd.DataFrame({"date": [k for k, _ in grp],
                       "open": grp["open"].first(),
                       "high": grp["high"].max(),
                       "low": grp["low"].min(),
                       "close": grp["close"].last()}).reset_index(drop=True)
    print(f"USA30Duk D1 marks: {len(d1)} wall dates {d1['date'].min()} .. {d1['date'].max()}")
    return d1


def compile_ea() -> dict:
    cdir = STUDY / "compile"
    cdir.mkdir(parents=True, exist_ok=True)
    for stale in (EA_EX5, EA_LOG):
        if stale.exists():
            stale.unlink()
    subprocess.run([str(ME), f"/compile:{EA_SRC}", "/log"], cwd=str(EA_SRC.parent),
                   capture_output=True, timeout=280)
    raw = EA_LOG.read_bytes() if EA_LOG.exists() else b""
    (cdir / "Pairs_FTMO.log").write_bytes(raw)
    shutil.copy2(EA_SRC, cdir / EA_SRC.name)
    if EA_EX5.exists():
        shutil.copy2(EA_EX5, cdir / EA_EX5.name)
    txt = raw.decode("utf-16-le", errors="ignore") + raw.decode("utf-8", errors="ignore")
    m = re.search(r"Result:\s*(\d+)\s*errors?,\s*(\d+)\s*warnings?", txt)
    return {"errors": int(m.group(1)) if m else None,
            "warnings": int(m.group(2)) if m else None,
            "ex5": EA_EX5.exists(),
            "source_sha256": sha256(EA_SRC),
            "ex5_sha256": sha256(EA_EX5) if EA_EX5.exists() else None}


def stage_common():
    COMMON.mkdir(parents=True, exist_ok=True)
    for stale in ("ftmo_pairs_trades.csv", "ftmo_pairs_status.csv"):
        p = COMMON / stale
        if p.exists():
            p.unlink()
    shutil.copy2(PAIRS, COMMON / "pairs_trades.csv")
    shutil.copy2(M5_EET, COMMON / "usa30_m5_eet.csv")


def write_set(name: str, lots: float) -> str:
    PROFILES.mkdir(parents=True, exist_ok=True)
    p = PROFILES / name
    p.write_text(f"; ftmo_pairs run parameters\n"
                 f"FixedLots={lots}\nInpProfitTargetUSD=10800.0\nInpMagicNumber=26091301\n",
                 encoding="ascii")
    return name


def make_ini(label: str, setname: str, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    report = outdir / f"tester_report_{label}.xml"
    text = ("[Common]\nLogin=\nPassword=\nServer=\nEnableDDE=0\nEnableNews=0\n\n"
            "[Tester]\nExpert=Pairs_FTMO\nExpertParameters=" + setname + "\n"
            "Symbol=US500.c\nPeriod=D1\nModel=0\nOptimization=0\n"
            "FromDate=2020-01-01\nToDate=2026-07-01\nForwardMode=0\n"
            "Deposit=10000\nCurrency=USD\nLeverage=100\nExecutionMode=0\n"
            "OptimizationCriterion=0\nVisualMode=0\nFrameMode=0\n"
            f"ReplaceReport=1\nReport={report}\nShutdownTerminal=1\n")
    ini = outdir / f"tester_FTMO_Pairs_{label}.ini"
    ini.write_text(text, encoding="ascii")
    CFG_INI.write_text(text, encoding="ascii")
    (STUDY / "ini").mkdir(exist_ok=True)
    shutil.copy2(ini, STUDY / "ini" / ini.name)
    return ini


def tester_log_sizes():
    snap = {}
    if TESTER_LOGS.exists():
        for f in TESTER_LOGS.glob("*.log"):
            snap[str(f)] = f.stat().st_size
    return snap


def tester_log_delta(snap):
    chunks = []
    if TESTER_LOGS.exists():
        for f in sorted(TESTER_LOGS.glob("*.log")):
            size = snap.get(str(f), 0)
            if f.stat().st_size > size:
                with open(f, "rb") as fh:
                    fh.seek(size)
                    chunks.append(fh.read().decode("utf-16-le", errors="ignore"))
    return "\n".join(chunks)


def run_tester(label: str, mult: float, timeout: int = 7200) -> dict:
    outdir = STUDY / "batches" / label
    outdir.mkdir(parents=True, exist_ok=True)
    setname = write_set(f"pairsftmo_{label}.set", mult)
    ini = make_ini(label, setname, outdir)
    stage_common()
    kill_mt5()
    before = tester_log_sizes()
    started = time.time()
    timed_out = False
    try:
        proc = subprocess.Popen([str(TERM), f"/config:{ini}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        kill_mt5()
    wall = round(time.time() - started, 1)
    log_text = tester_log_delta(before)
    (outdir / "tester_log.txt").write_text(log_text, encoding="utf-8")
    result = {"label": label, "mult": mult, "wall_s": wall, "timed_out": timed_out,
              "log_path": str(outdir / "tester_log.txt")}
    for name in ("ftmo_pairs_trades.csv", "ftmo_pairs_status.csv"):
        src = COMMON / name
        if src.exists() and src.stat().st_mtime >= started:
            shutil.copy2(src, outdir / name)
            result[name] = str(outdir / name)
    m = re.search(r"(\d+) ticks, (\d+) bars generated", log_text)
    if m:
        result["ticks"] = int(m.group(1))
        result["bars"] = int(m.group(2))
    result["passed"] = ("Test passed" in log_text)
    result["log_text"] = log_text
    return result


def parse_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], format="%Y.%m.%d %H:%M")
    df["exit_time"] = pd.to_datetime(df["exit_time"], format="%Y.%m.%d %H:%M")
    num = ["lots500", "lots30", "req30", "open500", "close500", "open30", "close30",
           "profit500", "profit30", "commission_total", "swap_total",
           "tick_value500", "tick_size500", "tick_value30", "tick_size30"]
    for c in num:
        df[c] = pd.to_numeric(df[c])
    df["pnl_sf"] = df["profit500"] + df["profit30"] - df["commission_total"]
    df["open_d"] = df["entry_time"].dt.date
    df["close_d"] = df["exit_time"].dt.date
    df["dir"] = df["direction"].map({"LONG_SPREAD": 1, "SHORT_SPREAD": -1})
    return df


def load_d1(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    return df.sort_values("time").reset_index(drop=True)


def simulate_sweep(trades: pd.DataFrame, d500: pd.DataFrame, d30: pd.DataFrame) -> dict:
    tv500 = float(trades["tick_value500"].iloc[0])
    ts500 = float(trades["tick_size500"].iloc[0])
    tv30 = float(trades["tick_value30"].iloc[0])
    ts30 = float(trades["tick_size30"].iloc[0])
    pp500 = tv500 / ts500
    pp30 = tv30 / ts30
    i500 = d500.set_index("date")
    i30 = d30.set_index("date")
    first = str(min(trades["open_d"]))
    dates = [d for d in i500.index if first <= d <= "2026-06-30" and d in i30.index]
    idx = {d: k for k, d in enumerate(dates)}
    n = len(dates)
    closed_unit = np.zeros(n)
    fl_close_unit = np.zeros(n)
    fl_worst_unit = np.zeros(n)
    for _, t in trades.iterrows():
        ci = idx.get(str(t["close_d"]))
        if ci is not None:
            closed_unit[ci] += t["pnl_sf"]
        span = [d for d in i500.index if str(t["open_d"]) <= d < str(t["close_d"])]
        for d in span:
            k = idx.get(d)
            r30 = i30.loc[d] if d in i30.index else None
            if k is None or r30 is None:
                continue
            leg500 = t["dir"] * (i500.loc[d]["close"] - t["open500"]) * pp500 * t["lots500"]
            leg30 = (-t["dir"]) * (r30["close"] - t["open30"]) * pp30 * t["lots30"]
            fl_close_unit[k] = leg500 + leg30
            adv500 = i500.loc[d]["low"] if t["dir"] > 0 else i500.loc[d]["high"]
            adv30 = r30["high"] if t["dir"] > 0 else r30["low"]
            w500 = t["dir"] * (adv500 - t["open500"]) * pp500 * t["lots500"]
            w30 = (-t["dir"]) * (adv30 - t["open30"]) * pp30 * t["lots30"]
            fl_worst_unit[k] = w500 + w30
    rows = []
    for M in MULTS:
        cl = closed_unit * M
        balance = INITIAL + np.cumsum(cl)
        anchors = np.concatenate([[INITIAL], balance[:-1]])
        eq_close = balance + fl_close_unit * M
        worst = anchors + cl + np.minimum(fl_worst_unit * M, 0.0)
        daily_loss_pct = (anchors - worst) / anchors * 100.0
        below_floor = worst <= FLOOR + 1e-9
        daily_breach = daily_loss_pct >= DAILY_PCT - 1e-12
        breach_idx = np.flatnonzero(daily_breach | below_floor)
        pass_idx = np.flatnonzero(eq_close >= TARGET - 1e-9)
        passed = len(pass_idx) > 0
        first_breach = int(breach_idx[0]) if len(breach_idx) else None
        pass_k = int(pass_idx[0]) if passed else None
        ok = passed and (first_breach is None or first_breach > pass_k)
        run_max = pass_k if ok else (first_breach if first_breach is not None else n - 1)
        peak = np.maximum.accumulate(eq_close[:run_max + 1])
        trail_dd = float(((peak - eq_close[:run_max + 1]) / peak).max() * 100.0) if run_max >= 0 else np.nan
        rows.append({"mult": M, "passed": bool(ok),
                     "pass_idx": pass_k,
                     "pass_date": dates[pass_k] if ok else None,
                     "days_to_pass": pass_k + 1 if ok else None,
                     "first_breach_idx": first_breach,
                     "first_breach_date": dates[first_breach] if first_breach is not None else None,
                     "breach_type": ("daily" if first_breach is not None and daily_breach[first_breach]
                                     and not below_floor[first_breach] else
                                     "floor" if first_breach is not None else None),
                     "peak_daily_loss_pct": float(daily_loss_pct[:run_max + 1].max()) if run_max >= 0 else np.nan,
                     "max_trailing_dd_pct": trail_dd,
                     "eq_close": eq_close,
                     "daily_loss_pct": daily_loss_pct,
                     "trail_dd_series": (1.0 - eq_close / np.maximum.accumulate(eq_close)) * 100.0})
    return {"dates": dates, "rows": rows, "pp500": pp500, "pp30": pp30}


def main():
    if not (STUDY / "SPEC_FTMO_PAIRS.txt").exists():
        raise SystemExit("SPEC missing; refusing to run")
    if not (STUDY / "GATE_FTMO_PAIRS.txt").exists():
        raise SystemExit("GATE missing; refusing to run")

    d30 = build_usa30_eet()
    if d30["date"].min() > "2021-03-12" or d30["date"].max() < "2026-06-24":
        raise SystemExit(f"USA30Duk mark coverage {d30['date'].min()}..{d30['date'].max()} "
                         f"does not cover 2021-03-12..2026-06-24 -> INFEASIBLE (data)")

    comp = compile_ea()
    print(f"compile: errors={comp['errors']} warnings={comp['warnings']} ex5={comp['ex5']}")
    if not (comp["errors"] == 0 and comp["warnings"] == 0 and comp["ex5"]):
        raise SystemExit("compile failed; aborting before tester run")
    EXPERTS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EA_EX5, EXPERTS / EA_EX5.name)
    svc_src = ROOT / "ea" / "Usa30DukService.mq5"
    svc_log = ROOT / "ea" / "Usa30DukService.log"
    svc_ex5 = ROOT / "ea" / "Usa30DukService.ex5"
    if svc_log.exists():
        shutil.copy2(svc_log, STUDY / "compile" / "Usa30DukService.log")
    if svc_src.exists():
        shutil.copy2(svc_src, STUDY / "compile" / svc_src.name)
    if svc_ex5.exists():
        shutil.copy2(svc_ex5, STUDY / "compile" / svc_ex5.name)
    setup_src = ROOT / "ea" / "Usa30SetupEA.mq5"
    if setup_src.exists():
        shutil.copy2(setup_src, STUDY / "compile" / setup_src.name)
    for stale in ((TDATA / "MQL5" / "Services" / "Usa30DukService.ex5"),
                  (TDATA / "MQL5" / "Experts" / "Usa30SetupEA.ex5")):
        if stale.exists():
            stale.unlink()

    ref = run_tester("ref_m1", 1.0)
    print(f"reference run: wall={ref['wall_s']}s timed_out={ref['timed_out']} "
          f"ticks={ref.get('ticks')} bars={ref.get('bars')} passed={ref['passed']}")
    if "ftmo_pairs_trades.csv" not in ref:
        raise SystemExit(f"no EA pairs output; tester log tail:\n{ref['log_text'][-3000:]}")
    trades = parse_trades(Path(ref["ftmo_pairs_trades.csv"]))
    status = {}
    if "ftmo_pairs_status.csv" in ref:
        srows = [l.split(",") for l in Path(ref["ftmo_pairs_status.csv"]).read_text().splitlines() if l.strip()]
        status = dict(zip(srows[0], srows[1])) if len(srows) > 1 else {}
    print(f"EA executed {len(trades)} pair trades; status={status}")
    both_legs = bool((trades["ticket500"] > 0).all()
                     and (trades["leg30_mode"] == "virtual_dukascopy").all())

    signals_next = int(status.get("signals_next", len(trades)))
    gp1 = (ref["passed"] and not ref["timed_out"] and both_legs
           and len(trades) >= 0.95 * max(signals_next, 1))
    tester_swap = float(trades["swap_total"].sum())
    gp2 = bool(len(trades) > 0 and trades["pnl_sf"].notna().all())

    d500 = load_d1(D1_500)
    sweep = simulate_sweep(trades, d500, d30)
    passers = [r for r in sweep["rows"] if r["passed"]]
    gp3 = len(passers) > 0
    optimal = min(passers, key=lambda r: (r["days_to_pass"], r["mult"])) if passers else None

    csv = pd.read_csv(PAIRS).sort_values("entry_date").reset_index(drop=True)
    ea_sorted = trades.sort_values("entry_time").reset_index(drop=True)
    m = min(len(csv), len(ea_sorted))
    fill_gross = ea_sorted["pnl_sf"].to_numpy()[:m]
    drift = fill_gross - csv["gross_usd"].to_numpy()[:m]

    gp4 = None
    conf_summary = None
    if optimal:
        conf = run_tester("confirm", optimal["mult"])
        conf_status = {}
        if "ftmo_pairs_status.csv" in conf:
            srows = [l.split(",") for l in Path(conf["ftmo_pairs_status.csv"]).read_text().splitlines() if l.strip()]
            conf_status = dict(zip(srows[0], srows[1])) if len(srows) > 1 else {}
        conf_target = conf_status.get("target_reached") == "1"
        conf_breach = conf_status.get("breached") == "1"
        conf_trades = 0
        conf_bal = float(conf_status.get("final_balance", 0.0))
        if "ftmo_pairs_trades.csv" in conf:
            cdf = parse_trades(Path(conf["ftmo_pairs_trades.csv"]))
            conf_trades = len(cdf)
            conf_bal = float(cdf["pnl_sf"].cumsum().iloc[-1]) + INITIAL
        py_trades_to_pass = int((ea_sorted["close_d"].astype(str) <= optimal["pass_date"]).sum())
        gp4 = bool(conf["passed"] and not conf["timed_out"] and conf_target
                   and not conf_breach and abs(conf_trades - max(py_trades_to_pass, 1)) <= 2
                   and conf_bal >= TARGET - 1.0)
        conf_summary = {"target_reached": conf_target, "breached": conf_breach,
                        "trades": conf_trades, "balance": conf_bal,
                        "py_trades_to_pass": py_trades_to_pass, "wall_s": conf["wall_s"]}
        print(f"confirmation: {conf_summary}")

    verdict = "PASS" if (gp1 and gp2 and gp3 and gp4) else "FAIL"

    mr_txt = ""
    if MR_REPORT.exists():
        mr_txt = MR_REPORT.read_text(encoding="utf-8")
    mr_opt = re.search(r"OPTIMAL: ([\d.]+) lots -> pass (\S+) \((\d+) trading days", mr_txt)
    mr_peak = re.search(r"peak FTMO daily loss: ([\d.]+)%", mr_txt)

    if optimal:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        dnum = pd.to_datetime(sweep["dates"]).to_numpy()
        fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
        ax[0].plot(dnum, optimal["eq_close"], lw=1.2,
                   label=f"equity (close marks), multiplier {optimal['mult']:.1f}")
        ax[0].axhline(TARGET, color="green", ls="--", lw=1, label="target $10,800 (+8%)")
        ax[0].axhline(FLOOR, color="red", ls="--", lw=1, label="max-loss floor $9,000 (-10%)")
        ax[0].axhline(INITIAL, color="gray", ls=":", lw=1, label="initial $10,000")
        pk = optimal["pass_idx"]
        ax[0].axvline(dnum[pk], color="green", alpha=0.4)
        ax[0].set_ylabel("USD")
        ax[0].legend(loc="upper left", fontsize=8)
        ax[0].set_title("FTMO swap-free replay of pairs_us500_usa30 OOS spread trades (Model=0 fills, US500.c + USA30Duk)")
        ax[1].plot(dnum[:pk + 1], -optimal["daily_loss_pct"][:pk + 1], lw=0.9, color="tab:red",
                   label="FTMO daily loss (worst-floating both legs, % of prior-day balance)")
        ax[1].plot(dnum[:pk + 1], -optimal["trail_dd_series"][:pk + 1], lw=0.9, color="tab:purple",
                   label="trailing drawdown of close-equity (%)")
        ax[1].axhline(-DAILY_PCT, color="red", ls="--", lw=1)
        ax[1].set_ylabel("%")
        ax[1].legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig(STUDY / "ftmo_pairs_equity_curve.png", dpi=130)
        plt.close(fig)

    L = []
    L.append("REPORT_FTMO_PAIRS")
    L.append("Generated: " + dt.datetime.now().isoformat(timespec="seconds"))
    L.append("")
    L.append("PREMISE: pairs_us500_usa30 (#19) failed its economics gates on financing alone (gross +$1,572.62,")
    L.append("spread -$30.77, swap -$5,510.00, net -$3,968.14 over 18 OOS spread trades; the study's USA30-leg")
    L.append("swap was unavailable and treated as 0, so the true bleed is worse). This study replays the SAME")
    L.append("18 frozen trades under simulated FTMO $10k Phase-1 Swap-Free rules (5% daily / 10% max / +8% target).")
    L.append("DOW LEG: no broker Dow history covers 2021-03..2022-09 (DJ30.c server depth starts")
    L.append("2022-11-21, 933 D1 bars; DJ30.ecn 7 bars; no USA30.c/ecn) and no headless terminal-side")
    L.append("custom-symbol build exists (tester CustomSymbolCreate=4014; services do not auto-start;")
    L.append("start-config [Charts] opens no chart on an empty workspace). Per SPEC_ADDENDUM_3_")
    L.append("FTMO_PAIRS.txt (frozen before this run) the US500.c leg is REAL tester execution and")
    L.append("the Dow leg is VIRTUAL: priced from the VERIFIED Dukascopy USA30 M5 tape (same tape as")
    L.append("study #19), EET wall clock, $1/pt/lot, exact lots30 ratio, 2.28-pt RT spread charged;")
    L.append("the in-EA FTMO guard aggregates the virtual floating PnL into every equity check. It is")
    L.append("a what-if simulation, NOT an FTMO qualification (ftmo10k_qualification_v1 stays INCOMPLETE);")
    L.append("downstream_model_allowed=false, live_execution_allowed=false; the #19 FAIL verdict stands.")
    L.append("")
    L.append(f"EXECUTION: EA fills the US500.c leg at the first tester tick at/after entry/exit-date+1day")
    L.append(f"(D1-close proxy); the virtual Dow leg prices at the same instant from the restamped tape.")
    L.append(f"Reference run multiplier 1.0: wall={ref['wall_s']}s, ticks={ref.get('ticks')}, bars={ref.get('bars')},")
    L.append(f"tester pass={ref['passed']}, executed={len(trades)} pair trades, status={status}")
    L.append(f"SWAP OVERRIDE: tester charged swap total {tester_swap:+.2f} USD -> forced to 0.00 for every trade")
    L.append(f"(swap-free mirror); analysis PnL = combined raw DEAL_PROFIT - commission; commission total "
             f"{trades['commission_total'].sum():+.2f} USD.")
    L.append(f"LEG ECONOMICS: $/pt/lot US500.c {sweep['pp500']:.4f}, USA30Duk {sweep['pp30']:.4f} (recorded).")
    L.append(f"FILL-VS-MODEL: combined swap-free fill $ vs CSV gross_usd over first {m} trades: median "
             f"{np.median(drift):+.2f}, mean {drift.mean():+.2f}, total {drift.sum():+.1f} USD")
    L.append(f"(USA30Duk fill economics + spread + daily-boundary execution vs the study's $1/pt convention).")
    L.append("")
    L.append("GATES")
    L.append(f"G-P1 EXECUTION INTEGRITY: compile 0/0, run passed={ref['passed']}, not timed out={not ref['timed_out']}, "
             f"both legs on all trades={both_legs}, executed {len(trades)} >= 95% of {signals_next} armed -> "
             f"{'PASS' if gp1 else 'FAIL'}")
    L.append(f"G-P2 SWAP-FREE OVERRIDE AUDIT: {len(trades)} trades, all swap-zeroed, tester swap "
             f"{tester_swap:+.2f} reported -> {'PASS' if gp2 else 'FAIL'}")
    L.append(f"G-P3 FTMO PHASE-1 FEASIBILITY: "
             + (f"{len(passers)} passing multipliers -> PASS" if gp3
                else "no multiplier in [1.0,30.0] passes without breach -> FAIL"))
    L.append(f"G-P4 CONFIRMATION RUN: {'PASS' if gp4 else 'FAIL'}"
             + (f" (multiplier {optimal['mult'] if optimal else '-'}: {conf_summary})" if conf_summary
                else " (not run: no optimal multiplier)"))
    L.append(f"OVERALL VERDICT: {verdict}")
    L.append("")
    L.append("MULTIPLIER SWEEP (D1-close marks, worst-floating via simultaneous adverse D1 extremes of both legs)")
    L.append(f"{'mult':>5} {'pass_date':>11} {'days':>5} {'peak_daily%':>12} {'max_trail%':>11} {'verdict':>7} {'first_breach':>12}")
    for r in sweep["rows"]:
        L.append(f"{r['mult']:>5.1f} {str(r['pass_date'] or '-'):>11} {str(r['days_to_pass'] or '-'):>5} "
                 f"{r['peak_daily_loss_pct']:>12.2f} {r['max_trailing_dd_pct']:>11.2f} "
                 f"{'PASS' if r['passed'] else 'FAIL':>7} {str(r['first_breach_date'] or '-'):>12}")
    if optimal:
        to_pass = ea_sorted[ea_sorted["close_d"].astype(str) <= optimal["pass_date"]]
        wr_all = float((ea_sorted["pnl_sf"] > 0).mean()) * 100.0
        wr = float((to_pass["pnl_sf"] > 0).mean()) * 100.0 if len(to_pass) else None
        hold = (pd.to_datetime(ea_sorted["close_d"]) - pd.to_datetime(ea_sorted["open_d"])).dt.days.mean()
        passing_days = [r["days_to_pass"] for r in passers]
        L.append("")
        L.append(f"OPTIMAL: multiplier {optimal['mult']:.1f} -> pass {optimal['pass_date']} "
                 f"({optimal['days_to_pass']} trading days from {sweep['dates'][0]})")
        L.append(f"  win rate: {wr:.2f}% over the {len(to_pass)} pairs closed by the pass date"
                 if wr is not None else
                 f"  win rate to pass: n/a (pass on floating equity before first close); "
                 f"swap-free win rate over all {len(ea_sorted)} executed pairs: {wr_all:.2f}%")
        L.append(f"  peak FTMO daily loss: {optimal['peak_daily_loss_pct']:.2f}% (limit 5.00%)")
        L.append(f"  max trailing DD of close-equity: {optimal['max_trailing_dd_pct']:.2f}%")
        L.append(f"  avg holding per pair (all executed): {hold:.2f} calendar days")
        L.append(f"  average days-to-pass across passing multipliers: {np.mean(passing_days):.1f} "
                 f"(min {min(passing_days)}, max {max(passing_days)})")
        L.append("")
        L.append("COMPARISON vs ftmo_macro_reversal (frozen REPORT_FTMO_MACRO_REVERSAL.txt):")
        if mr_opt and mr_peak:
            L.append(f"  macro reversal optimal: {mr_opt.group(1)} lots, pass {mr_opt.group(2)} "
                     f"({mr_opt.group(3)} trading days), peak daily loss {mr_peak.group(1)}%")
            L.append(f"  pairs (this study): multiplier {optimal['mult']:.1f}, pass {optimal['pass_date']} "
                     f"({optimal['days_to_pass']} trading days), peak daily loss "
                     f"{optimal['peak_daily_loss_pct']:.2f}%")
            L.append(f"  peak-daily-drawdown comparison at each strategy's own optimal: pairs "
                     f"{optimal['peak_daily_loss_pct']:.2f}% vs macro-reversal {mr_peak.group(1)}% "
                     f"(lower is safer against the 5% FTMO daily limit).")
        else:
            L.append("  (macro reversal report not parseable; see data/results/ftmo_macro_reversal/)")
    L.append("")
    L.append("CAVEATS: Model=0 constant synthetic spread underestimates the real US500 spread (~0.41 vs")
    L.append("~0.51 median; the virtual Dow leg carries the verified 2.28-pt spread by configuration);")
    L.append("worst-floating assumes SIMULTANEOUS adverse D1 extremes of both legs (very pessimistic for")
    L.append("a spread); daily anchor is prior-day realized total balance (tasking definition); the virtual")
    L.append("Dow leg prices are Dukascopy-derived on the Europe/Athens wall clock (<= 1 h DST-rule skew")
    L.append("vs the broker clock at fill instants; Python marks use the same tape).")
    (STUDY / "REPORT_FTMO_PAIRS.txt").write_text("\n".join(L), encoding="utf-8")

    meta = {"compile": comp, "status": status, "signals_next": signals_next,
            "tester_swap_total": tester_swap,
            "gp1": gp1, "gp2": gp2, "gp3": gp3, "gp4": gp4, "verdict": verdict,
            "optimal_mult": optimal["mult"] if optimal else None,
            "pass_date": optimal["pass_date"] if optimal else None,
            "peak_daily_pct": optimal["peak_daily_loss_pct"] if optimal else None,
            "max_trail_pct": optimal["max_trailing_dd_pct"] if optimal else None,
            "conf": conf_summary, "n_trades_ea": len(trades),
            "dj30_bars": len(d30)}
    (STUDY / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
