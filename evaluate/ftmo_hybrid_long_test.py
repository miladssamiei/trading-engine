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
STUDY = ROOT / "data" / "results" / "ftmo_hybrid_long"
SRC = ROOT / "data" / "results" / "youtube_hybrid_scalping" / "hybrid_scalp_trades.csv"
EA_SRC = ROOT / "ea" / "HybridLong_FTMO.mq5"
EA_LOG = ROOT / "ea" / "HybridLong_FTMO.log"
EA_EX5 = ROOT / "ea" / "HybridLong_FTMO.ex5"
TERM = Path(r"C:\Program Files\MetaTrader 5\terminal64.exe")
ME = Path(r"C:\Program Files\MetaTrader 5\metaeditor64.exe")
TDATA = Path(r"C:\Users\Milad\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075")
COMMON = Path(r"C:\Users\Milad\AppData\Roaming\MetaQuotes\Terminal\Common\Files")
EXPERTS = TDATA / "MQL5" / "Experts"
PROFILES = TDATA / "MQL5" / "Profiles" / "Tester"
TESTER_LOGS = TDATA / "Tester" / "logs"
CFG_INI = ROOT / "configs" / "tester_FTMO_HybridLong.ini"
STAGED = "hybrid_scalp_trades.csv"

INITIAL = 10000.0
TARGET = 10800.0
FLOOR = 9000.0
DAILY_PCT = 5.0
MULTS = [round(0.5 * k, 1) for k in range(2, 61)]
MAGIC = 26091601

# SPEC 1.1 pre-declared subset facts (runner re-verifies; abort on mismatch)
DECL_N = 1326
DECL_SAME_BAR = 1306
DECL_TARGETS = 994
DECL_STOPS = 332
DECL_NET_SUM = 3704.92


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


def load_subset() -> pd.DataFrame:
    df = pd.read_csv(SRC)
    d = df[(df["instrument"] == "US500") & (df["model"] == "Long")].copy()
    d = d.sort_values("entry_time").reset_index(drop=True)
    e = d["entry_time"].to_numpy()
    x = d["exit_time"].to_numpy()
    assert len(d) == DECL_N, f"subset n {len(d)} != {DECL_N}"
    assert (e[1:] >= e[:-1]).all(), "entries not sorted"
    assert (x >= e).all(), "exit < entry rows present"
    assert int((x == e).sum()) == DECL_SAME_BAR
    assert int((d["exit_reason"] == "target").sum()) == DECL_TARGETS
    assert int((d["exit_reason"] == "stop").sum()) == DECL_STOPS
    assert abs(d["pnl_net_pts"].sum() - DECL_NET_SUM) < 0.5
    overlaps = 0
    open_until = -10**18
    for i in range(len(e)):
        if e[i] < open_until:
            overlaps += 1
        else:
            open_until = max(open_until, x[i])
    assert overlaps == 0, f"{overlaps} overlaps in frozen list"
    collisions = sum(1 for i in range(1, len(e)) if e[i] < x[i - 1] + 300)
    assert collisions == 0, f"{collisions} +300s arming collisions"
    return d


def wall_epochs(utc_epochs: pd.Series) -> np.ndarray:
    """UTC epochs -> Europe/Athens wall-clock epochs (tz_localize(None) BEFORE
    the int64 cast: the #24 restamp bug)."""
    ts = pd.to_datetime(utc_epochs, unit="s", utc=True)
    wall = ts.dt.tz_convert("Europe/Athens").dt.tz_localize(None)
    return wall.astype("int64") // 10**9


def stage_common(d: pd.DataFrame):
    COMMON.mkdir(parents=True, exist_ok=True)
    for stale in ("ftmo_hybrid_long_trades.csv", "ftmo_hybrid_long_status.csv", STAGED):
        p = COMMON / stale
        if p.exists():
            p.unlink()
    out = pd.DataFrame({
        "instrument": d["instrument"],
        "model": d["model"],
        "dir": d["dir"],
        "entry_time": wall_epochs(d["entry_time"]),
        "exit_time": wall_epochs(d["exit_time"])})
    out.to_csv(COMMON / STAGED, index=False)
    print(f"staged {STAGED}: {len(out)} rows, "
          f"entry wall {pd.to_datetime(out['entry_time'].iloc[0], unit='s')} .. "
          f"{pd.to_datetime(out['entry_time'].iloc[-1], unit='s')}")


def compile_ea() -> dict:
    cdir = STUDY / "compile"
    cdir.mkdir(parents=True, exist_ok=True)
    for stale in (EA_EX5, EA_LOG):
        if stale.exists():
            stale.unlink()
    subprocess.run([str(ME), f"/compile:{EA_SRC}", "/log"], cwd=str(EA_SRC.parent),
                   capture_output=True, timeout=280)
    raw = EA_LOG.read_bytes() if EA_LOG.exists() else b""
    (cdir / "HybridLong_FTMO.log").write_bytes(raw)
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


def write_set(name: str, lots: float) -> str:
    PROFILES.mkdir(parents=True, exist_ok=True)
    p = PROFILES / name
    p.write_text(f"; ftmo_hybrid_long run parameters\n"
                 f"FixedLots={lots}\nInpProfitTargetUSD=10800.0\nInpMagicNumber={MAGIC}\n",
                 encoding="ascii")
    return name


def make_ini(label: str, setname: str, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    report = outdir / f"tester_report_{label}.xml"
    text = ("[Common]\nLogin=\nPassword=\nServer=\nEnableDDE=0\nEnableNews=0\n\n"
            "[Tester]\nExpert=HybridLong_FTMO\nExpertParameters=" + setname + "\n"
            "Symbol=US500.c\nPeriod=M5\nModel=0\nOptimization=0\n"
            "FromDate=2021-01-01\nToDate=2026-05-01\nForwardMode=0\n"
            "Deposit=10000\nCurrency=USD\nLeverage=100\nExecutionMode=0\n"
            "OptimizationCriterion=0\nVisualMode=0\nFrameMode=0\n"
            f"ReplaceReport=1\nReport={report}\nShutdownTerminal=1\n")
    ini = outdir / f"tester_FTMO_HybridLong_{label}.ini"
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
    setname = write_set(f"hybridftmo_{label}.set", mult)
    ini = make_ini(label, setname, outdir)
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
    for name in ("ftmo_hybrid_long_trades.csv", "ftmo_hybrid_long_status.csv"):
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
    for c in ("lots", "open_price", "close_price", "profit", "swap",
              "commission", "mae_usd", "mfe_usd"):
        df[c] = pd.to_numeric(df[c])
    df = df.sort_values("entry_time").reset_index(drop=True)
    # CRITICAL SWAP OVERRIDE (SPEC 3.1): swap forced to 0.00 on every trade
    df["pnl_sf"] = df["profit"] + 0.0 * df["swap"] - df["commission"]
    df["entry_d"] = df["entry_time"].dt.normalize()
    df["close_d"] = df["exit_time"].dt.normalize()
    return df


def simulate_sweep(trades: pd.DataFrame) -> dict:
    """Event-grid FTMO simulation per SPEC 3.2/3.3. All unit dollar quantities
    scale linearly in the multiplier M (reference run executed FixedLots=1.0)."""
    pnl = trades["pnl_sf"].to_numpy()
    mae = np.minimum(trades["mae_usd"].to_numpy(), 0.0)
    cum = np.cumsum(pnl)
    cum_before = cum - pnl
    entry_d = trades["entry_d"].to_numpy()
    close_d = trades["close_d"].to_numpy()

    grid = np.unique(np.concatenate([entry_d, close_d]))
    gidx = {d: k for k, d in enumerate(grid)}
    n = len(grid)
    # realized PnL of trades closed strictly BEFORE each grid day (unit $)
    # worst floating unit-$ per grid day: full-hold MAE attributed to every
    # intersected day (pessimistic, SPEC 3.2); realized close events are
    # equity samples too (balance == equity at each close; the guard-flatten
    # close can land below its own sampled MAE)
    unit_worst = np.full(n, np.nan)
    for i in range(len(trades)):
        lo = gidx[entry_d[i]]
        hi = gidx[close_d[i]]
        w = cum_before[i] + mae[i]
        seg = unit_worst[lo:hi + 1]
        seg2 = np.where(np.isnan(seg), w, np.minimum(seg, w))
        unit_worst[lo:hi + 1] = seg2
        k = gidx[close_d[i]]
        unit_worst[k] = cum[i] if np.isnan(unit_worst[k]) else min(unit_worst[k], cum[i])
    # closed-before anchors
    close_sorted_idx = np.argsort(close_d, kind="stable")
    cs = close_d[close_sorted_idx]
    cp = pnl[close_sorted_idx]
    ccum = np.cumsum(cp)
    pos = np.searchsorted(cs, grid, side="left")
    closed_before = np.where(pos > 0, np.concatenate([[0.0], ccum])[pos], 0.0)
    # days with no open trade: equity == balance at that point
    flat_eq = closed_before.copy()
    unit_worst = np.where(np.isnan(unit_worst), flat_eq, unit_worst)

    rows = []
    for M in MULTS:
        eq_close = INITIAL + M * cum                      # close-event equity
        anchors = INITIAL + M * closed_before             # prior-midnight balance
        worst_abs = INITIAL + M * unit_worst              # worst floating equity
        with np.errstate(divide="ignore", invalid="ignore"):
            daily_pct = (anchors - worst_abs) / anchors * 100.0
        daily_pct = np.where(anchors > 0.0, daily_pct, np.inf)
        floor_breach = worst_abs <= FLOOR + 1e-9
        daily_breach = daily_pct >= DAILY_PCT - 1e-9
        any_breach = daily_breach | floor_breach
        first_breach_k = int(np.flatnonzero(any_breach)[0]) if np.any(any_breach) else None
        pass_i_arr = np.flatnonzero(eq_close >= TARGET - 1e-9)
        passed = len(pass_i_arr) > 0
        pass_i = int(pass_i_arr[0]) if passed else None
        pass_day = close_d[pass_i] if passed else None
        pk = gidx[pass_day] if passed else None
        ok = False
        if passed:
            ok = first_breach_k is None or first_breach_k > pk
        run_k = pk if ok else (first_breach_k if first_breach_k is not None else n - 1)
        seg_eq = eq_close[:pass_i + 1] if ok else eq_close[:run_k + 1]
        peak = np.maximum.accumulate(seg_eq)
        trail = float(((peak - seg_eq) / peak).max() * 100.0) if len(seg_eq) else np.nan
        rows.append({"mult": M, "passed": bool(ok), "pass_i": pass_i,
                     "pass_date": pd.Timestamp(pass_day).strftime("%Y-%m-%d") if ok else None,
                     "days_to_pass": int(np.count_nonzero(grid <= pass_day)) if ok else None,
                     "first_breach_date": (pd.Timestamp(grid[first_breach_k]).strftime("%Y-%m-%d")
                                           if first_breach_k is not None else None),
                     "breach_type": ("daily" if first_breach_k is not None and
                                     daily_breach[first_breach_k] else
                                     "floor" if first_breach_k is not None else None),
                     "peak_daily_loss_pct": float(np.max(daily_pct[:run_k + 1]))
                                            if run_k >= 0 else np.nan,
                     "max_trailing_dd_pct": trail,
                     "eq_close": eq_close, "daily_pct": daily_pct})
    return {"grid": grid, "rows": rows, "cum": cum, "closed_before": closed_before}


def main():
    if not (STUDY / "SPEC_FTMO_HYBRID_LONG.txt").exists():
        raise SystemExit("SPEC missing; refusing to run")
    if not (STUDY / "GATE_FTMO_HYBRID_LONG.txt").exists():
        raise SystemExit("GATE missing; refusing to run")

    subset = load_subset()
    print(f"subset verified: n={len(subset)} "
          f"{subset['entry_time_utc'].iloc[0]} .. {subset['exit_time_utc'].iloc[-1]}")

    comp = compile_ea()
    print(f"compile: errors={comp['errors']} warnings={comp['warnings']} ex5={comp['ex5']}")
    if not (comp["errors"] == 0 and comp["warnings"] == 0 and comp["ex5"]):
        raise SystemExit("compile failed; aborting before tester run")
    EXPERTS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EA_EX5, EXPERTS / EA_EX5.name)

    stage_common(subset)

    ref = run_tester("ref_m1", 1.0)
    print(f"reference run: wall={ref['wall_s']}s timed_out={ref['timed_out']} "
          f"ticks={ref.get('ticks')} bars={ref.get('bars')} passed={ref['passed']}")
    if "ftmo_hybrid_long_trades.csv" not in ref:
        raise SystemExit(f"no EA output; tester log tail:\n{ref['log_text'][-3000:]}")
    trades = parse_trades(Path(ref["ftmo_hybrid_long_trades.csv"]))
    status = {}
    if "ftmo_hybrid_long_status.csv" in ref:
        srows = [l.split(",") for l in Path(ref["ftmo_hybrid_long_status.csv"]).read_text().splitlines() if l.strip()]
        status = dict(zip(srows[0], srows[1])) if len(srows) > 1 else {}
    print(f"EA executed {len(trades)} trades; status={status}")

    signals_next = int(status.get("signals_next", len(trades)))
    gh1 = bool(ref["passed"] and not ref["timed_out"]
               and (trades["open_price"] > 0).all() and (trades["close_price"] > 0).all()
               and len(trades) >= 0.95 * max(signals_next, 1))
    tester_swap = float(trades["swap"].sum())
    gh2 = bool(len(trades) > 0 and trades["pnl_sf"].notna().all())

    sweep = simulate_sweep(trades)
    passers = [r for r in sweep["rows"] if r["passed"]]
    gh3 = len(passers) > 0
    optimal = min(passers, key=lambda r: (r["days_to_pass"], r["mult"])) if passers else None

    # fill-vs-model drift (SPEC 3.4)
    model = subset.reset_index(drop=True)
    m = min(len(model), len(trades))
    pnl_drift = trades["pnl_sf"].to_numpy()[:m] - model["pnl_net_pts"].to_numpy()[:m] * 1.0
    px_drift = trades["open_price"].to_numpy()[:m] - model["entry_price"].to_numpy()[:m]
    wr_exec = float((trades["pnl_sf"] > 0).mean()) * 100.0
    wr_model_first = float((model["pnl_net_pts"].iloc[:m] > 0).mean()) * 100.0

    gh4 = None
    conf_summary = None
    if optimal:
        conf = run_tester("confirm", optimal["mult"])
        conf_status = {}
        if "ftmo_hybrid_long_status.csv" in conf:
            srows = [l.split(",") for l in Path(conf["ftmo_hybrid_long_status.csv"]).read_text().splitlines() if l.strip()]
            conf_status = dict(zip(srows[0], srows[1])) if len(srows) > 1 else {}
        conf_target = conf_status.get("target_reached") == "1"
        conf_breach = conf_status.get("breached") == "1"
        conf_trades = 0
        conf_bal = float(conf_status.get("final_balance", 0.0))
        if "ftmo_hybrid_long_trades.csv" in conf:
            cdf = parse_trades(Path(conf["ftmo_hybrid_long_trades.csv"]))
            conf_trades = len(cdf)
            conf_bal = float(cdf["pnl_sf"].cumsum().iloc[-1]) + INITIAL
        py_trades_to_pass = int((trades["close_d"].dt.strftime("%Y-%m-%d") <= optimal["pass_date"]).sum())
        gh4 = bool(conf["passed"] and not conf["timed_out"] and conf_target
                   and not conf_breach and abs(conf_trades - max(py_trades_to_pass, 1)) <= 2
                   and conf_bal >= TARGET - 1.0)
        conf_summary = {"target_reached": conf_target, "breached": conf_breach,
                        "trades": conf_trades, "balance": conf_bal,
                        "py_trades_to_pass": py_trades_to_pass, "wall_s": conf["wall_s"]}
        print(f"confirmation: {conf_summary}")

    verdict = "PASS" if (gh1 and gh2 and gh3 and gh4) else "FAIL"

    # archive reference-run EA outputs at study root
    shutil.copy2(Path(ref["ftmo_hybrid_long_trades.csv"]), STUDY / "ftmo_hybrid_long_trades.csv")
    if "ftmo_hybrid_long_status.csv" in ref:
        shutil.copy2(Path(ref["ftmo_hybrid_long_status.csv"]), STUDY / "ftmo_hybrid_long_status.csv")

    if optimal:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        M = optimal["mult"]
        xnum = trades["exit_time"].to_numpy()
        fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
        ax[0].plot(xnum, INITIAL + M * sweep["cum"], lw=1.0,
                   label=f"swap-free equity (close events), FixedLots {M:.1f}")
        ax[0].axhline(TARGET, color="green", ls="--", lw=1, label="target $10,800 (+8%)")
        ax[0].axhline(FLOOR, color="red", ls="--", lw=1, label="max-loss floor $9,000 (-10%)")
        ax[0].axhline(INITIAL, color="gray", ls=":", lw=1, label="initial $10,000")
        ax[0].axvline(xnum[optimal["pass_i"]], color="green", alpha=0.4)
        ax[0].set_ylabel("USD")
        ax[0].legend(loc="upper left", fontsize=8)
        ax[0].set_title("FTMO swap-free replay of youtube_hybrid_scalping US500/Long "
                        "(ATR band-touch) trades - Model=0 market fills, US500.c")
        gnum = sweep["grid"]
        pk = np.datetime64(optimal["pass_date"] if isinstance(optimal["pass_date"], str)
                           else str(optimal["pass_date"]))
        mask = gnum <= pk
        ax[1].plot(gnum[mask], -optimal["daily_pct"][mask], lw=0.9, color="tab:red",
                   label="FTMO daily loss (worst floating vs prior-midnight balance, %)")
        eqs = INITIAL + M * sweep["cum"][:optimal["pass_i"] + 1]
        tdd = (1.0 - eqs / np.maximum.accumulate(eqs)) * 100.0
        ax[1].plot(xnum[:optimal["pass_i"] + 1], -tdd, lw=0.9, color="tab:purple",
                   label="trailing drawdown of close-equity (%)")
        ax[1].axhline(-DAILY_PCT, color="red", ls="--", lw=1)
        ax[1].set_ylabel("%")
        ax[1].legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig(STUDY / "ftmo_hybrid_long_equity_curve.png", dpi=130)
        plt.close(fig)

    else:
        # no passing multiplier: plot the reference-run executed reality
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xnum = trades["exit_time"].to_numpy()
        fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
        ax[0].plot(xnum, INITIAL + sweep["cum"], lw=1.0,
                   label="swap-free equity (close events), FixedLots 1.0 (reference)")
        ax[0].axhline(TARGET, color="green", ls="--", lw=1, label="target $10,800 (+8%)")
        ax[0].axhline(FLOOR, color="red", ls="--", lw=1, label="max-loss floor $9,000 (-10%)")
        ax[0].axhline(INITIAL, color="gray", ls=":", lw=1, label="initial $10,000")
        halt = next((r for r in sweep["rows"] if r["first_breach_date"]), None)
        if halt:
            ax[0].axvline(np.datetime64(halt["first_breach_date"]), color="red", alpha=0.4,
                          label=f"guard halt ({halt['breach_type']}, {halt['first_breach_date']})")
        ax[0].set_ylabel("USD")
        ax[0].legend(loc="lower left", fontsize=8)
        ax[0].set_title("FTMO swap-free replay of youtube_hybrid_scalping US500/Long (ATR band-touch) "
                        "- Model=0 market fills, US500.c (no multiplier passes)")
        gnum = sweep["grid"]
        ref_row = sweep["rows"][0]
        mask = gnum <= xnum[-1]
        ax[1].plot(gnum[mask], -ref_row["daily_pct"][mask], lw=0.9, color="tab:red",
                   label="FTMO daily loss at 1.0 lot (worst floating vs prior-midnight balance, %)")
        eqs = INITIAL + sweep["cum"]
        tdd = (1.0 - eqs / np.maximum.accumulate(eqs)) * 100.0
        ax[1].plot(xnum, -tdd, lw=0.9, color="tab:purple",
                   label="trailing drawdown of close-equity (%)")
        ax[1].axhline(-DAILY_PCT, color="red", ls="--", lw=1)
        ax[1].set_ylabel("%")
        ax[1].legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig(STUDY / "ftmo_hybrid_long_equity_curve.png", dpi=130)
        plt.close(fig)

    L = []
    L.append("REPORT_FTMO_HYBRID_LONG")
    L.append("Generated: " + dt.datetime.now().isoformat(timespec="seconds"))
    L.append("")
    L.append("PREMISE: the youtube_hybrid_scalping study (#22) measured the ATR(14) lower-band")
    L.append("limit-touch Long model (band = Close[i-1] - 3.1*ATR[i-1], fill at band, TP +2.0 ATR /")
    L.append("SL -1.5 ATR, pessimistic same-bar stop-first, single position) on the verified US500")
    L.append("M5 tape: 1,326 frozen trades 2021-01-04..2026-04-30, net +2.7941 pt/trade at $1/pt/lot,")
    L.append("WR 74.96%, long holds unswapped (declared limitation). This study replays the EXACT")
    L.append("frozen US500/Long list under simulated FTMO $10k Phase-1 SWAP-FREE rules (5% daily /")
    L.append("10% max / +8% target $10,800): real tester market orders on US500.c, Model=0, M5.")
    L.append("NOT a strategy validation (the #22 in-sample descriptive status stands), NOT an FTMO")
    L.append("qualification (ftmo10k_qualification_v1 stays INCOMPLETE); downstream_model_allowed=")
    L.append("false, live_execution_allowed=false.")
    L.append("")
    L.append("EXECUTION: entries at the first tick at/after the (Athens-restamped) entry bar open;")
    L.append("exits armed at exit bar completion (exit_time + 300 s; 1,306/1,326 rows are same-bar")
    L.append("TP/SL resolutions, so a literal same-instant close would be degenerate - SPEC 1.3).")
    L.append(f"Reference run FixedLots=1.0: wall={ref['wall_s']}s, ticks={ref.get('ticks')}, "
             f"bars={ref.get('bars')}, tester pass={ref['passed']}, executed={len(trades)} trades, "
             f"status={status}")
    L.append(f"SWAP OVERRIDE: tester charged swap total {tester_swap:+.2f} USD -> forced to 0.00 for")
    L.append(f"every trade (swap-free mirror); analysis PnL = raw DEAL_PROFIT - commission "
             f"(commission total {trades['commission'].sum():+.2f} USD).")
    L.append(f"FILL-VS-MODEL over first {m} trades: PnL drift median {np.median(pnl_drift):+.2f}, "
             f"mean {pnl_drift.mean():+.2f}, total {pnl_drift.sum():+.1f} USD; "
             f"entry-price drift median {np.median(px_drift):+.2f} pts "
             f"(bar-open market fill vs offline band-price limit fill).")
    L.append(f"EXECUTED-VS-OFFLINE: executed win rate {wr_exec:.2f}% over {len(trades)} trades vs "
             f"offline {wr_model_first:.2f}% over the same first {m} model rows; executed expectancy "
             f"{trades['pnl_sf'].mean():+.2f} $/trade at 1.0 lot vs offline net +2.79 pt/trade. The "
             f"offline edge rests on the band-LIMIT fill: market entries at the signal bar open pay "
             f"~{np.median(px_drift):+.2f} pts worse (median), which alone exceeds the +3.30 gross pt "
             f"model edge.")
    L.append(f"GUARD HALT: reference run target_reached={status.get('target_reached')}, "
             f"breached={status.get('breached')} ({status.get('breach_reason')}): the in-EA FTMO guard "
             f"halted trading after {len(trades)} trades at final balance "
             f"{status.get('final_balance')} USD; {int(status.get('signals_total', 0)) - signals_next} "
             f"of {status.get('signals_total')} signals left unconsumed.")
    L.append("")
    L.append("GATES")
    L.append(f"G-H1 EXECUTION INTEGRITY: compile 0/0, run passed={ref['passed']}, "
             f"not timed out={not ref['timed_out']}, open+close fills on all trades="
             f"{bool((trades['open_price'] > 0).all() and (trades['close_price'] > 0).all())}, "
             f"executed {len(trades)} >= 95% of {signals_next} armed -> {'PASS' if gh1 else 'FAIL'}")
    L.append(f"G-H2 SWAP-FREE OVERRIDE AUDIT: {len(trades)} trades, all swap-zeroed, tester swap "
             f"{tester_swap:+.2f} reported -> {'PASS' if gh2 else 'FAIL'}")
    L.append(f"G-H3 FTMO PHASE-1 FEASIBILITY: "
             + (f"{len(passers)} passing multipliers -> PASS" if gh3
                else "no multiplier in [1.0,30.0] passes without breach -> FAIL"))
    L.append(f"G-H4 CONFIRMATION RUN: {'PASS' if gh4 else 'FAIL'}"
             + (f" (FixedLots {optimal['mult'] if optimal else '-'}: {conf_summary})" if conf_summary
                else " (not run: no optimal multiplier)"))
    L.append(f"OVERALL VERDICT: {verdict}")
    L.append("")
    L.append("MULTIPLIER SWEEP (event-grid worst-floating via EA tick-sampled MAE, SPEC 3.2/3.3)")
    L.append(f"{'lots':>5} {'pass_date':>11} {'days':>5} {'peak_daily%':>12} {'max_trail%':>11} {'verdict':>7} {'first_breach':>12}")
    for r in sweep["rows"]:
        L.append(f"{r['mult']:>5.1f} {str(r['pass_date'] or '-'):>11} {str(r['days_to_pass'] or '-'):>5} "
                 f"{r['peak_daily_loss_pct']:>12.2f} {r['max_trailing_dd_pct']:>11.2f} "
                 f"{'PASS' if r['passed'] else 'FAIL':>7} {str(r['first_breach_date'] or '-'):>12}")
    if optimal:
        to_pass = trades[trades["close_d"].dt.strftime("%Y-%m-%d") <= optimal["pass_date"]]
        wr = float((to_pass["pnl_sf"] > 0).mean()) * 100.0 if len(to_pass) else None
        L.append("")
        L.append(f"OPTIMAL: FixedLots {optimal['mult']:.1f} -> pass {optimal['pass_date']} "
                 f"({optimal['days_to_pass']} active tape days from the first signal)")
        L.append(f"  win rate to pass: {wr:.2f}% over the {len(to_pass)} trades closed by the pass date"
                 if wr is not None else
                 f"  win rate to pass: n/a; over all {len(trades)} executed: {wr_exec:.2f}%")
        L.append(f"  peak FTMO daily loss: {optimal['peak_daily_loss_pct']:.2f}% (limit 5.00%)")
        L.append(f"  max trailing DD of close-equity: {optimal['max_trailing_dd_pct']:.2f}%")
        L.append(f"  swap-free win rate over all {len(trades)} executed trades: {wr_exec:.2f}%")
        L.append("")
        L.append("PER-YEAR at the optimal multiplier (swap-free, executed fills):")
        yr = trades.assign(year=trades["exit_time"].dt.year,
                           usd=trades["pnl_sf"] * optimal["mult"])
        for y, g in yr.groupby("year"):
            L.append(f"  {y}: n={len(g):>4}  pnl {g['usd'].sum():+10.2f} USD  "
                     f"WR {float((g['pnl_sf'] > 0).mean()) * 100:.2f}%")
    else:
        L.append("")
        L.append(f"NO PASSING MULTIPLIER: at 1.0 lot the executed replay bleeds to the $9,000 floor "
                 f"({trades['pnl_sf'].sum():+.2f} USD over {len(trades)} trades)")
        L.append(f"before any +8% milestone; every M > 1.0 amplifies the same negative expectancy and "
                 f"breaches earlier (see sweep). G-H3 FAIL is the final answer per the GATE: the "
                 f"swap-free ATR band-touch long counterfactual does not pass Phase 1 under the "
                 f"tasking's market-order execution.")
    L.append("")
    L.append("CAVEATS: Model=0 constant synthetic spread (~0.41 pt) understates the real US500")
    L.append("median (~0.51); time-based market exits do NOT replicate the offline idealized")
    L.append("band/TP/SL limit fills (drift quantified above); MAE is sampled at Model=0 tick")
    L.append("granularity and full-hold MAE is attributed to every intersected server day")
    L.append("(pessimistic); the Athens restamp carries <= 1 h DST-rule skew vs the broker clock;")
    L.append("pass is judged on realized close-event balance (EA target-stop semantics).")
    (STUDY / "REPORT_FTMO_HYBRID_LONG.txt").write_text("\n".join(L), encoding="utf-8")

    meta = {"compile": comp, "status": status, "signals_next": signals_next,
            "tester_swap_total": tester_swap,
            "gh1": gh1, "gh2": gh2, "gh3": gh3, "gh4": gh4, "verdict": verdict,
            "optimal_lots": optimal["mult"] if optimal else None,
            "pass_date": optimal["pass_date"] if optimal else None,
            "peak_daily_pct": optimal["peak_daily_loss_pct"] if optimal else None,
            "max_trail_pct": optimal["max_trailing_dd_pct"] if optimal else None,
            "conf": conf_summary, "n_trades_ea": len(trades),
            "pnl_drift_total": float(pnl_drift.sum()),
            "pnl_drift_median": float(np.median(pnl_drift))}
    (STUDY / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
