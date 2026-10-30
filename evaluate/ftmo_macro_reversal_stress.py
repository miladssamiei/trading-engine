import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\Milad\Desktop\RR")
STUDY = ROOT / "data" / "results" / "ftmo_macro_reversal_stress"
G2 = ROOT / "data" / "results" / "short_term_reversal" / "g2_trades.csv"
D1 = ROOT / "data" / "results" / "momentum_multihorizon" / "us500c_d1.csv"

INITIAL = 10000.0
TARGET = 10800.0
FLOOR = 9000.0
DAILY_PCT = 5.0
SPREAD_PT = 0.51
LOTS = 3.0
PT_USD = 1.0
N_MC = 10000
SEED = 20260921
DECL_N = 458
DECL_SF_TOTAL = 2467.60


def load():
    d1 = pd.read_csv(D1).sort_values("time").reset_index(drop=True)
    d1["date"] = pd.to_datetime(d1["time"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    dates = d1["date"].tolist()
    idx = {d: k for k, d in enumerate(dates)}
    close = d1["close"].to_numpy()
    high = d1["high"].to_numpy()
    low = d1["low"].to_numpy()

    g = pd.read_csv(G2)
    assert len(g) == DECL_N, f"trade count {len(g)} != {DECL_N}"
    e = [idx[x] for x in g["entry"]]
    x = [idx[x] for x in g["exit"]]
    sig = g["signal"].to_numpy()
    gross = g["gross_pt"].to_numpy()
    sf_total = float((gross - SPREAD_PT).sum())
    assert abs(sf_total - DECL_SF_TOTAL) < 0.01, f"swap-free total {sf_total}"
    recon = sig * (close[np.array(x)] - close[np.array(e)])
    recon_med = float(np.median(np.abs(recon - gross)))
    return dates, idx, close, high, low, e, x, sig, gross, sf_total, recon_med


def build_paths(e, x, sig, close, high, low, gross):
    """Per-trade precomputed blocks: span (exposure days), worst_pts[rel-1]
    for rel=1..span (adverse D1 extreme vs entry close), close_pts[rel-1] for
    rel=1..span-1 (floating close mark), realized pts (frozen CSV value)."""
    blocks = []
    for i in range(len(e)):
        span = x[i] - e[i]
        assert span >= 1, f"trade {i} span {span}"
        entry_px = close[e[i]]
        days = np.arange(e[i] + 1, x[i] + 1)
        ext = np.where(sig[i] > 0, low[days], high[days])
        worst = sig[i] * (ext - entry_px)
        cl = sig[i] * (close[e[i] + 1:x[i]] - entry_px)
        blocks.append({"span": span, "worst": worst, "close": cl,
                       "realized": float(gross[i] - SPREAD_PT),
                       "entry_day": e[i]})
    return blocks


def run_challenge(blocks, schedule, first_day, n_days):
    """FTMO Phase-1 sim per SPEC 1.4/1.5. schedule = [(entry_day, trade_i)]
    in chronological order. Returns (outcome, fail_type, pass_day,
    days_elapsed, peak_dd_pct)."""
    k = 0
    cum = 0.0
    in_trade = False
    b = None
    rel = 0
    peak = -1e18
    max_dd = 0.0
    for d in range(first_day, n_days):
        anchor = INITIAL + PT_USD * LOTS * cum
        if not in_trade:
            while k < len(schedule) and schedule[k][0] < d:
                k += 1
            if k < len(schedule) and schedule[k][0] == d:
                b = blocks[schedule[k][1]]
                rel = 0
                in_trade = True
                k += 1
                eq_close = anchor
                worst_eq = anchor
            else:
                eq_close = anchor
                worst_eq = anchor
        else:
            rel += 1
            worst_eq = INITIAL + PT_USD * LOTS * (cum + b["worst"][rel - 1])
            if rel == b["span"]:
                eq_close = INITIAL + PT_USD * LOTS * (cum + b["realized"])
                cum += b["realized"]
                in_trade = False
                # chained same-day re-entry (rolling calendar: next entry may
                # equal this exit day)
                while k < len(schedule) and schedule[k][0] < d:
                    k += 1
                if k < len(schedule) and schedule[k][0] == d:
                    b = blocks[schedule[k][1]]
                    rel = 0
                    in_trade = True
                    k += 1
            else:
                eq_close = INITIAL + PT_USD * LOTS * (cum + b["close"][rel - 1])

        if worst_eq <= FLOOR + 1e-9:
            return "FAIL", "floor", d, d - first_day + 1, max_dd
        if anchor > 0.0 and (anchor - worst_eq) / anchor * 100.0 >= DAILY_PCT - 1e-9:
            return "FAIL", "daily", d, d - first_day + 1, max_dd
        if worst_eq > peak:
            peak = worst_eq
        dd = (peak - worst_eq) / peak * 100.0
        if dd > max_dd:
            max_dd = dd
        if eq_close >= TARGET - 1e-9:
            return "PASS", "", d, d - first_day + 1, max_dd
    return "UNRESOLVED", "", n_days - 1, n_days - first_day, max_dd


def main():
    if not (STUDY / "SPEC_STRESS_TEST.txt").exists():
        raise SystemExit("SPEC missing; refusing to run")
    if not (STUDY / "GATE_STRESS_TEST.txt").exists():
        raise SystemExit("GATE missing; refusing to run")

    dates, idx, close, high, low, e, x, sig, gross, sf_total, recon_med = load()
    n_days = len(dates)
    blocks = build_paths(e, x, sig, close, high, low, gross)
    print(f"loaded: {len(blocks)} trades, D1 grid {n_days} days "
          f"({dates[0]}..{dates[-1]}), swap-free total {sf_total:+.2f} pts "
          f"(${sf_total * PT_USD * LOTS:+.2f} at {LOTS} lots), "
          f"reconstruction |diff| median {recon_med:.2f} pts")

    # ---- Stress Test A: rolling starts -----------------------------------
    start_dates = sorted({dates[e[i]] for i in range(len(e))
                          if dates[e[i]] <= "2025-12-31"})
    roll = []
    for s in start_dates:
        s_idx = idx[s]
        schedule = [(dates[e[i]], i) for i in range(len(e)) if e[i] >= s_idx]
        schedule = [(idx[d0], i) for d0, i in schedule]
        out, ftype, day, elapsed, dd = run_challenge(blocks, schedule, s_idx, n_days)
        roll.append({"start": s, "outcome": out, "fail_type": ftype,
                     "end": dates[day], "days": elapsed if out == "PASS" else None,
                     "elapsed": elapsed, "peak_dd": dd,
                     "year": s[:4]})
    df = pd.DataFrame(roll)
    n_start = len(df)
    n_pass = int((df["outcome"] == "PASS").sum())
    n_fail = int((df["outcome"] == "FAIL").sum())
    n_unres = int((df["outcome"] == "UNRESOLVED").sum())
    assert n_pass + n_fail + n_unres == n_start
    fails = df[df["outcome"] == "FAIL"]
    n_floor = int((fails["fail_type"] == "floor").sum())
    n_daily = int((fails["fail_type"] == "daily").sum())
    resolved = n_pass + n_fail
    rate_all = n_pass / n_start * 100.0
    rate_res = n_pass / resolved * 100.0 if resolved else float("nan")
    passes = df[df["outcome"] == "PASS"]
    avg_days = float(passes["days"].mean()) if n_pass else float("nan")
    med_days = float(passes["days"].median()) if n_pass else float("nan")
    pass_dd_med = float(passes["peak_dd"].median()) if n_pass else float("nan")
    pass_dd_p95 = float(passes["peak_dd"].quantile(0.95)) if n_pass else float("nan")
    print(f"rolling: {n_start} starts, pass {n_pass} ({rate_all:.2f}% of all / "
          f"{rate_res:.2f}% of resolved), fail {n_fail} "
          f"(daily {n_daily} / floor {n_floor}), unresolved {n_unres}, "
          f"avg days-to-pass {avg_days:.1f}")

    # ---- Stress Test B: Monte Carlo shuffles ------------------------------
    rng = np.random.default_rng(SEED)
    mc = []
    order_all = np.arange(len(blocks))
    for it in range(N_MC):
        order = rng.permutation(order_all)
        schedule = []
        cursor = 0
        for i in order:
            schedule.append((cursor, int(i)))
            cursor += blocks[i]["span"] + 1
        out, ftype, day, elapsed, dd = run_challenge(blocks, schedule, 0, cursor)
        mc.append({"outcome": out, "fail_type": ftype, "days": elapsed if out == "PASS" else None,
                   "elapsed": elapsed, "peak_dd": dd})
    mdf = pd.DataFrame(mc)
    mc_pass = int((mdf["outcome"] == "PASS").sum())
    mc_fail = int((mdf["outcome"] == "FAIL").sum())
    mc_unres = int((mdf["outcome"] == "UNRESOLVED").sum())
    mc_res = mc_pass + mc_fail
    mc_rate_all = mc_pass / N_MC * 100.0
    mc_rate_res = mc_pass / mc_res * 100.0 if mc_res else float("nan")
    mc_fails = mdf[mdf["outcome"] == "FAIL"]
    mc_floor = int((mc_fails["fail_type"] == "floor").sum())
    mc_daily = int((mc_fails["fail_type"] == "daily").sum())
    dd_med = float(mdf["peak_dd"].median())
    dd_p95 = float(mdf["peak_dd"].quantile(0.95))
    mc_pass_days = mdf[mdf["outcome"] == "PASS"]["days"]
    mc_pass_med = float(mc_pass_days.median()) if mc_pass else float("nan")
    print(f"monte carlo: pass {mc_pass} ({mc_rate_all:.2f}% of all / "
          f"{mc_rate_res:.2f}% of resolved), fail {mc_fail} "
          f"(daily {mc_daily} / floor {mc_floor}), unresolved {mc_unres}, "
          f"peak DD median {dd_med:.2f}% p95 {dd_p95:.2f}%")

    # ---- Gates ------------------------------------------------------------
    gt1 = bool(len(blocks) == DECL_N and abs(sf_total - DECL_SF_TOTAL) < 0.01)
    gt2 = bool(n_pass + n_fail + n_unres == n_start and n_start == len(start_dates))
    gt3 = bool(resolved > 0 and rate_res >= 50.0)
    gt4 = bool(mc_res > 0 and mc_rate_res >= 50.0 and dd_p95 < 10.0)
    verdict = "PASS" if (gt1 and gt2 and gt3 and gt4) else "FAIL"

    # ---- Plot -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    if n_pass:
        ax[0].hist(passes["days"], bins=40, alpha=0.55, color="tab:green",
                   label=f"rolling starts (passes n={n_pass})")
    if mc_pass:
        ax[0].hist(mc_pass_days, bins=40, alpha=0.55, color="tab:blue",
                   label=f"Monte Carlo (passes n={mc_pass})")
    ax[0].set_xlabel("trading days to pass")
    ax[0].set_ylabel("runs")
    ax[0].set_title("Days to $10,800 (passing runs)")
    ax[0].legend(fontsize=8)
    ax[1].hist(mdf["peak_dd"], bins=60, color="tab:purple", alpha=0.8)
    ax[1].axvline(dd_med, color="black", ls="--", lw=1,
                  label=f"median {dd_med:.2f}%")
    ax[1].axvline(dd_p95, color="red", ls="--", lw=1,
                  label=f"p95 {dd_p95:.2f}%")
    ax[1].axvline(10.0, color="darkred", ls=":", lw=1.5, label="10% max-loss")
    ax[1].set_xlabel("peak drawdown of worst-path equity (%)")
    ax[1].set_ylabel("runs")
    ax[1].set_title(f"Monte Carlo peak drawdowns (n={N_MC}, FixedLots {LOTS})")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(STUDY / "stress_test_distributions.png", dpi=130)
    plt.close(fig)

    # ---- Report -----------------------------------------------------------
    L = []
    L.append("REPORT_STRESS_TEST")
    L.append("Generated: " + dt.datetime.now().isoformat(timespec="seconds"))
    L.append("")
    L.append("PREMISE: ftmo_macro_reversal (#23) showed the swap-free macro short-term reversal")
    L.append("counterfactual passes FTMO Phase 1 at FixedLots 3.0 from a January 2021 start")
    L.append("(pass 2021-04-01, 63 trading days, peak daily 4.61%) - a single-path result. This study")
    L.append("stresses that pass at the same frozen economics: (A) rolling fresh challenges on every")
    L.append("trade entry date 2021-01..2025-12, and (B) 10,000 seeded shuffles of the 458-trade list.")
    L.append("NOT new strategy research; the short_term_reversal G-R2 FAIL stands; not an FTMO")
    L.append("qualification; downstream_model_allowed=false, live_execution_allowed=false.")
    L.append("")
    L.append("ECONOMICS: 458 frozen g2 trades; swap-free PnL = (gross_pt - 0.51 spread) x $1/pt/lot")
    L.append(f"x {LOTS} lots; swap-free point total {sf_total:+.2f} pts "
             f"(${sf_total * PT_USD * LOTS:+.2f} at {LOTS} lots). D1-close marks from us500c_d1;")
    L.append("worst floating via adverse D1 extremes; daily anchor = prior-day realized balance")
    L.append("(5.0% of anchor, the frozen #23/#24 convention); static $9,000 floor; same-day breach")
    L.append(f"voids a pass. Reconstruction audit: |signal*(close_exit-close_entry) - gross_pt| median")
    L.append(f"{recon_med:.2f} pts (marks are consistent with the frozen PnL).")
    L.append("")
    L.append("TEST A - ROLLING CHALLENGE STARTS (fresh $10,000 per start date)")
    L.append(f"  start dates simulated: {n_start}")
    L.append(f"  Rolling Start Pass Rate: {rate_all:.2f}% of all starts "
             f"({n_pass}/{n_start}); {rate_res:.2f}% of resolved ({n_pass}/{resolved})")
    L.append(f"  Average Days to Pass: {avg_days:.1f} trading days (median {med_days:.0f})")
    L.append(f"  Failures: {n_fail} -> 5% Daily Loss {n_daily} "
             f"({n_fail and n_daily / n_fail * 100:.1f}%) | 10% Max Loss floor {n_floor} "
             f"({n_fail and n_floor / n_fail * 100:.1f}%)")
    L.append(f"  Unresolved (tape end before pass/breach): {n_unres} "
             f"({n_unres / n_start * 100:.1f}%)")
    if n_pass:
        L.append(f"  Peak drawdown of passing runs: median {pass_dd_med:.2f}%, "
                 f"p95 {pass_dd_p95:.2f}%")
    L.append("  Pass rate by entry year:")
    for y, gy in df.groupby("year"):
        L.append(f"    {y}: starts {len(gy):>3}  pass {int((gy['outcome'] == 'PASS').sum()):>3} "
                 f"({float((gy['outcome'] == 'PASS').mean()) * 100:>5.1f}%)  "
                 f"fail {int((gy['outcome'] == 'FAIL').sum()):>3}  "
                 f"unres {int((gy['outcome'] == 'UNRESOLVED').sum()):>3}")
    L.append("")
    L.append("TEST B - MONTE CARLO SEQUENCING (10,000 shuffles, seed 20260921)")
    L.append("  Sequencing: trades re-sequenced back-to-back on consecutive trading days, each")
    L.append("  carrying its original holding span and daily adverse/close path (chronology")
    L.append("  destroyed, per-trade risk preserved - SPEC 3.2).")
    L.append(f"  Monte Carlo Pass Rate: {mc_rate_all:.2f}% of all runs ({mc_pass}/{N_MC}); "
             f"{mc_rate_res:.2f}% of resolved ({mc_pass}/{mc_res})")
    L.append(f"  Failures: {mc_fail} -> daily {mc_daily} | floor {mc_floor}; unresolved {mc_unres}")
    L.append(f"  Median Peak Drawdown: {dd_med:.2f}%   95th Percentile Peak Drawdown: {dd_p95:.2f}%")
    if mc_pass:
        L.append(f"  Median days-to-pass (passing runs): {mc_pass_med:.0f} trading days")
    L.append("")
    L.append("GATES")
    L.append(f"G-T1 DATA INTEGRITY: 458 trades={len(blocks) == DECL_N}, dates aligned, "
             f"swap-free total {sf_total:+.2f} (expect {DECL_SF_TOTAL:+.2f}), "
             f"$1/pt/lot, 0.51 pt spread -> {'PASS' if gt1 else 'FAIL'}")
    L.append(f"G-T2 ROLLING COVERAGE: {n_pass}+{n_fail}+{n_unres}={n_start} simulated "
             f"start dates -> {'PASS' if gt2 else 'FAIL'}")
    L.append(f"G-T3 ROLLING ROBUSTNESS: resolved pass rate {rate_res:.2f}% >= 50% -> "
             f"{'PASS' if gt3 else 'FAIL'}")
    L.append(f"G-T4 MONTE-CARLO ROBUSTNESS: resolved pass rate {mc_rate_res:.2f}% >= 50% AND "
             f"p95 peak DD {dd_p95:.2f}% < 10.0% -> {'PASS' if gt4 else 'FAIL'}")
    L.append(f"OVERALL VERDICT: {verdict}"
             + (" - the FixedLots=3.0 pass is start-luck/sequencing fragile"
                if verdict == "FAIL" else " - the pass survives rolling starts and shuffles"))
    L.append("")
    L.append("CAVEATS: worst floating uses D1 adverse extremes (pessimistic, no intraday path);")
    L.append("the daily limit is 5% of the prior-day balance anchor (FTMO's live rule is a fixed")
    L.append("$500 = 5% of initial - slightly stricter at grown balances); Monte Carlo blocks are a")
    L.append("sequencing device, not a calendar simulation (market-regime dependence is destroyed")
    L.append("by design, so MC pass rates measure sequencing luck only); offline D1-close fills vs")
    L.append("the #23 EA fills showed median -0.30 pt drift; unresolved runs are non-passes in the")
    L.append("primary rates.")
    (STUDY / "REPORT_STRESS_TEST.txt").write_text("\n".join(L), encoding="utf-8")

    meta = {"gt1": gt1, "gt2": gt2, "gt3": gt3, "gt4": gt4, "verdict": verdict,
            "lots": LOTS, "sf_total_pts": sf_total, "recon_median_pt": recon_med,
            "rolling": {"starts": n_start, "pass": n_pass, "fail": n_fail,
                        "daily": n_daily, "floor": n_floor, "unresolved": n_unres,
                        "rate_all_pct": rate_all, "rate_resolved_pct": rate_res,
                        "avg_days_to_pass": avg_days, "median_days_to_pass": med_days,
                        "pass_dd_median_pct": pass_dd_med, "pass_dd_p95_pct": pass_dd_p95},
            "mc": {"runs": N_MC, "seed": SEED, "pass": mc_pass, "fail": mc_fail,
                   "daily": mc_daily, "floor": mc_floor, "unresolved": mc_unres,
                   "rate_all_pct": mc_rate_all, "rate_resolved_pct": mc_rate_res,
                   "dd_median_pct": dd_med, "dd_p95_pct": dd_p95,
                   "pass_days_median": mc_pass_med}}
    (STUDY / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
