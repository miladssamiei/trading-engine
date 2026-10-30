import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\Milad\Desktop\RR")
STUDY = ROOT / "data" / "results" / "ftmo_pairs_stress"
PAIRS = ROOT / "data" / "results" / "pairs_us500_usa30" / "pairs_trades.csv"
D1_500 = ROOT / "data" / "results" / "momentum_multihorizon" / "us500c_d1.csv"
M5_30_EET = ROOT / "data" / "results" / "ftmo_pairs_us500_usa30" / "usa30_m5_eet.csv"

INITIAL = 10000.0
TARGET = 10800.0
FLOOR = 9000.0
DAILY_PCT = 5.0
MULT = 5.0
N_MC = 10000
SEED = 20260921
DECL_N = 18
DECL_SF_TOTAL = 1541.85


def load():
    d1 = pd.read_csv(D1_500).sort_values("time").reset_index(drop=True)
    d1["date"] = pd.to_datetime(d1["time"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    i500 = d1.set_index("date")

    m5 = pd.read_csv(M5_30_EET)
    wall = pd.to_datetime(m5["time"], unit="s")
    dates30 = wall.dt.strftime("%Y-%m-%d")
    grp = m5.assign(date=dates30).groupby("date")
    i30 = pd.DataFrame({"open": grp["open"].first(), "high": grp["high"].max(),
                        "low": grp["low"].min(), "close": grp["close"].last()})

    grid = sorted(set(i500.index) & set(i30.index))
    idx = {d: k for k, d in enumerate(grid)}
    g500 = i500.loc[grid]
    g30 = i30.loc[grid]

    g = pd.read_csv(PAIRS)
    assert len(g) == DECL_N, f"trade count {len(g)} != {DECL_N}"
    g["entry"] = pd.to_datetime(g["entry_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    g["exit"] = pd.to_datetime(g["exit_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    missing = [x for x in list(g["entry"]) + list(g["exit"]) if x not in idx]
    assert not missing, f"dates missing from common grid: {missing[:5]}"
    sf_total = float((g["gross_usd"] - g["spread_cost_usd"]).sum())
    assert abs(sf_total - DECL_SF_TOTAL) < 0.01, f"swap-free total {sf_total}"

    e = np.array([idx[x] for x in g["entry"]])
    x = np.array([idx[x] for x in g["exit"]])
    d500 = g["direction"].map({"LONG_SPREAD": 1, "SHORT_SPREAD": -1}).to_numpy()
    lots30 = g["lots30"].to_numpy()
    p500e = g["p500_entry"].to_numpy()
    p30e = g["p30_entry"].to_numpy()
    realized = (g["gross_usd"] - g["spread_cost_usd"]).to_numpy()

    # reconstruction audit (grid closes vs frozen gross_usd), reported only
    rc = np.empty(DECL_N)
    for i in range(DECL_N):
        c500 = g500["close"].iloc[x[i]]
        c30 = g30["close"].iloc[x[i]]
        rc[i] = d500[i] * (c500 - p500e[i]) + (-d500[i]) * (c30 - p30e[i]) * lots30[i]
    recon_med = float(np.median(np.abs(rc - g["gross_usd"].to_numpy())))

    # per-trade unit-multiplier daily paths (SPEC 1.6)
    c500a = g500["close"].to_numpy()
    l500a = g500["low"].to_numpy()
    h500a = g500["high"].to_numpy()
    c30a = g30["close"].to_numpy()
    l30a = g30["low"].to_numpy()
    h30a = g30["high"].to_numpy()
    blocks = []
    for i in range(DECL_N):
        span = int(x[i] - e[i])
        assert span >= 1, f"trade {i} span {span}"
        days = np.arange(e[i] + 1, x[i] + 1)
        cl500 = c500a[days]
        adv500 = l500a[days] if d500[i] > 0 else h500a[days]
        cl30 = c30a[days]
        adv30 = h30a[days] if d500[i] > 0 else l30a[days]
        worst = d500[i] * (adv500 - p500e[i]) + (-d500[i]) * (adv30 - p30e[i]) * lots30[i]
        cl = d500[i] * (cl500[:-1] - p500e[i]) + (-d500[i]) * (cl30[:-1] - p30e[i]) * lots30[i]
        blocks.append({"span": span, "worst": worst, "close": cl,
                       "realized": float(realized[i]), "entry_day": int(e[i])})
    return grid, idx, blocks, sf_total, recon_med, g


def run_challenge(blocks, schedule, first_day, n_days):
    """FTMO Phase-1 sim per SPEC 1.7 (realized-balance pass semantics).
    schedule = [(entry_day, trade_i)] in chronological order."""
    k = 0
    cum = 0.0
    in_trade = False
    b = None
    rel = 0
    peak = -1e18
    max_dd = 0.0
    for d in range(first_day, n_days):
        anchor = INITIAL + MULT * cum
        if not in_trade:
            while k < len(schedule) and schedule[k][0] < d:
                k += 1
            if k < len(schedule) and schedule[k][0] == d:
                b = blocks[schedule[k][1]]
                rel = 0
                in_trade = True
                k += 1
            worst_eq = anchor
        else:
            rel += 1
            worst_eq = INITIAL + MULT * (cum + b["worst"][rel - 1])
            if rel == b["span"]:
                cum += b["realized"]
                in_trade = False
                while k < len(schedule) and schedule[k][0] < d:
                    k += 1
                if k < len(schedule) and schedule[k][0] == d:
                    b = blocks[schedule[k][1]]
                    rel = 0
                    in_trade = True
                    k += 1
        bal = INITIAL + MULT * cum

        if worst_eq <= FLOOR + 1e-9:
            return "FAIL", "floor", d, d - first_day + 1, max_dd
        if anchor > 0.0 and (anchor - worst_eq) / anchor * 100.0 >= DAILY_PCT - 1e-9:
            return "FAIL", "daily", d, d - first_day + 1, max_dd
        if worst_eq > peak:
            peak = worst_eq
        dd = (peak - worst_eq) / peak * 100.0
        if dd > max_dd:
            max_dd = dd
        if bal >= TARGET - 1e-9:
            return "PASS", "", d, d - first_day + 1, max_dd
    return "UNRESOLVED", "", n_days - 1, n_days - first_day, max_dd


def main():
    if not (STUDY / "SPEC_PAIRS_STRESS.txt").exists():
        raise SystemExit("SPEC missing; refusing to run")
    if not (STUDY / "GATE_PAIRS_STRESS.txt").exists():
        raise SystemExit("GATE missing; refusing to run")

    grid, idx, blocks, sf_total, recon_med, g = load()
    n_days = len(grid)
    print(f"loaded: {len(blocks)} pairs, common grid {n_days} days "
          f"({grid[0]}..{grid[-1]}), swap-free total ${sf_total:+.2f} at 1x "
          f"(${sf_total * MULT:+.2f} at {MULT}x), reconstruction |diff| median "
          f"${recon_med:.2f}")

    # ---- Stress Test A: rolling starts -----------------------------------
    roll = []
    for i in range(len(blocks)):
        s_idx = blocks[i]["entry_day"]
        schedule = [(blocks[j]["entry_day"], j)
                    for j in range(len(blocks)) if blocks[j]["entry_day"] >= s_idx]
        out, ftype, day, elapsed, dd = run_challenge(blocks, schedule, s_idx, n_days)
        roll.append({"start": grid[s_idx], "outcome": out, "fail_type": ftype,
                     "end": grid[day], "days": elapsed if out == "PASS" else None,
                     "elapsed": elapsed, "peak_dd": dd})
    df = pd.DataFrame(roll)
    n_start = len(df)
    n_pass = int((df["outcome"] == "PASS").sum())
    n_fail = int((df["outcome"] == "FAIL").sum())
    n_unres = int((df["outcome"] == "UNRESOLVED").sum())
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
          f"{rate_res:.2f}% of resolved), fail {n_fail} (daily {n_daily} / floor "
          f"{n_floor}), unresolved {n_unres}, avg days-to-pass {avg_days:.1f}")

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
        mc.append({"outcome": out, "fail_type": ftype,
                   "days": elapsed if out == "PASS" else None,
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
          f"{mc_rate_res:.2f}% of resolved), fail {mc_fail} (daily {mc_daily} / "
          f"floor {mc_floor}), unresolved {mc_unres}, peak DD median "
          f"{dd_med:.2f}% p95 {dd_p95:.2f}%")

    # ---- Gates ------------------------------------------------------------
    gs1 = bool(len(blocks) == DECL_N and abs(sf_total - DECL_SF_TOTAL) < 0.01
               and n_pass + n_fail + n_unres == n_start == DECL_N)
    gs2 = bool(resolved > 0 and rate_res >= 50.0)
    gs3 = bool(mc_res > 0 and mc_rate_res >= 50.0 and dd_p95 < 10.0)
    verdict = "PASS" if (gs1 and gs2 and gs3) else "FAIL"

    # ---- Plot -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    if n_pass:
        ax[0].hist(passes["days"], bins=20, alpha=0.55, color="tab:green",
                   label=f"rolling starts (passes n={n_pass})")
    if mc_pass:
        ax[0].hist(mc_pass_days, bins=40, alpha=0.55, color="tab:blue",
                   label=f"Monte Carlo (passes n={mc_pass})")
    ax[0].set_xlabel("trading days to pass")
    ax[0].set_ylabel("runs")
    ax[0].set_title("Days to $10,800 realized (passing runs)")
    ax[0].legend(fontsize=8)
    ax[1].hist(mdf["peak_dd"], bins=60, color="tab:purple", alpha=0.8)
    ax[1].axvline(dd_med, color="black", ls="--", lw=1,
                  label=f"median {dd_med:.2f}%")
    ax[1].axvline(dd_p95, color="red", ls="--", lw=1,
                  label=f"p95 {dd_p95:.2f}%")
    ax[1].axvline(10.0, color="darkred", ls=":", lw=1.5, label="10% max-loss")
    ax[1].set_xlabel("peak drawdown of worst-path equity (%)")
    ax[1].set_ylabel("runs")
    ax[1].set_title(f"Monte Carlo peak drawdowns (n={N_MC}, Multiplier {MULT})")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(STUDY / "pairs_stress_distributions.png", dpi=130)
    plt.close(fig)

    # ---- Report -----------------------------------------------------------
    L = []
    L.append("REPORT_PAIRS_STRESS")
    L.append("Generated: " + dt.datetime.now().isoformat(timespec="seconds"))
    L.append("")
    L.append("PREMISE: ftmo_pairs_us500_usa30 (#24) showed the swap-free US500/USA30 pairs")
    L.append("counterfactual passes FTMO Phase 1 at Multiplier 5.0 from the March 2021 list")
    L.append("start (pass 2021-05-11, 42 trading days, peak daily 3.72%) - a single-path result")
    L.append("on 18 trades. This study stresses that pass at the same frozen economics: (A) rolling")
    L.append("fresh challenges on every trade entry date, and (B) 10,000 seeded shuffles of the")
    L.append("18-trade list. Methodological twin of ftmo_macro_reversal_stress (#26). NOT new")
    L.append("strategy research; the pairs #19 FAIL stands; not an FTMO qualification;")
    L.append("downstream_model_allowed=false, live_execution_allowed=false.")
    L.append("")
    L.append("ECONOMICS: 18 frozen pairs (2021-03-12..2026-06-23); swap-free realized = (gross_usd -")
    L.append(f"spread_cost_usd) x {MULT} (verified spreads 0.51/2.28 pt embedded; swap dropped); total")
    L.append(f"${sf_total:+.2f} at 1x -> ${sf_total * MULT:+.2f} at {MULT}x. US500 marks us500c_d1;")
    L.append("Dow marks = D1 aggregation of the #24 restamped Athens-wall M5 tape (common grid =")
    L.append("intersection). Worst floating = SIMULTANEOUS adverse D1 extremes of both legs (#24).")
    L.append("Daily anchor = prior-day realized balance (5.0% of anchor); static $9,000 floor; pass")
    L.append("judged on REALIZED balance >= $10,800 (tasking definition); same-day breach voids a")
    L.append(f"pass. Reconstruction audit: |grid-close gross - gross_usd| median ${recon_med:.2f}.")
    L.append("")
    L.append("TEST A - ROLLING CHALLENGE STARTS (fresh $10,000 per start date)")
    L.append(f"  start dates simulated: {n_start}")
    L.append(f"  Rolling Start Pass Rate: {rate_all:.2f}% of all starts "
             f"({n_pass}/{n_start}); {rate_res:.2f}% of resolved ({n_pass}/{resolved})")
    L.append(f"  Average Days to Pass: {avg_days:.1f} trading days (median {med_days:.0f})")
    L.append(f"  Failures: {n_fail} -> 5% Daily Loss {n_daily} "
             f"({n_fail and n_daily / n_fail * 100:.1f}%) | 10% Max Loss floor {n_floor} "
             f"({n_fail and n_floor / n_fail * 100:.1f}%)")
    L.append(f"  Unresolved (tape end before pass/breach): {n_unres}")
    if n_pass:
        L.append(f"  Peak drawdown of passing runs: median {pass_dd_med:.2f}%, "
                 f"p95 {pass_dd_p95:.2f}%")
    L.append("  Per-start outcomes:")
    for _, r in df.iterrows():
        L.append(f"    {r['start']}: {r['outcome']:<10} "
                 f"{('' if r['outcome'] != 'FAIL' else '(' + r['fail_type'] + ')'):<8} "
                 f"end {r['end']}  elapsed {r['elapsed']:>4} d"
                 + (f"  pass in {r['days']} d" if r["outcome"] == "PASS" else "")
                 + (f"  peakDD {r['peak_dd']:.2f}%" if r["outcome"] == "PASS" else ""))
    L.append("")
    L.append("TEST B - MONTE CARLO SEQUENCING (10,000 shuffles, seed 20260921)")
    L.append("  Sequencing: trades re-sequenced back-to-back on consecutive common-grid days,")
    L.append("  each carrying its original holding span and dual-leg daily adverse/close path")
    L.append("  (chronology destroyed, per-trade risk preserved - SPEC 3.2).")
    L.append(f"  Monte Carlo Pass Rate: {mc_rate_all:.2f}% of all runs ({mc_pass}/{N_MC}); "
             f"{mc_rate_res:.2f}% of resolved ({mc_pass}/{mc_res})")
    L.append(f"  Failures: {mc_fail} -> daily {mc_daily} | floor {mc_floor}; unresolved {mc_unres}")
    L.append(f"  Median Peak Drawdown: {dd_med:.2f}%   95th Percentile Peak Drawdown: {dd_p95:.2f}%")
    if mc_pass:
        L.append(f"  Median days-to-pass (passing runs): {mc_pass_med:.0f} trading days")
    L.append("")
    L.append("GATES")
    L.append(f"G-S1 DATA INTEGRITY: 18 trades={len(blocks) == DECL_N}, dates on common grid, "
             f"swap-free total ${sf_total:+.2f} (expect ${DECL_SF_TOTAL:+.2f}), spreads retained, "
             f"{n_pass}+{n_fail}+{n_unres}={n_start} starts -> {'PASS' if gs1 else 'FAIL'}")
    L.append(f"G-S2 ROLLING PASS RATE: resolved pass rate {rate_res:.2f}% >= 50% -> "
             f"{'PASS' if gs2 else 'FAIL'}")
    L.append(f"G-S3 MONTE-CARLO ROBUSTNESS: resolved pass rate {mc_rate_res:.2f}% >= 50% AND "
             f"p95 peak DD {dd_p95:.2f}% < 10.0% -> {'PASS' if gs3 else 'FAIL'}")
    L.append(f"OVERALL VERDICT: {verdict}"
             + (" - the Multiplier=5.0 pass is start-luck/sequencing fragile"
                if verdict == "FAIL" else " - the pass survives rolling starts and shuffles"))
    L.append("")
    L.append("CAVEATS: worst floating assumes SIMULTANEOUS adverse D1 extremes of both legs")
    L.append("(very pessimistic for a spread); the daily limit is 5% of the prior-day balance")
    L.append("anchor (FTMO's live rule is a fixed $500); pass requires REALIZED balance (floating")
    L.append("gains cannot trigger a pass); Monte Carlo blocks are a sequencing device, not a")
    L.append("calendar simulation; offline D1-close marks vs the #24 EA fills differed by median")
    L.append("-$7.02 per pair (overnight-gap execution + Dow spread); the Athens restamp carries")
    L.append("<= 1 h DST-rule skew vs the US500 server clock; unresolved runs are non-passes in")
    L.append("the primary rates.")
    (STUDY / "REPORT_PAIRS_STRESS.txt").write_text("\n".join(L), encoding="utf-8")

    meta = {"gs1": gs1, "gs2": gs2, "gs3": gs3, "verdict": verdict,
            "mult": MULT, "sf_total_usd_1x": sf_total, "recon_median_usd": recon_med,
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
