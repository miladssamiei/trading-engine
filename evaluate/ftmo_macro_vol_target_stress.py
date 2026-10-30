import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\Milad\Desktop\RR")
STUDY = ROOT / "data" / "results" / "ftmo_macro_vol_target"
G2 = ROOT / "data" / "results" / "short_term_reversal" / "g2_trades.csv"
D1 = ROOT / "data" / "results" / "momentum_multihorizon" / "us500c_d1.csv"
BASE26_META = ROOT / "data" / "results" / "ftmo_macro_reversal_stress" / "run_meta.json"

INITIAL = 10000.0
TARGET = 10800.0
FLOOR = 9000.0
DAILY_PCT = 5.0
SPREAD_PT = 0.51
PT_USD = 1.0
BASE_LOTS = 1.0
LOT_MIN = 0.5
LOT_MAX = 15.0
TV_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
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

    lr = np.diff(np.log(close))
    rv20 = np.full(len(close), np.nan)
    v = pd.Series(lr).rolling(20).var(ddof=0).to_numpy()
    rv20[20:] = np.sqrt(252.0 * v[19:])
    rv_entry = rv20[np.array(e)]
    assert not np.isnan(rv_entry).any(), "RV20 NaN at a trade entry day"
    return dates, close, high, low, e, x, sig, gross, sf_total, rv_entry


def build_blocks(e, x, sig, gross, close, high, low, lots):
    """Per-trade blocks in USD at the trade's own DynamicLots (SPEC 1.4/1.5):
    worst[rel-1] for rel=1..span (adverse D1 extreme vs entry close),
    close[rel-1] for rel=1..span-1 (floating close mark), realized USD."""
    blocks = []
    for i in range(len(e)):
        span = x[i] - e[i]
        assert span >= 1, f"trade {i} span {span}"
        entry_px = close[e[i]]
        days = np.arange(e[i] + 1, x[i] + 1)
        ext = np.where(sig[i] > 0, low[days], high[days])
        worst = sig[i] * (ext - entry_px) * PT_USD * lots[i]
        cl = sig[i] * (close[e[i] + 1:x[i]] - entry_px) * PT_USD * lots[i]
        blocks.append({"span": span, "worst": worst, "close": cl,
                       "realized": float(gross[i] - SPREAD_PT) * PT_USD * lots[i],
                       "entry_day": e[i]})
    return blocks


def run_challenge(blocks, schedule, first_day, n_days):
    """FTMO Phase-1 sim per SPEC 1.5 (#26 conventions; pass on D1-close
    equity). Returns (outcome, fail_type, end_day, days_elapsed, peak_dd_pct,
    peak_daily_pct)."""
    k = 0
    cum = 0.0
    in_trade = False
    b = None
    rel = 0
    peak = -1e18
    max_dd = 0.0
    max_daily = 0.0
    for d in range(first_day, n_days):
        anchor = INITIAL + cum
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
            rel += 1
            worst_eq = INITIAL + cum + b["worst"][rel - 1]
            if rel == b["span"]:
                eq_close = INITIAL + cum + b["realized"]
                cum += b["realized"]
                in_trade = False
                while k < len(schedule) and schedule[k][0] < d:
                    k += 1
                if k < len(schedule) and schedule[k][0] == d:
                    b = blocks[schedule[k][1]]
                    rel = 0
                    in_trade = True
                    k += 1
            else:
                eq_close = INITIAL + cum + b["close"][rel - 1]

        if worst_eq <= FLOOR + 1e-9:
            return "FAIL", "floor", d, d - first_day + 1, max_dd, max_daily
        dl = (anchor - worst_eq) / anchor * 100.0 if anchor > 0.0 else 0.0
        if dl > max_daily:
            max_daily = dl
        if dl >= DAILY_PCT - 1e-9:
            return "FAIL", "daily", d, d - first_day + 1, max_dd, max_daily
        if worst_eq > peak:
            peak = worst_eq
        dd = (peak - worst_eq) / peak * 100.0
        if dd > max_dd:
            max_dd = dd
        if eq_close >= TARGET - 1e-9:
            return "PASS", "", d, d - first_day + 1, max_dd, max_daily
    return "UNRESOLVED", "", n_days - 1, n_days - first_day, max_dd, max_daily


def main():
    if not (STUDY / "SPEC_VOL_TARGET_STRESS.txt").exists():
        raise SystemExit("SPEC missing; refusing to run")
    if not (STUDY / "GATE_VOL_TARGET_STRESS.txt").exists():
        raise SystemExit("GATE missing; refusing to run")

    dates, close, high, low, e, x, sig, gross, sf_total, rv_entry = load()
    n_days = len(dates)
    rv_dec = rv_entry * 100.0
    print(f"loaded: {DECL_N} trades, D1 grid {n_days} days, swap-free total "
          f"{sf_total:+.2f} pts; RV20 at entries: min {rv_dec.min():.1f}% "
          f"median {np.median(rv_dec):.1f}% max {rv_dec.max():.1f}%")

    # ---- TargetVol sweep on the base path (#26 baseline start) -----------
    sweep = []
    blocks_by_tv = {}
    for tv in TV_GRID:
        lots = np.clip(BASE_LOTS * tv / rv_entry, LOT_MIN, LOT_MAX)
        assert (lots >= LOT_MIN - 1e-12).all() and (lots <= LOT_MAX + 1e-12).all()
        blocks = build_blocks(e, x, sig, gross, close, high, low, lots)
        blocks_by_tv[tv] = (blocks, lots)
        schedule = [(e[i], i) for i in range(len(e))]
        out, ftype, day, elapsed, dd, dl = run_challenge(blocks, schedule, e[0], n_days)
        row = {"tv": tv, "outcome": out, "end": dates[day],
               "days": elapsed if out == "PASS" else None,
               "pass_day_idx": day if out == "PASS" else 10**9,
               "peak_dd": dd, "peak_daily": dl,
               "lots_mean": float(lots.mean()), "lots_min": float(lots.min()),
               "lots_max": float(lots.max()),
               "frac_floor": float((lots <= LOT_MIN + 1e-12).mean()),
               "frac_cap": float((lots >= LOT_MAX - 1e-12).mean())}
        sweep.append(row)
        print(f"TV {tv * 100:.0f}%: {out} end {row['end']} "
              f"({'days ' + str(row['days']) if row['days'] else ftype}), "
              f"peakDD {dd:.2f}% peakDaily {dl:.2f}%, lots mean {row['lots_mean']:.2f} "
              f"[{row['lots_min']:.2f}..{row['lots_max']:.2f}] "
              f"floor {row['frac_floor'] * 100:.0f}% cap {row['frac_cap'] * 100:.0f}%")
    passers = [r for r in sweep if r["outcome"] == "PASS"]
    optimal = min(passers, key=lambda r: (r["pass_day_idx"], r["tv"])) if passers else None
    if optimal is None:
        raise SystemExit("no TargetVol passes the base path; G-V2/G-V3 not run")

    tv = optimal["tv"]
    blocks, lots = blocks_by_tv[tv]

    # ---- Stress Test A: rolling starts -----------------------------------
    start_dates = sorted({dates[e[i]] for i in range(len(e))
                          if dates[e[i]] <= "2025-12-31"})
    roll = []
    for s in start_dates:
        s_idx = dates.index(s)
        schedule = [(e[i], i) for i in range(len(e)) if e[i] >= s_idx]
        out, ftype, day, elapsed, dd, dl = run_challenge(blocks, schedule, s_idx, n_days)
        roll.append({"start": s, "outcome": out, "fail_type": ftype,
                     "end": dates[day], "days": elapsed if out == "PASS" else None,
                     "elapsed": elapsed, "peak_dd": dd, "year": s[:4]})
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
    print(f"rolling @TV {tv * 100:.0f}%: {n_start} starts, pass {n_pass} "
          f"({rate_all:.2f}% all / {rate_res:.2f}% resolved), fail {n_fail} "
          f"(daily {n_daily} / floor {n_floor}), unresolved {n_unres}, "
          f"avg days {avg_days:.1f}")

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
        out, ftype, day, elapsed, dd, dl = run_challenge(blocks, schedule, 0, cursor)
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
    print(f"monte carlo @TV {tv * 100:.0f}%: pass {mc_pass} ({mc_rate_all:.2f}% all / "
          f"{mc_rate_res:.2f}% resolved), fail {mc_fail} (daily {mc_daily} / floor "
          f"{mc_floor}), unresolved {mc_unres}, peak DD median {dd_med:.2f}% "
          f"p95 {dd_p95:.2f}%")

    # ---- Gates ------------------------------------------------------------
    gv1 = bool(len(blocks) == DECL_N and abs(sf_total - DECL_SF_TOTAL) < 0.01
               and not np.isnan(rv_entry).any()
               and (lots >= LOT_MIN - 1e-12).all() and (lots <= LOT_MAX + 1e-12).all()
               and n_pass + n_fail + n_unres == n_start)
    gv2 = bool(resolved > 0 and rate_res >= 50.0)
    gv3 = bool(mc_res > 0 and mc_rate_res >= 50.0 and dd_p95 < 10.0)
    verdict = "PASS" if (gv1 and gv2 and gv3) else "FAIL"

    b26 = json.loads(BASE26_META.read_text()) if BASE26_META.exists() else {}

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
    ax[0].set_title(f"Days to $10,800 (passing runs, TargetVol {tv * 100:.0f}%)")
    ax[0].legend(fontsize=8)
    ax[1].hist(mdf["peak_dd"], bins=60, color="tab:purple", alpha=0.8)
    ax[1].axvline(dd_med, color="black", ls="--", lw=1,
                  label=f"median {dd_med:.2f}%")
    ax[1].axvline(dd_p95, color="red", ls="--", lw=1,
                  label=f"p95 {dd_p95:.2f}%")
    ax[1].axvline(10.0, color="darkred", ls=":", lw=1.5, label="10% max-loss")
    ax[1].set_xlabel("peak drawdown of worst-path equity (%)")
    ax[1].set_ylabel("runs")
    ax[1].set_title(f"Monte Carlo peak drawdowns (n={N_MC}, TargetVol {tv * 100:.0f}%)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(STUDY / "vol_target_stress_distributions.png", dpi=130)
    plt.close(fig)

    # ---- Report -----------------------------------------------------------
    L = []
    L.append("REPORT_VOL_TARGET_STRESS")
    L.append("Generated: " + dt.datetime.now().isoformat(timespec="seconds"))
    L.append("")
    L.append("PREMISE: #26 stressed the #23 swap-free macro short-term reversal replay at FIXED")
    L.append("3.0 lots: rolling pass 53.60%, MC pass 47.31%, p95 peak DD 10.62% - fragile. This")
    L.append("study tests the classic VOLATILITY-TARGETING sizing overlay on the SAME frozen")
    L.append("458-trade list (no signal change): DynamicLots = clip(1.0 x TargetVol / RV20,")
    L.append("0.5, 15.0), RV20 = annualized stdev of the 20 daily log returns ending at the")
    L.append("entry day (ddof=0, observable at the entry close), lots frozen per trade at their")
    L.append("historical entry-day value. NOT new signal research; the short_term_reversal")
    L.append("G-R2 FAIL stands; not an FTMO qualification; flags false.")
    L.append("")
    L.append(f"ECONOMICS: 458 trades; swap-free (gross_pt - 0.51) x $1/pt/lot x DynamicLots;")
    L.append(f"swap-free point total {sf_total:+.2f} pts. RV20 at trade entries: min "
             f"{rv_dec.min():.1f}% / median {np.median(rv_dec):.1f}% / max {rv_dec.max():.1f}%.")
    L.append("FTMO rules and pessimistic conventions identical to #26 (adverse D1 extremes,")
    L.append("prior-day balance anchor at 5.0%, static $9,000 floor, pass on D1-close equity,")
    L.append("same-day breach voids).")
    L.append("")
    L.append("TARGETVOL SWEEP (base path = challenge starting at the first trade, 2021-01-04)")
    L.append(f"{'TV%':>4} {'outcome':>10} {'end':>11} {'days':>5} {'peakDD%':>8} "
             f"{'peakDly%':>9} {'lots_mean':>9} {'lots_min':>8} {'lots_max':>8} "
             f"{'@floor':>7} {'@cap':>6}")
    for r in sweep:
        L.append(f"{r['tv'] * 100:>4.0f} {r['outcome']:>10} {str(r['days'] or r['end']):>11} "
                 f"{str(r['days'] or '-'):>5} {r['peak_dd']:>8.2f} {r['peak_daily']:>9.2f} "
                 f"{r['lots_mean']:>9.2f} {r['lots_min']:>8.2f} {r['lots_max']:>8.2f} "
                 f"{r['frac_floor'] * 100:>6.1f}% {r['frac_cap'] * 100:>5.1f}%")
    L.append(f"OPTIMAL TargetVol = {tv * 100:.0f}% (earliest base-path pass "
             f"{optimal['end']} in {optimal['days']} trading days; ties toward smaller TV)")
    L.append("")
    L.append(f"TEST A - ROLLING CHALLENGE STARTS at TargetVol {tv * 100:.0f}%")
    L.append(f"  start dates simulated: {n_start}")
    L.append(f"  Rolling Start Pass Rate: {rate_all:.2f}% of all starts "
             f"({n_pass}/{n_start}); {rate_res:.2f}% of resolved ({n_pass}/{resolved})")
    L.append(f"  Average Days to Pass: {avg_days:.1f} trading days (median {med_days:.0f})")
    L.append(f"  Failures: {n_fail} -> 5% Daily Loss {n_daily} "
             f"({n_fail and n_daily / n_fail * 100:.1f}%) | 10% Max Loss floor {n_floor} "
             f"({n_fail and n_floor / n_fail * 100:.1f}%)")
    L.append(f"  Unresolved: {n_unres} ({n_unres / n_start * 100:.1f}%)")
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
    L.append(f"TEST B - MONTE CARLO SEQUENCING at TargetVol {tv * 100:.0f}% "
             f"(10,000 shuffles, seed {SEED})")
    L.append(f"  Monte Carlo Pass Rate: {mc_rate_all:.2f}% of all runs ({mc_pass}/{N_MC}); "
             f"{mc_rate_res:.2f}% of resolved ({mc_pass}/{mc_res})")
    L.append(f"  Failures: {mc_fail} -> daily {mc_daily} | floor {mc_floor}; "
             f"unresolved {mc_unres}")
    L.append(f"  Median Peak Drawdown: {dd_med:.2f}%   95th Percentile Peak Drawdown: "
             f"{dd_p95:.2f}%")
    if mc_pass:
        L.append(f"  Median days-to-pass (passing runs): {mc_pass_med:.0f} trading days")
    L.append("")
    if b26:
        L.append("COMPARISON vs #26 (FixedLots 3.0, frozen run_meta.json, identical "
                 "conventions and seed):")
        L.append(f"  {'metric':<34}{'#26 fixed 3.0':>15}{'#28 vol-target':>15}")
        L.append(f"  {'rolling pass rate (resolved)':<34}"
                 f"{b26['rolling']['rate_resolved_pct']:>14.2f}%{rate_res:>14.2f}%")
        L.append(f"  {'rolling avg days to pass':<34}"
                 f"{b26['rolling']['avg_days_to_pass']:>15.1f}{avg_days:>15.1f}")
        L.append(f"  {'MC pass rate (resolved)':<34}"
                 f"{b26['mc']['rate_resolved_pct']:>14.2f}%{mc_rate_res:>14.2f}%")
        L.append(f"  {'MC p95 peak drawdown':<34}"
                 f"{b26['mc']['dd_p95_pct']:>14.2f}%{dd_p95:>14.2f}%")
        L.append(f"  {'MC median peak drawdown':<34}"
                 f"{b26['mc']['dd_median_pct']:>14.2f}%{dd_med:>14.2f}%")
        L.append("")
    L.append("GATES")
    L.append(f"G-V1 DATA/VOL INTEGRITY: 458 trades, dates aligned, swap-free total "
             f"{sf_total:+.2f}, RV20 at all entries (min {rv_dec.min():.1f}%), lots in "
             f"[{lots.min():.2f}, {lots.max():.2f}] within [0.5, 15.0], coverage "
             f"{n_pass}+{n_fail}+{n_unres}={n_start} -> {'PASS' if gv1 else 'FAIL'}")
    L.append(f"G-V2 ROLLING PASS RATE: resolved pass rate {rate_res:.2f}% >= 50% -> "
             f"{'PASS' if gv2 else 'FAIL'}")
    L.append(f"G-V3 MONTE-CARLO ROBUSTNESS: resolved pass rate {mc_rate_res:.2f}% >= 50% "
             f"AND p95 peak DD {dd_p95:.2f}% < 10.0% -> {'PASS' if gv3 else 'FAIL'}")
    L.append(f"OVERALL VERDICT: {verdict}"
             + (" - volatility targeting does not make the replay robust"
                if verdict == "FAIL" else
                " - the sizing overlay survived this stress at the tested targets "
                "(no strategy validation)"))
    L.append("")
    L.append("CAVEATS: RV20 is close-to-close (no intraday vol); lots clamped hard at")
    L.append("0.5/15.0; worst floating via D1 adverse extremes (pessimistic); daily limit")
    L.append("is 5% of the prior-day balance anchor vs FTMO's fixed $500; MC blocks are a")
    L.append("sequencing device, not a calendar simulation; sizing overlays multiply")
    L.append("frozen trade PnL and cannot repair a negative-expectancy or fragile signal -")
    L.append("the #26/#28 family measures FTMO pass robustness only; unresolved runs are")
    L.append("non-passes in the primary rates.")
    (STUDY / "REPORT_VOL_TARGET_STRESS.txt").write_text("\n".join(L), encoding="utf-8")

    meta = {"gv1": gv1, "gv2": gv2, "gv3": gv3, "verdict": verdict,
            "optimal_tv": tv, "sweep": sweep,
            "sf_total_pts": sf_total,
            "rv_entry_pct": {"min": float(rv_dec.min()),
                             "median": float(np.median(rv_dec)),
                             "max": float(rv_dec.max())},
            "rolling": {"starts": n_start, "pass": n_pass, "fail": n_fail,
                        "daily": n_daily, "floor": n_floor, "unresolved": n_unres,
                        "rate_all_pct": rate_all, "rate_resolved_pct": rate_res,
                        "avg_days_to_pass": avg_days, "median_days_to_pass": med_days,
                        "pass_dd_median_pct": pass_dd_med,
                        "pass_dd_p95_pct": pass_dd_p95},
            "mc": {"runs": N_MC, "seed": SEED, "pass": mc_pass, "fail": mc_fail,
                   "daily": mc_daily, "floor": mc_floor, "unresolved": mc_unres,
                   "rate_all_pct": mc_rate_all, "rate_resolved_pct": mc_rate_res,
                   "dd_median_pct": dd_med, "dd_p95_pct": dd_p95,
                   "pass_days_median": mc_pass_med},
            "baseline_26": b26}
    (STUDY / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
