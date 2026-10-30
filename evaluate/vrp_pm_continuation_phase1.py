import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\Milad\Desktop\RR")
STUDY = ROOT / "data" / "results" / "vrp_pm_continuation"
M1_CSV = ROOT / "data" / "results" / "vwap_2021_2026" / "us500tick_m1.csv"
ARM_A_CSV = ROOT / "data" / "results" / "orb_vrp_filter" / "orb_vrp_filtered_trades.csv"

SQ252 = float(np.sqrt(252.0))
OR_LO, OR_HI = 780, 795
SIG_LO, SIG_HI = 796, 929
SESS_END_MOD = 959
COST_PT = 0.51
BOOT_ITERS = 10000
SEED = 20260924
G1_MIN = 50
G3_MAX_P = 0.20


def main():
    p = print
    STUDY.mkdir(parents=True, exist_ok=True)

    p("[1/5] Arm A dates + tape ...")
    arm = pd.read_csv(ARM_A_CSV)
    arm_dates = sorted(arm["ny_date"].unique())
    ny_dir = dict(zip(arm["ny_date"], arm["type"].map({"BUY": "LONG", "SELL": "SHORT"})))
    p(f"  Arm A dates: {len(arm_dates)} ({arm_dates[0]} .. {arm_dates[-1]})")

    df = pd.read_csv(M1_CSV)
    ts = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("America/New_York")
    ny = ts.dt.strftime("%Y-%m-%d").to_numpy()
    keep = np.isin(ny, np.array(arm_dates))
    df = df[keep].reset_index(drop=True)
    ny = ny[keep]
    mods = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()[keep]
    tstr = ts.dt.strftime("%Y-%m-%d %H:%M").to_numpy()[keep]
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    del df, ts

    p("[2/5] PM OR + signals + walks ...")
    funnel = {"n_arm_dates": len(arm_dates), "n_days_no_or": 0,
              "n_cand_long": 0, "n_cand_short": 0, "n_degenerate": 0, "n_no_entry": 0}
    rows = []
    change = np.flatnonzero(ny[1:] != ny[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(ny)]
    for s, e in zip(starts, ends):
        date = ny[s]
        mod = mods[s:e]
        orm = (mod >= OR_LO) & (mod < OR_HI)
        if not orm.any():
            funnel["n_days_no_or"] += 1
            continue
        orh = float(h[s:e][orm].max())
        orl = float(l[s:e][orm].min())
        win = (mod >= SIG_LO) & (mod <= SIG_HI)
        wpos = np.flatnonzero(win)
        last_sess = s + int(np.sum(mod < 960)) - 1
        for direction, cond, sl in ((1, c[s:e][win] > orh, orl), (-1, c[s:e][win] < orl, orh)):
            if not cond.any():
                continue
            funnel["n_cand_long" if direction == 1 else "n_cand_short"] += 1
            sig_g = s + int(wpos[int(np.argmax(cond))])
            j = sig_g + 1
            if j > last_sess:
                funnel["n_no_entry"] += 1
                continue
            entry = float(o[j])
            risk = (entry - sl) * direction
            if risk <= 0:
                funnel["n_degenerate"] += 1
                continue
            exit_idx, exit_price, exit_kind = None, None, None
            for k in range(j, last_sess + 1):
                if direction == 1:
                    if o[k] < sl:
                        exit_idx, exit_price, exit_kind = k, float(o[k]), "STOP_GAP"
                        break
                    if l[k] <= sl:
                        exit_idx, exit_price, exit_kind = k, float(sl), "STOP"
                        break
                else:
                    if o[k] > sl:
                        exit_idx, exit_price, exit_kind = k, float(o[k]), "STOP_GAP"
                        break
                    if h[k] >= sl:
                        exit_idx, exit_price, exit_kind = k, float(sl), "STOP"
                        break
            if exit_idx is None:
                exit_idx, exit_price, exit_kind = last_sess, float(c[last_sess]), "TIME"
            if direction == 1:
                mfe = (float(h[j:exit_idx + 1].max()) - entry)
                mae = (entry - float(l[j:exit_idx + 1].min()))
            else:
                mfe = (entry - float(l[j:exit_idx + 1].min()))
                mae = (float(h[j:exit_idx + 1].max()) - entry)
            gross = direction * (exit_price - entry)
            rows.append({
                "ny_date": date, "dir": "LONG" if direction == 1 else "SHORT",
                "signal_time": tstr[sig_g], "entry_time": tstr[j],
                "or_high": orh, "or_low": orl, "entry": entry, "sl": float(sl),
                "risk_pt": risk, "exit_kind": exit_kind, "exit_time": tstr[exit_idx],
                "exit": exit_price, "gross": gross, "net": gross - COST_PT,
                "mfe_pt": mfe, "mae_pt": mae,
                "walk_bars": int(exit_idx - j + 1),
                "ny_orb_dir": ny_dir.get(date, ""),
            })

    sig = pd.DataFrame(rows)
    n_tr = len(sig)
    p(f"  executed trades: {n_tr}")

    p("[3/5] aggregates ...")
    if n_tr:
        agg = {
            "n": n_tr,
            "net_mean": float(sig["net"].mean()), "net_median": float(sig["net"].median()),
            "net_total": float(sig["net"].sum()), "win": float((sig["net"] > 0).mean()),
            "mfe": float(sig["mfe_pt"].mean()), "mae": float(sig["mae_pt"].mean()),
            "sharpe": float(sig["net"].mean() / sig["net"].std(ddof=1) * SQ252),
        }
        by_dir = {d: {"n": len(gg), "win": float((gg["net"] > 0).mean()),
                      "net_mean": float(gg["net"].mean()),
                      "mfe": float(gg["mfe_pt"].mean()), "mae": float(gg["mae_pt"].mean())}
                  for d, gg in sig.groupby("dir")}
        exit_mix = {k: {"n": int(v), "share": float(v) / n_tr,
                        "net_mean": float(sig.loc[sig["exit_kind"] == k, "net"].mean())}
                    for k, v in sig["exit_kind"].value_counts().items()}
        by_year = sig.groupby(sig["ny_date"].str[:4]).size().to_dict()
        agree = sig["dir"] == sig["ny_orb_dir"]
        overlap = {
            "n_agree": int(agree.sum()), "n_disagree": int((~agree).sum()),
            "agree_rate": float(agree.mean()),
            "net_agree": float(sig.loc[agree, "net"].mean()),
            "net_disagree": float(sig.loc[~agree, "net"].mean()),
        }
    else:
        agg = {"n": 0}
        by_dir, exit_mix, by_year, overlap = {}, {}, {}, {}

    p("[4/5] day-block bootstrap (%d resamples) ..." % BOOT_ITERS)
    if n_tr:
        rng = np.random.default_rng(SEED)
        diff = sig["net"]
        sums = diff.groupby(sig["ny_date"]).sum().to_numpy(float)
        cnts = sig.groupby("ny_date").size().to_numpy(float)
        n_blocks = len(sums)
        idx = rng.integers(0, n_blocks, size=(BOOT_ITERS, n_blocks))
        means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
        pneg = float((means <= 0).mean())
    else:
        pneg, n_blocks = 1.0, 0
    p("  P(pooled mean Net <= 0) = %.4f over %d day blocks" % (pneg, n_blocks))

    p("[5/5] gates / report / csv ...")
    g1 = bool(n_tr >= G1_MIN)
    g2 = bool(agg.get("n", 0) > 0 and agg["sharpe"] > 0)
    g3 = bool(pneg <= G3_MAX_P)
    overall = bool(g1 and g2 and g3)
    verdict = "PASS" if overall else "FAIL"

    ts_now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = []
    A = L.append
    A("REPORT: VRP-CONDITIONED PM CONTINUATION BREAKOUT - PHASE 1 (HYPOTHESIS #44)")
    A(f"Generated: {ts_now}  |  script: evaluate/vrp_pm_continuation_phase1.py")
    A("SPEC/GATE: data/results/vrp_pm_continuation/{SPEC,GATE}_VRP_PM_CONT.txt (frozen before code)")
    A("")
    A(f"VERDICT: {verdict}   (G-PM1={g1}, G-PM2={g2}, G-PM3={g3})")
    A("")
    A("1. FUNNEL")
    A(f"   Arm A dates (from #31 artifact): {funnel['n_arm_dates']} ({arm_dates[0]} .. {arm_dates[-1]})")
    A(f"   days without [13:00,13:15) OR bars: {funnel['n_days_no_or']}")
    A(f"   candidates: LONG {funnel['n_cand_long']} / SHORT {funnel['n_cand_short']}")
    A(f"   dropped: degenerate {funnel['n_degenerate']}, no-entry {funnel['n_no_entry']}")
    A(f"   EXECUTED TRADES: {n_tr} (LONG {by_dir.get('LONG', {}).get('n', 0)} / SHORT {by_dir.get('SHORT', {}).get('n', 0)})")
    if by_year:
        A("   by year: " + ", ".join(f"{y}={m}" for y, m in sorted(by_year.items())))
    A("   exit mix: " + ", ".join(f"{k}={v['n']}" for k, v in sorted(exit_mix.items())))
    A("")
    A("2. NET ECONOMICS (points; net = gross - 0.51 spread floor)")
    if agg.get("n"):
        A(f"   n={agg['n']}  net/trade {agg['net_mean']:+.2f} (median {agg['net_median']:+.2f}, total {agg['net_total']:+.1f}, win {agg['win']*100:.1f}%)")
        A(f"   MFE {agg['mfe']:.2f} vs MAE {agg['mae']:.2f} pt  |  Net Sharpe (per-trade x sqrt252) {agg['sharpe']:.3f}")
        for d in sorted(by_dir):
            bb = by_dir[d]
            A(f"   {d}: n={bb['n']} win {bb['win']*100:.0f}% net {bb['net_mean']:+.2f}  MFE {bb['mfe']:.2f} vs MAE {bb['mae']:.2f}")
        for k in sorted(exit_mix):
            A(f"   exit {k}: {exit_mix[k]['n']} ({exit_mix[k]['share']*100:.1f}%) mean net {exit_mix[k]['net_mean']:+.2f}")
    else:
        A("   n=0 (gates not evaluable)")
    A("")
    A("3. OVERLAP WITH THE AM ORB (#31 Arm A same-date NY-cash trades)")
    if overlap:
        A(f"   direction agreement with same-date AM ORB trade: {overlap['n_agree']}/{n_tr} ({overlap['agree_rate']*100:.1f}%)")
        A(f"   mean net | agree {overlap['net_agree']:+.2f} vs disagree {overlap['net_disagree']:+.2f} pt")
    A("")
    A("4. GATES (frozen verbatim in GATE file)")
    A(f"   G-PM1 >= {G1_MIN} executed: {n_tr} -> {g1}")
    A(f"   G-PM2 Net Sharpe > 0: {agg.get('sharpe', float('nan')):.3f} -> {g2}")
    A(f"   G-PM3 bootstrap P(mean Net <= 0) <= {G3_MAX_P:.2f}: {pneg:.4f} (n_day_blocks={n_blocks}) -> {g3}")
    A(f"   OVERALL PHASE-1: {verdict}")
    A("")
    A("5. NOTES AND CAVEATS")
    A("   - Arm A date set is in-sample conditioning realized through #31 trade dates; all metrics")
    A("     inherit that optimism; #31's overall FAIL (G-VRP3) stands, not re-litigated.")
    A("   - Net = gross - 0.51 pt verified spread floor; no swap (intraday).")
    A("   - MFE/MAE include the exit bar's full M1 bar; gap-through stops fill at the open.")
    A("   - Frozen archetype parameters (PM OR window, signal window, stop rule); no search.")
    A("   - Null result is final for this artifact.")
    report = "\n".join(L) + "\n"
    (STUDY / "REPORT_VRP_PM_CONT_PHASE1.txt").write_text(report, encoding="utf-8")
    if n_tr:
        sig.to_csv(STUDY / "vrp_pm_signals.csv", index=False)
    meta = {
        "generated": ts_now,
        "script": "vrp_pm_continuation_phase1.py",
        "funnel": funnel,
        "aggregate": agg,
        "by_dir": by_dir,
        "exit_mix": exit_mix,
        "by_year": by_year,
        "overlap": overlap,
        "pneg": pneg, "n_day_blocks": n_blocks,
        "gates": {"G-PM1": g1, "G-PM2": g2, "G-PM3": g3},
        "overall": overall, "verdict": verdict,
        "params": {"or_window_et": "[13:00,13:15)", "signal_window_et": "(13:15,15:30) strict",
                   "stop": "opposite OR boundary", "time_exit": "16:00 ET cash close",
                   "cost_pt": COST_PT, "boot_iters": BOOT_ITERS, "seed": SEED},
    }
    (STUDY / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    p(f"VERDICT: {verdict}  (G-PM1={g1} n={n_tr}, G-PM2={g2}, G-PM3={g3} p={pneg:.4f})")
    if agg.get("n"):
        p(f"net {agg['net_mean']:+.2f} pt/trade, win {agg['win']*100:.1f}%, sharpe {agg['sharpe']:.3f} | agree {overlap['agree_rate']*100:.1f}%")
    p(f"funnel: {funnel}")


if __name__ == "__main__":
    main()
