import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\Milad\Desktop\RR")
STUDY = ROOT / "data" / "results" / "asian_range_breakout"
M1_CSV = ROOT / "data" / "results" / "eurusd_onboarding" / "eurusdtick_m1.csv"

PIP = 10000.0
ASIAN_END = 420
BRK_START = 420
BRK_END = 600
EXIT_MOD = 1020
MAX_RANGE_PIPS = 30.0
COST_FLOOR_PIPS = 0.3
BOOT_ITERS = 10000
SEED = 20260923
G1_MIN = 150
G3_MAX_P = 0.20


def main():
    p = print
    STUDY.mkdir(parents=True, exist_ok=True)

    p("[1/6] loading tape ...")
    df = pd.read_csv(M1_CSV)
    ts = pd.to_datetime(df["time"], unit="s", utc=True)
    assert ts.is_monotonic_increasing, "tape not monotonic"
    dates = ts.dt.strftime("%Y-%m-%d").to_numpy()
    mods = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
    tstr = ts.dt.strftime("%Y-%m-%d %H:%M").to_numpy()
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    del df, ts

    change = np.flatnonzero(dates[1:] != dates[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(dates)]

    funnel = {
        "n_dates_total": len(starts),
        "n_morning_dates": 0, "n_no_asian_bars": 0, "n_range_blocked": 0,
        "n_range_valid": 0, "n_no_brk_bars": 0,
        "n_cand_long": 0, "n_cand_short": 0,
        "n_no_entry": 0, "n_degenerate": 0,
    }
    range_pips_valid = []
    rows = []

    p("[2/6] Asian ranges + tight-range filter ...")
    for s, e in zip(starts, ends):
        mod = mods[s:e]
        if not (mod < EXIT_MOD).any():
            continue
        funnel["n_morning_dates"] += 1
        asian = mod < ASIAN_END
        if not asian.any():
            funnel["n_no_asian_bars"] += 1
            continue
        ah = float(h[s:e][asian].max())
        al = float(l[s:e][asian].min())
        rng_pt = round((ah - al) * PIP * 10.0)
        if rng_pt > 300:
            funnel["n_range_blocked"] += 1
            continue
        funnel["n_range_valid"] += 1
        range_pips_valid.append(rng_pt / 10.0)
        brk = (mod >= BRK_START) & (mod < BRK_END)
        if not brk.any():
            funnel["n_no_brk_bars"] += 1
            continue
        close_b = c[s:e][brk]
        up = close_b > ah
        dn = close_b < al
        first_exit = s + int(np.argmax(mod >= EXIT_MOD)) if (mod >= EXIT_MOD).any() else -1
        w_end = (first_exit - 1) if first_exit != -1 else (e - 1)
        for direction, hit, level, sl in ((1, up, ah, al), (-1, dn, al, ah)):
            if not hit.any():
                continue
            funnel["n_cand_long" if direction == 1 else "n_cand_short"] += 1
            sig_local = int(np.argmax(hit))
            sig_g = s + int(np.argmax(brk)) + sig_local
            j = sig_g + 1
            if j > w_end:
                funnel["n_no_entry"] += 1
                continue
            entry = float(o[j])
            risk_pips = (entry - sl) * PIP * direction
            if risk_pips <= 0:
                funnel["n_degenerate"] += 1
                continue
            hs = h[j:w_end + 1]
            ls = l[j:w_end + 1]
            if direction == 1:
                mfe = (float(hs.max()) - entry) * PIP
                mae = (entry - float(ls.min())) * PIP
            else:
                mfe = (entry - float(ls.min())) * PIP
                mae = (float(hs.max()) - entry) * PIP
            if first_exit != -1:
                exit_price = float(o[first_exit])
                exit_time = tstr[first_exit]
                exit_kind = "TIME"
            else:
                exit_price = float(c[w_end])
                exit_time = tstr[w_end]
                exit_kind = "TIME_FALLBACK"
            raw_pnl = direction * (exit_price - entry) * PIP
            rows.append({
                "utc_date": dates[s], "dir": "LONG" if direction == 1 else "SHORT",
                "signal_time": tstr[sig_g], "entry_time": tstr[j],
                "entry": entry, "sl": float(sl), "asian_high": ah, "asian_low": al,
                "asian_range_pips": rng_pt / 10.0, "risk_pips": risk_pips,
                "mfe_pips": mfe, "mae_pips": mae,
                "mfe_r": mfe / risk_pips, "mae_r": mae / risk_pips,
                "exit_kind": exit_kind, "exit_time": exit_time,
                "exit_price": exit_price, "raw_pnl_pips": raw_pnl,
            })

    sig = pd.DataFrame(rows)
    n_sig = len(sig)
    p(f"  final signals: {n_sig}")

    p("[3/6] aggregates ...")
    if n_sig:
        agg = {
            "n": n_sig,
            "mfe": float(sig["mfe_pips"].mean()), "mae": float(sig["mae_pips"].mean()),
            "diff": float((sig["mfe_pips"] - sig["mae_pips"]).mean()),
            "mfe_med": float(sig["mfe_pips"].median()), "mae_med": float(sig["mae_pips"].median()),
            "mfe_gt_mae_share": float((sig["mfe_pips"] > sig["mae_pips"]).mean()),
            "mfe_r": float(sig["mfe_r"].mean()), "mae_r": float(sig["mae_r"].mean()),
            "raw_pnl": float(sig["raw_pnl_pips"].mean()),
            "raw_pnl_med": float(sig["raw_pnl_pips"].median()),
            "raw_pnl_win": float((sig["raw_pnl_pips"] > 0).mean()),
        }
        by_dir = {}
        for d, g in sig.groupby("dir"):
            by_dir[d] = {
                "n": len(g), "mfe": float(g["mfe_pips"].mean()), "mae": float(g["mae_pips"].mean()),
                "diff": float((g["mfe_pips"] - g["mae_pips"]).mean()),
                "raw_pnl": float(g["raw_pnl_pips"].mean()),
            }
        by_year = sig.groupby(sig["utc_date"].str[:4]).size().to_dict()
        rng_stats = {
            "n_valid_days": len(range_pips_valid),
            "mean_range_pips": float(np.mean(range_pips_valid)),
            "median_range_pips": float(np.median(range_pips_valid)),
        }
    else:
        agg = {"n": 0}
        by_dir, by_year, rng_stats = {}, {}, {"n_valid_days": len(range_pips_valid)}

    p("[4/6] day-block bootstrap (%d resamples) ..." % BOOT_ITERS)
    if n_sig:
        rng = np.random.default_rng(SEED)
        diff = (sig["mfe_pips"] - sig["mae_pips"])
        sums = diff.groupby(sig["utc_date"]).sum().to_numpy(float)
        cnts = sig.groupby("utc_date").size().to_numpy(float)
        n_blocks = len(sums)
        idx = rng.integers(0, n_blocks, size=(BOOT_ITERS, n_blocks))
        means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
        pneg = float((means <= 0).mean())
    else:
        pneg, n_blocks = 1.0, 0
    p("  P(pooled mean(MFE-MAE) <= 0) = %.4f over %d day blocks" % (pneg, n_blocks))

    p("[5/6] gates ...")
    g1 = bool(n_sig >= G1_MIN)
    g2 = bool(agg.get("n", 0) > 0 and agg["mfe"] > agg["mae"])
    g3 = bool(pneg <= G3_MAX_P)
    overall = bool(g1 and g2 and g3)
    verdict = "PASS" if overall else "FAIL"

    p("[6/6] report / csv / meta ...")
    ts_now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = []
    A = L.append
    A("REPORT: ASIAN RANGE LONDON BREAKOUT - PHASE 1 (HYPOTHESIS #38)")
    A(f"Generated: {ts_now}  |  script: evaluate/asian_range_breakout_eurusd_phase1.py")
    A("SPEC/GATE: data/results/asian_range_breakout/{SPEC,GATE}_ASIAN_BREAKOUT.txt (frozen before code)")
    A("")
    A(f"VERDICT: {verdict}   (G-AS1={g1}, G-AS2={g2}, G-AS3={g3})")
    A("")
    A("1. FUNNEL (UTC dates, tape eurusdtick_m1.csv)")
    A(f"   dates in tape: {funnel['n_dates_total']}")
    A(f"   morning-active dates (>=1 bar before 17:00 UTC): {funnel['n_morning_dates']}")
    A(f"   no Asian-window bars: {funnel['n_no_asian_bars']}")
    A(f"   Asian range > 30 pips (blocked): {funnel['n_range_blocked']}")
    A(f"   tight-range valid days: {funnel['n_range_valid']}" + (f"  (mean {rng_stats.get('mean_range_pips', float('nan')):.1f} / median {rng_stats.get('median_range_pips', float('nan')):.1f} pips)" if rng_stats.get("n_valid_days") else ""))
    A(f"   valid days with no breakout-window bars: {funnel['n_no_brk_bars']}")
    A(f"   candidates: LONG {funnel['n_cand_long']} / SHORT {funnel['n_cand_short']}")
    A(f"   dropped no-entry: {funnel['n_no_entry']}  degenerate-risk: {funnel['n_degenerate']}")
    A(f"   FINAL SIGNALS: {n_sig}  (LONG {by_dir.get('LONG', {}).get('n', 0)} / SHORT {by_dir.get('SHORT', {}).get('n', 0)})")
    if by_year:
        A("   by year: " + ", ".join(f"{y}={n}" for y, n in sorted(by_year.items())))
    A("")
    A("2. AGGREGATE EXCURSIONS (pips, pre-cost)")
    if agg.get("n"):
        A(f"   n={agg['n']}  MFE {agg['mfe']:.2f} vs MAE {agg['mae']:.2f}  diff {agg['diff']:+.2f}")
        A(f"   medians: MFE {agg['mfe_med']:.2f} vs MAE {agg['mae_med']:.2f}  MFE>MAE share {agg['mfe_gt_mae_share']*100:.1f}%")
        A(f"   risk multiples: MFE/R {agg['mfe_r']:.3f} vs MAE/R {agg['mae_r']:.3f}")
        for d in ("LONG", "SHORT"):
            if d in by_dir:
                b = by_dir[d]
                A(f"   {d}: n={b['n']}  MFE {b['mfe']:.2f} vs MAE {b['mae']:.2f}  diff {b['diff']:+.2f}")
    else:
        A("   n=0 (no gates evaluable)")
    A("")
    A("3. RAW TIME-EXIT EXPECTANCY (descriptive, pre-cost)")
    if agg.get("n"):
        A(f"   mean raw PnL {agg['raw_pnl']:+.3f} pips (median {agg['raw_pnl_med']:+.3f}, win {agg['raw_pnl_win']*100:.1f}%)")
        A(f"   vs verified London/NY overlap cost floor {COST_FLOOR_PIPS:.1f} pips -> mean {'clears' if agg['raw_pnl'] > COST_FLOOR_PIPS else 'does NOT clear'} the floor")
        A(f"   by direction: " + ", ".join(f"{d} {by_dir[d]['raw_pnl']:+.3f}" for d in by_dir))
    A("")
    A("4. GATES (frozen verbatim in GATE file)")
    A(f"   G-AS1 >= {G1_MIN} signals: {n_sig} -> {g1}")
    A(f"   G-AS2 mean MFE > mean MAE (strict): {agg.get('mfe', float('nan')):.3f} vs {agg.get('mae', float('nan')):.3f} -> {g2}")
    A(f"   G-AS3 bootstrap P(mean(MFE-MAE)<=0) <= {G3_MAX_P:.2f}: {pneg:.4f} (n_day_blocks={n_blocks}) -> {g3}")
    A(f"   OVERALL PHASE-1: {verdict}")
    A("")
    A("5. NOTES AND CAVEATS")
    A("   - Pre-cost Bid-tape screen; 0.3-pip overlap median floor reported alongside, never netted in gates.")
    A("   - Tight-range filter (>30 pips blocked) is the frozen archetype condition; no range-grid search.")
    A("   - First close-breakout per direction per day; dropped candidates (no-entry/degenerate) do not re-arm.")
    A("   - Null result is final for this artifact.")
    report = "\n".join(L) + "\n"
    (STUDY / "REPORT_ASIAN_BREAKOUT_PHASE1.txt").write_text(report, encoding="utf-8")
    if n_sig:
        sig.to_csv(STUDY / "asian_breakout_signals.csv", index=False)
    meta = {
        "generated": ts_now,
        "script": "asian_range_breakout_eurusd_phase1.py",
        "funnel": funnel,
        "range_stats": rng_stats,
        "aggregate": agg,
        "by_dir": by_dir,
        "by_year": by_year,
        "pneg": pneg, "n_day_blocks": n_blocks,
        "gates": {"G-AS1": g1, "G-AS2": g2, "G-AS3": g3},
        "overall": overall, "verdict": verdict,
        "params": {"asian_window": "[00:00,07:00) UTC", "breakout_window": "[07:00,10:00) UTC",
                   "time_exit": "17:00 UTC", "max_range_pips": MAX_RANGE_PIPS,
                   "pip": 0.0001, "boot_iters": BOOT_ITERS, "seed": SEED},
    }
    (STUDY / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    p(f"VERDICT: {verdict}  (G-AS1={g1} n={n_sig}, G-AS2={g2}, G-AS3={g3} p={pneg:.4f})")
    if agg.get("n"):
        p(f"MFE {agg['mfe']:.2f} vs MAE {agg['mae']:.2f} (diff {agg['diff']:+.2f} pips) | raw time-exit {agg['raw_pnl']:+.3f} pips vs 0.3 floor")
    p(f"funnel: {funnel}")


if __name__ == "__main__":
    main()
