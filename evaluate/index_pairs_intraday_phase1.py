import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\Milad\Desktop\RR")
STUDY = ROOT / "data" / "results" / "index_pairs_intraday"
US500_M1 = ROOT / "data" / "results" / "vwap_2021_2026" / "us500tick_m1.csv"
USA30_M1 = ROOT / "data" / "results" / "usa30_onboarding" / "usa30tick_m1.csv"

SESS_OPEN_MOD = 570
SESS_END_MOD = 955
SIG_LO, SIG_HI = 631, 929
ROLL_N = 60
Z_T = 2.0
USD_PER_LEG = 10000.0
SPREAD_500_PT, SPREAD_30_PT = 0.51, 2.28
BOOT_ITERS = 10000
SEED = 20260924
G1_MIN = 150
G3_MAX_P = 0.20


def main():
    p = print
    STUDY.mkdir(parents=True, exist_ok=True)

    p("[1/6] loading + joining tapes ...")
    a = pd.read_csv(US500_M1)
    b = pd.read_csv(USA30_M1)
    m = a.merge(b, on="time", suffixes=("_500", "_30"))
    assert m["time"].is_monotonic_increasing
    ts = pd.to_datetime(m["time"], unit="s", utc=True).dt.tz_convert("America/New_York")
    m = m.assign(ny_date=ts.dt.strftime("%Y-%m-%d"), mod=(ts.dt.hour * 60 + ts.dt.minute).to_numpy())
    sess = m[(m["mod"] >= SESS_OPEN_MOD) & (m["mod"] <= SESS_END_MOD)].reset_index(drop=True)
    n_us500, n_usa30 = len(a), len(b)
    del a, b, m, ts
    p(f"  us500 bars {n_us500}, usa30 bars {n_usa30}, joined bars {len(sess)} (session-filtered)")

    p("[2/6] state machine + signals ...")
    funnel = {
        "n_us500_bars": int(n_us500), "n_usa30_bars": int(n_usa30), "n_joined_session_bars": int(len(sess)),
        "n_cash_days": 0, "n_signal_days": 0, "n_days_no_breach": 0, "n_no_entry": 0,
        "n_fired_short_div": 0, "n_fired_long_div": 0,
    }
    rows = []
    g = sess.groupby("ny_date", sort=False)
    for date, day in g:
        if int(day["mod"].iloc[0]) != SESS_OPEN_MOD:
            continue
        funnel["n_cash_days"] += 1
        c500 = day["close_500"].to_numpy(float)
        c30 = day["close_30"].to_numpy(float)
        mods = day["mod"].to_numpy(int)
        tstr = pd.to_datetime(day["time"], unit="s", utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d %H:%M").to_numpy()
        spread = (c500 / c500[0] - 1.0) - (c30 / c30[0] - 1.0)
        s = pd.Series(spread)
        mean60 = s.rolling(ROLL_N, min_periods=ROLL_N).mean().to_numpy()
        std60 = s.rolling(ROLL_N, min_periods=ROLL_N).std(ddof=1).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (spread - mean60) / std60
        z[std60 <= 1e-12] = np.nan
        win = (mods >= SIG_LO) & (mods <= SIG_HI) & ~np.isnan(z)
        if not win.any():
            funnel["n_days_no_breach"] += 1
            continue
        zw = np.where(win, z, 0.0)
        breach_pos = int(np.argmax(np.abs(zw) > Z_T)) if (np.abs(zw) > Z_T).any() else -1
        if breach_pos < 0:
            funnel["n_days_no_breach"] += 1
            continue
        funnel["n_signal_days"] += 1
        direction = "SHORT_DIV" if zw[breach_pos] > 0 else "LONG_DIV"
        funnel["n_fired_short_div" if direction == "SHORT_DIV" else "n_fired_long_div"] += 1
        d500 = -1.0 if direction == "SHORT_DIV" else 1.0
        d30 = -d500
        e = breach_pos + 1
        if e >= len(day) or mods[e] > SESS_END_MOD:
            funnel["n_no_entry"] += 1
            continue
        o500 = float(day["open_500"].iloc[e])
        o30 = float(day["open_30"].iloc[e])
        r500 = c500[e:] / o500 - 1.0
        r30 = c30[e:] / o30 - 1.0
        pnl = USD_PER_LEG * (d500 * r500 + d30 * r30)
        ze = z[e:]
        if direction == "SHORT_DIV":
            cross = np.nonzero(~np.isnan(ze) & (ze <= 0.0))[0]
        else:
            cross = np.nonzero(~np.isnan(ze) & (ze >= 0.0))[0]
        if cross.size:
            off = int(cross[0])
            exit_kind = "REVERSION"
        else:
            off = len(pnl) - 1
            exit_kind = "TIME"
        mfe = float(pnl[:off + 1].max())
        mae = float(-pnl[:off + 1].min())
        exit_pnl = float(pnl[off])
        rows.append({
            "ny_date": date, "dir": direction,
            "signal_time": tstr[breach_pos], "entry_time": tstr[e],
            "signal_mod": int(mods[breach_pos]), "z_signal": float(z[breach_pos]),
            "entry_500": o500, "entry_30": o30,
            "mfe_usd": mfe, "mae_usd": mae,
            "exit_kind": exit_kind, "exit_time": tstr[e + off],
            "exit_pnl_usd": exit_pnl, "walk_bars": off + 1,
        })

    sig = pd.DataFrame(rows)
    n_sig = len(sig)
    p(f"  final signals: {n_sig}")

    p("[3/6] aggregates ...")
    if n_sig:
        agg = {
            "n": n_sig,
            "mfe": float(sig["mfe_usd"].mean()), "mae": float(sig["mae_usd"].mean()),
            "diff": float((sig["mfe_usd"] - sig["mae_usd"]).mean()),
            "mfe_med": float(sig["mfe_usd"].median()), "mae_med": float(sig["mae_usd"].median()),
            "mfe_gt_mae_share": float((sig["mfe_usd"] > sig["mae_usd"]).mean()),
            "exit_pnl": float(sig["exit_pnl_usd"].mean()),
            "exit_pnl_med": float(sig["exit_pnl_usd"].median()),
            "exit_win": float((sig["exit_pnl_usd"] > 0).mean()),
        }
        exit_mix = {}
        for k, gg in sig.groupby("exit_kind"):
            exit_mix[k] = {"n": len(gg), "share": len(gg) / n_sig, "mean_exit_pnl": float(gg["exit_pnl_usd"].mean())}
        by_dir = {}
        for d, gg in sig.groupby("dir"):
            by_dir[d] = {
                "n": len(gg), "mfe": float(gg["mfe_usd"].mean()), "mae": float(gg["mae_usd"].mean()),
                "diff": float((gg["mfe_usd"] - gg["mae_usd"]).mean()),
                "exit_pnl": float(gg["exit_pnl_usd"].mean()),
            }
        by_year = sig.groupby(sig["ny_date"].str[:4]).size().to_dict()
        med_c500 = float(np.median(sig["entry_500"]))
        med_c30 = float(np.median(sig["entry_30"]))
        cost_floor_usd = USD_PER_LEG * (SPREAD_500_PT / med_c500 + SPREAD_30_PT / med_c30)
        med_walk = float(sig["walk_bars"].median())
    else:
        agg = {"n": 0}
        exit_mix, by_dir, by_year = {}, {}, {}
        cost_floor_usd, med_walk = None, None

    p("[4/6] day-block bootstrap (%d resamples) ..." % BOOT_ITERS)
    if n_sig:
        rng = np.random.default_rng(SEED)
        diff = sig["mfe_usd"] - sig["mae_usd"]
        sums = diff.groupby(sig["ny_date"]).sum().to_numpy(float)
        cnts = sig.groupby("ny_date").size().to_numpy(float)
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
    A("REPORT: US500/USA30 INTRADAY RELATIVE VALUE - PHASE 1 (HYPOTHESIS #39)")
    A(f"Generated: {ts_now}  |  script: evaluate/index_pairs_intraday_phase1.py")
    A("SPEC/GATE: data/results/index_pairs_intraday/{SPEC,GATE}_PAIRS_INTRADAY.txt (frozen before code)")
    A("")
    A(f"VERDICT: {verdict}   (G-P1={g1}, G-P2={g2}, G-P3={g3})")
    A("")
    A("1. FUNNEL")
    A(f"   us500 tape bars: {funnel['n_us500_bars']}  |  usa30 tape bars: {funnel['n_usa30_bars']} (data/results/usa30_onboarding/usa30tick_m1.csv - tasked vwap_2021_2026 path does not exist, substitution declared in SPEC)")
    A(f"   joined session bars (09:30..15:55): {funnel['n_joined_session_bars']}")
    A(f"   cash days (09:30 anchor present): {funnel['n_cash_days']}")
    A(f"   days without |Z|>2 breach in signal window: {funnel['n_days_no_breach']}")
    A(f"   signal days: {funnel['n_signal_days']}  (SHORT_DIV {funnel['n_fired_short_div']} / LONG_DIV {funnel['n_fired_long_div']})")
    A(f"   no-entry drops: {funnel['n_no_entry']}")
    A(f"   FINAL SIGNALS: {n_sig}")
    if by_year:
        A("   by year: " + ", ".join(f"{y}={n}" for y, n in sorted(by_year.items())))
    A("")
    A("2. AGGREGATE PAIR EXCURSIONS ($, unlevered $10k/leg, gross pre-cost)")
    if agg.get("n"):
        A(f"   n={agg['n']}  MFE {agg['mfe']:.2f} vs MAE {agg['mae']:.2f}  diff {agg['diff']:+.2f}")
        A(f"   medians: MFE {agg['mfe_med']:.2f} vs MAE {agg['mae_med']:.2f}  MFE>MAE share {agg['mfe_gt_mae_share']*100:.1f}%")
        for d in sorted(by_dir):
            bb = by_dir[d]
            A(f"   {d}: n={bb['n']}  MFE {bb['mfe']:.2f} vs MAE {bb['mae']:.2f}  diff {bb['diff']:+.2f}  exit {bb['exit_pnl']:+.2f}")
    else:
        A("   n=0 (no gates evaluable)")
    A("")
    A("3. EXIT MIX AND GROSS EDGE")
    if agg.get("n"):
        for k in sorted(exit_mix):
            x = exit_mix[k]
            A(f"   {k}: {x['n']} ({x['share']*100:.1f}%)  mean exit PnL {x['mean_exit_pnl']:+.2f}")
        A(f"   gross exit PnL/trade {agg['exit_pnl']:+.2f} (median {agg['exit_pnl_med']:+.2f}, win {agg['exit_win']*100:.1f}%), median walk {med_walk:.0f} bars")
        A(f"   descriptive cost floor: {cost_floor_usd:.2f} $/pair round trip (0.51pt US500 + 2.28pt USA30 spreads at median signal prices) -> gross mean {'clears' if agg['exit_pnl'] > cost_floor_usd else 'does NOT clear'} the floor")
    A("")
    A("4. GATES (frozen verbatim in GATE file)")
    A(f"   G-P1 >= {G1_MIN} signals: {n_sig} -> {g1}")
    A(f"   G-P2 mean MFE > mean MAE (strict, $): {agg.get('mfe', float('nan')):.2f} vs {agg.get('mae', float('nan')):.2f} -> {g2}")
    A(f"   G-P3 bootstrap P(mean(MFE-MAE)<=0) <= {G3_MAX_P:.2f}: {pneg:.4f} (n_day_blocks={n_blocks}) -> {g3}")
    A(f"   OVERALL PHASE-1: {verdict}")
    A("")
    A("5. NOTES AND CAVEATS")
    A("   - Pre-cost gross screen: both legs assumed filled at printed next-bar opens, zero slippage;")
    A("     the CFD-spread cost floor is reported alongside, never netted into gates.")
    A("   - MFE/MAE on per-bar closes (synchronous intrabar pair extremes unobservable from M1 OHLC).")
    A("   - Frozen archetype parameters (60-bar rolling Z, |Z|>2, 10:30/15:30 window, 15:55 exit); no search.")
    A("   - Daily pairs line (#19) verdict unaffected; this is a different (intraday, session-anchored) mechanism.")
    A("   - Null result is final for this artifact.")
    report = "\n".join(L) + "\n"
    (STUDY / "REPORT_PAIRS_INTRADAY_PHASE1.txt").write_text(report, encoding="utf-8")
    if n_sig:
        sig.to_csv(STUDY / "pairs_intraday_signals.csv", index=False)
    meta = {
        "generated": ts_now,
        "script": "index_pairs_intraday_phase1.py",
        "funnel": funnel,
        "aggregate": agg,
        "exit_mix": exit_mix,
        "by_dir": by_dir,
        "by_year": by_year,
        "cost_floor_usd": cost_floor_usd,
        "pneg": pneg, "n_day_blocks": n_blocks,
        "gates": {"G-P1": g1, "G-P2": g2, "G-P3": g3},
        "overall": overall, "verdict": verdict,
        "params": {"roll_n": ROLL_N, "z_threshold": Z_T, "signal_window": "(10:30,15:30) ET",
                   "time_exit": "15:55 ET", "usd_per_leg": USD_PER_LEG,
                   "boot_iters": BOOT_ITERS, "seed": SEED,
                   "usa30_substitution": "data/results/usa30_onboarding/usa30tick_m1.csv"},
    }
    (STUDY / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    p(f"VERDICT: {verdict}  (G-P1={g1} n={n_sig}, G-P2={g2}, G-P3={g3} p={pneg:.4f})")
    if agg.get("n"):
        p(f"MFE {agg['mfe']:.2f} vs MAE {agg['mae']:.2f} (diff {agg['diff']:+.2f} $) | exit PnL {agg['exit_pnl']:+.2f} $ vs floor {cost_floor_usd:.2f}")
        p("exit mix: " + ", ".join(f"{k}={v['n']}" for k, v in exit_mix.items()))
    p(f"funnel: {funnel}")


if __name__ == "__main__":
    main()
