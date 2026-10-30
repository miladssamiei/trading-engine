import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\Milad\Desktop\RR")
STUDY = ROOT / "data" / "results" / "index_pairs_daily_cointegration"
US500_M1 = ROOT / "data" / "results" / "vwap_2021_2026" / "us500tick_m1.csv"
USA30_M1 = ROOT / "data" / "results" / "usa30_onboarding" / "usa30tick_m1.csv"

OLS_N = 252
Z_N = 252
Z_T = 2.0
TIME_EXIT_BARS = 45
NOTIONAL_Y = 10000.0
SPREAD_500_PT, SPREAD_30_PT = 0.51, 2.28
G1_MIN = 30


def build_d1(csv_path):
    df = pd.read_csv(csv_path, usecols=["time", "open", "high", "low", "close"])
    wall = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("America/New_York")
    d1_date = (wall - pd.Timedelta(hours=17)) + pd.Timedelta(days=1)
    df = df.assign(d1=d1_date.dt.strftime("%Y-%m-%d"))
    g = df.groupby("d1")
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(), "bars": g.size(),
    }).reset_index()
    return out


def main():
    p = print
    STUDY.mkdir(parents=True, exist_ok=True)

    p("[1/5] D1 aggregation + join ...")
    d500 = build_d1(US500_M1)
    d30 = build_d1(USA30_M1)
    m = d500.merge(d30, on="d1", suffixes=("_y", "_x")).sort_values("d1").reset_index(drop=True)
    dates = m["d1"].to_numpy()
    y_o = m["open_y"].to_numpy(float)
    x_o = m["open_x"].to_numpy(float)
    y_c = m["close_y"].to_numpy(float)
    x_c = m["close_x"].to_numpy(float)
    n = len(m)
    p(f"  D1 days: us500 {len(d500)}, usa30 {len(d30)}, joined {n} ({dates[0]} .. {dates[-1]})")

    p("[2/5] rolling OLS + spread Z ...")
    sy = pd.Series(y_c)
    sx = pd.Series(x_c)
    beta = (sy.rolling(OLS_N).cov(sx) / sx.rolling(OLS_N).var()).to_numpy()
    alpha = (sy.rolling(OLS_N).mean() - beta * sx.rolling(OLS_N).mean()).to_numpy()
    spread = pd.Series(y_c - beta * x_c - alpha)
    z_mean = spread.rolling(Z_N, min_periods=Z_N).mean()
    z_std = spread.rolling(Z_N, min_periods=Z_N).std(ddof=1)
    z = ((spread - z_mean) / z_std).to_numpy()
    armed = ~np.isnan(z)
    first_armed = int(np.argmax(armed))
    p(f"  first armed day idx {first_armed} ({dates[first_armed]}), armed days {int(armed.sum())}")
    p(f"  beta on armed days: mean {np.nanmean(beta[armed]):.4f} min {np.nanmin(beta[armed]):.4f} max {np.nanmax(beta[armed]):.4f}")

    p("[3/5] sequential signal state machine ...")
    trades = []
    open_trade = {"LONG_SPREAD": None, "SHORT_SPREAD": None}
    n_breach_long = n_breach_short = 0
    for i in range(first_armed + 1, n):
        for k in ("LONG_SPREAD", "SHORT_SPREAD"):
            t = open_trade[k]
            if t is None or t["exit_idx"] is not None:
                continue
            if (k == "SHORT_SPREAD" and z[i] <= 0.0) or (k == "LONG_SPREAD" and z[i] >= 0.0):
                t["exit_idx"] = i
                t["exit_kind"] = "REVERSION"
            elif i - t["entry_idx"] >= TIME_EXIT_BARS:
                t["exit_idx"] = min(i, n - 1)
                t["exit_kind"] = "TIME"
            elif i == n - 1:
                t["exit_idx"] = i
                t["exit_kind"] = "EOD"
        prev = i - 1
        if armed[prev] and z[prev] > Z_T and open_trade["SHORT_SPREAD"] is None:
            n_breach_short += 1
            open_trade["SHORT_SPREAD"] = {"dir": "SHORT_SPREAD", "signal_idx": prev, "entry_idx": i,
                                          "beta_s": float(beta[prev]), "exit_idx": None, "exit_kind": None}
        elif armed[prev] and z[prev] < -Z_T and open_trade["LONG_SPREAD"] is None:
            n_breach_long += 1
            open_trade["LONG_SPREAD"] = {"dir": "LONG_SPREAD", "signal_idx": prev, "entry_idx": i,
                                         "beta_s": float(beta[prev]), "exit_idx": None, "exit_kind": None}
        for k in ("LONG_SPREAD", "SHORT_SPREAD"):
            t = open_trade[k]
            if t is not None and t["exit_idx"] is None and t["entry_idx"] == i:
                if (k == "SHORT_SPREAD" and z[i] <= 0.0) or (k == "LONG_SPREAD" and z[i] >= 0.0):
                    t["exit_idx"] = i
                    t["exit_kind"] = "REVERSION"
        for k in ("LONG_SPREAD", "SHORT_SPREAD"):
            t = open_trade[k]
            if t is not None and t["exit_idx"] is not None and not t.get("finalized"):
                trades.append(t)
                t["finalized"] = True
                open_trade[k] = None
    for k in ("LONG_SPREAD", "SHORT_SPREAD"):
        t = open_trade[k]
        if t is not None and t["exit_idx"] is None:
            t["exit_idx"] = n - 1
            t["exit_kind"] = "EOD"
            trades.append(t)
            t["finalized"] = True

    p("[4/5] economics ...")
    rows = []
    for t in trades:
        e, x_i = t["entry_idx"], t["exit_idx"]
        dy, dx = (-1.0, 1.0) if t["dir"] == "SHORT_SPREAD" else (1.0, -1.0)
        ry = y_c[e:x_i + 1] / y_o[e] - 1.0
        rx = x_c[e:x_i + 1] / x_o[e] - 1.0
        pnl = NOTIONAL_Y * (dy * ry + dx * t["beta_s"] * rx)
        rows.append({
            "d1_date_signal": dates[t["signal_idx"]], "d1_date_entry": dates[e],
            "dir": t["dir"], "z_signal": float(z[t["signal_idx"]]), "beta_signal": t["beta_s"],
            "alpha_signal": float(alpha[t["signal_idx"]]),
            "entry_y": float(y_o[e]), "entry_x": float(x_o[e]),
            "d1_date_exit": dates[x_i], "exit_kind": t["exit_kind"],
            "bars_in_trade": int(x_i - e + 1),
            "mfe_usd": float(pnl.max()), "mae_usd": float(-pnl.min()),
            "exit_pnl_usd": float(pnl[-1]),
            "cost_floor_usd": NOTIONAL_Y * SPREAD_500_PT / y_o[e] + NOTIONAL_Y * t["beta_s"] * SPREAD_30_PT / x_o[e],
        })
    sig = pd.DataFrame(rows)
    n_sig = len(sig)

    p("[5/5] gates / report / csv ...")
    if n_sig:
        agg = {
            "n": n_sig,
            "mfe": float(sig["mfe_usd"].mean()), "mae": float(sig["mae_usd"].mean()),
            "mfe_med": float(sig["mfe_usd"].median()), "mae_med": float(sig["mae_usd"].median()),
            "exit_pnl": float(sig["exit_pnl_usd"].mean()), "exit_pnl_med": float(sig["exit_pnl_usd"].median()),
            "win": float((sig["exit_pnl_usd"] > 0).mean()),
            "mfe_gt_mae_share": float((sig["mfe_usd"] > sig["mae_usd"]).mean()),
            "mean_bars": float(sig["bars_in_trade"].mean()), "median_bars": float(sig["bars_in_trade"].median()),
            "mean_cost_floor": float(sig["cost_floor_usd"].mean()),
        }
        exit_mix = {k: {"n": len(gg), "share": float(len(gg)) / n_sig,
                        "mean_pnl": float(gg["exit_pnl_usd"].mean())}
                    for k, gg in sig.groupby("exit_kind")}
        by_dir = {k: {"n": len(gg), "mfe": float(gg["mfe_usd"].mean()), "mae": float(gg["mae_usd"].mean()),
                      "exit_pnl": float(gg["exit_pnl_usd"].mean())} for k, gg in sig.groupby("dir")}
        reversion_rate = float((sig["exit_kind"] == "REVERSION").mean())
        mean_beta = float(sig["beta_signal"].mean())
    else:
        agg = {"n": 0}
        exit_mix, by_dir, reversion_rate, mean_beta = {}, {}, float("nan"), float("nan")
    g1 = bool(n_sig >= G1_MIN)
    g2 = bool(agg.get("n", 0) > 0 and agg["mfe"] > agg["mae"])
    g3 = bool(agg.get("n", 0) > 0 and agg["exit_pnl"] > 0)
    overall = bool(g1 and g2 and g3)
    verdict = "PASS" if overall else "FAIL"

    ts_now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = []
    A = L.append
    A("REPORT: US500/USA30 DAILY OLS COINTEGRATION - PHASE 1 (HYPOTHESIS #41)")
    A(f"Generated: {ts_now}  |  script: evaluate/index_pairs_daily_cointegration_phase1.py")
    A("SPEC/GATE: data/results/index_pairs_daily_cointegration/{SPEC,GATE}_PAIRS_D1_COINT.txt (frozen before code)")
    A("")
    A(f"VERDICT: {verdict}   (G-DC1={g1}, G-DC2={g2}, G-DC3={g3})")
    A("")
    A("1. FUNNEL")
    A(f"   D1 days: us500 {len(d500)} / usa30 {len(d30)} -> joined {n} ({dates[0]} .. {dates[-1]})")
    A(f"   two-stage warmup: first regression day 252, Z armed at joined day {first_armed + 1} ({dates[first_armed]})")
    A(f"   armed days {int(armed.sum())}; direction breaches: SHORT_SPREAD {n_breach_short} / LONG_SPREAD {n_breach_long} (first-open gating)")
    A(f"   FINAL SIGNALS: {n_sig}" + (f"  (SHORT {by_dir.get('SHORT_SPREAD', {}).get('n', 0)} / LONG {by_dir.get('LONG_SPREAD', {}).get('n', 0)})" if n_sig else ""))
    A(f"   mean signal-day beta {mean_beta:.4f}")
    A("")
    A("2. AGGREGATE PAIR EXCURSIONS + GROSS EDGE ($10k US500 leg, beta-weighted USA30 leg, gross)")
    if agg.get("n"):
        A(f"   n={agg['n']}  MFE {agg['mfe']:.2f} vs MAE {agg['mae']:.2f}  (medians {agg['mfe_med']:.2f} / {agg['mae_med']:.2f})  MFE>MAE share {agg['mfe_gt_mae_share']*100:.1f}%")
        A(f"   gross exit PnL/trade {agg['exit_pnl']:+.2f} (median {agg['exit_pnl_med']:+.2f}, win {agg['win']*100:.1f}%)")
        for k in sorted(exit_mix):
            x = exit_mix[k]
            A(f"   {k}: {x['n']} ({x['share']*100:.1f}%)  mean PnL {x['mean_pnl']:+.2f}")
        A(f"   reversion (target-hit) rate: {reversion_rate*100:.1f}%")
        A(f"   time in market: mean {agg['mean_bars']:.1f} / median {agg['median_bars']:.0f} bars (45-bar time cap)")
        for k in sorted(by_dir):
            bb = by_dir[k]
            A(f"   {k}: n={bb['n']}  MFE {bb['mfe']:.2f} vs MAE {bb['mae']:.2f}  exit {bb['exit_pnl']:+.2f}")
        A(f"   descriptive cost floor: mean {agg['mean_cost_floor']:.2f} $/pair round trip (0.51pt + 2.28pt CFD spreads, never netted)")
    else:
        A("   n=0 (gates not evaluable)")
    A("")
    A("4. GATES (frozen verbatim in GATE file)")
    A(f"   G-DC1 >= {G1_MIN} signals: {n_sig} -> {g1}")
    A(f"   G-DC2 mean MFE > mean MAE: {agg.get('mfe', float('nan')):.2f} vs {agg.get('mae', float('nan')):.2f} -> {g2}")
    A(f"   G-DC3 mean gross PnL > 0: {agg.get('exit_pnl', float('nan')):+.2f} -> {g3}")
    A(f"   OVERALL PHASE-1: {verdict}")
    A("")
    A("5. NOTES AND CAVEATS")
    A("   - Price-level OLS on non-stationary index levels carries a spurious-regression risk; the")
    A("     two-stage z (each day's own beta/alpha, then trailing spread stats) is the tasked mechanic.")
    A("   - Family lineage: #19 (log-OLS daily) FAIL, #39 (intraday 60-bar Z) FAIL - context, not evidence.")
    A("   - Gross only; fills at printed D1 opens/closes, no slippage; cost floor reported alongside.")
    A("   - Null result is final for this artifact.")
    report = "\n".join(L) + "\n"
    (STUDY / "REPORT_PAIRS_D1_COINT_PHASE1.txt").write_text(report, encoding="utf-8")
    if n_sig:
        sig.to_csv(STUDY / "pairs_d1_coint_signals.csv", index=False)
    meta = {
        "generated": ts_now,
        "script": "index_pairs_daily_cointegration_phase1.py",
        "funnel": {"d1_us500": len(d500), "d1_usa30": len(d30), "joined": n,
                   "first_armed_idx": first_armed, "first_armed_date": dates[first_armed],
                   "breaches_short": n_breach_short, "breaches_long": n_breach_long, "final": n_sig},
        "aggregate": agg, "exit_mix": exit_mix, "by_dir": by_dir, "reversion_rate": reversion_rate,
        "gates": {"G-DC1": g1, "G-DC2": g2, "G-DC3": g3},
        "overall": overall, "verdict": verdict,
        "params": {"ols_n": OLS_N, "z_n": Z_N, "z_threshold": Z_T, "time_exit_bars": TIME_EXIT_BARS,
                   "notional_y": NOTIONAL_Y, "d1_convention": "(D-1 17:00, D 17:00] ET"},
    }
    (STUDY / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    p(f"VERDICT: {verdict}  (G-DC1={g1} n={n_sig}, G-DC2={g2}, G-DC3={g3})")
    if agg.get("n"):
        p(f"MFE {agg['mfe']:.2f} vs MAE {agg['mae']:.2f} | exit {agg['exit_pnl']:+.2f} | reversion {reversion_rate*100:.1f}% | mean bars {agg['mean_bars']:.1f}")
    p(f"exit mix: {exit_mix}")


if __name__ == "__main__":
    main()
