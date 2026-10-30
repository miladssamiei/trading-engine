import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\Milad\Desktop\RR")
STUDY = ROOT / "data" / "results" / "fx_trend_d1"
D1_CSV = ROOT / "data" / "results" / "eurusd_onboarding" / "eurusdtick_d1.csv"

PIP = 10000.0
DON_N = 20
ATR_N = 14
WARMUP = 21
ATR_MULT = 2.0
COST_FLOOR_PIPS = 0.3
BOOT_ITERS = 10000
BOOT_BLOCK = 5
SEED = 20260924
G1_MIN = 30
G3_MAX_P = 0.20


def wilder_atr(h, l, c):
    n = len(c)
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = np.full(n, np.nan)
    if n >= ATR_N:
        atr[ATR_N - 1] = tr[:ATR_N].mean()
        for i in range(ATR_N, n):
            atr[i] = (atr[i - 1] * (ATR_N - 1) + tr[i]) / ATR_N
    return atr


def main():
    p = print
    STUDY.mkdir(parents=True, exist_ok=True)

    p("[1/5] tape + causal indicators ...")
    df = pd.read_csv(D1_CSV, dtype={"date": str}).sort_values("date").reset_index(drop=True)
    assert df["date"].is_monotonic_increasing
    dates = df["date"].to_numpy()
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(df)
    don_hi = pd.Series(h).rolling(DON_N).max().shift(1).to_numpy()
    don_lo = pd.Series(l).rolling(DON_N).min().shift(1).to_numpy()
    atr = wilder_atr(h, l, c)

    p("[2/5] sequential state machine ...")
    funnel = {"n_bars": n, "n_ignored_while_open": 0, "n_signals_long": 0,
              "n_signals_short": 0, "n_eod": 0}
    trades = []
    i = WARMUP
    while i < n:
        can_long = not np.isnan(don_hi[i]) and not np.isnan(atr[i])
        sig = 0
        if can_long and c[i] > don_hi[i]:
            sig = 1
        elif can_long and c[i] < don_lo[i]:
            sig = -1
        if sig == 0:
            i += 1
            continue
        funnel["n_signals_long" if sig == 1 else "n_signals_short"] += 1
        if i + 1 >= n:
            i += 1
            continue
        e = i + 1
        entry = float(o[e])
        sl = entry - sig * ATR_MULT * atr[i]
        atr_e = float(atr[i])
        exit_idx, exit_price, exit_kind = None, None, None
        j = e
        while j < n:
            close_sl = c[j] - sig * ATR_MULT * atr[j] if not np.isnan(atr[j]) else None
            if close_sl is not None:
                sl = max(sl, close_sl) if sig == 1 else min(sl, close_sl)
            k = j + 1
            if k >= n:
                break
            if sig == 1 and o[k] < sl:
                exit_idx, exit_price, exit_kind = k, float(o[k]), "STOP_GAP"
                break
            if sig == 1 and l[k] <= sl:
                exit_idx, exit_price, exit_kind = k, float(sl), "STOP"
                break
            if sig == -1 and o[k] > sl:
                exit_idx, exit_price, exit_kind = k, float(o[k]), "STOP_GAP"
                break
            if sig == -1 and h[k] >= sl:
                exit_idx, exit_price, exit_kind = k, float(sl), "STOP"
                break
            j = k
        if exit_idx is None:
            exit_idx, exit_price, exit_kind = n - 1, float(c[n - 1]), "EOD"
            funnel["n_eod"] += 1
        if sig == 1:
            mfe = (float(h[e:exit_idx + 1].max()) - entry) * PIP
            mae = (entry - float(l[e:exit_idx + 1].min())) * PIP
        else:
            mfe = (entry - float(l[e:exit_idx + 1].min())) * PIP
            mae = (float(h[e:exit_idx + 1].max()) - entry) * PIP
        gross = sig * (exit_price - entry) * PIP
        for k in range(e, exit_idx):
            if not np.isnan(don_hi[k]) and (c[k] > don_hi[k] or c[k] < don_lo[k]):
                funnel["n_ignored_while_open"] += 1
        trades.append({
            "date_signal": dates[i], "date_entry": dates[e],
            "dir": "LONG" if sig == 1 else "SHORT",
            "don_high": float(don_hi[i]), "don_low": float(don_lo[i]),
            "atr_signal": atr_e, "entry": entry, "initial_sl": entry - sig * ATR_MULT * atr_e,
            "final_sl": float(sl), "exit_price": exit_price, "exit_kind": exit_kind,
            "date_exit": dates[exit_idx], "bars_in_trade": int(exit_idx - e + 1),
            "mfe_pips": mfe, "mae_pips": mae, "gross_pips": gross,
        })
        i = exit_idx

    sig = pd.DataFrame(trades)
    n_tr = len(sig)
    p(f"  executed trades: {n_tr}")

    p("[3/5] aggregates ...")
    if n_tr:
        wins = sig[sig["gross_pips"] > 0]
        losses = sig[sig["gross_pips"] <= 0]
        agg = {
            "n": n_tr,
            "win": float(len(wins) / n_tr),
            "gross_mean": float(sig["gross_pips"].mean()),
            "gross_median": float(sig["gross_pips"].median()),
            "gross_total": float(sig["gross_pips"].sum()),
            "mfe": float(sig["mfe_pips"].mean()), "mae": float(sig["mae_pips"].mean()),
            "mfe_med": float(sig["mfe_pips"].median()), "mae_med": float(sig["mae_pips"].median()),
            "mfe_gt_mae_share": float((sig["mfe_pips"] > sig["mae_pips"]).mean()),
            "avg_win": float(wins["gross_pips"].mean()) if len(wins) else float("nan"),
            "avg_loss": float(losses["gross_pips"].mean()) if len(losses) else float("nan"),
            "payoff": float(wins["gross_pips"].mean() / abs(losses["gross_pips"].mean())) if len(wins) and len(losses) else float("nan"),
            "mean_bars": float(sig["bars_in_trade"].mean()),
            "median_bars": float(sig["bars_in_trade"].median()),
        }
        by_dir = {d: {"n": len(gg), "win": float((gg["gross_pips"] > 0).mean()),
                      "gross_mean": float(gg["gross_pips"].mean()),
                      "mfe": float(gg["mfe_pips"].mean()), "mae": float(gg["mae_pips"].mean())}
                  for d, gg in sig.groupby("dir")}
        by_year = sig.groupby(sig["date_signal"].str[:4]).size().to_dict()
        exit_mix = {k: int(v) for k, v in sig["exit_kind"].value_counts().items()}
    else:
        agg = {"n": 0}
        by_dir, by_year, exit_mix = {}, {}, {}

    p("[4/5] moving-block bootstrap (block=%d, %d resamples) ..." % (BOOT_BLOCK, BOOT_ITERS))
    if n_tr >= BOOT_BLOCK:
        rng = np.random.default_rng(SEED)
        g = sig["gross_pips"].to_numpy(float)
        nblocks = int(np.ceil(n_tr / BOOT_BLOCK))
        starts = rng.integers(0, n_tr - BOOT_BLOCK + 1, size=(BOOT_ITERS, nblocks))
        idx = starts[:, :, None] + np.arange(BOOT_BLOCK)[None, None, :]
        means = g[idx].mean(axis=(1, 2))
        pneg = float((means <= 0).mean())
    else:
        pneg = 1.0
    p("  P(block mean gross <= 0) = %.4f" % pneg)

    p("[5/5] gates / report / csv ...")
    g1 = bool(n_tr >= G1_MIN)
    g2 = bool(agg.get("n", 0) > 0 and agg["mfe"] > agg["mae"])
    g3 = bool(pneg <= G3_MAX_P)
    overall = bool(g1 and g2 and g3)
    verdict = "PASS" if overall else "FAIL"

    ts_now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = []
    A = L.append
    A("REPORT: EURUSD D1 TREND FOLLOWING - PHASE 1 (HYPOTHESIS #42)")
    A(f"Generated: {ts_now}  |  script: evaluate/fx_trend_d1_phase1.py")
    A("SPEC/GATE: data/results/fx_trend_d1/{SPEC,GATE}_FX_TREND_D1.txt (frozen before code)")
    A("")
    A(f"VERDICT: {verdict}   (G-FX1={g1}, G-FX2={g2}, G-FX3={g3})")
    A("")
    A("1. FUNNEL")
    A(f"   D1 bars {n} ({dates[0]} .. {dates[-1]}); warmup {WARMUP}; armed bars {n - WARMUP}")
    A(f"   breakout signals: LONG {funnel['n_signals_long']} / SHORT {funnel['n_signals_short']}; ignored while open: {funnel['n_ignored_while_open']}")
    A(f"   EXECUTED TRADES: {n_tr} (EOD force-closes {funnel['n_eod']})")
    if by_year:
        A("   by signal year: " + ", ".join(f"{y}={m}" for y, m in sorted(by_year.items())))
    A(f"   exit mix: {exit_mix}")
    A("")
    A("2. AGGREGATE EXCURSIONS AND GROSS EDGE (pips, pre-cost)")
    if agg.get("n"):
        A(f"   n={agg['n']}  MFE {agg['mfe']:.1f} vs MAE {agg['mae']:.1f}  (medians {agg['mfe_med']:.1f} / {agg['mae_med']:.1f})  MFE>MAE share {agg['mfe_gt_mae_share']*100:.1f}%")
        A(f"   gross PnL/trade {agg['gross_mean']:+.1f} pips (median {agg['gross_median']:+.1f}, total {agg['gross_total']:+.0f}, win {agg['win']*100:.1f}%)")
        A(f"   avg winner {agg['avg_win']:+.1f} vs avg loser {agg['avg_loss']:+.1f} pips -> payoff ratio {agg['payoff']:.2f}")
        A(f"   time in trade: mean {agg['mean_bars']:.1f} / median {agg['median_bars']:.0f} bars")
        for d in sorted(by_dir):
            bb = by_dir[d]
            A(f"   {d}: n={bb['n']} win {bb['win']*100:.0f}%  gross {bb['gross_mean']:+.1f}  MFE {bb['mfe']:.1f} vs MAE {bb['mae']:.1f}")
        A(f"   vs verified 0.3-pip overlap median floor: mean gross {'clears' if agg['gross_mean'] > COST_FLOOR_PIPS else 'does NOT clear'} the floor")
    else:
        A("   n=0 (gates not evaluable)")
    A("")
    A("3. GATES (frozen verbatim in GATE file)")
    A(f"   G-FX1 >= {G1_MIN} executed: {n_tr} -> {g1}")
    A(f"   G-FX2 mean MFE > mean MAE (strict): {agg.get('mfe', float('nan')):.1f} vs {agg.get('mae', float('nan')):.1f} -> {g2}")
    A(f"   G-FX3 block-bootstrap P(mean gross <= 0) <= {G3_MAX_P:.2f}: {pneg:.4f} -> {g3}")
    A(f"   OVERALL PHASE-1: {verdict}")
    A("")
    A("4. NOTES AND CAVEATS")
    A("   - Pre-cost Bid-tape screen; 0.3-pip floor reported alongside, never netted.")
    A("   - MFE/MAE include the exit day's full bar (D1 approximation); gap-through stops fill at open.")
    A("   - Frozen archetype parameters (Donchian 20 / ATR 14 / 2.0x multiple / single position); no search.")
    A("   - Null result is final for this artifact.")
    report = "\n".join(L) + "\n"
    (STUDY / "REPORT_FX_TREND_D1_PHASE1.txt").write_text(report, encoding="utf-8")
    if n_tr:
        sig.to_csv(STUDY / "fx_trend_d1_signals.csv", index=False)
    meta = {
        "generated": ts_now,
        "script": "fx_trend_d1_phase1.py",
        "funnel": funnel,
        "aggregate": agg,
        "by_dir": by_dir,
        "by_year": by_year,
        "exit_mix": exit_mix,
        "pneg": pneg,
        "gates": {"G-FX1": g1, "G-FX2": g2, "G-FX3": g3},
        "overall": overall, "verdict": verdict,
        "params": {"donchian_n": DON_N, "atr_n": ATR_N, "atr_mult": ATR_MULT,
                   "warmup": WARMUP, "boot_iters": BOOT_ITERS, "boot_block": BOOT_BLOCK,
                   "seed": SEED, "pip": 0.0001},
    }
    (STUDY / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    p(f"VERDICT: {verdict}  (G-FX1={g1} n={n_tr}, G-FX2={g2}, G-FX3={g3} p={pneg:.4f})")
    if agg.get("n"):
        p(f"gross {agg['gross_mean']:+.1f} pips/trade, win {agg['win']*100:.1f}%, payoff {agg['payoff']:.2f}, MFE {agg['mfe']:.1f} vs MAE {agg['mae']:.1f}")
    p(f"funnel: {funnel}")


if __name__ == "__main__":
    main()
