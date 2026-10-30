import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\Milad\Desktop\RR")
STUDY = ROOT / "data" / "results" / "ls9_liquidity_hunt"
M5 = ROOT / "data" / "results" / "usa30_onboarding" / "usa30tick_m5.csv"

WIN_START = int(dt.datetime(2025, 7, 1, tzinfo=dt.timezone.utc).timestamp())
WIN_END = int(dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc).timestamp())
BUCKET = 900
SESS_START_M = 8 * 60 + 30      # 08:30 UTC
SESS_END_M = 16 * 60 + 15       # 16:15 UTC (exclusive)
WALK_END_M = 20 * 60            # 20:00 UTC (exclusive)
ATR_P = 14
MAX_SWINGS = 3
N_BOOT = 10000
SEED = 20260921
THR = [0.5, 0.75, 1.0]


def load_m15() -> pd.DataFrame:
    df = pd.read_csv(M5)
    df = df[(df["time"] >= WIN_START) & (df["time"] < WIN_END)].copy()
    df = df.sort_values("time").drop_duplicates("time")
    print(f"M5 bars in window: {len(df)} "
          f"({pd.to_datetime(df['time'].min(), unit='s', utc=True)} .. "
          f"{pd.to_datetime(df['time'].max(), unit='s', utc=True)})")
    df["bucket"] = (df["time"] // BUCKET) * BUCKET
    g = df.groupby("bucket")
    m15 = pd.DataFrame({"time": g["bucket"].first(),
                        "open": g["open"].first(),
                        "high": g["high"].max(),
                        "low": g["low"].min(),
                        "close": g["close"].last(),
                        "nbars": g.size()}).reset_index(drop=True)
    print(f"M15 buckets: {len(m15)}; bars-per-bucket: "
          f"1:{int((m15['nbars'] == 1).sum())} 2:{int((m15['nbars'] == 2).sum())} "
          f"3:{int((m15['nbars'] == 3).sum())}")
    return m15


def wilder_atr(h, l, c):
    n = len(c)
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = np.full(n, np.nan)
    atr[ATR_P - 1] = tr[:ATR_P].mean()
    for i in range(ATR_P, n):
        atr[i] = (atr[i - 1] * (ATR_P - 1) + tr[i]) / ATR_P
    return atr


def run(m15: pd.DataFrame):
    t = m15["time"].to_numpy()
    o = m15["open"].to_numpy()
    h = m15["high"].to_numpy()
    l = m15["low"].to_numpy()
    c = m15["close"].to_numpy()
    n = len(t)
    atr = wilder_atr(h, l, c)
    green = c > o
    red = c < o
    day = t // 86400
    minute_of_day = (t % 86400) // 60

    swing_hi = []   # [formation_idx, level]
    swing_lo = []
    n_swing_hi = 0
    n_swing_lo = 0
    n_invalid_hi = 0
    n_invalid_lo = 0
    n_ls_shape = 0
    n_session = 0
    dropped_no_walk = 0
    sigs = []

    for i in range(1, n):
        # ---- (a) trigger with lists valid at close of i-1 (SPEC 2.4) ------
        if minute_of_day[i] >= SESS_START_M and minute_of_day[i] < SESS_END_M \
           and not np.isnan(atr[i]):
            body = abs(c[i] - o[i])
            uw = h[i] - max(c[i], o[i])
            lw = min(c[i], o[i]) - l[i]
            bear = uw > body and uw > lw and lw <= 7.0 * body
            bull = lw > body and lw > uw and uw <= 7.0 * body
            swept_level = 0.0
            n_swept = 0
            direction = 0
            if bear and swing_hi:
                recent = [s for s in swing_hi if h[i] >= s[1]]
                if recent:
                    n_swept = len(recent)
                    swept_level = recent[-1][1]
                    direction = -1
            elif bull and swing_lo:
                recent = [s for s in swing_lo if l[i] <= s[1]]
                if recent:
                    n_swept = len(recent)
                    swept_level = recent[-1][1]
                    direction = +1
            if direction != 0:
                n_ls_shape += 1
                n_session += 1
                # excursion walk from t+1 until 20:00 UTC of signal day
                walk_hi = []
                walk_lo = []
                j = i + 1
                while j < n and day[j] == day[i] and minute_of_day[j] < WALK_END_M:
                    walk_hi.append(h[j])
                    walk_lo.append(l[j])
                    j += 1
                if walk_hi and j > i + 1:
                    entry = o[i + 1]
                    wh = np.array(walk_hi)
                    wl = np.array(walk_lo)
                    if direction < 0:
                        mfe = entry - wl.min()
                        mae = wh.max() - entry
                    else:
                        mfe = wh.max() - entry
                        mae = entry - wl.min()
                    sigs.append({"time_utc": pd.to_datetime(t[i], unit="s", utc=True)
                                 .strftime("%Y-%m-%d %H:%M"),
                                 "utc_date": pd.to_datetime(t[i], unit="s", utc=True)
                                 .strftime("%Y-%m-%d"),
                                 "dir": "SHORT" if direction < 0 else "LONG",
                                 "entry": float(entry),
                                 "atr14": float(atr[i]),
                                 "swept_level": float(swept_level),
                                 "n_swept": int(n_swept),
                                 "walk_bars": len(walk_hi),
                                 "mfe_pts": float(mfe),
                                 "mae_pts": float(mae),
                                 "mfe_atr": float(mfe / atr[i]),
                                 "mae_atr": float(mae / atr[i]),
                                 "diff_atr": float((mfe - mae) / atr[i])})
                elif not walk_hi or j == i + 1:
                    dropped_no_walk += 1

        # ---- (b) update swing lists with bar i (SPEC 2.2-2.4) -------------
        if green[i - 1] and red[i]:
            swing_hi.append([i, max(h[i - 1], h[i])])
            n_swing_hi += 1
        if red[i - 1] and green[i]:
            swing_lo.append([i, min(l[i - 1], l[i])])
            n_swing_lo += 1
        before = len(swing_hi)
        swing_hi = [s for s in swing_hi if c[i] <= s[1]]
        n_invalid_hi += before - len(swing_hi)
        before = len(swing_lo)
        swing_lo = [s for s in swing_lo if c[i] >= s[1]]
        n_invalid_lo += before - len(swing_lo)
        if len(swing_hi) > MAX_SWINGS:
            swing_hi = swing_hi[-MAX_SWINGS:]
        if len(swing_lo) > MAX_SWINGS:
            swing_lo = swing_lo[-MAX_SWINGS:]

    funnel = {"m15_bars": n, "swing_highs_formed": n_swing_hi,
              "swing_lows_formed": n_swing_lo,
              "swing_highs_invalidated": n_invalid_hi,
              "swing_lows_invalidated": n_invalid_lo,
              "session_ls_signals": n_session,
              "dropped_no_walk": dropped_no_walk,
              "excursion_valid": len(sigs)}
    return pd.DataFrame(sigs), funnel


def main():
    if not (STUDY / "SPEC_LS9_LIQUIDITY_HUNT.txt").exists():
        raise SystemExit("SPEC missing; refusing to run")
    if not (STUDY / "GATE_LS9_LIQUIDITY_HUNT.txt").exists():
        raise SystemExit("GATE missing; refusing to run")

    m15 = load_m15()
    sig, funnel = run(m15)
    print(f"funnel: {funnel}")

    n = len(sig)
    ga1 = bool(n >= 150)
    thr_rows = []
    ga2 = n > 0
    for scope, sub in [("AGG", sig), ("LONG", sig[sig["dir"] == "LONG"]),
                       ("SHORT", sig[sig["dir"] == "SHORT"])]:
        for tau in THR:
            mf = float(np.maximum(sub["mfe_atr"] - tau, 0.0).mean()) if len(sub) else float("nan")
            ma = float(np.maximum(sub["mae_atr"] - tau, 0.0).mean()) if len(sub) else float("nan")
            ok = bool(len(sub) > 0 and mf > ma)
            ga2 = ga2 and ok
            thr_rows.append({"scope": scope, "tau": tau, "mfe_exc": mf,
                             "mae_exc": ma, "ok": ok})

    ga3 = None
    p_le0 = float("nan")
    if n > 0:
        by_day = sig.groupby("utc_date")["diff_atr"].apply(np.array).to_dict()
        days = list(by_day.keys())
        arrays = [by_day[d] for d in days]
        rng = np.random.default_rng(SEED)
        k = len(days)
        cnt = 0
        for _ in range(N_BOOT):
            idx = rng.integers(0, k, k)
            pooled = np.concatenate([arrays[i] for i in idx])
            if pooled.mean() <= 0.0:
                cnt += 1
        p_le0 = cnt / N_BOOT
        ga3 = bool(p_le0 <= 0.20)
    verdict = "PASS" if (ga1 and ga2 and bool(ga3)) else "FAIL"

    sig.to_csv(STUDY / "ls9_phase1_signals.csv", index=False)

    def stats(sub):
        if not len(sub):
            return "n=0"
        return (f"n={len(sub):>4}  MFE {sub['mfe_pts'].mean():>7.2f}pt / "
                f"{sub['mfe_atr'].mean():>5.3f}atr   MAE {sub['mae_pts'].mean():>7.2f}pt / "
                f"{sub['mae_atr'].mean():>5.3f}atr   diff {sub['diff_atr'].mean():>+6.3f}atr")

    L = []
    L.append("REPORT_LS9_PHASE1")
    L.append("Generated: " + dt.datetime.now().isoformat(timespec="seconds"))
    L.append("")
    L.append("PREMISE: mechanical interpretation of the LS7/LS9 Liquidity Hunt video rules on the")
    L.append("USA30 M15 tape (M5 -> M15 aggregation), session 12:00-19:45 Iran Time = [08:30,")
    L.append("16:15) UTC setup window, excursion walk to 20:00 UTC. Phase-1 MFE/MAE signal-")
    L.append("quality screen ONLY: no costs (USA30 RT spread is 2.28 pt, NOT subtracted), no")
    L.append("exit logic, MFE/MAE not simultaneously achievable. Descriptive regardless of")
    L.append("verdict; downstream_model_allowed=false, live_execution_allowed=false; the")
    L.append("closed intraday families are not reopened by this screen.")
    L.append("")
    L.append("FUNNEL")
    L.append(f"  M15 bars in window:            {funnel['m15_bars']}")
    L.append(f"  swing highs formed/invalidated:{funnel['swing_highs_formed']} / "
             f"{funnel['swing_highs_invalidated']}")
    L.append(f"  swing lows formed/invalidated: {funnel['swing_lows_formed']} / "
             f"{funnel['swing_lows_invalidated']}")
    L.append(f"  session LS signals (trigger):  {funnel['session_ls_signals']}")
    L.append(f"  dropped (no walk):             {funnel['dropped_no_walk']}")
    L.append(f"  excursion-valid signals:       {funnel['excursion_valid']}")
    L.append("")
    L.append("RAW EXCURSION STATISTICS")
    L.append(f"  AGG    {stats(sig)}")
    L.append(f"  LONG   {stats(sig[sig['dir'] == 'LONG'])}")
    L.append(f"  SHORT  {stats(sig[sig['dir'] == 'SHORT'])}")
    L.append("")
    L.append("G-A2 THRESHOLD TABLE  mean((MFE_atr - tau)+) vs mean((MAE_atr - tau)+)")
    L.append(f"  {'scope':>6} {'tau':>5} {'MFE_exc':>8} {'MAE_exc':>8} {'ok':>4}")
    for r in thr_rows:
        L.append(f"  {r['scope']:>6} {r['tau']:>5.2f} {r['mfe_exc']:>8.4f} "
                 f"{r['mae_exc']:>8.4f} {str(r['ok']):>5}")
    L.append("")
    L.append("GATES")
    L.append(f"G-A1 SAMPLE SIZE: {n} excursion-valid signals >= 150 -> "
             f"{'PASS' if ga1 else 'FAIL'}")
    L.append(f"G-A2 EXCURSION EDGE: all 9 threshold comparisons "
             f"(agg/long/short x 0.5/0.75/1.0 ATR) -> {'PASS' if ga2 else 'FAIL'}")
    L.append(f"G-A3 DAY-BLOCK BOOTSTRAP: P(mean(MFE_atr - MAE_atr) <= 0) = {p_le0:.4f} "
             f"<= 0.20 -> {'PASS' if ga3 else 'FAIL'} "
             f"({N_BOOT} resamples, seed {SEED}, {sig['utc_date'].nunique()} UTC-date blocks)")
    L.append(f"OVERALL VERDICT: {verdict}"
             + (" (descriptive screen pass only; no Phase-2 authorization)"
                if verdict == "PASS" else " (final for this artifact)"))
    L.append("")
    L.append("CAVEATS: single window 2025-07..2026-08, in-sample; excursions are upper bounds")
    L.append("on tradable edge (no exit logic, no spread/financing); mechanical choices per")
    L.append("SPEC 2.1-2.4/4.4 (doji handling, trigger-then-update order, 3-swing recency cap,")
    L.append("strict inequalities, 20:00 UTC walk bound, dropped no-walk signals); M15 buckets")
    L.append("with 1-2 M5 bars retained; ATR Wilder on the full continuous tape.")
    (STUDY / "REPORT_LS9_PHASE1.txt").write_text("\n".join(L), encoding="utf-8")

    meta = {"ga1": ga1, "ga2": ga2, "ga3": ga3, "verdict": verdict,
            "funnel": funnel, "n_signals": n,
            "p_le0_bootstrap": p_le0, "seed": SEED, "n_boot": N_BOOT,
            "mean_mfe_pts": float(sig["mfe_pts"].mean()) if n else None,
            "mean_mae_pts": float(sig["mae_pts"].mean()) if n else None,
            "mean_diff_atr": float(sig["diff_atr"].mean()) if n else None}
    (STUDY / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
