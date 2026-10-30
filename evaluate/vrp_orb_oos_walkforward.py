import os
import json
import time
import hashlib
import urllib.request

import numpy as np
import pandas as pd

SEED = 20260924
BOOT_ITERS = 10000
SPREAD = 0.51
OOS_START = pd.Timestamp("2024-01-01")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAPE = os.path.join(ROOT, "data", "results", "vwap_2021_2026", "us500tick_m1.csv")
OUT = os.path.join(ROOT, "data", "results", "vrp_orb_oos")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_vix():
    try:
        import yfinance as yf

        df = yf.download("^VIX", start="2020-01-01", progress=False, auto_adjust=True)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
        csv_path = os.path.join(OUT, "vix_yahoo_raw.csv")
        close.rename("vix_close").to_csv(csv_path)
        return close, f"yfinance (archived vix_yahoo_raw.csv sha256={sha256_file(csv_path)})"
    except Exception as e:
        print(f"yfinance unavailable ({type(e).__name__}); falling back to Yahoo v8 chart API")
    p2 = int(time.time())
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX"
        f"?period1=1577836800&period2={p2}&interval=1d"
    )
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    raw = urllib.request.urlopen(req, timeout=60).read()
    raw_path = os.path.join(OUT, "vix_yahoo_raw.json")
    with open(raw_path, "wb") as f:
        f.write(raw)
    d = json.loads(raw)
    res = d["chart"]["result"][0]
    off = res["meta"]["gmtoffset"]
    ts = np.asarray(res["timestamp"]) + off
    cl = res["indicators"]["quote"][0]["close"]
    idx = pd.to_datetime(ts, unit="s").tz_localize(None).normalize()
    s = pd.Series(cl, index=idx, name="vix_close").dropna()
    s = s[~s.index.duplicated(keep="last")]
    return s, f"yahoo_v8_chart_api (archived vix_yahoo_raw.json sha256={sha256_file(raw_path)})"


def net_sharpe(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252.0))


def block_stats(x):
    x = np.asarray(x, dtype=float)
    return dict(
        n=len(x),
        mean=float(x.mean()) if len(x) else float("nan"),
        total=float(x.sum()) if len(x) else float("nan"),
        win=float((x > 0).mean()) if len(x) else float("nan"),
        sharpe=net_sharpe(x),
    )


def day_block_bootstrap(frame):
    g = frame.groupby("ny_date")["net"]
    sums = g.sum().to_numpy(dtype=float)
    cnts = g.count().to_numpy(dtype=float)
    n_blocks = len(sums)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n_blocks, size=(BOOT_ITERS, n_blocks))
    means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return float((means <= 0.0).mean()), n_blocks


def main():
    os.makedirs(OUT, exist_ok=True)
    vix, vix_src = fetch_vix()
    print(f"VIX: {len(vix)} rows {vix.index[0].date()}..{vix.index[-1].date()} via {vix_src}")

    m1 = pd.read_csv(TAPE)
    dt = pd.to_datetime(m1["time"], unit="s", utc=True).dt.tz_convert("America/New_York")
    m1["ny_date"] = pd.to_datetime(dt.dt.date)
    m1["wall"] = dt.dt.hour * 60 + dt.dt.minute
    cash = m1[(m1["wall"] >= 570) & (m1["wall"] < 960)].copy()
    d1 = cash.groupby("ny_date")["close"].last()
    uni = pd.Index(m1.loc[m1["wall"] == 570, "ny_date"].unique())
    d1 = d1[d1.index.isin(uni)]
    grid = d1.index.intersection(vix.index)
    px = d1.loc[grid]
    vx = vix.loc[grid]

    ret = np.log(px).diff()
    rv21 = ret.rolling(21).std(ddof=1) * np.sqrt(252.0)
    vrp_raw = vx - 100.0 * rv21
    vrp_shift = vrp_raw.shift(1)
    p50 = vrp_shift.rolling(252, min_periods=252).median()
    armed = vrp_shift.where((vrp_shift.notna()) & (p50.notna()) & (vrp_shift > p50)).dropna()
    p50_valid_from = p50.dropna().index[0]
    print(
        f"grid sessions={len(grid)} {grid[0].date()}..{grid[-1].date()} | "
        f"P50 valid from {p50_valid_from.date()} | armed={len(armed)}"
    )

    day_groups = {k: v.sort_values("time") for k, v in cash.groupby("ny_date")}
    rows = []
    void_last_bar = 0
    for d in armed.index:
        if d not in day_groups:
            continue
        b = day_groups[d]
        w = b["wall"].to_numpy()
        o = b["open"].to_numpy(dtype=float)
        h = b["high"].to_numpy(dtype=float)
        l = b["low"].to_numpy(dtype=float)
        c = b["close"].to_numpy(dtype=float)
        orm = (w >= 570) & (w < 585)
        if not orm.any():
            continue
        hi = h[orm].max()
        lo = l[orm].min()
        trig = -1
        side = 0
        for i in np.where(w >= 585)[0]:
            if c[i] > hi:
                trig, side = i, 1
                break
            if c[i] < lo:
                trig, side = i, -1
                break
        if trig < 0:
            continue
        if trig == len(w) - 1:
            void_last_bar += 1
            continue
        ei = trig + 1
        entry = o[ei]
        sl = lo if side == 1 else hi
        exit_px = None
        reason = None
        for i in range(ei, len(w)):
            if side == 1:
                if o[i] <= sl:
                    exit_px, reason = o[i], "STOP_GAP"
                    break
                if l[i] <= sl:
                    exit_px, reason = sl, "STOP"
                    break
            else:
                if o[i] >= sl:
                    exit_px, reason = o[i], "STOP_GAP"
                    break
                if h[i] >= sl:
                    exit_px, reason = sl, "STOP"
                    break
        if exit_px is None:
            exit_px, reason = c[-1], "EOD"
        gross = side * (exit_px - entry)
        rows.append(
            dict(
                ny_date=d,
                side="LONG" if side == 1 else "SHORT",
                trigger_wall=int(w[trig]),
                entry=entry,
                sl=sl,
                exit=exit_px,
                exit_reason=reason,
                gross=gross,
                net=gross - SPREAD,
            )
        )

    t = pd.DataFrame(rows)
    t["date_str"] = t["ny_date"].dt.strftime("%Y-%m-%d")
    cal = t[t["ny_date"] < OOS_START].copy()
    oos = t[t["ny_date"] >= OOS_START].copy()
    cal_armed = int((armed.index < OOS_START).sum())
    oos_armed = int((armed.index >= OOS_START).sum())
    cs = block_stats(cal["net"].to_numpy())
    os_ = block_stats(oos["net"].to_numpy())
    p_oos, n_blocks = day_block_bootstrap(oos)
    yr = (
        oos.assign(year=oos["ny_date"].dt.year)
        .groupby("year")["net"]
        .agg(["count", "sum", "mean"])
    )
    yr["win"] = oos.assign(win=oos["net"] > 0).groupby(oos["ny_date"].dt.year)["win"].mean()

    g1 = os_["n"] >= 40
    g2 = os_["sharpe"] > 0
    g3 = p_oos <= 0.20
    verdict = "PASS" if (g1 and g2 and g3) else "FAIL"

    sig_cols = ["date_str", "side", "trigger_wall", "entry", "sl", "exit", "exit_reason", "gross", "net"]
    oos[sig_cols].to_csv(os.path.join(OUT, "vrp_oos_signals.csv"), index=False)
    cal[sig_cols].to_csv(os.path.join(OUT, "vrp_cal_signals.csv"), index=False)

    meta = dict(
        study="vrp_orb_oos",
        hypothesis=45,
        generated_at=pd.Timestamp.utcnow().isoformat(),
        script_sha256=sha256_file(os.path.abspath(__file__)),
        seed=SEED,
        boot_iters=BOOT_ITERS,
        spread_floor=SPREAD,
        oos_start=str(OOS_START.date()),
        tape=dict(path=TAPE, rows=int(len(m1)), cash_days=int(len(uni)), first=str(grid[0].date()), last=str(grid[-1].date())),
        vix_source=vix_src,
        grid_sessions=int(len(grid)),
        p50_valid_from=str(p50_valid_from.date()),
        armed_cal=cal_armed,
        armed_oos=oos_armed,
        trades_cal=cs["n"],
        trades_oos=os_["n"],
        void_last_bar_trigger=void_last_bar,
        gates=dict(G_OOS1=bool(g1), G_OOS2=bool(g2), G_OOS3=bool(g3), p_oos3=p_oos, n_blocks=n_blocks),
        verdict=verdict,
    )
    with open(os.path.join(OUT, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    L = []
    L.append("REPORT - HYPOTHESIS #45: VRP-ORB STRICT OUT-OF-SAMPLE WALK-FORWARD VERIFICATION")
    L.append(f"Generated: {meta['generated_at']}  |  Verdict: {verdict}")
    L.append("")
    L.append("1. DATA INTEGRITY")
    L.append(f"   Tape: {meta['tape']['rows']} M1 rows, {meta['tape']['cash_days']} cash days, {meta['tape']['first']}..{meta['tape']['last']}")
    L.append(f"   VIX : {vix_src}")
    L.append(f"   Session grid: {len(grid)} sessions | P50 valid from {meta['p50_valid_from']}")
    L.append(f"   Armed sessions: cal={cal_armed} oos={oos_armed} | last-bar voids: {void_last_bar}")
    L.append("")
    L.append("2. CALIBRATION BLOCK (< 2024-01-01) - EXPLORATORY, NOT GATED")
    L.append(f"   trades={cs['n']}  armed->trade={100.0 * cs['n'] / max(cal_armed, 1):.1f}%")
    L.append(f"   net mean/trade={cs['mean']:.4f} pt  total={cs['total']:.2f} pt  win={100.0 * cs['win']:.1f}%  sharpe={cs['sharpe']:.3f}")
    L.append("")
    L.append("3. OUT-OF-SAMPLE BLOCK (>= 2024-01-01) - GATED")
    L.append(f"   trades={os_['n']}  armed->trade={100.0 * os_['n'] / max(oos_armed, 1):.1f}%")
    L.append(f"   net mean/trade={os_['mean']:.4f} pt  total={os_['total']:.2f} pt  win={100.0 * os_['win']:.1f}%  sharpe={os_['sharpe']:.3f}")
    mix = oos["exit_reason"].value_counts().to_dict() if os_["n"] else {}
    side_mix = oos["side"].value_counts().to_dict() if os_["n"] else {}
    L.append(f"   exits={mix}  sides={side_mix}")
    L.append("")
    L.append("4. IN-SAMPLE vs OUT-OF-SAMPLE DEGRADATION")
    L.append(f"   net mean/trade: cal {cs['mean']:.4f} -> oos {os_['mean']:.4f} pt (delta {os_['mean'] - cs['mean']:.4f})")
    L.append(f"   net sharpe    : cal {cs['sharpe']:.3f} -> oos {os_['sharpe']:.3f} (delta {os_['sharpe'] - cs['sharpe']:.3f})")
    L.append("")
    L.append("5. YEAR-BY-YEAR OOS NET (sign inspection, per SPEC 9.4)")
    for y, r in yr.iterrows():
        L.append(f"   {y}: n={int(r['count']):3d}  total={r['sum']:9.2f} pt  mean={r['mean']:7.4f}  win={100.0 * r['win']:.1f}%")
    L.append("")
    L.append("6. GATES (frozen in GATE_VRP_OOS.txt, OOS block only)")
    L.append(f"   G-OOS1 trades >= 40                     : {os_['n']:3d}      -> {'PASS' if g1 else 'FAIL'}")
    L.append(f"   G-OOS2 OOS net Sharpe > 0               : {os_['sharpe']:7.3f} -> {'PASS' if g2 else 'FAIL'}")
    L.append(f"   G-OOS3 bootstrap P(mean<=0) <= 0.20     : {p_oos:7.4f} -> {'PASS' if g3 else 'FAIL'}  ({n_blocks} day blocks, {BOOT_ITERS} resamples, seed {SEED})")
    L.append(f"   VERDICT: {verdict}")
    L.append("")
    L.append("7. HONESTY NOTES (SPEC 9.3)")
    L.append("   The 2024+ block is a temporal split, NOT a virgin holdout: the #31 family was")
    L.append("   screened over the full 2021-2026 tape (verdict FAIL at G-VRP3). A PASS here is")
    L.append("   necessary-but-not-sufficient; the Phase-2 review reopening path applies in full.")
    report = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "REPORT_VRP_OOS.txt"), "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
