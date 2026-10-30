import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\Milad\Desktop\RR")
STUDY = ROOT / "data" / "results" / "cross_asset_vrp_orb"
GVZ_JSON = STUDY / "gvz_yahoo_raw.json"
M1_CSV = ROOT / "data" / "results" / "xauusd_onboarding" / "xauusdtick_m1.csv"
US500_TRADES = ROOT / "data" / "results" / "orb_vrp_filter" / "orb_vrp_filtered_trades.csv"

SQ252 = float(np.sqrt(252.0))
OR_LO, OR_HI = 500, 515
TRIG_LO, TRIG_HI = 515, 960
GOLD_COST = 0.41
RV_N = 21
PCT_N = 252
G1_MIN = 100
G3_MIN_TPY = 120.0


def sharpe_per_trade(net_return):
    x = np.asarray(net_return, dtype=float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * SQ252)


def stream_stats(tr):
    n = len(tr)
    return {
        "n": int(n),
        "win": float((tr["net"] > 0).mean()) if n else float("nan"),
        "total_net": float(tr["net"].sum()) if n else float("nan"),
        "mean_net": float(tr["net"].mean()) if n else float("nan"),
        "sharpe": sharpe_per_trade(tr["net_return"]) if n else float("nan"),
    }


def main():
    p = print
    STUDY.mkdir(parents=True, exist_ok=True)

    p("[1/7] GVZ source ...")
    gvz_sha = hashlib.sha256(GVZ_JSON.read_bytes()).hexdigest()
    raw = json.loads(GVZ_JSON.read_text())
    res = raw["chart"]["result"][0]
    gts = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert("America/New_York")
    gdf = pd.DataFrame({
        "date": gts.strftime("%Y-%m-%d"),
        "gvz": [x if x is not None else np.nan for x in res["indicators"]["quote"][0]["close"]],
    }).dropna()
    gvz = pd.Series(gdf["gvz"].to_numpy(), index=gdf["date"].to_numpy())
    p(f"  GVZ daily {gvz.index[0]} .. {gvz.index[-1]}  n={len(gvz)}  sha256 {gvz_sha[:12]}...")

    p("[2/7] XAUUSD M1 tape ...")
    df = pd.read_csv(M1_CSV)
    ts = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("America/New_York")
    ny = ts.dt.strftime("%Y-%m-%d").to_numpy()
    mods = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    del df, ts

    p("[3/7] gold ORB extraction ...")
    change = np.flatnonzero(ny[1:] != ny[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(ny)]
    orb_funnel = {"n_days_tape": len(starts), "n_days_or": 0, "n_days_no_trigger": 0,
                  "n_degenerate": 0}
    d1_close = {}
    trades = []
    for s, e in zip(starts, ends):
        date = ny[s]
        d1_close[date] = float(c[e - 1])
        mod = mods[s:e]
        orm = (mod >= OR_LO) & (mod < OR_HI)
        if not orm.any():
            continue
        orb_funnel["n_days_or"] += 1
        orh = float(h[s:e][orm].max())
        orl = float(l[s:e][orm].min())
        scan = (mod >= TRIG_LO) & (mod < TRIG_HI)
        sig = -1
        direction = 0
        for k in np.flatnonzero(scan):
            if c[s + k] > orh:
                sig, direction = int(k), 1
                break
            if c[s + k] < orl:
                sig, direction = int(k), -1
                break
        if sig < 0:
            orb_funnel["n_days_no_trigger"] += 1
            continue
        entry = float(c[s + sig])
        sl = orl if direction == 1 else orh
        risk = (entry - sl) * direction
        if risk <= 0:
            orb_funnel["n_degenerate"] += 1
            continue
        exit_price, exit_kind = None, "TIME"
        for k in range(sig + 1, e - s):
            if mods[s + k] >= TRIG_HI:
                break
            bo = o[s + k]
            if direction == 1:
                if bo <= sl:
                    exit_price, exit_kind = float(bo), "STOP_GAP"
                    break
                if l[s + k] <= sl:
                    exit_price, exit_kind = float(sl), "STOP"
                    break
            else:
                if bo >= sl:
                    exit_price, exit_kind = float(bo), "STOP_GAP"
                    break
                if h[s + k] >= sl:
                    exit_price, exit_kind = float(sl), "STOP"
                    break
        if exit_price is None:
            last = max(k2 for k2 in range(e - s) if mods[s + k2] < TRIG_HI) if any(mods[s:e] < TRIG_HI) else None
            if last is None:
                orb_funnel["n_degenerate"] += 1
                continue
            exit_price, exit_kind = float(c[s + last]), "TIME"
        gross = direction * (exit_price - entry)
        net = gross - GOLD_COST
        trades.append({
            "ny_date": date, "dir": "LONG" if direction == 1 else "SHORT",
            "or_high": orh, "or_low": orl, "entry": entry, "sl": float(sl),
            "exit": exit_price, "exit_kind": exit_kind,
            "gross": gross, "net": net, "net_return": net / entry,
        })
    gold = pd.DataFrame(trades)
    p(f"  gold ORB trades: {len(gold)}  funnel {orb_funnel}")

    p("[4/7] VRP features + causal percentiles ...")
    d1 = pd.Series(d1_close).sort_index()
    lr = np.log(d1).diff()
    rv = lr.rolling(RV_N).std(ddof=1) * SQ252
    j = pd.DataFrame({"gvz": gvz, "rv": rv}).dropna()
    vrp = j["gvz"] - j["rv"] * 100.0
    S = vrp.shift(1)
    p50 = S.rolling(PCT_N, min_periods=PCT_N).median()
    lab = pd.Series("WARMUP", index=S.index)
    armed = S.notna() & p50.notna()
    lab[armed & (S > p50)] = "A"
    lab[armed & (S < p50)] = "D"
    lab[armed & (S == p50)] = "FLAT"
    gold["arm"] = gold["ny_date"].map(lab).fillna("NOFEAT")
    gold["vrp_shifted"] = gold["ny_date"].map(S)
    gold["p50"] = gold["ny_date"].map(p50)
    eligible = gold[gold["arm"].isin(["A", "D"])]
    arm_a = gold[gold["arm"] == "A"]
    arm_d = gold[gold["arm"] == "D"]
    first_elig = lab[lab.isin(["A", "D", "FLAT"])].index.min()
    p(f"  joined VRP dates {len(j)}, first arm-eligible {first_elig}")
    p(f"  gold trades: all {len(gold)} | eligible {len(eligible)} (A {len(arm_a)} / D {len(arm_d)} / FLAT {(gold['arm']=='FLAT').sum()} / warmup {(gold['arm']=='WARMUP').sum()} / nofeat {(gold['arm']=='NOFEAT').sum()})")

    p("[5/7] US500 #31 Arm A stream ...")
    u5 = pd.read_csv(US500_TRADES)
    u5 = u5.rename(columns={"type": "side"})
    u5["dir"] = u5["side"].map({"BUY": "LONG", "SELL": "SHORT"})
    u5["net_return"] = u5["net"] / u5["open_price"]
    u5a = u5[["ny_date", "dir", "open_price", "close_price", "net", "net_return"]].copy()
    us500_stats = stream_stats(u5a)

    p("[6/7] portfolio + gates ...")
    gold_stats_full = stream_stats(gold)
    gold_stats_base = stream_stats(eligible)
    gold_stats_a = stream_stats(arm_a)
    gold_stats_d = stream_stats(arm_d)
    gx1 = bool(len(arm_a) >= G1_MIN)
    gx2 = bool(gx1 and gold_stats_a["sharpe"] > gold_stats_base["sharpe"])
    members = [("US500_A31", u5a)]
    if gx1 and gx2:
        members.append(("XAUUSD_A", arm_a))
    all_tr = pd.concat([t.assign(stream=nm) for nm, t in members], ignore_index=True)
    by_day = all_tr.groupby(["stream", "ny_date"])["net_return"].sum().unstack(level=0)
    by_day = by_day.fillna(0.0)
    port_daily = by_day.sum(axis=1)
    port_sharpe = float(port_daily.mean() / port_daily.std(ddof=1) * SQ252) if len(port_daily) > 1 else float("nan")
    span_years = (pd.Timestamp(all_tr["ny_date"].max()) - pd.Timestamp(all_tr["ny_date"].min())).days / 365.25
    n_comb = len(all_tr)
    tpy = n_comb / span_years if span_years > 0 else float("nan")
    tpw = tpy / 52.0 if span_years > 0 else float("nan")
    gx3 = bool(tpy >= G3_MIN_TPY)
    corr = by_day.corr() if by_day.shape[1] > 1 else None
    overall = bool(gx1 and gx2 and gx3)

    p("[7/7] report / csv / meta ...")
    ts_now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = []
    A = L.append
    A("REPORT: CROSS-ASSET VRP-CONDITIONED ORB - PHASE 1 (HYPOTHESIS #40)")
    A(f"Generated: {ts_now}  |  script: evaluate/cross_asset_vrp_orb_phase1.py")
    A("SPEC/GATE: data/results/cross_asset_vrp_orb/{SPEC,GATE}_CROSS_ASSET_VRP.txt (frozen before code)")
    A("")
    A(f"VERDICT: {'PASS' if overall else 'FAIL'}   (G-XV1={gx1}, G-XV2={gx2}, G-XV3={gx3})")
    A("")
    A("1. FEASIBILITY AUDIT (declared in SPEC before code)")
    A("   USA100: NO local tape exists anywhere under data/results/ -> leg INFEASIBLE, VXN not fetched.")
    A("   Gold: verified onboarding M1 tape (2020-01..2026-09). GVZ fetched from Yahoo chart API (EOD,")
    A(f"   2020-01-02..2026-09-01, n={len(gvz)}), archived gvz_yahoo_raw.json, sha256 {gvz_sha[:16]}...")
    A("   US500: #31 Arm A trades reused verbatim (171 rows, net includes 0.51 pt spread).")
    A("")
    A("2. GOLD FEATURE PIPELINE")
    A(f"   D1 dates from M1: {len(d1)}; GVZ-joined VRP dates: {len(j)}; first arm-eligible date: {first_elig}")
    A("   RV = sqrt(252)*std(ln-ret, 21 obs, ddof=1); VRP = GVZ - RV*100; shifted s->s+1 on the joined grid;")
    A("   trailing_P50 over 252 obs inclusive of d (causal per #31 convention).")
    A("")
    A("3. GOLD ORB FUNNEL (OR [08:20,08:35) ET, first close-breakout/day, SL opposite OR side, 16:00 exit)")
    A(f"   days in tape: {orb_funnel['n_days_tape']} | with OR bars: {orb_funnel['n_days_or']} | no trigger: {orb_funnel['n_days_no_trigger']} | degenerate: {orb_funnel['n_degenerate']}")
    A(f"   trades: all {len(gold)} | arm-eligible {len(eligible)} = A {len(arm_a)} + D {len(arm_d)} (+FLAT {(gold['arm']=='FLAT').sum()}) | warmup {(gold['arm']=='WARMUP').sum()} | no-feature {(gold['arm']=='NOFEAT').sum()}")
    if len(gold):
        A(f"   exit mix: " + ", ".join(f"{k}={v}" for k, v in gold['exit_kind'].value_counts().items()))
    A("")
    A("4. PER-INSTRUMENT METRICS (net in $/unit; Sharpe = per-trade mean/std(net_return) x sqrt(252), #31 convention)")
    A(f"   GOLD all-history (descr):        n={gold_stats_full['n']}  win={gold_stats_full['win']*100:.1f}%  total={gold_stats_full['total_net']:.1f}  mean={gold_stats_full['mean_net']:.3f}  sharpe={gold_stats_full['sharpe']:.3f}")
    A(f"   GOLD baseline (arm-eligible):    n={gold_stats_base['n']}  win={gold_stats_base['win']*100:.1f}%  total={gold_stats_base['total_net']:.1f}  mean={gold_stats_base['mean_net']:.3f}  sharpe={gold_stats_base['sharpe']:.3f}")
    A(f"   GOLD Arm A (VRP > P50):          n={gold_stats_a['n']}  win={gold_stats_a['win']*100:.1f}%  total={gold_stats_a['total_net']:.1f}  mean={gold_stats_a['mean_net']:.3f}  sharpe={gold_stats_a['sharpe']:.3f}")
    A(f"   GOLD Arm D (control, VRP < P50): n={gold_stats_d['n']}  win={gold_stats_d['win']*100:.1f}%  total={gold_stats_d['total_net']:.1f}  mean={gold_stats_d['mean_net']:.3f}  sharpe={gold_stats_d['sharpe']:.3f}")
    if gold_stats_base["n"]:
        ec = (gold_stats_a["mean_net"] / gold_stats_base["mean_net"] - 1.0) * 100 if gold_stats_base["mean_net"] != 0 else float("nan")
        A(f"   edge concentration (Arm A mean vs baseline mean): {ec:+.1f}%")
    A(f"   US500 #31 Arm A (reused):        n={us500_stats['n']}  win={us500_stats['win']*100:.1f}%  total={us500_stats['total_net']:.1f}  mean={us500_stats['mean_net']:.3f}  sharpe={us500_stats['sharpe']:.3f}")
    A("")
    A("5. COMBINED PORTFOLIO (daily union calendar, net-return-on-notional streams)")
    A("   members: " + ", ".join(nm for nm, _ in members) + ("  (gold included: passed G-XV1+G-XV2)" if len(members) > 1 else "  (gold EXCLUDED: failed G-XV1/G-XV2)"))
    A(f"   span {all_tr['ny_date'].min()} .. {all_tr['ny_date'].max()} ({span_years:.2f}y), trades {n_comb}, active days {len(port_daily)}")
    A(f"   trades/year {tpy:.1f} ({tpw:.2f}/week)  vs G-XV3 floor {G3_MIN_TPY:.0f}/yr")
    A(f"   combined daily Net Sharpe: {port_sharpe:.3f}")
    if corr is not None:
        A("   daily-return correlation matrix:")
        for nm in corr.index:
            A("     " + nm + ": " + "  ".join(f"{v:+.3f}" for v in corr.loc[nm]))
    A("")
    A("6. GATES (frozen verbatim in GATE file)")
    A(f"   G-XV1 >= 1 new-instrument Arm A with >= {G1_MIN} trades post-warmup: gold A n={len(arm_a)} (USA100 infeasible) -> {gx1}")
    A(f"   G-XV2 qualifying Arm A Net Sharpe > baseline: {gold_stats_a['sharpe']:.3f} vs {gold_stats_base['sharpe']:.3f} -> {gx2}")
    A(f"   G-XV3 combined portfolio >= {G3_MIN_TPY:.0f} trades/yr: {tpy:.1f} -> {gx3}")
    A(f"   OVERALL PHASE-1: {'PASS' if overall else 'FAIL'}")
    A("")
    A("7. NOTES AND CAVEATS")
    A("   - G-XV2 IS A NEGATIVE-VS-NEGATIVE PASS: gold Arm A Sharpe -0.160 improves on the -0.352")
    A("     baseline but REMAINS negative-expectancy (mean -0.414 $/trade, win 33.8%, total -272.9);")
    A("     the conditioning shrinks losses, it does not create an edge (edge concentration is")
    A("     meaningless on two negative means). G-XV3's 149.3 tr/yr is carried by 659 gold trades")
    A("     with negative total PnL; the combined 0.214 daily Sharpe is attributable entirely to")
    A("     US500 #31 Arm A. NO ECONOMIC CASE FOR TRADING GOLD ORB EXISTS IN THIS ARTIFACT; the")
    A("     portfolio pass is a mechanical gate outcome, not a validated result.")
    A("   - Arm A conditioning is in-sample selection (#31 lineage); all portfolio metrics inherit that optimism;")
    A("     #31's overall FAIL (G-VRP3 win-rate gate) stands and is not re-litigated here.")
    A("   - Yahoo-sourced GVZ EOD is a third-party redistribution, not the official Cboe release (pinned, archived).")
    A("   - Gold fills: signal-bar-close entry, stop at SL with gap-through filled at the touching bar's open;")
    A("     0.41 round-trip cost floor; no swap. US500 stream is the #31 artifact with its own execution model.")
    A("   - Null result is final for this artifact.")
    report = "\n".join(L) + "\n"
    (STUDY / "REPORT_CROSS_ASSET_VRP_PHASE1.txt").write_text(report, encoding="utf-8")

    gold_out = gold[["ny_date", "dir", "or_high", "or_low", "entry", "sl", "exit", "exit_kind",
                     "gross", "net", "net_return", "vrp_shifted", "p50", "arm"]].copy()
    gold_out.insert(0, "instrument", "XAUUSD")
    u5_out = u5a[["ny_date", "dir", "open_price", "close_price", "net", "net_return"]].copy()
    u5_out.insert(0, "instrument", "US500")
    u5_out["arm"] = "A31"
    for col in gold_out.columns:
        if col not in u5_out.columns:
            u5_out[col] = ""
    for col in u5_out.columns:
        if col not in gold_out.columns:
            gold_out[col] = ""
    out_csv = pd.concat([gold_out[u5_out.columns], u5_out], ignore_index=True)
    out_csv.to_csv(STUDY / "cross_asset_vrp_trades.csv", index=False)

    meta = {
        "generated": ts_now,
        "script": "cross_asset_vrp_orb_phase1.py",
        "gvz_sha256": gvz_sha,
        "gvz_source": "Yahoo Finance chart API ^GVZ interval=1d (EOD redistribution)",
        "feasibility": {"usa100": "NO local tape; leg infeasible; VXN not fetched",
                        "gold": "xauusd_onboarding M1", "us500": "#31 Arm A reused"},
        "gold_orb_funnel": orb_funnel,
        "gold_stats": {"all": gold_stats_full, "baseline_eligible": gold_stats_base,
                       "arm_a": gold_stats_a, "arm_d": gold_stats_d},
        "us500_arm_a": us500_stats,
        "portfolio": {"members": [nm for nm, _ in members], "n_trades": n_comb,
                      "span_years": span_years, "trades_per_year": tpy, "trades_per_week": tpw,
                      "daily_sharpe": port_sharpe,
                      "correlation": None if corr is None else corr.to_dict()},
        "gates": {"G-XV1": gx1, "G-XV2": gx2, "G-XV3": gx3},
        "overall": overall,
        "params": {"rv_n": RV_N, "pct_n": PCT_N, "or_window_et": "[08:20,08:35)",
                   "trigger": "first close-breakout post-OR, 1/day", "exit": "SL opposite OR / 16:00 close",
                   "gold_cost": GOLD_COST, "sharpe_convention": "per-trade x sqrt(252)"},
    }
    (STUDY / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    p(f"VERDICT: {'PASS' if overall else 'FAIL'}  (G-XV1={gx1}, G-XV2={gx2}, G-XV3={gx3} tpy={tpy:.1f})")
    p(f"gold: baseline sharpe {gold_stats_base['sharpe']:.3f} | ArmA n={gold_stats_a['n']} sharpe {gold_stats_a['sharpe']:.3f} | ArmD sharpe {gold_stats_d['sharpe']:.3f}")
    p(f"portfolio: n={n_comb} over {span_years:.2f}y -> {tpy:.1f} tr/yr, daily sharpe {port_sharpe:.3f}")


if __name__ == "__main__":
    main()
