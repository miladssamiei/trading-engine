# -*- coding: utf-8 -*-
"""
crt_exploration_phase1.py - Phase 1 signal-quality measurement for the
CRT (Candle Range Theory) "Model 1" entry on US500-tick, per
data/results/crt_exploration/SPEC_CRT_EXPLORATION.txt and
GATE_CRT_EXPLORATION.txt (pre-declared 2026-09-09, before this script
existed).

POST-TERMINATION EXPLORATORY: this study officially violates the Path C
termination rule from SPEC_SHORT_TERM_REVERSAL and cannot yield a
deployable EA regardless of outcome. Gates grade raw signal quality
only; no Phase 2 may follow.

Mechanism (long side, short mirrored):
  reference 4H candle = 08:00-12:00 NY block (M5 tape);
  Candle 1 = first 15m candle closing after 12:00 with Low < ref Low;
  Candle 2 = subsequent 15m candle within 3h sweeping Low(C1) but
  closing above it (turtle soup), with the deepening/re-arm rules of
  SPEC 3.2; entry at the open of the adjacent next 15m candle, only if
  it starts <= 15:00 NY; MFE/MAE measured on M5 to the 15:55 session
  close, scaled by continuous-tape Wilder ATR14(15m) at Candle 2.

Outputs (data/results/crt_exploration/):
  phase1_signals.csv, REPORT_CRT_EXPLORATION.txt
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\crt_exploration")
M5 = Path(r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present\us500tick_m5.csv")
NY = "America/New_York"

REF_START = 480
REF_END = 720
REF_MIN_BARS = 24
SCAN_FROM = 720
WINDOW_MIN = 180
ENTRY_CUTOFF = 900
SESSION_END = 960
CASH_OPEN_WALL = 570
THRESHOLDS = (0.50, 0.75, 1.00)
N_BOOT = 10000
BOOT_SEED = 20260909
COST_FLOOR_PT = 0.51


def wilder_atr(h, l, c):
    n = len(h)
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    if n > 1:
        tr[1:] = np.maximum.reduce([h[1:] - l[1:],
                                    np.abs(h[1:] - c[:-1]),
                                    np.abs(l[1:] - c[:-1])])
    atr = np.full(n, np.nan)
    if n >= 14:
        atr[13] = tr[:14].mean()
        for i in range(14, n):
            atr[i] = (atr[i - 1] * 13.0 + tr[i]) / 14.0
    return atr


def ts_str(bucket):
    return pd.to_datetime(bucket, unit="s", utc=True).tz_convert(NY).strftime(
        "%Y.%m.%d %H:%M")


def scan_side(daylist, side, ref_hi, ref_lo, w15, h15, l15, c15, atr15):
    st = {"arm_ev": 0, "trig_ev": 0, "cutoff": 0, "gap": 0, "atr": 0, "sig": 0}
    armed_any = False
    trig_any = False
    state = 0
    c1 = -1
    for k, i in enumerate(daylist):
        sw = int(w15[i])
        if sw < SCAN_FROM:
            continue
        cw = sw + 15
        hi_ = float(h15[i])
        lo_ = float(l15[i])
        cl_ = float(c15[i])
        sweep = (lo_ < ref_lo) if side == "L" else (hi_ > ref_hi)
        if state == 0:
            if sweep:
                state = 1
                c1 = i
                st["arm_ev"] += 1
                armed_any = True
            continue
        c1_close = int(w15[c1]) + 15
        if cw - c1_close > WINDOW_MIN:
            state = 0
            if sweep:
                state = 1
                c1 = i
                st["arm_ev"] += 1
            continue
        lvl = float(l15[c1]) if side == "L" else float(h15[c1])
        if side == "L":
            hit = lo_ < lvl and cl_ > lvl
            deep = (not hit) and lo_ < lvl and cl_ <= lvl
        else:
            hit = hi_ > lvl and cl_ < lvl
            deep = (not hit) and hi_ > lvl and cl_ >= lvl
        if hit:
            st["trig_ev"] += 1
            trig_any = True
            if k + 1 >= len(daylist) or int(w15[daylist[k + 1]]) != cw:
                st["gap"] += 1
                state = 0
                continue
            nxt = daylist[k + 1]
            if int(w15[nxt]) > ENTRY_CUTOFF:
                st["cutoff"] += 1
                state = 0
                continue
            if i < 14 or not np.isfinite(atr15[i]) or atr15[i] <= 0:
                st["atr"] += 1
                state = 0
                continue
            st["sig"] += 1
            return st, (c1, i, nxt), armed_any, trig_any
        if deep:
            c1 = i
    return st, None, armed_any, trig_any


def boot_day_p(dates, vals, seed=BOOT_SEED):
    dm = defaultdict(list)
    for d, v in zip(dates, vals):
        dm[d].append(v)
    means = np.array([np.mean(x) for x in dm.values()])
    rng = np.random.default_rng(seed)
    draws = rng.choice(means, size=(N_BOOT, means.size), replace=True).mean(axis=1)
    return float((draws <= 0).mean()), means.size


def fmt(x, nd=3):
    return f"{x:.{nd}f}"


def main():
    for f in (BASE / "SPEC_CRT_EXPLORATION.txt", BASE / "GATE_CRT_EXPLORATION.txt"):
        if not f.exists():
            sys.exit(f"REFUSING TO RUN: {f.name} missing "
                     "(SPEC/GATE must exist before the data loop)")
    print("SPEC/GATE present - protocol satisfied")

    m5 = pd.read_csv(M5)
    ny = pd.to_datetime(m5["time"], unit="s", utc=True).dt.tz_convert(NY)
    m5["wall"] = ny.dt.hour * 60 + ny.dt.minute
    m5["ny_date"] = ny.dt.strftime("%Y%m%d")
    t0 = pd.to_datetime(m5["time"].iloc[0], unit="s", utc=True)
    t1 = pd.to_datetime(m5["time"].iloc[-1], unit="s", utc=True)
    W = m5["wall"].to_numpy()
    D = m5["ny_date"].to_numpy()
    H = m5["high"].to_numpy(float)
    L = m5["low"].to_numpy(float)
    C = m5["close"].to_numpy(float)
    daym5 = {d: np.asarray(ix) for d, ix in m5.groupby("ny_date").indices.items()}

    m5x = m5.assign(bucket=(m5["time"] // 900) * 900)
    g = (m5x.groupby("bucket")
         .agg(open=("open", "first"), high=("high", "max"),
              low=("low", "min"), close=("close", "last"),
              nbars=("open", "size"))
         .reset_index()
         .sort_values("bucket"))
    ny15 = pd.to_datetime(g["bucket"], unit="s", utc=True).dt.tz_convert(NY)
    B15 = g["bucket"].to_numpy(np.int64)
    O15 = g["open"].to_numpy(float)
    H15 = g["high"].to_numpy(float)
    L15 = g["low"].to_numpy(float)
    C15 = g["close"].to_numpy(float)
    W15 = (ny15.dt.hour * 60 + ny15.dt.minute).to_numpy()
    D15 = ny15.dt.strftime("%Y%m%d").to_numpy()
    ATR15 = wilder_atr(H15, L15, C15)
    day15 = defaultdict(list)
    for i, d in enumerate(D15):
        day15[d].append(i)
    print(f"M5 bars {len(m5)}  {t0.date()}..{t1.date()}  "
          f"15m candles {len(g)}")

    signals = []
    fun = {"L": defaultdict(int), "S": defaultdict(int)}
    sig_days = {"L": set(), "S": set()}
    n_days = len(daym5)
    n_cash = 0
    n_elig = 0
    n_refshort_cash = 0
    refcounts = []

    for d in sorted(daym5):
        ix = daym5[d]
        w = W[ix]
        if not (w == CASH_OPEN_WALL).any():
            continue
        n_cash += 1
        rsel = (w >= REF_START) & (w < REF_END)
        nref = int(rsel.sum())
        if nref < REF_MIN_BARS:
            n_refshort_cash += 1
            continue
        n_elig += 1
        refcounts.append(nref)
        rix = ix[rsel]
        ref_hi = float(H[rix].max())
        ref_lo = float(L[rix].min())
        for side in ("L", "S"):
            st, tup, a_any, t_any = scan_side(
                day15.get(d, []), side, ref_hi, ref_lo,
                W15, H15, L15, C15, ATR15)
            f = fun[side]
            for kk, vv in st.items():
                f[kk] += vv
            f["arm_days"] += int(a_any)
            f["trig_days"] += int(t_any)
            if tup is None:
                continue
            c1i, c2i, ent = tup
            ew = int(W15[ent])
            ep = float(O15[ent])
            wix = ix[(W[ix] >= ew) & (W[ix] < SESSION_END)]
            hh = H[wix]
            ll = L[wix]
            if side == "L":
                mfe = float(hh.max() - ep)
                mae = float(ep - ll.min())
            else:
                mfe = float(ep - ll.min())
                mae = float(hh.max() - ep)
            scix = ix[W[ix] < SESSION_END]
            session_close = float(C[scix[-1]]) if len(scix) else float("nan")
            atr = float(ATR15[c2i])
            signals.append({
                "ny_date": d,
                "year": int(d[:4]),
                "side": "long" if side == "L" else "short",
                "ref_high": ref_hi,
                "ref_low": ref_lo,
                "c1_time": ts_str(B15[c1i]),
                "c1_level": float(L15[c1i]) if side == "L" else float(H15[c1i]),
                "c2_time": ts_str(B15[c2i]),
                "entry_time": ts_str(B15[ent]),
                "entry_price": ep,
                "atr14": atr,
                "session_close": session_close,
                "window_bars": int(len(wix)),
                "mfe_pt": round(mfe, 3),
                "mae_pt": round(mae, 3),
                "mfe_atr": round(mfe / atr, 4),
                "mae_atr": round(mae / atr, 4),
                "diff_pt": round(mfe - mae, 3),
            })
            sig_days[side].add(d)

    for side in ("L", "S"):
        f = fun[side]
        assert f["sig"] == f["trig_ev"] - f["cutoff"] - f["gap"] - f["atr"]

    df = pd.DataFrame(signals)
    df.to_csv(BASE / "phase1_signals.csv", index=False)

    rep = []
    rep.append("REPORT -- CRT (CANDLE RANGE THEORY) MODEL 1 SIGNAL-QUALITY "
               "EXPLORATION, PHASE 1")
    rep.append("Run 2026-09-09 by evaluate/crt_exploration_phase1.py per "
               "SPEC_CRT_EXPLORATION.txt / GATE_CRT_EXPLORATION.txt")
    rep.append("(pre-declared before the data loop).")
    rep.append("")
    rep.append("STATUS: POST-TERMINATION EXPLORATORY. This study "
               "officially violates the Path C termination rule "
               "pre-declared in")
    rep.append("SPEC_SHORT_TERM_REVERSAL. Regardless of the verdicts below, "
               "no EA, no Phase 2, no parameter work and no re-test may")
    rep.append("follow from it. The gates grade raw signal quality only.")
    rep.append("")
    rep.append("1. DATA AND CONSTRUCTS")
    rep.append(f"   M5 bars: {len(m5)}  range {t0.date()} .. {t1.date()} "
               f"(US500-tick, verified export)")
    rep.append(f"   15m candles aggregated: {len(g)}")
    rep.append(f"   days in tape: {n_days}")
    rep.append(f"   cash days (09:30 bar exists): {n_cash}")
    rep.append(f"   eligible days (cash AND >= {REF_MIN_BARS}/48 reference "
               f"bars): {n_elig}")
    rep.append(f"   cash days skipped for short reference block: "
               f"{n_refshort_cash}")
    rc = np.array(refcounts)
    rep.append(f"   reference-block M5 bar count (cash days): min {rc.min()}, "
               f"p25 {fmt(np.percentile(rc,25),1)}, median "
               f"{fmt(np.percentile(rc,50),1)}, p75 "
               f"{fmt(np.percentile(rc,75),1)}, max {rc.max()}")
    rep.append("")
    rep.append("2. FUNNEL (per side; S mirrored)")
    rep.append("   side   arm-days  trig-days  trig-ev  cutoff  gap  atr-drop"
               "  signals")
    for side, nm in (("L", "long"), ("S", "short")):
        f = fun[side]
        rep.append(f"   {nm:5s}  {f['arm_days']:8d}  {f['trig_days']:9d} "
                   f"{f['trig_ev']:8d}  {f['cutoff']:7d}  {f['gap']:3d}  "
                   f"{f['atr']:9d}  {f['sig']:7d}")
    ntot = fun["L"]["sig"] + fun["S"]["sig"]
    both = len(sig_days["L"] & sig_days["S"])
    rep.append(f"   combined signals: {ntot}  (days with both sides "
               f"fired: {both})")
    rep.append("")

    if df.empty:
        rep.append("3. POPULATION STATISTICS: none (no signals)")
        rep.append("4. GATES")
        rep.append("   G-C1: FAIL (no signals; unevaluable counts as FAIL)")
        rep.append("   G-C2: FAIL (no signals; unevaluable counts as FAIL)")
        rep.append("   INFORMATIONAL COST CHECK: mean(MFE-MAE) undefined; "
               "no cost claim possible.")
        rep.append("")
        rep.append("VERDICT: PHASE 1 FAIL. Post-termination exploratory "
                   "artifact; nothing follows from it.")
        (BASE / "REPORT_CRT_EXPLORATION.txt").write_text(
            "\n".join(rep) + "\n", encoding="utf-8")
        print("No signals - FAIL report written")
        return

    mfe = df["mfe_pt"].to_numpy()
    mae = df["mae_pt"].to_numpy()
    atr = df["atr14"].to_numpy()
    mfe_a = df["mfe_atr"].to_numpy()
    mae_a = df["mae_atr"].to_numpy()
    diff = mfe - mae
    dates = df["ny_date"].to_numpy()

    rep.append("3. POPULATION STATISTICS (points unless stated)")
    for nm, sel in (("ALL", np.ones(len(df), bool)),
                    ("long", (df["side"] == "long").to_numpy()),
                    ("short", (df["side"] == "short").to_numpy())):
        if sel.sum() == 0:
            rep.append(f"   {nm:5s}  n=0")
            continue
        rep.append(f"   {nm:5s}  n={int(sel.sum()):4d}  "
                   f"meanMFE {fmt(mfe[sel].mean())}  "
                   f"meanMAE {fmt(mae[sel].mean())}  "
                   f"mean(MFE-MAE) {fmt(diff[sel].mean())}  "
                   f"med(MFE-MAE) {fmt(np.median(diff[sel]))}  "
                   f"meanMFE/ATR {fmt(mfe_a[sel].mean())}  "
                   f"meanMAE/ATR {fmt(mae_a[sel].mean())}")
    rep.append("   yearly breakdown of (MFE-MAE), informational:")
    for y in sorted(df["year"].unique()):
        sel = (df["year"] == y).to_numpy()
        rep.append(f"     {y}: n={int(sel.sum()):4d}  "
                   f"mean {fmt(diff[sel].mean())}  "
                   f"median {fmt(np.median(diff[sel]))}")
    rep.append("")

    rep.append("4. GATE EVALUATION (GATE_CRT_EXPLORATION.txt, frozen)")
    g_c1 = ntot >= 150
    rep.append(f"   G-C1 SAMPLE SIZE: total signals {ntot} vs >= 150 -> "
               f"{'PASS' if g_c1 else 'FAIL'}")
    mean_mfe = float(mfe.mean())
    mean_mae = float(mae.mean())
    mean_diff = float(diff.mean())
    c2i = mean_mfe > mean_mae
    rep.append(f"   G-C2(i) mean(MFE) {fmt(mean_mfe)} > mean(MAE) "
               f"{fmt(mean_mae)} -> {'PASS' if c2i else 'FAIL'} "
               f"[ATR-scaled: {fmt(mfe_a.mean())} vs {fmt(mae_a.mean())}, "
               f"non-binding]")
    rep.append("   G-C2(ii) threshold dominance "
               "(count MFE >= t*ATR vs count MAE >= t*ATR):")
    c2ii = True
    for t in THRESHOLDS:
        cm = int((mfe >= t * atr).sum())
        ca = int((mae >= t * atr).sum())
        ok = cm > ca
        c2ii = c2ii and ok
        rep.append(f"     t={t:.2f}: MFE {cm:4d}  MAE {ca:4d}  -> "
                   f"{'PASS' if ok else 'FAIL'}")
    p_raw, ndays = boot_day_p(dates, diff)
    p_atr, _ = boot_day_p(dates, diff / atr, seed=BOOT_SEED + 1)
    c2iii = p_raw <= 0.20
    rep.append(f"   G-C2(iii) day-block bootstrap P(mean(MFE-MAE) <= 0) = "
               f"{p_raw:.4f} over {ndays} day-means, {N_BOOT} draws, seed "
               f"{BOOT_SEED} vs <= 0.20 -> {'PASS' if c2iii else 'FAIL'}")
    rep.append(f"     [ATR-normalized variant P = {p_atr:.4f}, "
               f"non-binding]")
    g_c2 = c2i and c2ii and c2iii
    rep.append(f"   G-C2 GROSS EDGE (i AND ii AND iii) -> "
               f"{'PASS' if g_c2 else 'FAIL'}")
    rep.append("")
    rep.append("5. INFORMATIONAL COST CHECK vs TRANSACTION-COST FLOOR")
    rep.append(f"   raw point expectancy proxy mean(MFE - MAE) = "
               f"{fmt(mean_diff)} pt")
    if mean_diff < COST_FLOOR_PT:
        rep.append(f"   {fmt(mean_diff)} pt < {COST_FLOOR_PT} pt confirmed "
                   f"real-tick median round-trip spread floor ->")
        rep.append("   FLAG: ECONOMICALLY DEAD BEFORE COSTS. The excursion "
                   "proxy does not clear the spread floor, before any")
        rep.append("   slippage, financing or execution considerations.")
    else:
        rep.append(f"   {fmt(mean_diff)} pt >= {COST_FLOOR_PT} pt floor -> "
                   "not flagged dead-before-costs. CAUTION: MFE/MAE are")
        rep.append("   excursion maxima and overstate simultaneously "
                   "achievable PnL; this is NOT an after-cost edge claim,")
        rep.append("   and the study is post-termination exploratory.")
    rep.append("")
    rep.append("6. HONESTY NOTES")
    rep.append("   - One long + one short sample per day maximum; first "
               "valid trigger per side (SPEC 3.4).")
    rep.append("   - ATR14 uses Candle 2's close (last completed candle at "
               "decision); excursions walk the finer M5 tape;")
    rep.append("     no lookahead past the entry decision anywhere.")
    rep.append("   - Single frozen configuration; no parameter was tuned "
               "and no alternative was tried after seeing numbers.")
    rep.append("   - No MT5/tick-model involvement; offline Python on the "
               "verified M5 export only.")
    rep.append("   - Post-termination exploratory artifact: FAIL verdicts "
               "are final; a PASS would still license nothing.")
    verdict = "PASS" if (g_c1 and g_c2) else "FAIL"
    rep.append("")
    rep.append(f"PHASE 1 VERDICT: {verdict} "
               f"(G-C1 {'PASS' if g_c1 else 'FAIL'}, "
               f"G-C2 {'PASS' if g_c2 else 'FAIL'}). "
               "Post-termination exploratory;")
    rep.append("TRANSITION TO PATH C: RESEARCH EXHAUSTED remains the "
               "governing state of this program; this artifact does not")
    rep.append("reopen it.")

    (BASE / "REPORT_CRT_EXPLORATION.txt").write_text("\n".join(rep) + "\n",
                                                     encoding="utf-8")
    print(f"signals {ntot} (long {fun['L']['sig']}, short {fun['S']['sig']})")
    print(f"mean MFE {fmt(mean_mfe)}  mean MAE {fmt(mean_mae)}  "
          f"mean diff {fmt(mean_diff)} pt  boot p {p_raw:.4f}")
    print(f"G-C1 {'PASS' if g_c1 else 'FAIL'}  G-C2 "
          f"{'PASS' if g_c2 else 'FAIL'}  -> PHASE 1 {verdict}")
    print(f"artifacts: phase1_signals.csv, REPORT_CRT_EXPLORATION.txt in "
          f"{BASE}")


if __name__ == "__main__":
    main()
