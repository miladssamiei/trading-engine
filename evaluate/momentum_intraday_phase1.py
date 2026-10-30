# -*- coding: utf-8 -*-
"""
momentum_intraday_phase1.py — Phase 1 signal-quality test for the
project-original intraday multi-horizon trend alignment hypothesis.

Per SPEC.md / GATE.md in data/results/momentum_intraday_signal_quality/
(written BEFORE this run): at every eligible M5 bar, score = sum of
sign(close - close_{h ago}) over horizons {30m,1h,2h,4h,12h,24h}
(score6) and the fast subset {30m,1h,2h,4h} (score4); forward window =
next 18 M5 bars (90 min) truncated at tape session end, min 6 bars;
direction-adjusted uncensored MFE/MAE in % of entry; buckets by |score|;
gate = dominance + ordering + temporal stability (GATE.md).

Arms: NY (primary; signal bars 09:30..15:55 NY open-time) and ALL24
(secondary, information only). Provenance label: project-original
hypothesis — NOT a test of Lempérière et al. 2014 or the video rubric.

Outputs: phase1_buckets.csv, phase1_yearly.csv, phase1_signals_ny.csv,
phase1_stats.json (gate evaluation).
"""

import csv
import json
import time as timemod
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\momentum_intraday_signal_quality")
M5 = Path(r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present\us500tick_m5.csv")

HORIZONS = [1800, 3600, 7200, 14400, 43200, 86400]     # 30m 1h 2h 4h 12h 24h
FAST = HORIZONS[:4]
FWD_BARS = 18                                          # 90 min
MIN_FWD = 6                                            # 30 min
THRESHOLDS = (0.10, 0.15, 0.20)
NY_LO, NY_HI = 570, 955                                # 09:30 .. 15:55 open-time
SESSION_GAP = 3600                                     # >60 min tape gap = session end


def main():
    t0 = timemod.time()
    df = pd.read_csv(M5)
    t = df["time"].to_numpy(dtype=np.int64)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)
    print(f"M5 bars: {n}  {pd.Timestamp(t[0], unit='s', tz='UTC')} .. "
          f"{pd.Timestamp(t[-1], unit='s', tz='UTC')}")

    # ---- session ids / session-end index (tape gap > 60 min) ----
    sess = np.zeros(n, dtype=np.int64)
    sess[0] = 0
    sess[1:] = np.cumsum(np.diff(t) > SESSION_GAP)
    sess_end = np.zeros(n, dtype=np.int64)
    last = n - 1
    for i in range(n - 1, -1, -1):
        if i == n - 1 or sess[i] != sess[i + 1]:
            last = i
        sess_end[i] = last

    # ---- NY wall minutes ----
    idx = pd.DatetimeIndex(pd.to_datetime(t, unit="s", utc=True))
    ny = idx.tz_convert("America/New_York")
    ny_min = ny.hour * 60 + ny.minute
    ny_year = ny.year.to_numpy()
    ny_date = ny.strftime("%Y-%m-%d")

    # ---- horizon directions (vectorized) ----
    dirs = {}
    for hsec in HORIZONS:
        j = np.searchsorted(t, t - hsec, side="right") - 1
        ok = j >= 0
        d = np.zeros(n, dtype=np.int8)
        d[ok] = np.sign(c[ok] - c[j[ok]])
        dirs[hsec] = d
    ties = np.zeros(n, dtype=bool)
    for hsec in HORIZONS:
        ties |= (dirs[hsec] == 0) & (np.arange(n) > 0)
    print(f"bars with any horizon tie: {int(ties.sum())} (expected ~0)")

    score6 = sum(dirs[x] for x in HORIZONS).astype(np.int8)
    score4 = sum(dirs[x] for x in FAST).astype(np.int8)

    # ---- eligibility ----
    fwd_b = np.minimum(np.arange(n) + FWD_BARS, sess_end)
    fwd_len = fwd_b - np.arange(n)
    anchor_ok = np.ones(n, dtype=bool)
    for hsec in HORIZONS:
        anchor_ok &= np.searchsorted(t, t - hsec, side="right") - 1 >= 0
    eligible_ny = anchor_ok & (~ties) & (ny_min >= NY_LO) & (ny_min <= NY_HI) & (fwd_len >= MIN_FWD)
    eligible_all = anchor_ok & (~ties) & (fwd_len >= MIN_FWD)
    print(f"eligible NY bars: {int(eligible_ny.sum())}   ALL24: {int(eligible_all.sum())}")

    # ---- forward excursions + first-touch ----
    def walk(sel):
        recs = []
        ii = np.flatnonzero(sel)
        for i in ii:
            e = c[i]
            b = fwd_b[i]
            wmax = -1e18
            wmin = 1e18
            ft_up = {T: -1 for T in THRESHOLDS}
            ft_dn = {T: -1 for T in THRESHOLDS}
            for k in range(i + 1, b + 1):
                hk, lk = h[k], l[k]
                if hk > wmax:
                    wmax = hk
                if lk < wmin:
                    wmin = lk
                up = (hk - e) / e * 100.0
                dn = (e - lk) / e * 100.0
                for T in THRESHOLDS:
                    if ft_up[T] < 0 and up >= T:
                        ft_up[T] = k - i
                    if ft_dn[T] < 0 and dn >= T:
                        ft_dn[T] = k - i
            up_exc = (wmax - e) / e * 100.0
            dn_exc = (e - wmin) / e * 100.0
            recs.append((int(i), e, int(b - i), up_exc, dn_exc,
                         {T: (1 if (ft_up[T] >= 0 and (ft_dn[T] < 0 or ft_up[T] < ft_dn[T])) else 0)
                          for T in THRESHOLDS}))
        return recs

    def summarize_arm(name, sel):
        recs = walk(sel)
        rows = []
        sig_rows = []
        per_bar = {}
        for (i, e, fl, upx, dnx, fts) in recs:
            s6 = int(score6[i]); s4 = int(score4[i])
            d6 = 1 if s6 > 0 else (-1 if s6 < 0 else 1)
            d4 = 1 if s4 > 0 else (-1 if s4 < 0 else 1)
            mfe6 = upx if d6 > 0 else dnx
            mae6 = dnx if d6 > 0 else upx
            mfe4 = upx if d4 > 0 else dnx
            mae4 = dnx if d4 > 0 else upx
            per_bar[i] = (s6, s4, mfe6, mae6, mfe4, mae4, fts, d6, e, fl,
                          int(ny_year[i]), str(ny_date[i]), int(t[i]))
            if name == "NY":
                sig_rows.append([
                    int(t[i]), str(ny_date[i]), int(ny_year[i]), s6, s4,
                    round(e, 3), fl, round(mfe6, 4), round(mae6, 4),
                    round(mfe4, 4), round(mae4, 4),
                    fts[0.10], fts[0.15], fts[0.20]])
        out = []
        for sname, scores_idx in (("score6", 0), ("score4", 1)):
            buckets = {}
            for v in per_bar.values():
                s = v[scores_idx]
                buckets.setdefault(abs(s), []).append(v)
            for bkey in sorted(buckets):
                g = buckets[bkey]
                nn = len(g)
                i_mfe = 2 + scores_idx * 2
                i_mae = 3 + scores_idx * 2
                mfs = [x[i_mfe] for x in g]
                mas = [x[i_mae] for x in g]
                row = {"arm": name, "score_set": sname, "abs_score": bkey, "n": nn,
                       "mean_mfe": round(sum(mfs) / nn, 4),
                       "mean_mae": round(sum(mas) / nn, 4),
                       "mean_diff": round((sum(mfs) - sum(mas)) / nn, 4)}
                for T in THRESHOLDS:
                    cm = sum(1 for v2 in mfs if v2 >= T)
                    ca = sum(1 for v2 in mas if v2 >= T)
                    row[f"mfe_ge_{int(T * 100):03d}"] = cm
                    row[f"mae_ge_{int(T * 100):03d}"] = ca
                if sname == "score6":
                    row["ft_win_015"] = round(sum(x[6][0.15] for x in g) / nn, 4)
                out.append(row)
        return out, per_bar, sig_rows

    ny_rows, ny_perbar, sig_rows = summarize_arm("NY", eligible_ny)
    all_rows, _, _ = summarize_arm("ALL24", eligible_all)

    all_out = ny_rows + all_rows
    with open(BASE / "phase1_buckets.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_out[0].keys()))
        w.writeheader()
        w.writerows(all_out)

    with open(BASE / "phase1_signals_ny.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "ny_date", "year", "score6", "score4", "entry",
                    "fwd_len", "mfe6", "mae6", "mfe4", "mae4",
                    "ft6_010", "ft6_015", "ft6_020"])
        w.writerows(sig_rows)

    # ---- yearly: top-bucket dominance at 0.15% ----
    yearly_rows = []
    for sname, top in (("score6", 6), ("score4", 4)):
        i_mfe = 2 if sname == "score6" else 4
        i_mae = 3 if sname == "score6" else 5
        per_year = {}
        for v in ny_perbar.values():
            if abs(v[0 if sname == "score6" else 1]) == top:
                per_year.setdefault(v[10], []).append(v)
        for y in sorted(per_year):
            g = per_year[y]
            mfs = [x[i_mfe] for x in g]
            mas = [x[i_mae] for x in g]
            yearly_rows.append({
                "score_set": sname, "year": y, "n": len(g),
                "mfe_ge_015": sum(1 for z in mfs if z >= 0.15),
                "mae_ge_015": sum(1 for z in mas if z >= 0.15),
                "mean_diff": round((sum(mfs) - sum(mas)) / len(g), 4)})
    with open(BASE / "phase1_yearly.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(yearly_rows[0].keys()))
        w.writeheader()
        w.writerows(yearly_rows)

    # ---- gate evaluation (NY arm) ----
    def brow(sname, bkey):
        for r in ny_rows:
            if r["score_set"] == sname and r["abs_score"] == bkey:
                return r
        return None

    g1 = int(eligible_ny.sum()) >= 50000
    min_bucket = min(r["n"] for r in ny_rows)
    g1b = min_bucket >= 500
    g2, g2_detail = True, []
    for bkey in (4, 6):
        r = brow("score6", bkey)
        dom = all(r[f"mfe_ge_{int(T * 100):03d}"] > r[f"mae_ge_{int(T * 100):03d}"]
                  for T in THRESHOLDS)
        md = r["mean_diff"] > 0
        g2 = g2 and dom and md
        g2_detail.append({"bucket": bkey, "dom_all_t": dom, "mean_diff": r["mean_diff"]})
    g3, g3_detail = True, {}
    for sname, bkeys in (("score6", (0, 2, 4, 6)), ("score4", (0, 2, 4))):
        vals = [brow(sname, b)["mean_diff"] for b in bkeys]
        ties_tolerated = 0
        ok = True
        for a, b2 in zip(vals, vals[1:]):
            if b2 <= a:
                if b2 - a > -0.005 and ties_tolerated == 0:
                    ties_tolerated += 1
                else:
                    ok = False
        g3 = g3 and ok
        g3_detail[sname] = {"mean_diff_by_bucket": vals, "ordered": ok}
    g4_rows = [r for r in yearly_rows if r["score_set"] == "score6"]
    g4_years_pos = sum(1 for r in g4_rows
                       if r["mfe_ge_015"] > r["mae_ge_015"])
    g4 = g4_years_pos >= 5

    score6_pass = g1 and g1b and g2 and g3 and g4
    g2_4 = all(brow("score4", 4)[f"mfe_ge_{int(T * 100):03d}"] > brow("score4", 4)[f"mae_ge_{int(T * 100):03d}"]
               for T in THRESHOLDS) and brow("score4", 4)["mean_diff"] > 0
    g4_4_rows = [r for r in yearly_rows if r["score_set"] == "score4"]
    g4_4 = sum(1 for r in g4_4_rows if r["mfe_ge_015"] > r["mae_ge_015"]) >= 5
    score4_pass = g1 and g1b and g2_4 and g3_detail["score4"]["ordered"] and g4_4

    verdict = ("PASS score6" if score6_pass else
               "PASS score4 (12h/24h horizons failed)" if score4_pass else
               "FAIL -> STOP (null result)")

    stats = {
        "eligible_ny": int(eligible_ny.sum()), "eligible_all24": int(eligible_all.sum()),
        "g1_counts": bool(g1 and g1b), "min_bucket_n": min_bucket,
        "g2_score6_dominance": g2_detail, "g2_pass": bool(g2),
        "g3_ordering": g3_detail, "g3_pass": bool(g3),
        "g4_score6_years_positive": g4_years_pos, "g4_total_years": len(g4_rows),
        "g4_pass": bool(g4),
        "score6_pass": bool(score6_pass), "score4_pass": bool(score4_pass),
        "verdict": verdict,
        "runtime_s": round(timemod.time() - t0, 1),
    }
    with open(BASE / "phase1_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    print("\n--- NY buckets (score6) ---")
    for r in ny_rows:
        if r["score_set"] == "score6":
            print(f"  |s|={r['abs_score']}: n={r['n']:>7}  mfe {r['mean_mfe']:.4f}  "
                  f"mae {r['mean_mae']:.4f}  diff {r['mean_diff']:+.4f}  "
                  f"015: {r['mfe_ge_015']:>6} vs {r['mae_ge_015']:>6}  "
                  f"ft015 {r.get('ft_win_015', '-')}")
    print("\n--- NY buckets (score4) ---")
    for r in ny_rows:
        if r["score_set"] == "score4":
            print(f"  |s|={r['abs_score']}: n={r['n']:>7}  mfe {r['mean_mfe']:.4f}  "
                  f"mae {r['mean_mae']:.4f}  diff {r['mean_diff']:+.4f}  "
                  f"015: {r['mfe_ge_015']:>6} vs {r['mae_ge_015']:>6}")
    print("\nGATE:", json.dumps({k: v for k, v in stats.items()
                                 if k in ("g1_counts", "g2_pass", "g3_pass",
                                          "g4_pass", "g4_score6_years_positive")}))
    print("VERDICT:", verdict)
    print(f"\nwrote phase1_buckets.csv, phase1_yearly.csv, phase1_signals_ny.csv, "
          f"phase1_stats.json  ({stats['runtime_s']}s)")


if __name__ == "__main__":
    main()
