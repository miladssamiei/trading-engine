# -*- coding: utf-8 -*-
"""
metalabel_sp2l_phase1.py - Phase 1 of hypothesis #15: meta-label ML filter
(ALLOW/REDUCE/SKIP) on the frozen SP2L spike/gap EA, per
data/results/metalabel_sp2l/SPEC_METALABEL_SP2L.txt sections 3-6 and
GATE_METALABEL_SP2L.txt (M1-M5 pre-declared).

SP2L is NEVER modified: mt5_trades.csv is used exactly as validated
(1,343 trades, Sharpe 0.275 common-universe, maxDD 4.17%, 2025 -260.25).

Primary model: logistic regression (L2, C in {0.01, 0.1, 1.0} chosen by
nested walk-forward on strictly prior trades only, class_weight balanced,
solver lbfgs, max_iter 1000). Secondary: shallow random forest
(max_depth<=4, n_estimators=200, random_state=42), reported, never
substitutable.

Features (frozen, all causal, computed at the trade's SETUP bar t =
setup_time column, strictly before entry):
  F1 setup_type: "gap" if the setup's gap clause holds at 0.00% tolerance
     (literal price gap), else "spike" (gap passes only via the 0.04%
     tolerance). Pre-declared interpretation: the validated EA requires
     BOTH spike and gap clauses on every setup; the spike-vs-gap
     classification is the natural two-value split of SP2L's own gap
     condition.
  F2 session_tag: NY if NY wall-minute in [570, 930) (SP2L's own entry
     window); LON if London wall-minute in [480, 960) (SP2L's own LON
     variant window) and not NY; ASIAN otherwise. Pre-declared partition.
  F3 ema60_slope: EMA60(close_t) - EMA60(close_{t-1}) in points; EMA is
     the EA's incremental one (k = 2/61, seeded at first close, all M1
     bars) == pandas ewm(span=60, adjust=False).
  F4 adx_value: Wilder ADX(14) at bar t, verbatim from
     evaluate/sp2l_phase1.py calc_adx (source-exact).
  F5 regime_label: hypothesis #13 monthly label (regime_labels.csv,
     corrected UTC-midnight causal cutoff) for the month of the trade's
     entry date.
  F6 trailing_sp2l_sharpe_20: mean/std (ddof=1) of SP2L's last 20
     realized trade profits strictly before this one (0 if std==0 or no
     prior trades).
  F7 day_of_week: NY weekday of the entry date.
Label: y = 1 if realized net PnL > 0 else 0 (mt5_trades.csv exactly).

Decision rule (frozen): p >= 0.60 ALLOW (1.00 lot); 0.45 <= p < 0.60
REDUCE (0.50 lot); p < 0.45 SKIP. 2021 = burn-in: no predictions, all
trades taken unfiltered at full size. Single pre-declared exploratory
threshold pair (0.65/0.40) is tried ONLY if the primary fails, labeled
EXPLORATORY, never adoptable. Trades with uncomputable features are
taken unfiltered at full size (conservative; tallied).

Outputs (data/results/metalabel_sp2l/):
  metalabel_features.csv, metalabel_folds.csv,
  metalabel_filtered_trades.csv, yearly_breakdown_metalabel.csv,
  phase1_metalabel_stats.json
"""

import csv
import json
import math
import time as timemod
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, log_loss)

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\metalabel_sp2l")
SP2L = Path(r"C:\Users\Milad\Desktop\RR\data\results\sp2l_2021_2026")
ORB = Path(r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present")
M1_CSV = Path(r"C:\Users\Milad\Desktop\RR\data\results\vwap_2021_2026\us500tick_m1.csv")
RL_CSV = Path(r"C:\Users\Milad\Desktop\RR\data\results\regime_rotation\regime_labels.csv")

ACCOUNT = 10000.0
SQ252 = math.sqrt(252)
C_GRID = (0.01, 0.1, 1.0)
THR_ALLOW = 0.60
THR_REDUCE = 0.45
EXPL_THR = (0.65, 0.40)
FOLDS = [(2022, 1), (2022, 2), (2023, 1), (2023, 2), (2024, 1),
         (2024, 2), (2025, 1), (2025, 2), (2026, 1)]
FOLD_NAME = {1: "H1", 2: "H2"}
CAT_COLS = ["setup_type", "session_tag", "regime_label", "day_of_week"]
NUM_COLS = ["ema60_slope", "adx_value", "trailing_sharpe_20"]
SP2L_DOC = {"trades": 1343, "pnl": 281.84, "sharpe": 0.275, "dd_pct": 4.17}


def calc_adx(df, period=14):
    """Source-exact Wilder ADX (SP2L_Advanced_Bot.calculate_adx)."""
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0),
                        index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0),
                         index=df.index)
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()],
                   axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr
    den = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / den
    return dx.ewm(alpha=1.0 / period, adjust=False).mean().to_numpy(float)


def metrics(daily):
    s = pd.Series(daily, dtype=float)
    total = float(s.sum())
    cum = s.cumsum()
    mdd = float((cum.cummax() - cum).max())
    sd = float(s.std(ddof=1))
    return {"total": round(total, 2), "ret_pct": round(total / ACCOUNT * 100, 2),
            "sharpe": round(float(s.mean()) / sd * SQ252, 4) if sd > 0 else None,
            "mdd_pts": round(mdd, 2), "mdd_pct": round(mdd / ACCOUNT * 100, 2)}


def fold_of(ts):
    """Half-year fold key for a trade timestamp (NY). Returns
    (fold_name or None for 2021 burn-in, fold_start_ts)."""
    y, m = ts.year, ts.month
    if y < 2022:
        return None, None
    h = 1 if m <= 6 else 2
    return f"{y}{FOLD_NAME[h]}", pd.Timestamp(y, (h - 1) * 6 + 1, 1,
                                             tz="UTC")


def prep(Xtr, Xte):
    cat_tr = pd.get_dummies(Xtr[CAT_COLS]).astype(float)
    cat_te = pd.get_dummies(Xte[CAT_COLS]).astype(float).reindex(
        columns=cat_tr.columns, fill_value=0.0)
    mu = Xtr[NUM_COLS].mean()
    sd = Xtr[NUM_COLS].std().replace(0, 1.0)
    num_tr = (Xtr[NUM_COLS] - mu) / sd
    num_te = (Xte[NUM_COLS] - mu) / sd
    return (pd.concat([num_tr.reset_index(drop=True),
                       cat_tr.reset_index(drop=True)], axis=1),
            pd.concat([num_te.reset_index(drop=True),
                       cat_te.reset_index(drop=True)], axis=1),
            list(num_tr.columns) + list(cat_tr.columns))


def fit_lr(X, y, C):
    return LogisticRegression(C=C, class_weight="balanced", solver="lbfgs",
                              max_iter=1000).fit(X, y)


def fit_rf(X, y):
    return RandomForestClassifier(n_estimators=200, max_depth=4,
                                  random_state=42, n_jobs=-1).fit(X, y)


def choose_C(prior):
    """Nested C selection on STRICTLY prior trades via inner half-year
    walk-forward (mean log-loss; ties toward smaller C)."""
    halves = sorted(prior["half"].unique().tolist())
    results = {C: [] for C in C_GRID}
    for b in halves[1:]:
        b_start = pd.Timestamp(b[0], (b[1] - 1) * 6 + 1, 1, tz="UTC")
        tr_sel = prior[prior["open_t"] < b_start]
        te_sel = prior[prior["half"] == b]
        if len(tr_sel) < 50 or len(te_sel) < 20:
            continue
        if te_sel["y"].nunique() < 2 or tr_sel["y"].nunique() < 2:
            continue
        Xtr, Xte, _ = prep(tr_sel, te_sel)
        for C in C_GRID:
            m = fit_lr(Xtr, tr_sel["y"], C)
            results[C].append(log_loss(te_sel["y"], m.predict_proba(Xte)))
    best, best_v = None, None
    for C in C_GRID:
        if not results[C]:
            continue
        v = float(np.mean(results[C]))
        if best_v is None or v < best_v:
            best, best_v = C, v
    return (best if best is not None else 1.0), results


def diagnose(y_true, p, y_prob):
    y_hat = (p >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = None
    return {"n": int(len(y_true)),
            "accuracy": round(float(accuracy_score(y_true, y_hat)), 4),
            "precision": round(float(precision_score(y_true, y_hat,
                                                     zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_hat,
                                               zero_division=0)), 4),
            "auc": round(auc, 4) if auc is not None else None}


def decision(p, thr_allow=THR_ALLOW, thr_reduce=THR_REDUCE):
    if p >= thr_allow:
        return "ALLOW", 1.0
    if p >= thr_reduce:
        return "REDUCE", 0.5
    return "SKIP", 0.0


def main():
    t0 = timemod.time()
    out = {}

    # ---------------- trades + universe ----------------
    tr = pd.read_csv(SP2L / "mt5_trades.csv", sep=";")
    tr["open_t"] = pd.to_datetime(tr["open_time"], format="%Y.%m.%d %H:%M",
                                  utc=True)
    tr["setup_t"] = pd.to_datetime(tr["setup_time"], format="%Y.%m.%d %H:%M",
                                   utc=True)
    tr["profit"] = tr["profit"].astype(float)
    tr["y"] = (tr["profit"] > 0).astype(int)
    tr["ny_date"] = tr["open_t"].dt.tz_convert(
        "America/New_York").dt.strftime("%Y%m%d")
    tr["year"] = tr["ny_date"].str[:4].astype(int)
    tr["day_of_week"] = tr["open_t"].dt.tz_convert(
        "America/New_York").dt.dayofweek.map(
        {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"})
    tr["half"] = list(zip(tr["year"], (tr["open_t"].dt.month > 6).astype(int) + 1))

    od = pd.read_csv(ORB / "mt5_days.csv", sep=";", dtype=str)
    od["session_open"] = pd.to_numeric(od["session_open"], errors="coerce")
    od["prev_close"] = pd.to_numeric(od["prev_close"], errors="coerce")
    universe = sorted(od[(od["session_open"] > 0) |
                         (od["prev_close"] > 0)]["ny_date"])

    # unfiltered baseline (documented reproduction)
    unf_day = tr.groupby("ny_date")["profit"].sum().to_dict()
    unf_v = [unf_day.get(k, 0.0) for k in universe]
    m_unf = metrics(unf_v)
    unf_year = {y: round(float(g["profit"].sum()), 2)
                for y, g in tr.groupby("year")}
    rep = {"trades": len(tr), "pnl": m_unf["total"], "sharpe": m_unf["sharpe"],
           "mdd_pct": m_unf["mdd_pct"], "unf_2025": unf_year.get(2025)}
    rep_ok = (len(tr) == SP2L_DOC["trades"]
              and abs(m_unf["total"] - SP2L_DOC["pnl"]) <= 0.05
              and abs(m_unf["mdd_pct"] - SP2L_DOC["dd_pct"]) <= 0.03
              and m_unf["sharpe"] is not None
              and abs(m_unf["sharpe"] - SP2L_DOC["sharpe"]) <= 0.03)
    print(f"unfiltered reproduction: trades={len(tr)} pnl={m_unf['total']} "
          f"sharpe={m_unf['sharpe']} mdd%={m_unf['mdd_pct']} "
          f"2025={unf_year.get(2025)} ok={rep_ok}")
    if not rep_ok:
        print("REPRODUCTION GATE FAILED - abort (data problem, not a result)")
        with open(BASE / "phase1_metalabel_stats.json", "w") as fh:
            json.dump({"reproduction": rep, "ok": False}, fh, indent=2)
        return
    out["reproduction"] = rep
    out["baseline_unfiltered"] = m_unf
    out["baseline_unfiltered_yearly"] = unf_year

    # ---------------- M1 context arrays ----------------
    m1 = pd.read_csv(M1_CSV)
    t = pd.to_datetime(m1["time"], unit="s", utc=True)
    ny = t.dt.tz_convert("America/New_York")
    lon = t.dt.tz_convert("Europe/London")
    ny_wall = (ny.dt.hour * 60 + ny.dt.minute).to_numpy(np.int32)
    lon_wall = (lon.dt.hour * 60 + lon.dt.minute).to_numpy(np.int32)
    o = m1["open"].to_numpy(float)
    h = m1["high"].to_numpy(float)
    l = m1["low"].to_numpy(float)
    c = m1["close"].to_numpy(float)
    tsec = m1["time"].to_numpy(np.int64)
    ema = m1["close"].ewm(span=60, adjust=False).mean().to_numpy(float)
    adx = calc_adx(m1)
    del m1
    print(f"M1 context: {len(tsec)} bars, EMA/ADX computed "
          f"({timemod.time() - t0:.0f}s)")

    rl = pd.read_csv(RL_CSV)
    regime = dict(zip(rl["month"], rl["label"]))

    # ---------------- features ----------------
    tset = (tr["setup_t"].astype("int64") // 10**9).to_numpy(np.int64)
    idx = np.searchsorted(tsec, tset)
    bad = (idx >= len(tsec)) | (tsec[idx] != tset)
    n_miss = int(bad.sum())
    print(f"setup-bar matches: {len(tr) - n_miss}/{len(tr)} exact")

    i = idx
    btyp = (tr["type"] == "BUY").to_numpy()
    gap_lit = np.where(
        btyp,
        l[np.maximum(i - 1, 0)] > h[np.maximum(i - 3, 0)],
        h[np.maximum(i - 1, 0)] < l[np.maximum(i - 3, 0)])
    tr["setup_type"] = np.where(bad, "unknown", np.where(gap_lit, "gap", "spike"))
    w_ny = ny_wall[i]
    w_lon = lon_wall[i]
    tr["session_tag"] = np.where(
        bad, "unknown",
        np.where((w_ny >= 570) & (w_ny < 930), "NY",
                 np.where((w_lon >= 480) & (w_lon < 960), "LON", "ASIAN")))
    tr["ema60_slope"] = np.where(bad, np.nan,
                                 ema[i] - ema[np.maximum(i - 1, 0)])
    tr["adx_value"] = np.where(bad, np.nan, adx[i])
    tr["regime_label"] = tr["open_t"].dt.tz_convert(
        "America/New_York").dt.strftime("%Y-%m").map(regime)

    # trailing Sharpe over last 20 realized trades strictly before
    profits = tr["profit"].to_numpy(float)
    n_tr = len(tr)
    ts20 = np.zeros(n_tr)
    for k in range(n_tr):
        lo = max(0, k - 20)
        win = profits[lo:k]
        if len(win) >= 2 and win.std(ddof=1) > 0:
            ts20[k] = float(win.mean() / win.std(ddof=1))
    tr["trailing_sharpe_20"] = ts20

    feats = tr[["ticket", "open_time", "ny_date", "year", "half", "type",
                "profit", "y", "setup_type", "session_tag", "ema60_slope",
                "adx_value", "regime_label", "trailing_sharpe_20",
                "day_of_week"]].copy()
    feats["fold"] = [fold_of(ts_)[0] for ts_ in tr["open_t"]]
    feats.to_csv(BASE / "metalabel_features.csv", index=False)

    # ---------------- walk-forward models ----------------
    fold_rows = []
    preds = {}
    for model_name in ("lr", "rf"):
        preds[model_name] = {}
    diag = {"lr": {}, "rf": {}}
    chosen = {}
    for (fy, fh) in FOLDS:
        fname = f"{fy}{FOLD_NAME[fh]}"
        fstart = pd.Timestamp(fy, (fh - 1) * 6 + 1, 1, tz="UTC")
        fold_sel = tr["half"] == (fy, fh)
        prior = tr[tr["open_t"] < fstart].copy()
        te = tr[fold_sel].copy()
        print(f"\nfold {fname}: train={len(prior)} predict={len(te)}")
        if len(te) == 0:
            continue
        # ---- primary: nested C selection then fit ----
        C, inner_res = choose_C(prior)
        chosen[fname] = C
        Xtr, Xte, feat_cols = prep(prior, te)
        m_lr = fit_lr(Xtr, prior["y"], C)
        p_lr = m_lr.predict_proba(Xte)[:, 1]
        preds["lr"][fname] = (te.index.to_numpy(), p_lr)
        diag["lr"][fname] = diagnose(te["y"], p_lr, p_lr)
        fold_rows.append({"fold": fname, "model": "logistic_regression",
                          "C": C, "n_train": len(prior), "n_predict": len(te),
                          "intercept": round(float(m_lr.intercept_[0]), 6),
                          "coefficients": json.dumps(
                              {feat_cols[j]: round(float(m_lr.coef_[0][j]), 6)
                               for j in range(len(feat_cols))})})
        print(f"  LR C={C} diag={diag['lr'][fname]}")
        # ---- secondary: fixed RF ----
        m_rf = fit_rf(Xtr, prior["y"])
        p_rf = m_rf.predict_proba(Xte)[:, 1]
        preds["rf"][fname] = (te.index.to_numpy(), p_rf)
        diag["rf"][fname] = diagnose(te["y"], p_rf, p_rf)
        fold_rows.append({"fold": fname, "model": "random_forest",
                          "C": None, "n_train": len(prior),
                          "n_predict": len(te), "intercept": None,
                          "coefficients": None})
        print(f"  RF diag={diag['rf'][fname]}")

    with open(BASE / "metalabel_folds.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fold_rows[0].keys()))
        w.writeheader()
        w.writerows(fold_rows)

    # ---------------- decisions + filtered series ----------------
    def build_series(model_name, thr_allow=THR_ALLOW, thr_reduce=THR_REDUCE):
        dec = np.full(n_tr, 1.0)          # burn-in trades: full size
        kind = ["BURNIN"] * n_tr
        pvec = [None] * n_tr
        for fname, (idxarr, p) in preds[model_name].items():
            for k, pv in zip(idxarr, p):
                d, mult = decision(pv, thr_allow, thr_reduce)
                dec[k] = mult
                kind[k] = d
                pvec[k] = float(pv)
        sized = dec * profits
        day = defaultdict(float)
        for d, v in zip(tr["ny_date"], sized):
            day[d] += v
        series = [day.get(k, 0.0) for k in universe]
        return dec, kind, pvec, sized, metrics(series), day

    results = {}
    for model_name in ("lr", "rf"):
        dec, kind, pvec, sized, m, day = build_series(model_name)
        n_allow = sum(1 for k in kind if k == "ALLOW")
        n_reduce = sum(1 for k in kind if k == "REDUCE")
        n_skip = sum(1 for k in kind if k == "SKIP")
        n_remain = n_allow + n_reduce
        yearly = defaultdict(float)
        for d, v in day.items():
            yearly[int(d[:4])] += v
        results[model_name] = {"metrics": m, "decisions": kind,
                               "p": pvec, "sized": sized, "daily": day,
                               "tallies": {"allow": n_allow,
                                           "reduce": n_reduce,
                                           "skip": n_skip,
                                           "remain": n_remain,
                                           "burnin": sum(1 for k in kind
                                                         if k == "BURNIN")},
                               "yearly": {y: round(v, 2)
                                          for y, v in sorted(yearly.items())}}
        print(f"\n{model_name} filtered: {m} tallies="
              f"{results[model_name]['tallies']}")
        print(f"  yearly: {results[model_name]['yearly']}")

    # ---------------- gates (primary only) ----------------
    def gates(r, model_name):
        g = {}
        m = r["metrics"]
        g["M1"] = {"sharpe_f": m["sharpe"], "need_gt": SP2L_DOC["sharpe"],
                   "ok": m["sharpe"] is not None
                   and m["sharpe"] > SP2L_DOC["sharpe"]}
        g["M2"] = {"mdd_pct_f": m["mdd_pct"], "bound": SP2L_DOC["dd_pct"],
                   "ok": m["mdd_pct"] <= SP2L_DOC["dd_pct"]}
        g["M3"] = {"remain": r["tallies"]["remain"], "need": 100,
                   "ok": r["tallies"]["remain"] >= 100}
        fy = {y: v for y, v in r["yearly"].items() if y in (2021, 2022,
                                                            2023, 2024, 2025)}
        n_pos = sum(1 for v in fy.values() if v > 0)
        f2025 = r["yearly"].get(2025, 0.0)
        g["M4"] = {"positive_full_years": n_pos, "need": 4,
                   "f2025": f2025, "unf2025": unf_year.get(2025),
                   "ok": n_pos >= 4 and f2025 > unf_year.get(2025)}
        impr = {}
        for (fy, fh) in FOLDS:
            fname = f"{fy}{FOLD_NAME[fh]}"
            sel = tr["half"] == (fy, fh)
            f_impr = float((r["sized"][tr.index[sel].to_numpy()] -
                            tr.loc[sel, "profit"].to_numpy()).sum())
            impr[fname] = round(f_impr, 2)
        tot_impr = sum(impr.values())
        shares = {f: (impr[f] / tot_impr * 100 if tot_impr > 0 else None)
                  for f in impr}
        max_share = max((v for v in shares.values() if v is not None),
                        default=None)
        g["M5"] = {"fold_improvement": impr,
                   "total_improvement": round(tot_impr, 2),
                   "fold_share_pct": {f: (round(v, 2) if v is not None
                                          else None)
                                      for f, v in shares.items()},
                   "ok": tot_impr > 0 and max_share is not None
                   and max_share <= 50.0}
        passed = all(g[k]["ok"] for k in ("M1", "M2", "M3", "M4", "M5"))
        return g, passed

    g_lr, pass_lr = gates(results["lr"], "lr")
    g_rf, pass_rf = gates(results["rf"], "rf")
    for k in ("M1", "M2", "M3", "M4", "M5"):
        print(f"{k}: LR ok={g_lr[k]['ok']} {g_lr[k]}")
    print(f"\nPRIMARY (logistic regression) verdict: "
          f"{'PASS -> Phase 2' if pass_lr else 'FAIL -> STOP'}")

    # exploratory threshold pair ONLY if primary fails (labeled, not adoptable)
    expl = None
    if not pass_lr:
        dec_e, kind_e, pvec_e, sized_e, m_e, day_e = build_series(
            "lr", *EXPL_THR)
        expl = {"thresholds": list(EXPL_THR), "metrics": m_e,
                "tallies": {"remain": sum(1 for k in kind_e if k != "SKIP")},
                "note": "EXPLORATORY (primary failed); not adoptable"}
        print(f"\nEXPLORATORY thresholds {EXPL_THR}: {m_e} "
              f"tallies={expl['tallies']}")

    # ---------------- artifacts ----------------
    filt_rows = []
    for k in range(n_tr):
        for model_name in ("lr", "rf"):
            r = results[model_name]
            filt_rows.append({
                "ticket": tr["ticket"].iloc[k],
                "open_time": tr["open_time"].iloc[k],
                "model": model_name,
                "p_win": (round(r["p"][k], 6)
                          if r["p"][k] is not None else None),
                "decision": r["decisions"][k],
                "mult": (r["sized"][k] / profits[k]
                         if profits[k] != 0 else 0.0),
                "sized_pnl": round(r["sized"][k], 2),
                "profit": profits[k], "y": int(tr["y"].iloc[k])})
    pd.DataFrame(filt_rows).to_csv(BASE / "metalabel_filtered_trades.csv",
                                   index=False)

    y_rows = []
    for model_name, r in results.items():
        for y in sorted(r["yearly"]):
            y_rows.append({"model": model_name, "year": y,
                           "filtered_net": r["yearly"][y],
                           "unfiltered_net": unf_year.get(y),
                           "share_pct": round(
                               100.0 * r["yearly"][y] / r["metrics"]["total"], 1)
                           if r["metrics"]["total"] > 0 else None})
    pd.DataFrame(y_rows).to_csv(BASE / "yearly_breakdown_metalabel.csv",
                                index=False)

    stats = {
        "reproduction": rep,
        "baseline": {"unfiltered": m_unf,
                     "unfiltered_yearly": unf_year,
                     "sp2l_in_vol_cell_sharpe": 0.892,
                     "down_vol_cell_sharpe": 0.313,
                     "base_win_rate": round(float(tr["y"].mean()), 4),
                     "base_win_rate_pct": round(float(tr["y"].mean()) * 100, 2)},
        "feature_coverage": {"n_trades": n_tr,
                             "setup_bar_unmatched": n_miss,
                             "setup_type_dist":
                             dict(tr["setup_type"].value_counts()),
                             "session_tag_dist":
                             dict(tr["session_tag"].value_counts()),
                             "regime_label_dist":
                             dict(tr["regime_label"].value_counts())},
        "primary_lr": {"C_by_fold": chosen,
                       "decision_rule": {"allow": THR_ALLOW,
                                         "reduce": THR_REDUCE},
                       "results": results["lr"],
                       "gates": g_lr,
                       "verdict": "PASS -> Phase 2" if pass_lr
                       else "FAIL -> null result"},
        "secondary_rf": {"results": results["rf"], "gates": g_rf,
                         "verdict": "reported only (never substitutable)"},
        "exploratory_alt_thresholds": expl,
        "diagnostics": diag,
        "params": {"C_GRID": list(C_GRID), "rf_max_depth": 4,
                   "rf_n_estimators": 200, "rf_random_state": 42,
                   "universe_days": len(universe),
                   "folds": [f"{y}{FOLD_NAME[h]}" for y, h in FOLDS]},
        "runtime_s": round(timemod.time() - t0, 1)}
    with open(BASE / "phase1_metalabel_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2, default=str)
    print(f"\nwrote artifacts ({stats['runtime_s']}s)")


if __name__ == "__main__":
    main()
