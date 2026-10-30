"""Daily directional model per
data/results/daily_directional_model/SPEC_DAILY_DIRECTIONAL_MODEL.txt and
GATE_DAILY_DIRECTIONAL_MODEL.txt. CPCV with purging/embargo, two frozen
model classes, gates G-M1..G-M3. No deployment, no tester."""

import json
import math
import os
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.linear_model import ElasticNet

from daily_feature_engineering import SP_M5, sp_cash_from_m5

BASE = r"C:\Users\Milad\Desktop\RR\data\results\daily_directional_model"
FEATS_CSV = (r"C:\Users\Milad\Desktop\RR\data\results"
             r"\daily_feature_engineering\daily_features_stationary.csv")
HORIZONS = [1, 2, 3, 5]
N_GROUPS = 6
N_PATHS = 15
EMBARGO = 11
IC_FLOOR = 0.025
PBO_MAX = 0.20
SPREAD_PT = 0.51
SWAP_LONG_PT = 20.0
SWAP_SHORT_PT = 2.0
SEED = 42
MODELS = ["enet", "lgbm"]
ALPHAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
L1R = [0.1, 0.5, 0.9]

rep = []


def P(s=""):
    rep.append(str(s))
    print(s, flush=True)


def spearman(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 10:
        return float("nan")
    r = spearmanr(a[m], b[m]).statistic
    return float(r) if not np.isnan(r) else float("nan")


def purge_drop(spans):
    drop = set()
    for lo, hi, emb_hi in spans:
        for i in range(max(0, lo), emb_hi + 1):
            drop.add(i)
    return drop


def fit_enet(X, y, keep, gbr, h):
    def std_fit(idx):
        mu = X[idx].mean(axis=0)
        sd = X[idx].std(axis=0, ddof=0)
        sd[sd == 0] = 1.0
        return mu, sd

    uniq = sorted(set(gbr[i] for i in keep))
    best = None
    for alpha in ALPHAS:
        for l1r in L1R:
            ics = []
            for vg in uniq:
                vg_rows = np.arange(min(gbr), max(gbr) + 1)
                vg_rows = np.array(
                    [i for i in vg_rows if gbr.get(i) == vg])
                if len(vg_rows) == 0:
                    continue
                spans = [(int(vg_rows[0]) - h, int(vg_rows[-1]) + h,
                          int(vg_rows[-1]) + h + EMBARGO)]
                drop = purge_drop(spans)
                itr = np.array([i for i in keep
                                if gbr[i] != vg and i not in drop])
                iv = np.array([i for i in keep if gbr[i] == vg])
                if len(itr) < 50 or len(iv) < 50:
                    continue
                mu, sd = std_fit(itr)
                en = ElasticNet(alpha=alpha, l1_ratio=l1r,
                                max_iter=10000, tol=1e-8,
                                random_state=SEED)
                en.fit((X[itr] - mu) / sd, y[itr])
                pred = en.predict((X[iv] - mu) / sd)
                ics.append(spearman(pred, y[iv]))
            mic = float(np.nanmean(ics)) if ics else float("nan")
            if not np.isnan(mic) and (best is None or mic > best[0]):
                best = (mic, alpha, l1r)
    if best is None:
        best = (float("nan"), 0.1, 0.5)
    _, alpha, l1r = best
    mu, sd = std_fit(keep)
    en = ElasticNet(alpha=alpha, l1_ratio=l1r, max_iter=10000,
                    tol=1e-8, random_state=SEED)
    en.fit((X[keep] - mu) / sd, y[keep])

    def predict(Xte):
        return en.predict((Xte - mu) / sd)

    return predict, alpha, l1r


def fit_lgbm(X, y, keep):
    mdl = lgb.LGBMRegressor(
        objective="regression", max_depth=3,
        min_data_in_leaf=int(math.ceil(0.05 * len(keep))),
        feature_fraction=0.8, n_estimators=200, learning_rate=0.05,
        random_state=SEED, n_jobs=4, verbosity=-1)
    mdl.fit(X[keep], y[keep])
    return mdl.predict


def trade_series(dates, close, sig, h):
    rows = []
    i = 0
    while i + h < len(dates):
        s = int(sig.iloc[i])
        gross = 0.0
        fin = 0.0
        spread = 0.0
        if s != 0:
            gross = s * (close.iloc[i + h] - close.iloc[i])
            spread = SPREAD_PT
            nights = pd.date_range(dates.iloc[i],
                                   dates.iloc[i + h], freq="D")[1:]
            for d0 in nights:
                mult = 5 if d0.weekday() == 2 else 1
                fin += (SWAP_LONG_PT if s > 0
                        else -SWAP_SHORT_PT) * mult
        rows.append({"entry": dates.iloc[i].date(),
                     "exit": dates.iloc[i + h].date(),
                     "signal": s, "gross_pt": round(gross, 4),
                     "spread_pt": spread,
                     "financing_pt": round(fin, 4),
                     "net_pt": round(gross - spread - fin, 4),
                     "h": h})
        i += h
    return pd.DataFrame(rows)


def ann_sharpe(net, h):
    if len(net) < 3 or net.std(ddof=1) == 0:
        return float("nan")
    return float(net.mean() / net.std(ddof=1) * math.sqrt(252.0 / h))


def deflated_sharpe(net, sr_all, k):
    sr = float(net.mean() / net.std(ddof=1))
    n = len(net)
    sk = float(net.skew())
    ku = float(net.kurt())
    v = float(np.var(sr_all, ddof=1))
    if v <= 0 or k < 2:
        return None
    g = 0.5772156649
    sr0 = math.sqrt(v) * ((1 - g) * norm.ppf(1 - 1.0 / k)
                          + g * norm.ppf(1 - 1.0 / (k * math.e)))
    den = math.sqrt(max(1e-12, 1 - sk * sr
                        + (ku - 1) / 4.0 * sr * sr))
    return {"dsr": round(float(norm.cdf(
        (sr - sr0) * math.sqrt(n - 1) / den)), 5),
        "sr_hat": round(sr, 5), "sr0": round(sr0, 5),
        "skew": round(sk, 4), "kurt": round(ku, 4),
        "n_trials": k, "n_trades": int(n)}


def write_report(verdict_lines, summ):
    hdr = ["REPORT_DAILY_DIRECTIONAL_MODEL",
           f"run: {datetime.now(timezone.utc).isoformat()}",
           "CPCV: N=6 groups (np.array_split), 15 paths, purge "
           "[g_start-h, g_end+h] + embargo 11 rows per test group; "
           "the same purge+embargo applies to the inner LOGO folds of "
           "Model A. Model A: train-fold standardization, inner LOGO "
           "on mean Spearman IC over the pre-declared alpha/l1_ratio "
           "grid. Model B: LightGBM 4.6.0, max_depth=3, "
           "min_data_in_leaf=ceil(5% n_train), feature_fraction=0.8, "
           "200 trees, lr=0.05, seed 42, no search. Financing: "
           "-20/+2 pt per night, Wednesday-night starts x5, spread "
           "0.51 pt round trip.",
           "",
           "mean OOS IC per configuration (15 CPCV paths):",
           summ.round(5).to_string(),
           ""] + verdict_lines
    with open(os.path.join(BASE,
                           "REPORT_DAILY_DIRECTIONAL_MODEL.txt"),
              "w") as f:
        f.write("\n".join(rep) + "\n\n" + "\n".join(hdr) + "\n")


def main():
    feats = pd.read_csv(FEATS_CSV, index_col=0, parse_dates=True)
    feats = feats.sort_index()
    P(f"features: {feats.shape[0]} rows x {feats.shape[1]} cols "
      f"({feats.index.min().date()}..{feats.index.max().date()})")
    _, sp_close = sp_cash_from_m5(SP_M5)
    sp_close.index = pd.to_datetime(sp_close.index)
    sp_close = sp_close.sort_index()
    r_all = np.log(sp_close).diff()
    sigma = r_all.rolling(20).std(ddof=1).reindex(feats.index)
    close_f = sp_close.reindex(feats.index)
    if close_f.isna().any() or sigma.isna().any():
        raise SystemExit("price/sigma alignment failure")

    X = feats.values.astype(float)
    n = len(feats)
    groups = np.array_split(np.arange(n), N_GROUPS)
    P(f"group sizes: {[len(g) for g in groups]}")
    gbr = {}
    for gi, g in enumerate(groups):
        for i in g:
            gbr[int(i)] = gi
    pairs = [(a, b) for a in range(N_GROUPS)
             for b in range(a + 1, N_GROUPS)]
    assert len(pairs) == N_PATHS

    ic_rows = []
    pred_rows = []
    for H in HORIZONS:
        y = (np.log(close_f.shift(-H) / close_f)
             / (sigma * math.sqrt(H))).values
        valid = ~np.isnan(y)
        P(f"[h={H}] target valid rows: {int(valid.sum())} "
          f"(dropped {int((~valid).sum())})")
        for pi, (ga, gb) in enumerate(pairs):
            test_idx = np.array(sorted(
                i for i in np.concatenate([groups[ga], groups[gb]])
                if valid[i]))
            train_idx = np.array(
                [i for i in range(n)
                 if gbr[i] not in (ga, gb) and valid[i]])
            spans = [(int(groups[ga][0]) - H, int(groups[ga][-1]) + H,
                      int(groups[ga][-1]) + H + EMBARGO),
                     (int(groups[gb][0]) - H, int(groups[gb][-1]) + H,
                      int(groups[gb][-1]) + H + EMBARGO)]
            keep = np.array([i for i in train_idx
                             if i not in purge_drop(spans)])
            if len(keep) < 100:
                P(f"  path {pi} h={H}: train rows {len(keep)} < 100")
                continue
            fn_a, al, l1 = fit_enet(X, y, keep, gbr, H)
            pa = fn_a(X[test_idx])
            fn_b = fit_lgbm(X, y, keep)
            pb = fn_b(X[test_idx])
            ic_a = spearman(pa, y[test_idx])
            ic_b = spearman(pb, y[test_idx])
            ic_rows += [
                {"model": "enet", "h": H, "path": pi,
                 "test_groups": f"{ga}+{gb}", "n_train": len(keep),
                 "n_test": len(test_idx), "ic": ic_a,
                 "alpha": al, "l1_ratio": l1},
                {"model": "lgbm", "h": H, "path": pi,
                 "test_groups": f"{ga}+{gb}", "n_train": len(keep),
                 "n_test": len(test_idx), "ic": ic_b,
                 "alpha": None, "l1_ratio": None}]
            for i, ti in enumerate(test_idx):
                pred_rows += [
                    {"row": int(ti),
                     "date": feats.index[ti].date(), "path": pi,
                     "h": H, "model": "enet", "pred": pa[i],
                     "y": y[ti]},
                    {"row": int(ti),
                     "date": feats.index[ti].date(), "path": pi,
                     "h": H, "model": "lgbm", "pred": pb[i],
                     "y": y[ti]}]
            if (pi + 1) % 5 == 0:
                P(f"  h={H}: {pi + 1}/15 paths done "
                  f"(last IC enet={ic_a:.4f} lgbm={ic_b:.4f})")

    ics = pd.DataFrame(ic_rows)
    ics.to_csv(os.path.join(BASE, "cpcv_ic_results.csv"),
               index=False)
    pd.DataFrame(pred_rows).to_csv(
        os.path.join(BASE, "oos_predictions.csv"), index=False)

    summ = ics.groupby(["model", "h"])["ic"].agg(["mean", "std",
                                                  "count"])
    summ["t_stat"] = summ["mean"] / (summ["std"] / np.sqrt(
        summ["count"]))
    P("--- mean OOS IC per configuration ---")
    P(summ.round(4).to_string())

    P(f"G-M1 minimum IC floor (>= {IC_FLOOR}) ...")
    passers = []
    for (m_, h_), row in summ.iterrows():
        ok = row["mean"] >= IC_FLOOR
        P(f"  {m_} h={h_}: mean IC={row['mean']:.4f} "
          f"t={row['t_stat']:.3f} -> {'PASS' if ok else 'fail'}")
        if ok:
            passers.append((float(row["mean"]), m_, int(h_)))
    gm1 = len(passers) > 0
    if not gm1:
        write_report(
            [f"G-M1: FAIL - no configuration reached mean OOS IC >= "
             f"{IC_FLOOR}.",
             "Per SPEC 5.1 the study STOPS here: the features contain "
             "no tradable directional edge. G-M2/G-M3 not evaluated.",
             "VERDICT: FAIL (complete null deliverable)."], summ)
        P("G-M1 FAIL -> stopped per SPEC.")
        return

    passers.sort(reverse=True)
    _, M_SEL, H_SEL = passers[0]
    P(f"selected configuration: {M_SEL} h={H_SEL} "
      "(highest mean IC among G-M1 passers)")

    piv = ics.pivot_table(index="path", columns=["model", "h"],
                          values="ic")
    ranks = piv.rank(axis=1, method="average")
    omega = ranks / (len(piv.columns) + 1)
    lam = np.log(omega / (1 - omega))
    pbo_per_cfg = {f"{m_}|h={h_}": round(float((lam[m_, h_] < 0)
                                               .mean()), 4)
                   for m_, h_ in piv.columns}
    sel_pbo = float((lam[M_SEL, H_SEL] < 0).mean())
    P(f"G-M2 PBO for {M_SEL} h={H_SEL}: {sel_pbo:.4f} "
      f"(required <= {PBO_MAX})")
    gm2 = sel_pbo <= PBO_MAX
    if not gm2:
        write_report(
            [f"G-M1: PASS - {len(passers)} configuration(s) >= "
             f"{IC_FLOOR}.",
             f"G-M2: FAIL - selected {M_SEL} h={H_SEL} has "
             f"PBO={sel_pbo:.4f} > {PBO_MAX}.",
             "Per SPEC 5.2 the study FAILS at the highest gate "
             "reached; G-M3 not evaluated.",
             "VERDICT: FAIL (complete deliverable)."], summ)
        P("G-M2 FAIL -> stopped per SPEC.")
        return

    preds = pd.DataFrame(pred_rows)
    trade_net_by_cfg = {}
    sr_all = []
    for H in HORIZONS:
        for m_ in MODELS:
            sub = preds[(preds["model"] == m_) & (preds["h"] == H)]
            avgp = sub.groupby("row")["pred"].mean()
            sig = pd.Series(0.0, index=range(len(feats)))
            sig.loc[avgp.index] = np.sign(avgp)
            sig.index = feats.index
            ts = trade_series(pd.Series(feats.index),
                              pd.Series(close_f.values), sig, H)
            trade_net_by_cfg[(m_, H)] = ts
            net = ts["net_pt"]
            sr_all.append(float(net.mean() / net.std(ddof=1))
                          if net.std(ddof=1) > 0 else float("nan"))

    sel_ts = trade_net_by_cfg[(M_SEL, H_SEL)]
    sel_ts.to_csv(os.path.join(BASE, "g3_net_returns.csv"),
                  index=False)
    dsr = deflated_sharpe(sel_ts["net_pt"], sr_all, len(sr_all))
    gross_sr = ann_sharpe(sel_ts["gross_pt"], H_SEL)
    net_sr = ann_sharpe(sel_ts["net_pt"], H_SEL)
    yr = sel_ts.copy()
    yr["year"] = pd.to_datetime(yr["exit"]).dt.year
    per_year = yr.groupby("year")["net_pt"].sum()
    tot_cost = float(sel_ts["spread_pt"].sum()
                     + sel_ts["financing_pt"].sum())
    P(f"G-M3: trades={len(sel_ts)} gross SR={gross_sr:.4f} "
      f"net SR={net_sr:.4f} total costs={tot_cost:.1f} pt")
    P(f"per-year net pt: {per_year.round(1).to_dict()}")
    gm3 = (not np.isnan(net_sr)) and net_sr > 0

    with open(os.path.join(BASE, "pbo_evaluation.json"), "w") as f:
        json.dump({
            "selected_config": {"model": M_SEL, "h": H_SEL},
            "n_trials": len(sr_all),
            "pbo_per_config": pbo_per_cfg,
            "selected_pbo": sel_pbo,
            "selected_lambda_by_path":
                [round(float(v), 4) for v in lam[M_SEL, H_SEL]],
            "per_trade_sharpe_all_configs":
                {f"{m_}|h={h_}": (None if np.isnan(v)
                                  else round(v, 4))
                 for (m_, h_), v in
                 zip(trade_net_by_cfg.keys(), sr_all)},
            "deflated_sharpe": dsr,
            "gross_ann_sharpe": None if np.isnan(gross_sr)
            else round(gross_sr, 4),
            "net_ann_sharpe": None if np.isnan(net_sr)
            else round(net_sr, 4),
            "total_costs_pt": round(tot_cost, 2),
            "per_year_net_pt": {str(k): round(float(v), 2)
                                for k, v in per_year.items()}},
            f, indent=2)

    failed = [nm for nm, ok in (("G-M1", gm1), ("G-M2", gm2),
                                ("G-M3", gm3)) if not ok]
    if not failed:
        verdict = ("VERDICT: PASS - OOS IC clears the floor, PBO "
                   "within budget, net Sharpe positive after spread + "
                   "documented financing.")
    else:
        verdict = (f"VERDICT: FAIL (failed: {', '.join(failed)}) - "
                   "per SPEC 5.3 final for this artifact.")
    write_report([
        f"G-M1: PASS - {len(passers)} configuration(s) reached mean "
        f"IC >= {IC_FLOOR}.",
        f"G-M2: PASS - selected {M_SEL} h={H_SEL}, "
        f"PBO={sel_pbo:.4f} <= {PBO_MAX}.",
        f"G-M3: {'PASS' if gm3 else 'FAIL'} - {len(sel_ts)} trades, "
        f"gross SR={gross_sr:.4f}, net SR={net_sr:.4f} "
        f"(required > 0); total costs {tot_cost:.1f} pt.",
        verdict], summ)
    P(verdict)


if __name__ == "__main__":
    main()
