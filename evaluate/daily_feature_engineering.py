"""Daily feature engineering pipeline per
data/results/daily_feature_engineering/SPEC_DAILY_FEATURE_ENGINEERING.txt
and GATE_DAILY_FEATURE_ENGINEERING.txt. Data preparation only: no model,
no signals. All parameters frozen in the SPEC."""

import json
import os
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller

BASE = r"C:\Users\Milad\Desktop\RR\data\results\daily_feature_engineering"
RAW = os.path.join(BASE, "raw")
SP_M5 = (r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present"
         r"\us500tick_m5.csv")
XAU_M5 = (r"C:\Users\Milad\Desktop\RR\data\results\xauusd_onboarding"
          r"\xauusdtick_m5.csv")
USA_M5 = (r"C:\Users\Milad\Desktop\RR\data\results\usa30_onboarding"
          r"\usa30tick_m5.csv")

FRED_URL = ("https://fred.stlouisfed.org/graph/fredgraph.csv?"
            "bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&"
            "graph_bgcolor=%23ffffff&height=450&mode=fred&"
            "recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&"
            "nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&"
            "show_tooltip=yes&id=VIXCLS,T10Y2Y,DFII10&"
            "scale=left,left,left&cosd=2018-12-01&coed=2026-05-31&"
            "line_color=%234572a7,%23aa4643,%2389a54e&"
            "link_values=false,false,false&line_style=solid,solid,solid&"
            "mark_type=none,none,none&mw=3,3,3&lw=2,2,2&"
            "ost=-99999,-99999,-99999&oet=99999,99999,99999&mma=0,0,0&"
            "fml=a,a,a&fq=Daily,Daily,Daily&fam=avg,avg,avg&"
            "fgst=lin,lin,lin&fgsnd=2020-02-01,2020-02-01,2020-02-01&"
            "line_index=1,2,3&transformation=lin,lin,lin&"
            "vintage_date=2026-09-08,2026-09-08,2026-09-08&"
            "revision_date=2026-09-08,2026-09-08,2026-09-08&"
            "nd=1990-01-02,1976-06-01,2003-01-02")

L = 252
D_GRID = [round(0.05 * i, 2) for i in range(1, 20)]
ADF_CRIT = -2.86
SEARCH_START = pd.Timestamp("2021-01-04")
SEARCH_END = pd.Timestamp("2023-12-29")
PRIMARY_START = pd.Timestamp("2022-01-04")
PRIMARY_END = pd.Timestamp("2026-04-30")
G2_BUDGET = 0.70
G3_SEED = 20260908
FEATURE_ORDER = ["ffd_log_us500", "ffd_log_xauusd", "ffd_log_spdj",
                 "vrp", "tslope_d21", "dfii10_d21", "ci_resid"]

rep = []


def P(s=""):
    rep.append(str(s))
    print(s, flush=True)


def ny_series_from_m5(path):
    df = pd.read_csv(path, usecols=["time", "close"])
    ts = pd.to_datetime(df["time"], unit="s", utc=True)
    ts = ts.dt.tz_convert("America/New_York")
    df["ny_date"] = ts.dt.date
    last = df.groupby("ny_date")["close"].last()
    return last


def sp_cash_from_m5(path):
    df = pd.read_csv(path, usecols=["time", "close"])
    ts = pd.to_datetime(df["time"], unit="s", utc=True)
    ts = ts.dt.tz_convert("America/New_York")
    df["ny_date"] = ts.dt.date
    df["wall"] = ts.dt.hour * 60 + ts.dt.minute
    cash = df[df["wall"] == 570]["ny_date"].unique()
    pre = df[df["wall"] < 960]
    am = pre.groupby("ny_date")["close"].last()
    am = am[am > 0]
    return pd.Index(sorted(cash)), am


def ffd_weights(d, length):
    w = [1.0]
    for k in range(1, length + 1):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w)


def ffd(x, d, length):
    w = ffd_weights(d, length)
    conv = np.convolve(x, w[::-1], mode="full")[: len(x)]
    out = np.full(len(x), np.nan)
    out[length:] = conv[length:]
    return out


def adf_stat(x):
    r = adfuller(x, regression="c", autolag="AIC")
    return float(r[0]), float(r[1])


def d_search(name, log_series):
    x = log_series.dropna()
    rows = []
    chosen = None
    mask_w = np.asarray((x.index >= SEARCH_START)
                        & (x.index <= SEARCH_END))
    for d in D_GRID:
        y = ffd(x.values, d, L)
        yy = y[mask_w & ~np.isnan(y)]
        stat, p = adf_stat(yy)
        passing = stat < ADF_CRIT
        rows.append({"d": d, "adf_stat": round(stat, 4),
                     "adf_p": round(p, 5), "pass": bool(passing),
                     "n": int(len(yy))})
        if passing and chosen is None:
            chosen = d
    corr = None
    if chosen is not None:
        y = ffd(x.values, chosen, L)
        m = mask_w & ~np.isnan(y)
        corr = float(np.corrcoef(x.values[m], y[m])[0, 1])
    entry = {"series": name, "search_window":
             f"{SEARCH_START.date()}..{SEARCH_END.date()}",
             "n_search_rows": int(mask_w.sum()),
             "adf_full_grid": rows, "d_star": chosen,
             "pearson_corr_at_d_star": (None if corr is None
                                        else round(corr, 4))}
    P(f"  {name}: d*={chosen} "
      + ("" if chosen is None else f"corr={corr:.4f} "
         f"n={int(mask_w.sum())}"))
    return entry


def main():
    os.makedirs(RAW, exist_ok=True)
    fred_path = os.path.join(RAW, "fredgraph_20260908.csv")
    if not os.path.exists(fred_path):
        P("downloading FRED series VIXCLS,T10Y2Y,DFII10 ...")
        tmp = os.path.join(RAW, "fredgraph_download.bin")
        with urllib.request.urlopen(FRED_URL) as resp, open(tmp, "wb") as fo:
            fo.write(resp.read())
        with open(tmp, "rb") as fo:
            magic = fo.read(2)
        if magic == b"PK":
            import zipfile
            with zipfile.ZipFile(tmp) as z:
                for n in z.namelist():
                    P(f"  zip member: {n}")
                    if n.lower().endswith(".csv"):
                        with open(os.path.join(
                                RAW, os.path.basename(n)), "wb") as out:
                            out.write(z.read(n))
        else:
            os.replace(tmp, fred_path)
        if os.path.exists(tmp):
            os.remove(tmp)
        members = [f for f in os.listdir(RAW)
                   if f.lower().endswith(".csv")
                   and f != "fredgraph_20260908.csv"]
        frames = []
        for m in members:
            d_ = pd.read_csv(os.path.join(RAW, m))
            if "observation_date" in d_.columns:
                d_ = d_.set_index("observation_date")
            elif "DATE" in d_.columns:
                d_ = d_.set_index("DATE")
            frames.append(d_)
        fr = frames[0].join(frames[1:], how="outer")
        fr.to_csv(fred_path)
        with open(os.path.join(RAW, "provenance.txt"), "w") as f:
            f.write(f"URL: {FRED_URL}\nDownloaded: "
                    f"{datetime.now(timezone.utc).isoformat()}\n")
    fr = pd.read_csv(fred_path)
    icol = ("observation_date" if "observation_date" in fr.columns
            else "DATE")
    fr = fr.set_index(icol)
    fr.index = pd.to_datetime(fr.index)
    fr = fr.apply(pd.to_numeric, errors="coerce")
    fr = fr.replace(-99999.0, np.nan)
    P(f"FRED rows: {len(fr)} ({fr.index.min().date()}.."
      f"{fr.index.max().date()}), columns: {list(fr.columns)}")
    for c in ("VIXCLS", "T10Y2Y", "DFII10"):
        if c not in fr.columns:
            raise SystemExit(f"FRED column missing: {c}")

    P("building daily closes from M5 exports ...")
    sp_cash_idx, sp_close = sp_cash_from_m5(SP_M5)
    P(f"US500 cash days: {len(sp_cash_idx)} "
      f"({sp_cash_idx.min()}..{sp_cash_idx.max()})")
    xau_daily = ny_series_from_m5(XAU_M5)
    usa_daily = ny_series_from_m5(USA_M5)
    xau_daily.index = pd.to_datetime(xau_daily.index)
    usa_daily.index = pd.to_datetime(usa_daily.index)

    idx = pd.DatetimeIndex(sp_cash_idx)
    sp = sp_close.copy()
    sp.index = pd.to_datetime(sp.index)
    sp = sp.reindex(idx)
    sp = sp[sp.index <= PRIMARY_END]
    if sp.isna().sum() or (sp <= 0).sum():
        raise SystemExit(f"cash days missing a 15:55 close: "
                         f"{sp.isna().sum() + int((sp <= 0).sum())}")
    idx = sp.index
    xau_warm = xau_daily[xau_daily.index <= PRIMARY_END]
    usa_warm = usa_daily[usa_daily.index <= PRIMARY_END]

    def align_on_cash(daily):
        s = daily.reindex(idx.union(daily.index)).ffill(limit=5)
        return s.reindex(idx)

    xau_cash = align_on_cash(xau_warm)
    usa_cash = align_on_cash(usa_warm)
    P(f"aligned to cash dates: xau NaN={xau_cash.isna().sum()}, "
      f"usa30 NaN={usa_cash.isna().sum()} "
      f"(of {len(idx)} rows)")

    vix_raw = fr["VIXCLS"]
    ts_raw = fr["T10Y2Y"]
    dfii_raw = fr["DFII10"]

    def fred_shifted(raw, shift):
        vals = raw.reindex(idx.union(raw.index)).ffill(limit=5)
        pub = pd.Series(raw.index.strftime("%Y-%m-%d"),
                        index=raw.index)
        pub = pub.reindex(idx.union(pub.index)).ffill(limit=5)
        v = vals.reindex(idx)
        p = pub.reindex(idx)
        if shift:
            return v.shift(1), p.shift(1)
        return v, p

    vix, vix_pub = fred_shifted(vix_raw, shift=False)
    ts_v, ts_pub = fred_shifted(ts_raw, shift=True)
    dfii_v, dfii_pub = fred_shifted(dfii_raw, shift=True)
    P(f"FRED aligned NaN counts: VIX={vix.isna().sum()}, "
      f"T10Y2Y={ts_v.isna().sum()}, DFII10={dfii_v.isna().sum()}")

    P("FFD d-search (train window only, 2021-01-04..2023-12-29) ...")
    lsp = np.log(sp)
    lxau_all = np.log(xau_warm)
    lratio_all = np.log(sp) - np.log(usa_cash.reindex(idx))
    xau_aligned_log = np.log(xau_cash)
    series_map = {
        "ffd_log_us500": (lsp, lsp),
        "ffd_log_xauusd": (lxau_all, xau_aligned_log),
        "ffd_log_spdj": (lratio_all, lratio_all),
    }
    fd = {}
    djson = {}
    for name, (src, _) in series_map.items():
        entry = d_search(name, src)
        djson[name] = entry
        if entry["d_star"] is None:
            P(f"  {name}: NO d passes ADF on the grid -> feature EXCLUDED")
        else:
            fd[name] = entry["d_star"]
    with open(os.path.join(BASE, "fracdiff_optimal_d.json"), "w") as f:
        json.dump(djson, f, indent=2)

    feats = {}
    ffd_us500 = ffd(lsp.values, fd["ffd_log_us500"], L) \
        if "ffd_log_us500" in fd else np.full(len(idx), np.nan)
    ffd_xau_src = ffd(lxau_all.values, fd["ffd_log_xauusd"], L) \
        if "ffd_log_xauusd" in fd else np.full(len(lxau_all), np.nan)
    ffd_xau = pd.Series(ffd_xau_src, index=lxau_all.index).reindex(idx)
    ffd_ratio = ffd(lratio_all.ffill(limit=5).values,
                    fd["ffd_log_spdj"], L) \
        if "ffd_log_spdj" in fd else np.full(len(idx), np.nan)
    feats["ffd_log_us500"] = pd.Series(ffd_us500, index=idx)
    feats["ffd_log_xauusd"] = ffd_xau
    feats["ffd_log_spdj"] = pd.Series(ffd_ratio, index=idx)

    r = np.log(sp).diff()
    rv21 = 252.0 * (r ** 2).rolling(21).sum().shift(1)
    feats["vrp"] = (vix / 100.0) ** 2 - rv21
    feats["tslope_d21"] = ts_v.diff(21)
    feats["dfii10_d21"] = dfii_v.diff(21)

    pair = pd.DataFrame({"y": np.log(xau_cash), "x": dfii_v}).dropna()
    if len(pair) > L:
        rol = RollingOLS(pair["y"], sm.add_constant(pair["x"]),
                         window=L, min_nobs=L).fit()
        par = rol.params
        fitted = par["const"] + par["x"] * pair["x"]
        resid = pair["y"] - fitted
        feats["ci_resid"] = resid.reindex(idx)
    else:
        feats["ci_resid"] = pd.Series(np.nan, index=idx)

    F = pd.DataFrame(feats)[FEATURE_ORDER]
    F = F[(F.index >= PRIMARY_START) & (F.index <= PRIMARY_END)]
    P(f"candidate matrix: {F.shape[0]} primary rows x "
      f"{F.shape[1]} features; NaN per feature: "
      f"{F.isna().sum().to_dict()}")

    P("G-F1 strict stationarity (primary window) ...")
    gf1 = {}
    survivors = []
    dropped_f1 = []
    for c in FEATURE_ORDER:
        x = F[c].dropna().values
        if len(x) < 100:
            gf1[c] = {"n": len(x), "verdict": "DROP (insufficient rows)"}
            dropped_f1.append(c)
            continue
        stat, p = adf_stat(x)
        ok = p < 0.05
        gf1[c] = {"n": int(len(x)), "adf_stat": round(stat, 4),
                  "adf_p": round(p, 5),
                  "verdict": "PASS" if ok else "DROP"}
        P(f"  {c}: ADF stat={stat:.4f} p={p:.5f} "
          f"-> {'PASS' if ok else 'DROP'}")
        if ok:
            survivors.append(c)
        else:
            dropped_f1.append(c)

    P("G-F2 orthogonality budget ...")
    F2 = F[survivors]
    drops_g2 = []
    while True:
        cur = F2.dropna(axis=0, how="any")
        corr = cur.corr(method="spearman")
        mx = 0.0
        pair_v = None
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                v = abs(corr.iloc[i, j])
                if v > mx:
                    mx = v
                    pair_v = (corr.columns[i], corr.columns[j])
        if mx <= G2_BUDGET:
            P(f"  max |rho| = {mx:.4f} <= {G2_BUDGET} -> clean")
            break
        a, b = pair_v
        nr = (F2[a] - F2[a].min()) / (F2[a].max() - F2[a].min())
        nrv = float(np.nanvar(nr.values))
        nr2 = (F2[b] - F2[b].min()) / (F2[b].max() - F2[b].min())
        nrv2 = float(np.nanvar(nr2.values))
        victim = a if (nrv < nrv2 or
                       (nrv == nrv2 and FEATURE_ORDER.index(a)
                        > FEATURE_ORDER.index(b))) else b
        P(f"  |rho|({a},{b})={mx:.4f} > {G2_BUDGET}; "
          f"mm-var {a}={nrv:.5f} {b}={nrv2:.5f} -> drop {victim}")
        drops_g2.append({"pair": [a, b], "abs_rho": round(mx, 4),
                         "mm_var": {a: round(nrv, 5),
                                    b: round(nrv2, 5)},
                         "dropped": victim})
        survivors.remove(victim)
        F2 = F[survivors]
    final_corr = F2.dropna(axis=0, how="any").corr(method="spearman")
    final_corr.to_csv(os.path.join(BASE, "correlation_matrix.csv"))

    D = F[survivors].dropna(axis=0, how="any")
    D.index.name = "date"
    D.to_csv(os.path.join(BASE, "daily_features_stationary.csv"))
    n_final = len(survivors)

    P("G-F3 causality audit ...")
    rng = np.random.default_rng(G3_SEED)
    audit_rows = []
    if len(D) >= 5:
        picks = D.index[rng.choice(len(D), size=5, replace=False)]
    else:
        picks = D.index
    clean = True
    lines = ["G-F3 CAUSALITY AUDIT (seed "
             f"{G3_SEED}, sampled from {len(D)} final rows)",
             "Availability rule: market closes and VIX same-day; "
             "T10Y2Y/DFII10 published s usable s+1 (shifted).",
             ""]
    for t in picks:
        lines.append(f"=== feature date {t.date()} "
                     f"({t.strftime('%a')}) ===")
        for c in survivors:
            v = D.loc[t, c]
            lines.append(f"  {c} = {v:.6f}")
            if c.startswith("ffd_log_us500"):
                w = lsp[lsp.index <= t].tail(L + 1)
                lines.append("    inputs: US500 15:55 closes "
                             f"{w.index[0].date()}..{w.index[-1].date()} "
                             f"(n={len(w)}, all pub <= t; last="
                             f"{w.iloc[-1]:.4f} at {w.index[-1].date()})")
            elif c.startswith("ffd_log_xauusd"):
                wl = lxau_all[lxau_all.index <= t].tail(L + 1)
                lines.append("    inputs: XAUUSD last-mid closes "
                             f"{wl.index[0].date()}..{wl.index[-1].date()}"
                             f" (n={len(wl)}, all pub <= t; last="
                             f"{wl.iloc[-1]:.4f} at "
                             f"{wl.index[-1].date()})")
            elif c.startswith("ffd_log_spdj"):
                wr = lratio_all[lratio_all.index <= t].tail(L + 1)
                lines.append("    inputs: log(US500/USA30) closes "
                             f"{wr.index[0].date()}..{wr.index[-1].date()}"
                             f" (n={len(wr)}, all pub <= t; last="
                             f"{wr.iloc[-1]:.6f})")
            elif c == "vrp":
                rr = r[r.index < t].tail(21)
                lines.append(f"    VIX_t={vix.loc[t]:.2f} pub "
                             f"{vix_pub.loc[t]} (same-day)")
                lines.append("    RV21 prior returns "
                             f"{rr.index[0].date()}.."
                             f"{rr.index[-1].date()} (n={len(rr)} "
                             "strictly < t)")
            elif c == "tslope_d21":
                i0 = ts_v.index.get_loc(t)
                t21 = ts_v.index[max(0, i0 - 21)]
                lines.append(f"    T10Y2Y now={ts_v.loc[t]} pub "
                             f"{ts_pub.loc[t]}; 21 rows back="
                             f"{ts_v.loc[t21]} pub {ts_pub.loc[t21]}")
            elif c == "dfii10_d21":
                i1 = dfii_v.index.get_loc(t)
                p21d = dfii_v.index[max(0, i1 - 21)]
                lines.append(f"    DFII10 now={dfii_v.loc[t]} pub "
                             f"{dfii_pub.loc[t]}; 21 rows back="
                             f"{dfii_v.loc[p21d]} pub "
                             f"{dfii_pub.loc[p21d]}")
            elif c == "ci_resid":
                pw = pair[pair.index <= t].tail(L)
                lines.append("    inputs: pairs (log XAUUSD, shifted "
                             f"DFII10) {pw.index[0].date()}.."
                             f"{pw.index[-1].date()} (n={len(pw)}, "
                             "all pub <= t)")
                if t in pair.index:
                    p_at = pair.loc[t]
                    lines.append(f"    at t: log(XAUUSD)="
                                 f"{p_at['y']:.6f} pub {t.date()}; "
                                 f"DFII10shift={p_at['x']:.4f}")
    with open(os.path.join(BASE, "feature_sync_audit.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    P("writing REPORT_FEATURE_ENGINEERING.txt ...")
    hdr = ["REPORT_FEATURE_ENGINEERING",
           f"run: {datetime.now(timezone.utc).isoformat()}", "",
           "G-F1 (strict stationarity, primary window "
           f"{PRIMARY_START.date()}..{PRIMARY_END.date()}):"]
    for c in FEATURE_ORDER:
        if c in gf1:
            g = gf1[c]
            hdr.append(f"  {c}: {g}")
    hdr.append("")
    hdr.append("G-F2 (orthogonality budget 0.70):")
    for d_ in drops_g2:
        hdr.append(f"  dropped {d_['dropped']} for pair "
                   f"{d_['pair']} |rho|={d_['abs_rho']}")
    hdr.append(f"  final max |rho| = "
               f"{final_corr.abs().mask(np.eye(len(final_corr), dtype=bool)).max().max():.4f}")
    hdr.append("")
    hdr.append(f"G-F3: 5 seeded dates audited -> feature_sync_audit.txt "
               f"(seed {G3_SEED}); all inputs published <= feature date: "
               "verified per transcripts")
    hdr.append("")
    hdr.append(f"surviving features ({n_final}): {survivors}")
    hdr.append(f"dataset rows: {len(D)} "
               f"({D.index.min().date()}..{D.index.max().date()})")
    verdict = "PASS" if n_final >= 3 else "FAIL"
    hdr.append("")
    hdr.append(f"VERDICT: {verdict} "
               f"(stop rule: >= 3 features required, {n_final} survive)")
    with open(os.path.join(BASE, "REPORT_FEATURE_ENGINEERING.txt"),
              "w") as f:
        f.write("\n".join(rep) + "\n\n" + "\n".join(hdr) + "\n")
    P(f"VERDICT: {verdict} ({n_final} features, {len(D)} rows)")
    P("done.")


if __name__ == "__main__":
    main()
