"""Short-term reversal study per
data/results/short_term_reversal/SPEC_SHORT_TERM_REVERSAL.txt and
GATE_SHORT_TERM_REVERSAL.txt. Train 2015-2019, 2020 embargo, strict OOS
test 2021-01..2026-04. Trees barred. Terminal Path C stake."""

import csv
import datetime as dt
import json
import math
import os
import time
import urllib.request
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet, RidgeClassifier
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller

BASE = r"C:\Users\Milad\Desktop\RR\data\results\short_term_reversal"
RAW = os.path.join(BASE, "raw")
SPINE = (r"C:\Users\Milad\Desktop\RR\data\results\momentum_multihorizon"
         r"\us500c_d1.csv")
FRED_URL = ("https://fred.stlouisfed.org/graph/fredgraph.csv?"
            "bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&"
            "graph_bgcolor=%23ffffff&height=450&mode=fred&"
            "recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&"
            "nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&"
            "show_tooltip=yes&id=VIXCLS,T10Y2Y,DFII10&"
            "scale=left,left,left&cosd=2014-01-01&coed=2026-08-31&"
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
TRAIN_LO = pd.Timestamp("2015-01-01")
TRAIN_HI = pd.Timestamp("2019-12-31")
TEST_LO = pd.Timestamp("2021-01-01")
TEST_HI = pd.Timestamp("2026-04-30")
L = 252
D_GRID = [round(0.05 * i, 2) for i in range(1, 20)]
ADF_CRIT = -2.86
IC_FLOOR = 0.025
SPREAD_PT = 0.51
SWAP_LONG = 20.0
SWAP_SHORT = 2.0
HORIZONS = [1, 2, 3]
ALPHAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
L1R = [0.1, 0.5, 0.9]
RC_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
SEED = 42
TERMINAL = r"C:\Program Files\MetaTrader 5\terminal64.exe"

rep = []


def P(s=""):
    rep.append(str(s))
    print(s, flush=True)


def spearman(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 10:
        return float("nan")
    v = spearmanr(a[m], b[m]).statistic
    return float(v) if not np.isnan(v) else float("nan")


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


def download_fred():
    os.makedirs(RAW, exist_ok=True)
    merged = os.path.join(RAW, "fredgraph_2014_2026.csv")
    if os.path.exists(merged):
        return pd.read_csv(merged, index_col=0, parse_dates=True)
    tmp = os.path.join(RAW, "fredgraph_download.bin")
    with urllib.request.urlopen(FRED_URL) as r, open(tmp, "wb") as fo:
        fo.write(r.read())
    frames = []
    with open(tmp, "rb") as fo:
        magic = fo.read(2)
    if magic == b"PK":
        with zipfile.ZipFile(tmp) as z:
            for n in z.namelist():
                P(f"  zip member: {n}")
                if n.lower().endswith(".csv"):
                    p2 = os.path.join(RAW, os.path.basename(n))
                    with open(p2, "wb") as out:
                        out.write(z.read(n))
    else:
        os.replace(tmp, os.path.join(RAW, "fredgraph_direct.csv"))
    os.remove(tmp)
    for m in sorted(os.listdir(RAW)):
        if m.lower().endswith(".csv") and m != "fredgraph_2014_2026.csv":
            d_ = pd.read_csv(os.path.join(RAW, m))
            icol = ("observation_date" if "observation_date" in d_.columns
                    else "DATE")
            frames.append(d_.set_index(icol))
    fr = frames[0].join(frames[1:], how="outer")
    fr.index = pd.to_datetime(fr.index)
    fr = fr.apply(pd.to_numeric, errors="coerce").replace(-99999.0,
                                                          np.nan)
    fr.to_csv(merged)
    with open(os.path.join(RAW, "provenance.txt"), "w") as f:
        f.write(f"URL: {FRED_URL}\nDownloaded: "
                f"{datetime.now(timezone.utc).isoformat()}\n")
    return fr


def try_fetch_usa30():
    out_path = os.path.join(BASE, "usa30c_d1.csv")
    if os.path.exists(out_path):
        return out_path
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize(TERMINAL):
            P(f"USA30.c: mt5.initialize failed "
              f"({mt5.last_error()})")
            return None
        mt5.symbol_select("USA30.c", True)
        time.sleep(1.5)
        bars = None
        for attempt in range(3):
            bars = mt5.copy_rates_range(
                "USA30.c", mt5.TIMEFRAME_D1,
                dt.datetime(1990, 1, 1, tzinfo=dt.timezone.utc),
                dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc))
            if bars is not None and len(bars):
                break
            time.sleep(2.0 * (attempt + 1))
        mt5.shutdown()
        if bars is None or not len(bars):
            P("USA30.c: no D1 bars returned")
            return None
        with open(out_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["time", "open", "high", "low", "close",
                        "tick_volume"])
            for b in bars:
                w.writerow([int(b["time"]), float(b["open"]),
                            float(b["high"]), float(b["low"]),
                            float(b["close"]), int(b["tick_volume"])])
        t0 = dt.datetime.fromtimestamp(int(bars[0]["time"]),
                                       dt.timezone.utc)
        P(f"USA30.c: {len(bars)} D1 bars from {t0.date()}")
        return out_path
    except Exception as e:
        P(f"USA30.c fetch failed: {repr(e)[:160]}")
        return None


def d_search(name, log_series):
    x = log_series.dropna()
    mask = np.asarray((x.index >= TRAIN_LO) & (x.index <= TRAIN_HI))
    rows = []
    chosen = None
    for d in D_GRID:
        y = ffd(x.values, d, L)
        yy = y[mask & ~np.isnan(y)]
        if len(yy) < 100:
            rows.append({"d": d, "adf_stat": None, "pass": False,
                         "n": int(len(yy))})
            continue
        stat, p = adfuller(yy, regression="c", autolag="AIC")[:2]
        ok = stat < ADF_CRIT
        rows.append({"d": d, "adf_stat": round(float(stat), 4),
                     "adf_p": round(float(p), 5), "pass": bool(ok),
                     "n": int(len(yy))})
        if ok and chosen is None:
            chosen = d
    return {"series": name, "adf_full_grid": rows, "d_star": chosen}, chosen


def financing(entry_d, exit_d, s):
    fin = 0.0
    for d0 in pd.date_range(entry_d, exit_d, freq="D")[1:]:
        mult = 5 if d0.weekday() == 2 else 1
        fin += (SWAP_LONG if s > 0 else -SWAP_SHORT) * mult
    return fin


def main():
    fr = download_fred()
    P(f"FRED: {len(fr)} rows {fr.index.min().date()}.."
      f"{fr.index.max().date()} cols={list(fr.columns)}")

    sp = pd.read_csv(SPINE)
    sp["date"] = pd.to_datetime(sp["time"], unit="s", utc=True) \
        .dt.tz_convert(None).dt.normalize()
    sp = sp.set_index("date").sort_index()
    sp = sp[sp.index <= TEST_HI]
    P(f"spine: {len(sp)} D1 bars {sp.index.min().date()}.."
      f"{sp.index.max().date()}")
    idx = sp.index
    close = sp["close"]

    usa30_path = try_fetch_usa30()
    usa30_ok = False
    ratio_log = None
    if usa30_path:
        u = pd.read_csv(usa30_path)
        u["date"] = pd.to_datetime(u["time"], unit="s", utc=True) \
            .dt.tz_convert(None).dt.normalize()
        u = u.set_index("date").sort_index()
        u = u[u["close"] > 0]
        cover = u.index[u.index <= TRAIN_HI]
        if len(cover) and cover.min() <= TRAIN_LO:
            usa30_ok = True
            ua = u["close"].reindex(idx.union(u.index)).ffill(limit=5)
            ratio_log = np.log(close) - np.log(ua.reindex(idx))
            P(f"USA30.c ok: covers from {cover.min().date()} "
              f"({len(u)} bars)")
        else:
            P(f"USA30.c insufficient history "
              f"(first bar {cover.min() if len(cover) else 'n/a'}) -> "
              "CONTINGENCY: ffd_log_spdj EXCLUDED")
    else:
        P("USA30.c unavailable -> CONTINGENCY: ffd_log_spdj EXCLUDED")

    r = np.log(close).diff()
    sigma = r.rolling(20).std(ddof=1)
    rv21 = 252.0 * (r ** 2).rolling(21).sum().shift(1)

    def fred_on_spine(col, shift):
        s = fr[col].reindex(idx.union(fr.index)).ffill(limit=5)
        v = s.reindex(idx)
        return v.shift(1) if shift else v

    vix = fred_on_spine("VIXCLS", False)
    ts = fred_on_spine("T10Y2Y", True)
    dfii = fred_on_spine("DFII10", True)

    feats = {"vrp": (vix / 100.0) ** 2 - rv21,
             "tslope_d21": ts.diff(21),
             "dfii10_d21": dfii.diff(21)}

    djson = {}
    lsp = np.log(close)
    cand = {"ffd_log_us500": lsp}
    if usa30_ok:
        cand["ffd_log_spdj"] = ratio_log
    for name, series in cand.items():
        entry, chosen = d_search(name, series)
        djson[name] = entry
        if chosen is None:
            P(f"  {name}: NO passing d -> feature EXCLUDED")
            continue
        x = series.dropna()
        vals = pd.Series(ffd(x.values, chosen, L), index=x.index)
        feats[name] = vals.reindex(idx)
        P(f"  {name}: d*={chosen} "
          f"(train ADF pass, n={entry['adf_full_grid'][0]['n']})")
    with open(os.path.join(BASE, "fracdiff_reversal_d.json"),
              "w") as f:
        json.dump(djson, f, indent=2)

    F = pd.DataFrame(feats)
    train_mask = (idx >= TRAIN_LO) & (idx <= TRAIN_HI)
    test_mask = (idx >= TEST_LO) & (idx <= TEST_HI)
    eff_start = F[train_mask].dropna().index.min()
    P(f"features: {list(F.columns)}; effective train start "
      f"{eff_start.date()} (SPEC 2.1); train rows "
      f"{int((F.index >= eff_start).sum() & 0xFFFFFFFF) if False else ''}"
      f"{len(F[train_mask & (F.index >= eff_start)].dropna())} non-NaN; "
      f"test rows {len(F[test_mask].dropna())}")

    mu = F[(F.index >= eff_start) & train_mask].mean()
    sd = F[(F.index >= eff_start) & train_mask].std(ddof=0)
    sd[sd == 0] = 1.0
    Z = (F - mu) / sd

    tscv = TimeSeriesSplit(n_splits=5)
    pred_rows = []
    results = {}
    models_used = {}

    for H in HORIZONS:
        y = (np.log(close.shift(-H) / close)
             / (sigma * math.sqrt(H)))
        lab = np.sign(np.log(close.shift(-H) / close))
        tr = F[train_mask & (F.index >= eff_start)].dropna().index
        tr = tr[y.reindex(tr).notna()]
        te = F[test_mask].dropna().index
        te = te[y.reindex(te).notna()]
        Xtr = Z.reindex(tr).values
        ytr = y.reindex(tr).values
        ltr = lab.reindex(tr).values
        Xte = Z.reindex(te).values

        best_a = None
        for alpha in ALPHAS:
            for l1r in L1R:
                ics = []
                for tr_i, va_i in tscv.split(Xtr):
                    m_, s_ = Xtr[tr_i].mean(0), Xtr[tr_i].std(0)
                    s_[s_ == 0] = 1.0
                    en = ElasticNet(alpha=alpha, l1_ratio=l1r,
                                    max_iter=10000, tol=1e-8,
                                    random_state=SEED)
                    en.fit((Xtr[tr_i] - m_) / s_, ytr[tr_i])
                    pr = en.predict((Xtr[va_i] - m_) / s_)
                    ics.append(spearman(pr, ytr[va_i]))
                mic = float(np.nanmean(ics)) if ics else float("nan")
                if not np.isnan(mic) and (best_a is None
                                          or mic > best_a[0]):
                    best_a = (mic, alpha, l1r)
        _, al, l1r = best_a if best_a else (float("nan"), 0.1, 0.5)
        m_, s_ = Xtr.mean(0), Xtr.std(0)
        s_[s_ == 0] = 1.0
        en = ElasticNet(alpha=al, l1_ratio=l1r, max_iter=10000,
                        tol=1e-8, random_state=SEED)
        en.fit((Xtr - m_) / s_, ytr)
        pred_en = en.predict((Xte - m_) / s_)
        models_used[("enet", H)] = {"alpha": al, "l1_ratio": l1r}

        best_b = None
        cls_ok = ltr != 0
        Xc = Xtr[cls_ok]
        yc = ytr[cls_ok]
        lc = ltr[cls_ok]
        for ralpha in RC_ALPHAS:
            ics = []
            for tr_i, va_i in tscv.split(Xc):
                if len(set(lc[tr_i])) < 2:
                    continue
                m_, s_ = Xc[tr_i].mean(0), Xc[tr_i].std(0)
                s_[s_ == 0] = 1.0
                rc = RidgeClassifier(alpha=ralpha)
                rc.fit((Xc[tr_i] - m_) / s_, lc[tr_i])
                pr = rc.decision_function((Xc[va_i] - m_) / s_)
                ics.append(spearman(pr, yc[va_i]))
            mic = float(np.nanmean(ics)) if ics else float("nan")
            if not np.isnan(mic) and (best_b is None
                                      or mic > best_b[0]):
                best_b = (mic, ralpha)
        _, ralpha = best_b if best_b else (float("nan"), 1.0)
        rc = RidgeClassifier(alpha=ralpha)
        rc.fit((Xtr[cls_ok] - m_) / s_, ltr[cls_ok])
        pred_rc = rc.decision_function((Xte - m_) / s_)
        models_used[("ridge", H)] = {"alpha": ralpha}

        ic_en = spearman(pred_en, y.reindex(te).values)
        ic_rc = spearman(pred_rc, y.reindex(te).values)
        results[("enet", H)] = ic_en
        results[("ridge", H)] = ic_rc
        P(f"[h={H}] train n={len(tr)} test n={len(te)} | "
          f"enet IC={ic_en:+.4f} (alpha={al}, l1={l1r}) | "
          f"ridge IC={ic_rc:+.4f} (alpha={ralpha})")
        for i, d0 in enumerate(te):
            pred_rows.append({"date": d0.date(), "h": H,
                              "model": "enet", "pred": pred_en[i],
                              "y": y.loc[d0]})
            pred_rows.append({"date": d0.date(), "h": H,
                              "model": "ridge", "pred": pred_rc[i],
                              "y": y.loc[d0]})

    pd.DataFrame(pred_rows).to_csv(
        os.path.join(BASE, "reversal_oos_predictions.csv"), index=False)

    P(f"G-R1 IC floor (>= {IC_FLOOR}, one-sided positive) ...")
    passers = []
    for (m_, H), ic in results.items():
        ok = (not np.isnan(ic)) and ic >= IC_FLOOR
        P(f"  {m_} h={H}: IC={ic:+.4f} -> {'PASS' if ok else 'fail'}")
        if ok:
            passers.append((ic, m_, H))
    gm1 = len(passers) > 0
    gm2 = False
    econ = {}
    if gm1:
        passers.sort(reverse=True)
        _, M_SEL, H_SEL = passers[0]
        P(f"G-R2 selected: {M_SEL} h={H_SEL} (by test IC)")
        sub = pd.DataFrame([p for p in pred_rows
                            if p["model"] == M_SEL
                            and p["h"] == H_SEL])
        sub["date"] = pd.to_datetime(sub["date"])
        sig_by_date = sub.set_index("date")["pred"]
        pos = int(idx.searchsorted(TEST_LO))
        trades = []
        while pos + H_SEL < len(idx):
            d0 = idx[pos]
            if d0 in sig_by_date.index:
                s = int(np.sign(sig_by_date.loc[d0]))
                gross = s * (close.iloc[pos + H_SEL]
                             - close.iloc[pos])
                fin = financing(d0, idx[pos + H_SEL], s) if s != 0 \
                    else 0.0
                net = gross - (SPREAD_PT if s != 0 else 0.0) - fin
                trades.append({"entry": d0.date(),
                               "exit": idx[pos + H_SEL].date(),
                               "signal": s,
                               "gross_pt": round(gross, 4),
                               "financing_pt": round(fin, 4),
                               "net_pt": round(net, 4)})
            pos += H_SEL
        tdf = pd.DataFrame(trades)
        tdf.to_csv(os.path.join(BASE, "g2_trades.csv"), index=False)
        if len(tdf) < 3 or tdf["net_pt"].std(ddof=1) == 0:
            sharpe = float("nan")
        else:
            net = tdf["net_pt"]
            sharpe = float(net.mean() / net.std(ddof=1)
                           * math.sqrt(252.0 / H_SEL))
        if len(tdf):
            yr = tdf.copy()
            yr["year"] = pd.to_datetime(yr["exit"]).dt.year
            per_year = {str(k): round(float(v), 2) for k, v in
                        yr.groupby("year")["net_pt"].sum().items()}
            econ = {"n_trades": len(tdf),
                    "gross_pt": round(float(tdf["gross_pt"].sum()),
                                      2),
                    "financing_pt":
                        round(float(tdf["financing_pt"].sum()), 2),
                    "net_pt": round(float(tdf["net_pt"].sum()), 2),
                    "net_ann_sharpe": None if np.isnan(sharpe)
                    else round(sharpe, 4),
                    "per_year_net_pt": per_year}
        else:
            econ = {"n_trades": 0, "net_ann_sharpe": None}
        P(f"G-R2: {len(tdf)} trades, net {econ.get('net_pt')} pt, "
          f"net Sharpe {econ['net_ann_sharpe']} (must be > 0)")
        gm2 = (not np.isnan(sharpe)) and sharpe > 0

    usa30_status = ("ratio feature INCLUDED" if usa30_ok
                    else "ffd_log_spdj EXCLUDED "
                         "(pre-declared contingency)")
    lines = ["REPORT_SHORT_TERM_REVERSAL",
             f"run: {datetime.now(timezone.utc).isoformat()}",
             "",
             f"USA30.c contingency: {usa30_status}",
             f"effective train start: {eff_start.date()}",
             "d* (train-only search): "
             + json.dumps({k: v["d_star"]
                           for k, v in djson.items()}),
             "",
             "test-window IC (2021-01-01..2026-04-30):"]
    for (m_, H), ic in results.items():
        lines.append(f"  {m_} h={H}: {ic:+.5f} "
                     f"hp={models_used.get((m_, H))}")
    lines.append("")
    lines.append(f"G-R1: {'PASS' if gm1 else 'FAIL'} "
                 f"({len(passers)} of 6 configs >= {IC_FLOOR})")
    if gm1:
        lines.append(f"G-R2: {'PASS' if gm2 else 'FAIL'} "
                     f"{json.dumps(econ)}")
    else:
        lines.append("G-R2: not evaluated (G-R1 failed)")
    if gm1 and gm2:
        lines.append("VERDICT: PASS - reversal structure generalized "
                     "across the 2020 break.")
    else:
        lines.append("VERDICT: FAIL.")
        lines.append("TRANSITION TO PATH C: RESEARCH EXHAUSTED")
    with open(os.path.join(BASE, "REPORT_SHORT_TERM_REVERSAL.txt"),
              "w") as f:
        f.write("\n".join(rep) + "\n\n" + "\n".join(lines) + "\n")
    P(lines[-1])


if __name__ == "__main__":
    main()
