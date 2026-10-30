# -*- coding: utf-8 -*-
"""
fvg_continuation_correlations.py - Step 5 daily-P&L correlations of the
FVG continuation WF stream vs every prior hypothesis with a P&L stream
(pearson_zf convention: zero-filled union of days, >= 30 common days).
H1/H2, H2-morning and BB MR stopped at Phase 1 (no stream) - stated in
the report. Writes correlations.csv next to the WF artifacts.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RR = Path(r"C:\Users\Milad\Desktop\RR\data\results\fvg_continuation_rr")
ORB = Path(r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present\py_trades.csv")
VWAP = Path(r"C:\Users\Milad\Desktop\RR\data\results\vwap_2021_2026\py_days.csv")
TP = Path(r"C:\Users\Milad\Desktop\RR\data\results\vwap_trend_pullback_rr\wf_trades.csv")
SP2L = Path(r"C:\Users\Milad\Desktop\RR\data\results\sp2l_2021_2026\py_trades.csv")
IM = Path(r"C:\Users\Milad\Desktop\RR\data\results\momentum_intraday_rr\wf_trades.csv")
DP = Path(r"C:\Users\Milad\Desktop\RR\data\results\momentum_multihorizon\variant_p_daily_n5.csv")


def daily_from_trades(path, pnl_col=None):
    df = pd.read_csv(path, sep=None, engine="python")
    col = pnl_col or ("net" if "net" in df.columns else
                      "pnl_pts" if "pnl_pts" in df.columns else "profit")
    if "nd" in df.columns:
        keys = df["nd"].astype(str)
    else:
        keys = df["open_time"].astype(str).str[:10].str.replace(".", "")
    per = df.groupby(keys)[col].sum()
    return {str(k): float(v) for k, v in per.items()}


def pearson_zf(a, b):
    ks = sorted(set(a) | set(b))
    if len([k for k in ks if k in a and k in b]) < 30:
        return None
    xa = np.array([a.get(k, 0.0) for k in ks])
    xb = np.array([b.get(k, 0.0) for k in ks])
    if xa.std() == 0 or xb.std() == 0:
        return None
    return float(np.corrcoef(xa, xb)[0, 1])


def main():
    fvg = daily_from_trades(RR / "wf_trades.csv", pnl_col="net")
    vw = pd.read_csv(VWAP, sep=";")
    vwap = {str(int(r["ny_date"])): float(r["total_pnl"])
            for _, r in vw.iterrows()}
    dp = pd.read_csv(DP)
    varp = {str(d).replace("-", ""): float(v)
            for d, v in zip(dp["date"], dp["exfin"])}
    others = {
        "ORB": daily_from_trades(ORB),
        "VWAP_MR": vwap,
        "trend_pullback_WF": daily_from_trades(TP, pnl_col="pnl"),
        "SP2L": daily_from_trades(SP2L),
        "intraday_momentum_WF": daily_from_trades(IM, pnl_col="pnl_pts"),
        "daily_momentum_P": varp,
    }
    rows = []
    for name, s in others.items():
        r = pearson_zf(fvg, s)
        common = len(set(fvg) & set(s))
        rows.append({"vs": name, "pearson_zf": round(r, 4) if r is not None else None,
                     "common_days": common})
        print(rows[-1])
    with open(RR / "correlations.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["vs", "pearson_zf", "common_days"])
        w.writeheader()
        w.writerows(rows)
    with open(RR / "correlations.json", "w") as fh:
        json.dump({"n_fvg_days": len(fvg), "rows": rows}, fh, indent=2)


if __name__ == "__main__":
    main()
