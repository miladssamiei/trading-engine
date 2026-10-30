# -*- coding: utf-8 -*-
"""
children_reval_stats.py - Model=4 revalidation stats for ORB and SP2L
(SPEC_TICK_MODEL_REVALIDATION_CHILDREN), replicating the ORIGINAL
conventions:

  ORB (orb_stats.py): $10k fixed base, 1 lot, daily PnL zero-filled
  over Model=4 session days; Sharpe = mean/std*sqrt(252); maxDD on
  cumulative; CAGR; return/DD; BH from the verified M5 export
  (real-tick-derived; same Custom-base ticks the tester replays).
  SP2L (sp2l_stats/portfolio conventions): same, plus common-universe
  Sharpe zero-filled over the ORIGINAL ORB session-day universe (the
  documented 0.275 convention), win rate, per-trade expectancy,
  per-year with 2025 attention.

Writes stats JSON + yearly_breakdown CSVs + reconcile summary CSVs.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results"
            r"\tick_model_revalidation_children")
ORB_ORIG = Path(r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present")
ACCOUNT = 10000.0
SQ252 = math.sqrt(252)


def load_trades(path):
    df = pd.read_csv(path, sep=";")
    df["open_time"] = pd.to_datetime(df["open_time"],
                                     format="%Y.%m.%d %H:%M", utc=True)
    df["profit"] = df["profit"].astype(float)
    df["date"] = pd.to_datetime(df["open_time"].dt.tz_convert(
        "America/New_York").dt.strftime("%Y-%m-%d")).dt.date
    return df


def load_session_days(path):
    df = pd.read_csv(path, sep=";", dtype=str)
    if "session_open" in df.columns:
        df["session_open"] = pd.to_numeric(df["session_open"],
                                           errors="coerce")
        df["prev_close"] = pd.to_numeric(df["prev_close"],
                                         errors="coerce")
        sel = df[(df["session_open"] > 0) | (df["prev_close"] > 0)]
    else:
        # SP2L day schema: one row per NY date the EA saw bars for
        sel = df
    return sorted(pd.to_datetime(sel["ny_date"], format="%Y%m%d").dt.date)


def aggregated(trades, session_days):
    per_day = trades.groupby("date")["profit"].sum()
    vals = np.array([per_day.get(d, 0.0) for d in session_days])
    total = float(vals.sum())
    cum = np.cumsum(vals)
    mdd = float((np.maximum.accumulate(cum) - cum).max())
    s = pd.Series(vals)
    mean, std = float(s.mean()), float(s.std(ddof=1))
    years = len(session_days) / 252.0
    ret_pct = total / ACCOUNT * 100
    return {"trades": int(len(trades)), "trading_days": len(session_days),
            "total_pnl": round(total, 2),
            "total_return_pct": round(ret_pct, 2),
            "max_dd_usd": round(mdd, 2),
            "max_dd_pct": round(mdd / ACCOUNT * 100, 2),
            "daily_mean": round(mean, 4), "daily_std": round(std, 4),
            "sharpe": round(mean / std * SQ252, 4) if std > 0 else None,
            "cagr_pct": round(((1 + ret_pct / 100) ** (1 / years) - 1)
                              * 100, 2),
            "return_dd": round(ret_pct / (mdd / ACCOUNT * 100), 3)
            if mdd > 0 else None,
            "win_rate_pct": round(float((trades["profit"] > 0).mean())
                                  * 100, 2),
            "expectancy_per_trade": round(
                total / len(trades), 3) if len(trades) else None}


def yearly(trades):
    out = {}
    for y, g in trades.groupby(trades["open_time"].dt.year):
        out[int(y)] = {"trades": int(len(g)),
                       "pnl": round(float(g["profit"].sum()), 2)}
    return out


def orb_bh():
    m5 = pd.read_csv(BASE / "orb" / "us500tick_m5.csv")
    m5["time"] = pd.to_datetime(m5["time"], unit="s", utc=True)
    m5["date"] = m5["time"].dt.date
    first_price = float(m5.iloc[0]["open"])
    last_price = float(m5.iloc[-1]["close"])
    price_ret_usd = last_price - first_price
    ret_base = price_ret_usd / ACCOUNT * 100
    daily = m5.groupby("date")["close"].last()
    counts = m5.groupby("date").size()
    daily_full = daily[counts >= 50]
    eq = (daily_full / first_price - 1.0) * ACCOUNT
    mdd = float((eq.cummax() - eq).max())
    rets = daily_full.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std(ddof=1) * SQ252)
    open_dates = sorted(m5["date"].unique())
    swap = sum(-20.0 * (5 if pd.Timestamp(d).weekday() == 2 else 1)
               for d in open_dates)
    return {"first_price": first_price, "last_price": last_price,
            "total_return_pct_base10k": round(ret_base, 2),
            "max_dd_pct": round(mdd / ACCOUNT * 100, 2),
            "sharpe": round(sharpe, 4),
            "return_dd": round(ret_base / (mdd / ACCOUNT * 100), 3),
            "total_swap_usd": round(swap, 2),
            "with_financing_ret_pct": round(
                (price_ret_usd + swap) / ACCOUNT * 100, 2)}


def main():
    out = {}

    # ---------------- ORB ----------------
    orb_tr = load_trades(BASE / "orb" / "mt5_trades.csv")
    orb_days = load_session_days(BASE / "orb" / "mt5_days.csv")
    orb_m4 = aggregated(orb_tr, orb_days)
    orb_bh_res = orb_bh()
    orb_m0 = {"trades": 447, "total_pnl": 470.67, "sharpe": 0.3365,
              "max_dd_pct": 5.13, "return_dd": 0.917,
              "reconcile_pct": 93.1}
    orb_py = {"trades": 442, "total_pnl": 531.13}
    out["orb"] = {"model4": orb_m4, "model0_reference": orb_m0,
                  "python_reference": orb_py, "buy_and_hold": orb_bh_res,
                  "yearly": yearly(orb_tr)}

    # ---------------- SP2L ----------------
    sp_tr = load_trades(BASE / "sp2l" / "mt5_trades.csv")
    sp_days = load_session_days(BASE / "sp2l" / "mt5_days.csv")
    sp_own = aggregated(sp_tr, sp_days)
    orig_universe = load_session_days(ORB_ORIG / "mt5_days.csv")
    sp_common = aggregated(sp_tr, orig_universe)
    sp_m0 = {"trades": 1343, "total_pnl": 281.84,
             "sharpe_common": 0.275, "max_dd_pct": 4.17,
             "reconcile_pct": 89.7}
    out["sp2l"] = {"model4_own_universe": sp_own,
                   "model4_common_universe": sp_common,
                   "model0_reference": sp_m0,
                   "yearly": yearly(sp_tr)}

    # ---------------- reconcile summaries ----------------
    def orb_recon_summary():
        txt = (BASE / "orb" / "RECONCILE.md").read_text()
        import re
        m = re.search(r"\*\*(\d+)/(\d+) \(([\d.]+)%\)\*\*", txt)
        return {"matched": 424, "agreement_n": int(m.group(1)),
                "agreement_max": int(m.group(2)),
                "agreement_pct": float(m.group(3)),
                "note": "tables recomputed by orb_reconcile_fullrange.py;"
                        " the script's narrative verdict section is "
                        "hardcoded Model=0 text and is NOT valid for "
                        "Model=4"}

    def sp2l_recon_summary():
        txt = (BASE / "sp2l" / "RECONCILE.md").read_text()
        import re
        m = re.search(r"Matched \*\*(\d+)\*\* of py (\d+) / mt5 (\d+) "
                      r"trades; \*\*(\d+)/(\d+) = ([\d.]+)%\*\*", txt)
        return {"matched": int(m.group(1)), "py_n": int(m.group(2)),
                "mt5_n": int(m.group(3)), "agreement_n": int(m.group(4)),
                "agreement_matched": int(m.group(5)),
                "agreement_pct": float(m.group(6))}

    out["orb"]["reconcile"] = orb_recon_summary()
    out["sp2l"]["reconcile"] = sp2l_recon_summary()

    with open(BASE / "children_model4_stats.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)

    # ---------------- yearly breakdown CSVs ----------------
    for strat, key in (("orb", "orb"), ("sp2l", "sp2l")):
        rows = []
        m0y = {"orb": {2021: 74, 2022: 102, 2023: 87, 2024: 83,
                       2025: 74, 2026: 27},
               "sp2l": None}[strat]
        for y in sorted(out[strat]["yearly"]):
            rows.append({"year": y,
                         "model4_trades": out[strat]["yearly"][y]["trades"],
                         "model4_pnl": out[strat]["yearly"][y]["pnl"],
                         "model0_trades": (m0y or {}).get(y)})
        pd.DataFrame(rows).to_csv(
            BASE / f"yearly_breakdown_{key}_model4.csv", index=False)

    # ---------------- reconcile CSVs ----------------
    pd.DataFrame([out["orb"]["reconcile"]]).to_csv(
        BASE / "orb_model4_reconcile.csv", index=False)
    pd.DataFrame([out["sp2l"]["reconcile"]]).to_csv(
        BASE / "sp2l_model4_reconcile.csv", index=False)

    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
