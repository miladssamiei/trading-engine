# -*- coding: utf-8 -*-
"""
vwap_stats.py — monthly performance matrix, aggregated performance,
buy-and-hold benchmark, and ORB-vs-VWAP correlation/complementarity for
the full-range VWAP mean-reversion run (2021-01-03 .. 2026-04-30).

State base: $10,000 fixed account, 1 lot, no compounding (same as ORB).
Pure pandas + stdlib. Writes monthly_matrix.csv + aggregated.json.
"""

import json
import math
import os
from datetime import datetime, timezone

import pandas as pd

BASE = r"C:\Users\Milad\Desktop\RR\data\results\vwap_2021_2026"
ORB_BASE = r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present"
ACCOUNT_BASE = 10000.0
LOT = 1
USD_PER_POINT = 1.0

MONTHS = pd.period_range("2021-01", "2026-04", freq="M")
SWAP_LONG = -20.0    # points/day
WED_FACTOR = 5


def fmt(x, nd=2):
    if x is None:
        return "n/a"
    try:
        if math.isnan(x) or math.isinf(x):
            return "n/a"
    except TypeError:
        pass
    return f"{x:.{nd}f}"


def load_trades(side):
    df = pd.read_csv(os.path.join(BASE, f"{side}_trades.csv"), sep=";")
    df["open_time"] = pd.to_datetime(df["open_time"], format="%Y.%m.%d %H:%M", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], format="%Y.%m.%d %H:%M", utc=True)
    df["open_date"] = df["open_time"].dt.date
    df["month"] = df["open_time"].dt.strftime("%Y-%m")
    df["profit"] = df["profit"].astype(float)
    return df


def load_days(side):
    df = pd.read_csv(os.path.join(BASE, f"{side}_days.csv"), sep=";", dtype=str)
    df["date"] = pd.to_datetime(df["ny_date"], format="%Y%m%d", utc=True).dt.date
    df["month"] = pd.to_datetime(df["ny_date"], format="%Y%m%d").dt.strftime("%Y-%m")
    df["session_open"] = pd.to_numeric(df["session_open"], errors="coerce")
    df["skipped"] = pd.to_numeric(df["skipped"], errors="coerce").fillna(0)
    df["is_session"] = df["session_open"] > 0
    return df


def build_daily_pnl(side):
    trades = load_trades(side)
    days = load_days(side)
    sess = days[days["is_session"]].sort_values("date").reset_index(drop=True)
    per_day = trades.groupby("open_date")["profit"].sum()
    vals = sess["date"].map(per_day).fillna(0.0).to_numpy(dtype=float)
    return trades, days, sess, vals, per_day


def monthly_stats(side):
    trades, days, sess, vals, per_day_map = build_daily_pnl(side)
    rows = []
    for m in MONTHS:
        mkey = str(m)
        sess_m = sess[sess["month"] == mkey]
        tr_m = trades[trades["month"] == mkey]
        n_tr = int(len(tr_m))
        winrate = (float((tr_m["profit"] > 0).mean()) * 100) if n_tr else 0.0
        pnl = float(tr_m["profit"].sum())
        if len(sess_m) == 0:
            maxdd = 0.0
        else:
            dvals = sess_m["date"].map(per_day_map).fillna(0.0).to_numpy(dtype=float)
            cum = pd.Series(dvals).cumsum()
            maxdd = float((cum.cummax() - cum).max())
        rows.append({
            "month": mkey,
            "trading_days": int(len(sess_m)),
            "skipped_total": int(((days["month"] == mkey) & (days["skipped"] == 1)).sum()),
            "trades": n_tr,
            "winrate_pct": winrate,
            "pnl": pnl,
            "pnl_pct": pnl / ACCOUNT_BASE * 100,
            "maxdd": maxdd,
        })
    return rows


def aggregated(side, years):
    trades, days, sess, vals, per_day_map = build_daily_pnl(side)
    total_pnl = float(vals.sum())
    total_return_pct = total_pnl / ACCOUNT_BASE * 100
    cum = pd.Series(vals).cumsum()
    maxdd = float((cum.cummax() - cum).max())
    maxdd_pct = maxdd / ACCOUNT_BASE * 100
    s = pd.Series(vals)
    mean = float(s.mean())
    std = float(s.std(ddof=1))
    sharpe = (mean / std * math.sqrt(252)) if std > 0 else None
    cagr = ((1 + total_return_pct / 100) ** (1 / years) - 1) * 100
    rdd = (total_return_pct / maxdd_pct) if maxdd_pct > 0 else None
    wins = int((trades["profit"] > 0).sum())
    return {
        "side": side,
        "trades": int(len(trades)),
        "trading_days": int(len(sess)),
        "win_rate_pct": 100.0 * wins / len(trades) if len(trades) else None,
        "total_pnl": total_pnl,
        "total_return_pct": total_return_pct,
        "max_dd_usd": maxdd,
        "max_dd_pct": maxdd_pct,
        "daily_pnl_mean": mean,
        "daily_pnl_std": std,
        "annualized_sharpe": sharpe,
        "cagr_pct": cagr,
        "return_dd_ratio": rdd,
    }


def bh_metrics(m1):
    first = m1.iloc[0]
    last = m1.iloc[-1]
    first_price = float(first["open"])
    last_price = float(last["close"])
    first_ts = first["time"]
    last_ts = last["time"]
    years = (last_ts - first_ts).total_seconds() / 86400.0 / 365.25

    price_ret_usd = (last_price - first_price) * LOT * USD_PER_POINT
    bh_ret_price = (last_price / first_price - 1) * 100
    bh_ret_base = price_ret_usd / ACCOUNT_BASE * 100
    cagr_price = ((1 + bh_ret_price / 100) ** (1 / years) - 1) * 100
    cagr_base = ((1 + bh_ret_base / 100) ** (1 / years) - 1) * 100

    daily = m1.groupby("date")["close"].last()
    counts = m1.groupby("date").size()
    daily_full = daily[counts >= 240]     # M1: full day ~1080 bars; >=240 cuts partials
    cum_ret = daily_full / first_price - 1.0
    eq = cum_ret * ACCOUNT_BASE
    peak = eq.cummax()
    maxdd_usd = float((peak - eq).max())
    maxdd_pct = maxdd_usd / ACCOUNT_BASE * 100
    rets = daily_full.pct_change().dropna()
    sharpe = (float(rets.mean()) / float(rets.std(ddof=1)) * math.sqrt(252)) if len(rets) > 1 else None
    rdd = (bh_ret_base / maxdd_pct) if maxdd_pct > 0 else None

    m1m = m1.copy()
    m1m["month"] = m1m["time"].dt.strftime("%Y-%m")
    month_end = m1m.groupby("month")["close"].last().reindex([str(m) for m in MONTHS])
    prev = month_end.shift(1)
    prev.iloc[0] = first_price
    month_ret = (month_end / prev - 1) * 100
    cum_pct = ((1 + month_ret / 100).cumprod() - 1) * 100

    open_dates = sorted(m1["date"].unique())
    total_swap_usd = float(sum(
        SWAP_LONG * (WED_FACTOR if pd.Timestamp(d).weekday() == 2 else 1)
        for d in open_dates))
    with_fin_ret_pct = (price_ret_usd + total_swap_usd) / ACCOUNT_BASE * 100
    with_fin_cagr = None
    if 1 + with_fin_ret_pct / 100 > 0:
        with_fin_cagr = ((1 + with_fin_ret_pct / 100) ** (1 / years) - 1) * 100

    return {
        "first_price": first_price,
        "last_price": last_price,
        "first_ts": str(first_ts),
        "last_ts": str(last_ts),
        "years": years,
        "price_ret_usd": price_ret_usd,
        "total_return_pct_price": bh_ret_price,
        "total_return_pct_base10k": bh_ret_base,
        "cagr_pct_price": cagr_price,
        "cagr_pct_base10k": cagr_base,
        "daily_full_days": int(len(daily_full)),
        "max_dd_usd": maxdd_usd,
        "max_dd_pct": maxdd_pct,
        "annualized_sharpe": sharpe,
        "return_dd_ratio": rdd,
        "monthly_ret_pct": {str(m): float(month_ret.loc[str(m)]) for m in MONTHS},
        "monthly_cum_ret_pct": {str(m): float(cum_pct.loc[str(m)]) for m in MONTHS},
        "swap_days": len(open_dates),
        "wednesday_days": sum(1 for d in open_dates if pd.Timestamp(d).weekday() == 2),
        "total_swap_usd": total_swap_usd,
        "with_financing_long_total_return_pct": with_fin_ret_pct,
        "with_financing_long_cagr_pct": with_fin_cagr,
    }


def spearman(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    import numpy as np
    ra = (ra - ra.mean()) / ra.std()
    rb = (rb - rb.mean()) / rb.std()
    return float((ra * rb).mean())


def main():
    print("=" * 78)
    print("VWAP mean-reversion full-range stats  2021-01-03 .. 2026-04-30")
    print("State base: $10,000 fixed, 1 lot, NO compounding.")
    print("=" * 78)

    m1 = pd.read_csv(os.path.join(BASE, "us500tick_m1.csv"))
    m1["time"] = pd.to_datetime(m1["time"], unit="s", utc=True)
    m1["date"] = m1["time"].dt.date
    bh = bh_metrics(m1)
    years = bh["years"]
    print(f"\nM1 bars: {len(m1)}  first {bh['first_ts']}  last {bh['last_ts']}")
    print(f"CAGR years: {years:.4f}")

    agg = {}
    for side in ("py", "mt5"):
        agg[side] = aggregated(side, years)
        a = agg[side]
        print(f"\n--- VWAP-{side.upper()} aggregated ---")
        print(f"  trades={a['trades']}  trading_days={a['trading_days']}  "
              f"win_rate={fmt(a['win_rate_pct'], 1)}%  total_pnl={a['total_pnl']:+.2f}")
        print(f"  total_return={a['total_return_pct']:.2f}%  CAGR={a['cagr_pct']:.2f}%  "
              f"maxDD=${a['max_dd_usd']:.2f} ({a['max_dd_pct']:.2f}%)")
        print(f"  daily mean={a['daily_pnl_mean']:.4f}  std={a['daily_pnl_std']:.4f}  "
              f"Sharpe={fmt(a['annualized_sharpe'], 3)}  return/DD={fmt(a['return_dd_ratio'])}")

    # ---------- monthly matrix ----------
    mp = {r["month"]: r for r in monthly_stats("py")}
    mm = {r["month"]: r for r in monthly_stats("mt5")}
    header = ("month;py_trading_days;py_trades;py_winrate_pct;py_pnl;py_pnl_pct;py_maxdd;"
              "mt5_trading_days;mt5_trades;mt5_winrate_pct;mt5_pnl;mt5_pnl_pct;mt5_maxdd;"
              "bh_month_ret_pct;bh_cum_ret_pct")
    print("\n" + "=" * 78)
    print("MONTHLY MATRIX (base $10,000, 1 lot, no compounding)")
    mrows = []
    for m in MONTHS:
        mkey = str(m)
        p, t = mp[mkey], mm[mkey]
        row = {
            "month": mkey,
            "py_trading_days": p["trading_days"], "py_trades": p["trades"],
            "py_winrate_pct": round(p["winrate_pct"], 1), "py_pnl": round(p["pnl"], 2),
            "py_pnl_pct": round(p["pnl_pct"], 2), "py_maxdd": round(p["maxdd"], 2),
            "mt5_trading_days": t["trading_days"], "mt5_trades": t["trades"],
            "mt5_winrate_pct": round(t["winrate_pct"], 1), "mt5_pnl": round(t["pnl"], 2),
            "mt5_pnl_pct": round(t["pnl_pct"], 2), "mt5_maxdd": round(t["maxdd"], 2),
            "bh_month_ret_pct": round(bh["monthly_ret_pct"][mkey], 2),
            "bh_cum_ret_pct": round(bh["monthly_cum_ret_pct"][mkey], 2),
        }
        mrows.append(row)
        print(f"{mkey}  py[{p['trading_days']:>3} d, {p['trades']:>3} tr, {p['winrate_pct']:>5.1f}%, "
              f"{p['pnl']:>+9.2f}]  mt5[{t['trading_days']:>3} d, {t['trades']:>3} tr, "
              f"{t['winrate_pct']:>5.1f}%, {t['pnl']:>+9.2f}]  "
              f"bh {bh['monthly_ret_pct'][mkey]:>+6.2f}%")
    pd.DataFrame(mrows).to_csv(os.path.join(BASE, "monthly_matrix.csv"), index=False)

    # ---------- buy & hold ----------
    print("\n" + "=" * 78)
    print("BUY & HOLD benchmark (same range, from the M1 export)")
    print(f"  first {bh['first_price']:.3f} -> last {bh['last_price']:.3f}")
    print(f"  price return {bh['total_return_pct_price']:.2f}% (CAGR {bh['cagr_pct_price']:.2f}%)")
    print(f"  $10k/1-lot ex-financing {bh['total_return_pct_base10k']:.2f}% (CAGR {bh['cagr_pct_base10k']:.2f}%)")
    print(f"  maxDD {bh['max_dd_pct']:.2f}%  Sharpe {fmt(bh['annualized_sharpe'], 3)}  "
          f"return/DD {fmt(bh['return_dd_ratio'])}")
    print(f"  with-financing LONG: swap days {bh['swap_days']} (Wed x{WED_FACTOR}), "
          f"swap ${bh['total_swap_usd']:.0f} -> total {bh['with_financing_long_total_return_pct']:.2f}%")

    # ---------- ORB comparison + correlation ----------
    print("\n" + "=" * 78)
    print("ORB vs VWAP — correlation & complementarity (daily PnL, session days)")
    orb_agg = json.load(open(os.path.join(ORB_BASE, "aggregated.json")))["aggregated"]

    def daily_series(side, base):
        trades = load_trades(side) if base == BASE else _load_orb_trades(side)
        days = load_days(side) if base == BASE else _load_orb_days(side)
        sess = days[days["is_session"]].sort_values("date").reset_index(drop=True)
        per_day = trades.groupby("open_date")["profit"].sum()
        return sess["date"].map(per_day).fillna(0.0)

    def _load_orb_trades(side):
        df = pd.read_csv(os.path.join(ORB_BASE, f"{side}_trades.csv"), sep=";")
        df["open_time"] = pd.to_datetime(df["open_time"], format="%Y.%m.%d %H:%M", utc=True)
        df["open_date"] = df["open_time"].dt.date
        df["profit"] = df["profit"].astype(float)
        return df

    def _load_orb_days(side):
        df = pd.read_csv(os.path.join(ORB_BASE, f"{side}_days.csv"), sep=";", dtype=str)
        df["date"] = pd.to_datetime(df["ny_date"], format="%Y%m%d", utc=True).dt.date
        df["session_open"] = pd.to_numeric(df["session_open"], errors="coerce")
        df["prev_close"] = pd.to_numeric(df["prev_close"], errors="coerce")
        df["is_session"] = (df["session_open"] > 0) | (df["prev_close"] > 0)
        return df

    series = {}
    for name, side, base in (("vwap_py", "py", BASE), ("vwap_mt5", "mt5", BASE),
                             ("orb_py", "py", ORB_BASE), ("orb_mt5", "mt5", ORB_BASE)):
        s = daily_series(side, base)
        series[name] = s
        print(f"  {name}: {len(s)} session days, total PnL {s.sum():+.2f}")

    import numpy as np
    corr = {}
    for a_name, b_name in (("vwap_mt5", "orb_mt5"), ("vwap_py", "orb_py")):
        a, b = series[a_name], series[b_name]
        j = pd.concat([a, b], axis=1, join="inner").dropna()
        ja, jb = j.iloc[:, 0].to_numpy(), j.iloc[:, 1].to_numpy()
        pearson = float(np.corrcoef(ja, jb)[0, 1])
        rho = spearman(ja, jb)
        corr[f"{a_name}_vs_{b_name}"] = {"pearson": pearson, "spearman": rho,
                                         "n_days": int(len(j))}
        print(f"  corr({a_name}, {b_name}) over {len(j)} common days: "
              f"Pearson {pearson:+.3f}, Spearman {rho:+.3f}")

    # combined portfolio (equal weight, both at 1 lot, daily PnL summed)
    combo = {}
    for ta, tb, label in (("mt5", "mt5", "combined_mt5"), ("py", "py", "combined_py")):
        va = series[f"vwap_{ta}"]
        ob = series[f"orb_{tb}"]
        j = pd.concat([va, ob], axis=1, join="inner").dropna()
        s = j.iloc[:, 0] + j.iloc[:, 1]
        mean, std = float(s.mean()), float(s.std(ddof=1))
        sharpe = mean / std * math.sqrt(252) if std > 0 else None
        cum = s.cumsum()
        maxdd = float((cum.cummax() - cum).max())
        combo[label] = {"total_pnl": float(s.sum()),
                        "sharpe": sharpe,
                        "max_dd_usd": maxdd,
                        "n_days": int(len(s)),
                        "corr_components": corr.get(
                            f"vwap_{ta}_vs_orb_{tb}")}
        print(f"  {label}: total {s.sum():+.2f}, Sharpe {fmt(sharpe, 3)}, maxDD ${maxdd:.2f}")

    # ---------- final comparison table ----------
    bh_ex = {"total_return_pct": bh["total_return_pct_base10k"],
             "cagr_pct": bh["cagr_pct_base10k"], "max_dd_pct": bh["max_dd_pct"],
             "annualized_sharpe": bh["annualized_sharpe"],
             "return_dd_ratio": bh["return_dd_ratio"]}
    print("\n" + "=" * 78)
    print("FINAL COMPARISON (returns % of $10k base / 1 lot)")
    cols = {
        "VWAP-MT5": {k: agg["mt5"][k] for k in
                     ("total_return_pct", "cagr_pct", "max_dd_pct",
                      "annualized_sharpe", "return_dd_ratio")},
        "VWAP-Python": {k: agg["py"][k] for k in
                        ("total_return_pct", "cagr_pct", "max_dd_pct",
                         "annualized_sharpe", "return_dd_ratio")},
        "ORB-MT5": {"total_return_pct": orb_agg["mt5"]["total_return_pct"],
                    "cagr_pct": orb_agg["mt5"]["cagr_pct"],
                    "max_dd_pct": orb_agg["mt5"]["max_dd_pct"],
                    "annualized_sharpe": orb_agg["mt5"]["annualized_sharpe"],
                    "return_dd_ratio": orb_agg["mt5"]["return_dd_ratio"]},
        "ORB-Python": {"total_return_pct": orb_agg["python"]["total_return_pct"],
                       "cagr_pct": orb_agg["python"]["cagr_pct"],
                       "max_dd_pct": orb_agg["python"]["max_dd_pct"],
                       "annualized_sharpe": orb_agg["python"]["annualized_sharpe"],
                       "return_dd_ratio": orb_agg["python"]["return_dd_ratio"]},
        "B&H ex-financing": bh_ex,
        "B&H w/ fin long": {"total_return_pct": bh["with_financing_long_total_return_pct"],
                            "cagr_pct": bh["with_financing_long_cagr_pct"],
                            "max_dd_pct": bh["max_dd_pct"],
                            "annualized_sharpe": bh["annualized_sharpe"],
                            "return_dd_ratio":
                                bh["with_financing_long_total_return_pct"] / bh["max_dd_pct"]},
    }
    print(f"{'metric':<22}" + "".join(f"{k:>18}" for k in cols))
    for metric in ("total_return_pct", "cagr_pct", "max_dd_pct",
                   "annualized_sharpe", "return_dd_ratio"):
        line = f"{metric:<22}"
        for k in cols:
            line += f"{fmt(cols[k][metric]):>18}"
        print(line)

    out = {
        "params": {"account_base_usd": ACCOUNT_BASE, "lot": LOT,
                   "usd_per_point": USD_PER_POINT, "compounding": False,
                   "data_range": [bh["first_ts"], bh["last_ts"]], "years": years},
        "aggregated": {"python": agg["py"], "mt5": agg["mt5"]},
        "buy_and_hold": bh,
        "orb_reference": {k: {m: orb_agg[k][m] for m in
                              ("total_return_pct", "cagr_pct", "max_dd_pct",
                               "annualized_sharpe", "return_dd_ratio", "trades")}
                          for k in ("python", "mt5")},
        "correlation": corr,
        "combined_portfolio": combo,
        "comparison": cols,
        "monthly_matrix": mrows,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(BASE, "aggregated.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWROTE: {os.path.join(BASE, 'monthly_matrix.csv')}")
    print(f"WROTE: {os.path.join(BASE, 'aggregated.json')}")
    print("=" * 78)
    print("DONE")


if __name__ == "__main__":
    main()
