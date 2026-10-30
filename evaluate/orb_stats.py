# -*- coding: utf-8 -*-
"""
orb_stats.py
============
Monthly performance matrix, aggregated performance and buy-and-hold benchmark
for the full-range ORB run (2021-01-03 23:00 UTC .. 2026-04-30 23:55 UTC).

State base: $10,000 fixed account base, 1 lot, no compounding.
PnL is USD at 1 lot (USD 1 per US500 point). All % figures are per-day PnL /
cumulative PnL expressed as a % of the $10,000 base (no compounding) unless a
price-based (index) figure is explicitly labelled.
Pure pandas + stdlib.
"""

import json
import math
import os

import pandas as pd

BASE = r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present"
ACCOUNT_BASE = 10000.0   # fixed $10,000 account base, 1 lot, no compounding
LOT = 1
USD_PER_POINT = 1.0

MONTHS = pd.period_range("2021-01", "2026-04", freq="M")
WED_FACTOR = 5   # rollover3days=5 : Wednesday swap counted 5x (spec formula)
SWAP_LONG = -20.0  # points/day, USD 1/point at 1 lot


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
    df["prev_close"] = pd.to_numeric(df["prev_close"], errors="coerce")
    df["skipped"] = pd.to_numeric(df["skipped"], errors="coerce").fillna(0)
    df["is_session"] = (df["session_open"] > 0) | (df["prev_close"] > 0)
    return df


def build_daily_pnl(side):
    """Full per-day PnL over ALL session days (zero-filled) in date order."""
    trades = load_trades(side)
    days = load_days(side)
    sess = days[days["is_session"]].sort_values("date").reset_index(drop=True)
    per_day = trades.groupby("open_date")["profit"].sum()
    vals = sess["date"].map(per_day).fillna(0.0).to_numpy(dtype=float)
    return trades, days, sess, vals, per_day


def monthly_stats(side, per_day_map):
    trades, days, sess, vals, per_day_map = build_daily_pnl(side)
    rows = []
    for m in MONTHS:
        mkey = str(m)
        sess_m = sess[sess["month"] == mkey]
        day_m = days[days["month"] == mkey]
        tr_m = trades[trades["month"] == mkey]
        n_tr = int(len(tr_m))
        winrate = (float((tr_m["profit"] > 0).mean()) * 100) if n_tr else 0.0
        pnl = float(tr_m["profit"].sum())
        pnl_pct = pnl / ACCOUNT_BASE * 100
        if len(sess_m) == 0:
            maxdd = 0.0
        else:
            dvals = sess_m["date"].map(per_day_map).fillna(0.0).to_numpy(dtype=float)
            cum = pd.Series(dvals).cumsum()
            maxdd = float((cum.cummax() - cum).max())
        rows.append({
            "month": mkey,
            "trading_days": int(len(sess_m)),
            "skipped_total": int((day_m["skipped"] == 1).sum()),
            "skipped_gap": int(((day_m["skipped"] == 1) & (day_m["skip_reason"] == "gap_filter")).sum()),
            "trades": n_tr,
            "winrate_pct": winrate,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
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
    return {
        "side": side,
        "trades": int(len(trades)),
        "trading_days": int(len(sess)),
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


def bh_metrics(m5, first_price, last_price, first_ts, last_ts, years):
    # --- headline price returns ---
    price_ret_usd = (last_price - first_price) * LOT * USD_PER_POINT
    bh_ret_price = (last_price / first_price - 1) * 100          # price / index return
    bh_ret_base = price_ret_usd / ACCOUNT_BASE * 100             # % of $10,000 base at 1 lot
    cagr_price = ((1 + bh_ret_price / 100) ** (1 / years) - 1) * 100
    cagr_base = ((1 + bh_ret_base / 100) ** (1 / years) - 1) * 100

    # --- daily B&H equity: resample M5 closes to daily last-close (UTC date),
    #     drop rows without a full day (Sunday-evening-only partial sessions,
    #     incl. the boundary first day 2021-01-03, have < 50 bars; every real
    #     trading day incl. holiday-shortened ones has >= 95 bars) ---
    daily = m5.groupby("date")["close"].last()
    counts = m5.groupby("date").size()
    daily_full = daily[counts >= 50]
    n_daily_full = int(len(daily_full))
    cum_ret = daily_full / first_price - 1.0
    eq = cum_ret * ACCOUNT_BASE
    peak = eq.cummax()
    maxdd_usd = float((peak - eq).max())
    maxdd_pct = maxdd_usd / ACCOUNT_BASE * 100
    rets = daily_full.pct_change().dropna()
    sharpe = (float(rets.mean()) / float(rets.std(ddof=1)) * math.sqrt(252)) if len(rets) > 1 else None
    rdd = (bh_ret_base / maxdd_pct) if maxdd_pct > 0 else None

    # --- monthly B&H returns ---
    m5m = m5.copy()
    m5m["month"] = m5m["time"].dt.strftime("%Y-%m")
    month_end = m5m.groupby("month")["close"].last().reindex([str(m) for m in MONTHS])
    prev = month_end.shift(1)
    prev.iloc[0] = first_price
    month_ret = (month_end / prev - 1) * 100
    cum_pct = (1 + month_ret / 100).cumprod() - 1
    cum_pct = cum_pct * 100

    # --- financing (long) ---
    open_dates = sorted(m5["date"].unique())
    swap_days = len(open_dates)
    wed_days = sum(1 for d in open_dates if pd.Timestamp(d).weekday() == 2)
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
        "total_return_pct_price": bh_ret_price,     # (last/first - 1)*100  index basis
        "total_return_pct_base10k": bh_ret_base,    # price_ret_usd/10000*100  1-lot on $10k
        "cagr_pct_price": cagr_price,
        "cagr_pct_base10k": cagr_base,
        "daily_full_days": n_daily_full,
        "max_dd_usd": maxdd_usd,
        "max_dd_pct": maxdd_pct,
        "annualized_sharpe": sharpe,
        "return_dd_ratio": rdd,
        "monthly_ret_pct": {str(m): float(month_ret.loc[str(m)]) for m in MONTHS},
        "monthly_cum_ret_pct": {str(m): float(cum_pct.loc[str(m)]) for m in MONTHS},
        "swap_days": swap_days,
        "wednesday_days": wed_days,
        "total_swap_usd": total_swap_usd,
        "with_financing_long_total_return_pct": with_fin_ret_pct,
        "with_financing_long_cagr_pct": with_fin_cagr,
    }


def verdict(a_name, a, b_name, b):
    """Independent numeric verdict statements on absolute return, Sharpe, return/DD."""
    lines = []
    cmp = lambda x, y: "higher" if x > y else ("lower" if x < y else "equal")
    lines.append(
        f"{a_name} total return {fmt(a['total_return_pct'])}% vs {b_name} total return "
        f"{fmt(b['total_return_pct'])}% -> "
        f"{b_name} {'higher' if b['total_return_pct'] > a['total_return_pct'] else 'NOT higher'} on absolute")
    lines.append(
        f"{a_name} Sharpe {fmt(a['annualized_sharpe'], 3)} vs {b_name} Sharpe "
        f"{fmt(b['annualized_sharpe'], 3)} -> "
        f"{a_name} {'higher' if a['annualized_sharpe'] > b['annualized_sharpe'] else 'NOT higher'} on Sharpe")
    rdd_a, rdd_b = a["return_dd_ratio"], b["return_dd_ratio"]
    if rdd_a is None or rdd_b is None:
        lines.append(f"{a_name} return/DD vs {b_name} return/DD -> cannot compare (n/a)")
    else:
        lines.append(
            f"{a_name} return/DD {fmt(rdd_a)} vs {b_name} return/DD {fmt(rdd_b)} -> "
            f"{a_name} {'higher' if rdd_a > rdd_b else 'NOT higher'} on return/DD")
    return lines


def main():
    print("=" * 78)
    print("ORB full-range performance stats  2021-01-03 .. 2026-04-30")
    print("State base: $10,000 fixed account base, 1 lot, NO compounding.")
    print("=" * 78)

    # ---------- M5 / buy & hold range ----------
    m5 = pd.read_csv(os.path.join(BASE, "us500tick_m5.csv"))
    m5["time"] = pd.to_datetime(m5["time"], unit="s", utc=True)
    m5["date"] = m5["time"].dt.date
    m5["month"] = m5["time"].dt.strftime("%Y-%m")
    first = m5.iloc[0]
    last = m5.iloc[-1]
    first_price = float(first["open"])
    last_price = float(last["close"])
    first_ts = first["time"]
    last_ts = last["time"]
    years = (last_ts - first_ts).total_seconds() / 86400.0 / 365.25
    print(f"\nM5 bars: {len(m5)}   first bar {first_ts} open={first_price}   "
          f"last bar {last_ts} close={last_price}")
    print(f"CAGR years (full data range first M5 bar -> last M5 bar): {years:.4f} "
          f"(stated 5.32)")

    # ---------- ORB aggregated ----------
    agg = {}
    for side in ("py", "mt5"):
        agg[side] = aggregated(side, years)
        a = agg[side]
        tr = load_trades(side)
        chk_pnl = float(tr["profit"].sum())
        print(f"\n--- ORB-{side.upper()} aggregated ---")
        print(f"  trades={a['trades']}  trading_days(session)={a['trading_days']}  "
              f"total_pnl={a['total_pnl']:.2f} (check trades sum {chk_pnl:.2f})")
        print(f"  total_return_pct={a['total_return_pct']:.2f}%  CAGR={a['cagr_pct']:.2f}%  "
              f"maxDD=${a['max_dd_usd']:.2f} ({a['max_dd_pct']:.2f}%)")
        print(f"  daily mean={a['daily_pnl_mean']:.4f}  std={a['daily_pnl_std']:.4f}  "
              f"Sharpe={a['annualized_sharpe']:.3f}  return/DD={fmt(a['return_dd_ratio'])}")

    # ---------- Monthly matrix ----------
    per_day_py = build_daily_pnl("py")[4]
    per_day_mt5 = build_daily_pnl("mt5")[4]
    mp = {r["month"]: r for r in monthly_stats("py", per_day_py)}
    mm = {r["month"]: r for r in monthly_stats("mt5", per_day_mt5)}
    bh = bh_metrics(m5, first_price, last_price, first_ts, last_ts, years)

    header = ("month;py_trading_days;py_skipped_total;py_skipped_gap;py_trades;"
              "py_winrate_pct;py_pnl;py_pnl_pct;py_maxdd;"
              "mt5_trading_days;mt5_skipped_total;mt5_skipped_gap;mt5_trades;"
              "mt5_winrate_pct;mt5_pnl;mt5_pnl_pct;mt5_maxdd;"
              "bh_month_ret_pct;bh_cum_ret_pct")
    print("\n" + "=" * 78)
    print("MONTHLY MATRIX (base $10,000, 1 lot, no compounding)")
    print(header.replace(";", " | "))
    mrows = []
    for m in MONTHS:
        mkey = str(m)
        p, t = mp[mkey], mm[mkey]
        bmr = bh["monthly_ret_pct"][mkey]
        bcr = bh["monthly_cum_ret_pct"][mkey]
        row = {
            "month": mkey,
            "py_trading_days": p["trading_days"], "py_skipped_total": p["skipped_total"],
            "py_skipped_gap": p["skipped_gap"], "py_trades": p["trades"],
            "py_winrate_pct": p["winrate_pct"], "py_pnl": p["pnl"],
            "py_pnl_pct": p["pnl_pct"], "py_maxdd": p["maxdd"],
            "mt5_trading_days": t["trading_days"], "mt5_skipped_total": t["skipped_total"],
            "mt5_skipped_gap": t["skipped_gap"], "mt5_trades": t["trades"],
            "mt5_winrate_pct": t["winrate_pct"], "mt5_pnl": t["pnl"],
            "mt5_pnl_pct": t["pnl_pct"], "mt5_maxdd": t["maxdd"],
            "bh_month_ret_pct": bmr, "bh_cum_ret_pct": bcr,
        }
        mrows.append(row)
        print(f"{mkey}  py[{p['trading_days']:>3} d, {p['skipped_total']:>2} sk, "
              f"{p['trades']:>3} tr, {p['winrate_pct']:>5.1f}%, {p['pnl']:>+9.2f}, "
              f"{p['pnl_pct']:>+6.2f}%, dd {p['maxdd']:>7.2f}]  "
              f"mt5[{t['trading_days']:>3} d, {t['skipped_total']:>2} sk, "
              f"{t['trades']:>3} tr, {t['winrate_pct']:>5.1f}%, {t['pnl']:>+9.2f}, "
              f"{t['pnl_pct']:>+6.2f}%, dd {t['maxdd']:>7.2f}]  "
              f"bh {bmr:>+6.2f}% (cum {bcr:>+7.2f}%)")

    pd.DataFrame(mrows).to_csv(os.path.join(BASE, "monthly_matrix.csv"), index=False)

    # months where py / mt5 materially differ
    print("\nMonths where PY vs MT5 materially differ (|pnl|>20 or winrate diff>5pp):")
    flag = False
    for m in MONTHS:
        mkey = str(m)
        p, t = mp[mkey], mm[mkey]
        if abs(p["pnl"] - t["pnl"]) > 20 or abs(p["winrate_pct"] - t["winrate_pct"]) > 5:
            print(f"  {mkey}: py_pnl={p['pnl']:+.2f} mt5_pnl={t['pnl']:+.2f} "
                  f"| py_wr={p['winrate_pct']:.1f}% mt5_wr={t['winrate_pct']:.1f}%")
            flag = True
    if not flag:
        print("  (none)")

    # ---------- Buy & hold ----------
    print("\n" + "=" * 78)
    print("BUY & HOLD benchmark (US500-tick, full data range)")
    print(f"  first_price (open 1st M5 bar 2021-01-03 23:00) = {first_price:.3f}")
    print(f"  last_price  (close last M5 bar 2026-04-30 23:55) = {last_price:.3f}")
    print(f"  price return (index basis)      = {bh['total_return_pct_price']:.2f}%  "
          f"CAGR {bh['cagr_pct_price']:.2f}%")
    print(f"  $10k base / 1-lot (no-financing) = {bh['total_return_pct_base10k']:.2f}%  "
          f"CAGR {bh['cagr_pct_base10k']:.2f}%  (optimistic upper bound)")
    print(f"  daily B&H: full days={bh['daily_full_days']}  maxDD=${bh['max_dd_usd']:.2f} "
          f"({bh['max_dd_pct']:.2f}%)  Sharpe={bh['annualized_sharpe']:.3f}  "
          f"return/DD={fmt(bh['return_dd_ratio'])}")
    print(f"  financing (LONG): swap_days={bh['swap_days']} (Wed={bh['wednesday_days']}, "
          f"x{WED_FACTOR})  total_swap_usd={bh['total_swap_usd']:.2f}")
    print(f"  with-financing-long total return (on $10k) = "
          f"{bh['with_financing_long_total_return_pct']:.2f}%")

    # ---------- Final comparison ----------
    bh_ex = {
        "total_return_pct": bh["total_return_pct_base10k"],
        "cagr_pct": bh["cagr_pct_base10k"],
        "max_dd_pct": bh["max_dd_pct"],
        "annualized_sharpe": bh["annualized_sharpe"],
        "return_dd_ratio": bh["return_dd_ratio"],
    }
    bh_wf = {
        "total_return_pct": bh["with_financing_long_total_return_pct"],
        "cagr_pct": bh["with_financing_long_cagr_pct"],
        "max_dd_pct": bh["max_dd_pct"],
        "annualized_sharpe": bh["annualized_sharpe"],
        "return_dd_ratio": bh["with_financing_long_total_return_pct"] / bh["max_dd_pct"] if bh["max_dd_pct"] > 0 else None,
    }
    cols = {
        "ORB-MT5": {"total_return_pct": agg["mt5"]["total_return_pct"],
                    "cagr_pct": agg["mt5"]["cagr_pct"],
                    "max_dd_pct": agg["mt5"]["max_dd_pct"],
                    "annualized_sharpe": agg["mt5"]["annualized_sharpe"],
                    "return_dd_ratio": agg["mt5"]["return_dd_ratio"]},
        "ORB-Python": {"total_return_pct": agg["py"]["total_return_pct"],
                       "cagr_pct": agg["py"]["cagr_pct"],
                       "max_dd_pct": agg["py"]["max_dd_pct"],
                       "annualized_sharpe": agg["py"]["annualized_sharpe"],
                       "return_dd_ratio": agg["py"]["return_dd_ratio"]},
        "B&H ex-financing": bh_ex,
        "B&H with-financing-long": bh_wf,
    }
    print("\n" + "=" * 78)
    print("FINAL COMPARISON (all returns % of $10,000 base / 1 lot unless noted)")
    print(f"{'metric':<22}" + "".join(f"{k:>20}" for k in cols))
    for metric in ("total_return_pct", "cagr_pct", "max_dd_pct", "annualized_sharpe", "return_dd_ratio"):
        label = {"total_return_pct": "total_return_pct",
                 "cagr_pct": "cagr_pct", "max_dd_pct": "max_dd_pct",
                 "annualized_sharpe": "annualized_sharpe",
                 "return_dd_ratio": "return_dd_ratio"}[metric]
        line = f"{label:<22}"
        for k in cols:
            v = cols[k][metric]
            line += f"{fmt(v):>20}"
        print(line)

    # ---------- Verdicts ----------
    print("\n" + "=" * 78)
    print("VERDICT STATEMENTS (each independent)")
    for a_side, a_name in (("mt5", "ORB MT5"), ("py", "ORB Python")):
        for b_name, b in (("B&H ex-financing", bh_ex), ("B&H with-financing-long", bh_wf)):
            for ln in verdict(a_name, agg[a_side], b_name, b):
                print("  " + ln)

    # ---------- Per-year sanity ----------
    print("\n" + "=" * 78)
    print("PER-YEAR SANITY")
    for side in ("py", "mt5"):
        tr = load_trades(side)
        gy = tr.groupby(tr["open_time"].dt.year).agg(n=("profit", "size"), pnl=("profit", "sum"))
        print(f"  {side.upper()}: year n / pnl")
        for y, r in gy.iterrows():
            print(f"    {y}: n={int(r['n']):>3}  pnl={r['pnl']:>+9.2f}")
        print(f"    TOTAL: n={len(tr)}  pnl={tr['profit'].sum():+.2f}")

    known = {"py": (442, 531.13), "mt5": (447, None)}
    for side, (n, pnl) in known.items():
        tr = load_trades(side)
        ok_n = len(tr) == n
        ok_p = (pnl is None) or abs(float(tr["profit"].sum()) - pnl) < 0.01
        print(f"  sanity {side}: trades {len(tr)} vs known {n} -> {'OK' if ok_n else 'MISMATCH'}"
              + (f"; pnl {tr['profit'].sum():.2f} vs known {pnl} -> {'OK' if ok_p else 'MISMATCH'}" if pnl is not None else ""))

    # reference comparison for python side (py_run.log: over all 1656 day records)
    print(f"\n  reference (py_run.log, all 1656 day records incl. evening-only): "
          f"daily mean 0.3207 std 14.7115 Sharpe 0.3461 maxDD 499.15")
    print(f"  here (session days only): py daily mean {agg['py']['daily_pnl_mean']:.4f} "
          f"std {agg['py']['daily_pnl_std']:.4f} Sharpe {agg['py']['annualized_sharpe']:.3f} "
          f"maxDD {agg['py']['max_dd_usd']:.2f}")

    # ---------- JSON ----------
    out = {
        "params": {
            "account_base_usd": ACCOUNT_BASE,
            "lot": LOT,
            "usd_per_point": USD_PER_POINT,
            "compounding": False,
            "data_range": [str(first_ts), str(last_ts)],
            "years": years,
            "years_stated": 5.32,
            "swap_long_points_per_day": SWAP_LONG,
            "swap_short_points_per_day": 2.0,
            "rollover3days": 5,
            "wednesday_swap_factor": WED_FACTOR,
            "full_day_min_bars": 50,
        },
        "aggregated": {"python": agg["py"], "mt5": agg["mt5"]},
        "buy_and_hold": bh,
        "comparison": cols,
        "verdicts": {
            "mt5_vs_bh_ex": verdict("ORB MT5", agg["mt5"], "B&H ex-financing", bh_ex),
            "mt5_vs_bh_wf": verdict("ORB MT5", agg["mt5"], "B&H with-financing-long", bh_wf),
            "py_vs_bh_ex": verdict("ORB Python", agg["py"], "B&H ex-financing", bh_ex),
            "py_vs_bh_wf": verdict("ORB Python", agg["py"], "B&H with-financing-long", bh_wf),
        },
        "per_year": {side: {str(int(y)): {"trades": int(r["n"]), "pnl": float(r["pnl"])}
                            for y, r in load_trades(side)
                                .groupby(load_trades(side)["open_time"].dt.year)
                                .agg(n=("profit", "size"), pnl=("profit", "sum")).iterrows()}
                     for side in ("py", "mt5")},
        "monthly_matrix": mrows,
    }
    with open(os.path.join(BASE, "aggregated.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWROTE: {os.path.join(BASE, 'monthly_matrix.csv')}")
    print(f"WROTE: {os.path.join(BASE, 'aggregated.json')}")
    print("=" * 78)
    print("DONE")


if __name__ == "__main__":
    main()
