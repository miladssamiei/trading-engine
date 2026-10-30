"""Rolling cointegration pairs study: US500 (US500.c D1) vs USA30 (onboarded
M1 tape aggregated to NY-cash-day closes).

Implements SPEC_PAIRS_US500_USA30 / GATE_PAIRS_US500_USA30 (pre-declared
2026-09-10) exactly: L=252 rolling log-OLS hedge, out-of-sample spread
Z-score, |z|>=2 entries at close, z-cross-0 exits with |z|>=4 hard stop,
0.51/2.28 pt spreads per side, documented US500 swaps and terminal-queried
USA30 swaps (Wednesday x5), gates G-P1/G-P2/G-P3.

Outputs: data/results/pairs_us500_usa30/pairs_trades.csv and
REPORT_PAIRS_US500_USA30.txt.
"""

import datetime as dt
import os

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

REPO = r"C:\Users\Milad\Desktop\RR"
STUDY = os.path.join(REPO, "data", "results", "pairs_us500_usa30")
D1_CSV = os.path.join(REPO, "data", "results", "momentum_multihorizon",
                      "us500c_d1.csv")
M1_CSV = os.path.join(REPO, "data", "results", "usa30_onboarding",
                      "usa30tick_m1.csv")
TERMINAL_EXE = r"C:\Program Files\MetaTrader 5\terminal64.exe"

L = 252
Z_ENTRY = 2.0
Z_STOP = 4.0
SPREAD_US500 = 0.51
SPREAD_USA30 = 2.28
SWAP_US500_LONG = -20.0
SWAP_US500_SHORT = 2.0
SWAP_WED_MULT = 5


def load_us500_daily():
    d1 = pd.read_csv(D1_CSV, usecols=["time", "close"])
    d1["date"] = pd.to_datetime(d1["time"], unit="s", utc=True).dt.date
    s = d1.groupby("date")["close"].last()
    return s


def load_usa30_daily():
    m1 = pd.read_csv(M1_CSV, usecols=["time", "close"])
    ts = pd.to_datetime(m1["time"], unit="s", utc=True).dt.tz_convert(
        "America/New_York")
    wall = ts.dt.hour * 60 + ts.dt.minute
    keep = m1[wall < 960].copy()
    keep["date"] = ts[wall < 960].dt.date
    s = keep.groupby("date")["close"].last()
    return s


def query_terminal():
    info = {}
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize(TERMINAL_EXE):
            return info, "initialize failed: %s" % (mt5.last_error(),)
        for sym, key in [("US500.c", "us500"), ("USA30.c", "usa30")]:
            mt5.symbol_select(sym, True)
            si = mt5.symbol_info(sym)
            if si is None:
                cands = (list(mt5.symbols_get("*US30*") or [])
                         + list(mt5.symbols_get("*USA30*") or [])
                         + list(mt5.symbols_get("*DJ*") or []))
                info[key + "_missing"] = {
                    "symbol": sym,
                    "candidates": sorted({s.name for s in cands})[:10],
                }
                continue
            tick_size = getattr(si, "trade_tick_size", 0.0) or 0.0
            tick_val = getattr(si, "trade_tick_value", 0.0) or 0.0
            info[key] = {
                "symbol": sym,
                "swap_long": getattr(si, "swap_long", None),
                "swap_short": getattr(si, "swap_short", None),
                "swap_mode": getattr(si, "swap_mode", None),
                "dollars_per_point": (tick_val / tick_size
                                      if tick_size > 0 else None),
            }
        mt5.shutdown()
        return info, "ok"
    except Exception as exc:
        return info, "exception: %s" % exc


def rolling_model(x, y):
    n = len(x)
    alpha = np.full(n, np.nan)
    beta = np.full(n, np.nan)
    sigma = np.full(n, np.nan)
    z = np.full(n, np.nan)
    adf_p = np.full(n, np.nan)
    for t in range(L, n):
        xs = x[t - L:t]
        ys = y[t - L:t]
        b, a = np.polyfit(xs, ys, 1)
        resid = ys - (a + b * xs)
        sd = resid.std(ddof=2)
        alpha[t], beta[t], sigma[t] = a, b, sd
        if sd > 0:
            z[t] = (y[t] - (a + b * x[t])) / sd
        try:
            adf_p[t] = adfuller(resid, regression="c", autolag="AIC")[1]
        except Exception:
            adf_p[t] = np.nan
    return alpha, beta, sigma, z, adf_p


def run_trades(df, z, beta, v500, v30, swap30_long, swap30_short):
    dates = list(df.index)
    p500 = df["us500"].to_numpy()
    p30 = df["usa30"].to_numpy()
    n = len(df)
    daily = pd.Series(0.0, index=dates)
    daily_gross = pd.Series(0.0, index=dates)
    trades = []
    pos = None
    for t in range(n):
        if np.isnan(z[t]):
            continue
        if pos is not None:
            d = pos["dir"]
            reason = None
            if d == "SHORT" and z[t] <= 0.0:
                reason = "MEAN_REVERT"
            elif d == "LONG" and z[t] >= 0.0:
                reason = "MEAN_REVERT"
            if reason is None and abs(z[t]) >= Z_STOP:
                reason = "HARD_STOP"
            if reason is None and t == n - 1:
                reason = "END"
            if reason is not None:
                ie, xe = pos["entry_i"], t
                s500 = -1.0 if d == "SHORT" else 1.0
                s30 = -s500
                lots30 = pos["lots30"]
                gross = (s500 * (p500[xe] - p500[ie]) * v500
                         + s30 * lots30 * (p30[xe] - p30[ie]) * v30)
                spread_cost = 2.0 * (SPREAD_US500 * v500
                                     + SPREAD_USA30 * v30 * lots30)
                swap_total = 0.0
                for nd in (dates[ie] + dt.timedelta(days=k)
                           for k in range(1, (dates[xe] - dates[ie]).days + 1)):
                    mult = SWAP_WED_MULT if nd.weekday() == 2 else 1
                    sw500 = (SWAP_US500_SHORT if s500 < 0 else
                             SWAP_US500_LONG) * v500
                    sw30 = ((swap30_short if s30 < 0 else swap30_long)
                            if swap30_long is not None else 0.0) * v30 * lots30
                    swap_total += mult * (sw500 + sw30)
                for k in range(ie + 1, xe + 1):
                    mtm = (s500 * (p500[k] - p500[k - 1]) * v500
                           + s30 * lots30 * (p30[k] - p30[k - 1]) * v30)
                    daily_gross.iloc[k] += mtm
                    daily.iloc[k] += mtm
                swap_daily_alloc = swap_total / max(1, (dates[xe] - dates[ie]).days)
                for k in range(ie + 1, xe + 1):
                    daily.iloc[k] += swap_daily_alloc
                daily.iloc[ie] -= spread_cost / 2.0
                daily.iloc[xe] -= spread_cost / 2.0
                net = gross - spread_cost + swap_total
                trades.append({
                    "entry_date": dates[ie].strftime("%Y%m%d"),
                    "exit_date": dates[xe].strftime("%Y%m%d"),
                    "direction": "SHORT_SPREAD" if d == "SHORT" else "LONG_SPREAD",
                    "z_entry": round(pos["z_entry"], 4),
                    "z_exit": round(z[t], 4),
                    "beta": round(pos["beta"], 6),
                    "lots30": round(lots30, 6),
                    "p500_entry": p500[ie], "p500_exit": p500[xe],
                    "p30_entry": p30[ie], "p30_exit": p30[xe],
                    "hold_nights": (dates[xe] - dates[ie]).days,
                    "exit_reason": reason,
                    "gross_usd": round(gross, 2),
                    "spread_cost_usd": round(spread_cost, 2),
                    "swap_usd": round(swap_total, 2),
                    "net_usd": round(net, 2),
                })
                pos = None
                continue
        if pos is None:
            if z[t] >= Z_ENTRY:
                pos = {"dir": "SHORT", "entry_i": t, "z_entry": z[t],
                       "beta": beta[t],
                       "lots30": beta[t] * p500[t] / p30[t]}
            elif z[t] <= -Z_ENTRY:
                pos = {"dir": "LONG", "entry_i": t, "z_entry": z[t],
                       "beta": beta[t],
                       "lots30": beta[t] * p500[t] / p30[t]}
    return trades, daily, daily_gross


def main():
    os.makedirs(STUDY, exist_ok=True)
    us500 = load_us500_daily()
    usa30 = load_usa30_daily()
    df = pd.DataFrame({"us500": us500, "usa30": usa30}).dropna()
    df.index.name = "ny_date"
    span = "%s..%s (%d days)" % (df.index[0], df.index[-1], len(df))
    print("overlap:", span)

    tinfo, tstatus = query_terminal()
    print("terminal:", tstatus)
    for k, v in tinfo.items():
        print("  %s: %s" % (k, v))
    v500 = 1.0
    v30 = 1.0
    swap30_long = None
    swap30_short = None
    if "us500" in tinfo and tinfo["us500"]["dollars_per_point"]:
        v500 = tinfo["us500"]["dollars_per_point"]
    if "usa30" in tinfo:
        si = tinfo["usa30"]
        if si["dollars_per_point"]:
            v30 = si["dollars_per_point"]
        if si["swap_long"] is not None:
            swap30_long = float(si["swap_long"])
            swap30_short = float(si["swap_short"])
    print("v500=%.4f v30=%.4f swap30_long=%s swap30_short=%s"
          % (v500, v30, swap30_long, swap30_short))

    x = np.log(df["usa30"].to_numpy())
    y = np.log(df["us500"].to_numpy())
    alpha, beta, sigma, z, adf_p = rolling_model(x, y)
    adf_frac = float(np.mean(adf_p[L:] < 0.05))
    print("ADF fraction p<0.05 over rolling windows: %.4f" % adf_frac)

    trades, daily, daily_gross = run_trades(df, z, beta, v500, v30,
                                            swap30_long, swap30_short)
    tdf = pd.DataFrame(trades)
    tdf.to_csv(os.path.join(STUDY, "pairs_trades.csv"), index=False)

    n_tr = len(tdf)
    gross_total = float(tdf["gross_usd"].sum()) if n_tr else 0.0
    gross_mean = float(tdf["gross_usd"].mean()) if n_tr else 0.0
    spread_total = float(tdf["spread_cost_usd"].sum()) if n_tr else 0.0
    swap_total = float(tdf["swap_usd"].sum()) if n_tr else 0.0
    net_total = float(tdf["net_usd"].sum()) if n_tr else 0.0
    sharpe_net = (daily.mean() / daily.std(ddof=1) * np.sqrt(252)
                  if daily.std(ddof=1) > 0 else 0.0)
    sharpe_gross = (daily_gross.mean() / daily_gross.std(ddof=1) * np.sqrt(252)
                    if daily_gross.std(ddof=1) > 0 else 0.0)
    win = float((tdf["net_usd"] > 0).mean()) if n_tr else 0.0
    avg_hold = float(tdf["hold_nights"].mean()) if n_tr else 0.0
    beta_mean = float(np.nanmean(beta[L:]))

    g1 = adf_frac >= 0.40
    g2 = gross_mean > 0.0
    g3 = sharpe_net > 0.0
    verdict = "PASS" if (g1 and g2 and g3) else "FAIL"

    lines = []
    lines.append("REPORT_PAIRS_US500_USA30")
    lines.append("Study dir: data/results/pairs_us500_usa30/")
    lines.append("Script: evaluate/pairs_us500_usa30.py")
    lines.append("Executed: %s UTC" % dt.datetime.utcnow().strftime(
        "%Y-%m-%d %H:%M:%S"))
    lines.append("SPEC: SPEC_PAIRS_US500_USA30.txt (pre-declared 2026-09-10)")
    lines.append("")
    lines.append("DATA")
    lines.append("  us500: us500c_d1.csv D1 closes (%d bars)" % len(us500))
    lines.append("  usa30: usa30tick_m1.csv M1 -> last-bar-before-16:00 NY"
                 " daily closes (%d days)" % len(usa30))
    lines.append("  joined NY cash dates: %s" % span)
    lines.append("  note: overlap extends past the expected 2026-04 boundary"
                 " if the refreshed D1 spine allows; actual intersection"
                 " governs")
    lines.append("  rolling mean beta: %.4f (min %.4f max %.4f)"
                 % (beta_mean, np.nanmin(beta[L:]), np.nanmax(beta[L:])))
    lines.append("")
    lines.append("COST INPUTS")
    lines.append("  spreads: US500 %.2f pt, USA30 %.2f pt (per side,"
                 " both legs)" % (SPREAD_US500, SPREAD_USA30))
    lines.append("  US500 swap: long %.0f / short %.0f pt per night,"
                 " Wednesday x%d" % (SWAP_US500_LONG, SWAP_US500_SHORT,
                                     SWAP_WED_MULT))
    if swap30_long is not None:
        lines.append("  USA30.c swap (terminal): long %.2f / short %.2f"
                     " (swap_mode=%s), Wednesday x%d assumed"
                     % (swap30_long, swap30_short,
                        tinfo["usa30"]["swap_mode"], SWAP_WED_MULT))
    else:
        cands = tinfo.get("usa30_missing", {}).get("candidates", [])
        lines.append("  USA30.c swap: MISSING (symbol not found in broker"
                     " symbol table; near-matches: %s) - USA30 swap"
                     " treated as 0.0 AND FLAGGED; net metrics are"
                     " net-of-spread only" % (cands or "none"))
    lines.append("  dollars/point: US500 %.4f, USA30 %.4f" % (v500, v30))
    lines.append("")
    lines.append("GATES")
    lines.append("  G-P1 ADF p<0.05 fraction %.4f >= 0.40 : %s"
                 % (adf_frac, "PASS" if g1 else "FAIL"))
    lines.append("  G-P2 gross expectancy per trade %.2f USD > 0 : %s"
                 % (gross_mean, "PASS" if g2 else "FAIL"))
    lines.append("  G-P3 net annualized Sharpe %.4f > 0 : %s"
                 % (sharpe_net, "PASS" if g3 else "FAIL"))
    lines.append("")
    lines.append("PERFORMANCE (USD, 1.0-lot US500 base, no compounding)")
    lines.append("  trades: %d" % n_tr)
    lines.append("  hold avg: %.1f nights" % avg_hold)
    lines.append("  gross total %.2f | spread total -%.2f | swap total %+.2f"
                 % (gross_total, spread_total, swap_total))
    lines.append("  net total %.2f | win rate %.1f%%" % (net_total,
                                                         win * 100))
    lines.append("  Sharpe gross %.4f | Sharpe net %.4f"
                 % (sharpe_gross, sharpe_net))
    if n_tr:
        lines.append("  exits: %s" % tdf["exit_reason"].value_counts()
                     .to_dict())
        lines.append("  by direction: %s" % tdf.groupby("direction")
                     ["net_usd"].agg(["count", "sum"]).to_dict("index"))
    lines.append("")
    lines.append("VERDICT: %s" % verdict)
    if not g3 and swap30_long is None:
        lines.append("CAVEAT: G-P3 evaluated with USA30 swap unavailable"
                     " (broker has no USA30.c); true net Sharpe can only be"
                     " worse than reported unless the USA30 leg swap is a"
                     " large credit, which is atypical.")
    lines.append("")
    lines.append("Artifacts: pairs_trades.csv (%d rows), this report."
                 % n_tr)
    with open(os.path.join(STUDY, "REPORT_PAIRS_US500_USA30.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nreport written; verdict=%s" % verdict)


if __name__ == "__main__":
    main()
