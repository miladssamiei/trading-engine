import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUT_DIR = os.path.join("data", "results", "pairs_futures")
RAW_CSV = os.path.join(OUT_DIR, "raw_eikon_data.csv")
TRADES_CSV = os.path.join(OUT_DIR, "futures_trades.csv")
REPORT_TXT = os.path.join(OUT_DIR, "REPORT_PAIRS_FUTURES.txt")
SPEC_TXT = os.path.join(OUT_DIR, "SPEC_PAIRS_FUTURES.txt")
GATE_TXT = os.path.join(OUT_DIR, "GATE_PAIRS_FUTURES.txt")

START_DATE = "2016-01-01"
END_DATE = "2026-09-01"
YEAR_FIRST = 2016
YEAR_LAST = 2026
RIC_ES = "ESc1"
RIC_YM = "YMc1"

L = 252
Z_ENTRY = 2.0
Z_STOP = 4.0
ES_MULT = 50.0
YM_MULT = 5.0
ES_SPREAD_PTS = 0.25
YM_SPREAD_PTS = 1.0
COMMISSION_RT = 4.0

ADF_WIN = 252
ADF_STEP = 21
ADF_PVAL = 0.05
ADF_FRAC_MIN = 0.30
NET_MIN = 0.0
SHARPE_MIN = 0.50

SWAP_LONG_PER_NIGHT = -20.0
SWAP_SHORT_PER_NIGHT = 2.0
SWAP_WED_X = 5.0

TRADE_COLS = [
    "Entry_Date", "Exit_Date", "Hold_Time_Days", "Direction", "ES_Contracts",
    "YM_Contracts", "Z_Entry", "Z_Exit", "Beta_Entry", "Gross_PnL_USD",
    "Costs_USD", "Net_PnL_USD",
]


def write_incomplete(reason):
    lines = [
        "REPORT_PAIRS_FUTURES",
        f"Written: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "VERDICT: INCOMPLETE - NO RESULTS",
        f"Reason: {reason}",
        "",
        "This is an execution/environment failure, NOT a strategy FAIL verdict.",
        "Required to complete the run:",
        "  1. python package 'eikon' installed in the running environment",
        "  2. EIKON_APP_KEY environment variable set (or interactive key entry)",
        "  3. Refinitiv Workspace / Eikon desktop running and signed in with Data API permissioning",
        "No substitute data source may be used without a new pre-declared SPEC.",
    ]
    with open(REPORT_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def get_app_key():
    key = os.environ.get("EIKON_APP_KEY", "").strip()
    if key:
        return key
    try:
        key = input("EIKON_APP_KEY not set. Enter Eikon App Key: ").strip()
    except EOFError:
        return ""
    return key


def fetch_close(ek, ric):
    frames = []
    for year in range(YEAR_FIRST, YEAR_LAST + 1):
        s = max(f"{year}-01-01", START_DATE)
        e = min(f"{year}-12-31", END_DATE)
        if s > e:
            continue
        ts = ek.get_timeseries(ric, fields="CLOSE", start_date=s, end_date=e, interval="daily")
        if ts is None or len(ts) == 0:
            continue
        frames.append(ts)
    if not frames:
        raise RuntimeError(f"no rows returned for {ric}")
    raw = pd.concat(frames)
    raw = raw.loc[~raw.index.duplicated(keep="last")].sort_index()
    ser = pd.to_numeric(raw.iloc[:, 0], errors="coerce").dropna()
    ser.name = ric
    return ser


def compute_model(df):
    n = len(df)
    les = np.log(df["ES"].to_numpy())
    lym = np.log(df["YM"].to_numpy())
    alpha = np.full(n, np.nan)
    beta = np.full(n, np.nan)
    sigma = np.full(n, np.nan)
    spread = np.full(n, np.nan)
    z = np.full(n, np.nan)
    for t in range(L, n):
        x = lym[t - L:t]
        y = les[t - L:t]
        b, a = np.polyfit(x, y, 1)
        resid = y - (a + b * x)
        s = float(np.std(resid, ddof=1))
        alpha[t] = a
        beta[t] = b
        sigma[t] = s
        spread[t] = les[t] - (a + b * lym[t])
        z[t] = spread[t] / s
    return alpha, beta, sigma, spread, z


def backtest(df, beta, z):
    n = len(df)
    es = df["ES"].to_numpy()
    ym = df["YM"].to_numpy()
    dates = df.index
    daily = pd.Series(0.0, index=dates)
    trades = []
    pos = None
    for t in range(L, n):
        pe = es[t]
        pm = ym[t]
        d = dates[t]
        zt = z[t]
        if pos is not None:
            daily.iat[t] += (pe - pos["prev_es"]) * ES_MULT * pos["es_dir"]
            daily.iat[t] += (pm - pos["prev_ym"]) * YM_MULT * pos["ym_c"] * pos["ym_dir"]
            hit_target = (zt <= 0.0) if pos["dir"] == "SHORT_SPREAD" else (zt >= 0.0)
            hit_stop = abs(zt) >= Z_STOP
            if hit_target or hit_stop:
                daily.iat[t] -= pos["cost"] / 2.0
                gross = (
                    (pe - pos["es_entry"]) * ES_MULT * pos["es_dir"]
                    + (pm - pos["ym_entry"]) * YM_MULT * pos["ym_c"] * pos["ym_dir"]
                )
                trades.append({
                    "Entry_Date": pos["date"],
                    "Exit_Date": d,
                    "Hold_Time_Days": (d - pos["date"]).days,
                    "Direction": pos["dir"],
                    "ES_Contracts": 1,
                    "YM_Contracts": pos["ym_c"],
                    "Z_Entry": pos["z_entry"],
                    "Z_Exit": float(zt),
                    "Beta_Entry": pos["beta"],
                    "Gross_PnL_USD": float(gross),
                    "Costs_USD": float(pos["cost"]),
                    "Net_PnL_USD": float(gross - pos["cost"]),
                })
                pos = None
        elif not np.isnan(zt) and (zt >= Z_ENTRY or zt <= -Z_ENTRY):
            if zt >= Z_ENTRY:
                pdir, es_dir, ym_dir = "SHORT_SPREAD", -1, 1
            else:
                pdir, es_dir, ym_dir = "LONG_SPREAD", 1, -1
            ym_c = int(round((1.0 * pe * ES_MULT) / (pm * YM_MULT) * beta[t]))
            cost = (
                (ES_SPREAD_PTS * ES_MULT + COMMISSION_RT) * 1.0
                + (YM_SPREAD_PTS * YM_MULT + COMMISSION_RT) * ym_c
            )
            pos = {
                "date": d,
                "dir": pdir,
                "es_dir": es_dir,
                "ym_dir": ym_dir,
                "ym_c": ym_c,
                "es_entry": pe,
                "ym_entry": pm,
                "prev_es": pe,
                "prev_ym": pm,
                "z_entry": float(zt),
                "beta": float(beta[t]),
                "cost": cost,
            }
            daily.iat[t] -= cost / 2.0
        if pos is not None:
            pos["prev_es"] = pe
            pos["prev_ym"] = pm
    return daily, trades


def adf_density(spread):
    from statsmodels.tsa.stattools import adfuller
    s = spread[L:]
    s = s[np.isfinite(s)]
    pvals = []
    for end in range(ADF_WIN, len(s) + 1, ADF_STEP):
        w = s[end - ADF_WIN:end]
        try:
            pvals.append(float(adfuller(w, autolag="AIC")[1]))
        except Exception:
            continue
    if not pvals:
        return float("nan"), 0
    arr = np.array(pvals)
    return float(np.mean(arr < ADF_PVAL)), len(pvals)


def cfd_swap_counterfactual(trades):
    total = 0.0
    for tr in trades:
        nights = tr["Hold_Time_Days"]
        if nights <= 0:
            continue
        weds = len(pd.date_range(tr["Entry_Date"], tr["Exit_Date"] - pd.Timedelta(days=1), freq="W-WED"))
        rate = SWAP_LONG_PER_NIGHT if tr["Direction"] == "LONG_SPREAD" else SWAP_SHORT_PER_NIGHT
        total += rate * (nights + (SWAP_WED_X - 1.0) * weds)
    return total


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for req in (SPEC_TXT, GATE_TXT):
        if not os.path.exists(req):
            write_incomplete(f"protocol violation: {req} missing; SPEC/GATE must pre-date the run")
            print(f"FATAL: {req} missing")
            sys.exit(2)
    key = get_app_key()
    if not key:
        write_incomplete("no Eikon App Key: EIKON_APP_KEY not set and no interactive input available")
        print("FATAL: no Eikon App Key (set EIKON_APP_KEY and re-run)")
        sys.exit(2)
    try:
        import eikon as ek
    except Exception as ex:
        write_incomplete(f"eikon package import failed: {ex}")
        print(f"FATAL: eikon import failed: {ex}")
        sys.exit(2)
    ek.set_app_key(key)
    print(f"pulling {RIC_ES} daily CLOSE {START_DATE}..{END_DATE}")
    try:
        es = fetch_close(ek, RIC_ES)
        print(f"{RIC_ES} rows: {len(es)}")
        ym = fetch_close(ek, RIC_YM)
        print(f"{RIC_YM} rows: {len(ym)}")
    except Exception as ex:
        write_incomplete(f"Eikon get_timeseries failed (desktop running? key permissioned?): {ex}")
        print(f"FATAL: {ex}")
        sys.exit(3)
    df = pd.concat([es, ym], axis=1, join="inner")
    df.columns = ["ES", "YM"]
    df = df.dropna()
    if len(df) < 2 * L:
        write_incomplete(f"insufficient aligned data: {len(df)} rows after inner join/dropna")
        print("FATAL: insufficient aligned data")
        sys.exit(3)
    df.to_csv(RAW_CSV, index_label="Date")
    print(f"aligned rows: {len(df)} ({df.index[0].date()} .. {df.index[-1].date()}) -> {RAW_CSV}")

    try:
        from statsmodels.tsa.stattools import adfuller
    except Exception as ex:
        write_incomplete(f"statsmodels unavailable for the ADF gate: {ex}")
        print(f"FATAL: statsmodels unavailable: {ex}")
        sys.exit(3)

    alpha, beta, sigma, spread, z = compute_model(df)
    daily, trades = backtest(df, beta, z)
    frac, nwin = adf_density(spread)

    d = daily.iloc[L:]
    net_total = float(d.sum())
    sd = float(d.std(ddof=1))
    sharpe = float(d.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")
    cum = d.cumsum()
    maxdd = float((cum - cum.cummax()).min())
    yearly = d.groupby(d.index.year).sum()
    swap_total = cfd_swap_counterfactual(trades)

    n_tr = len(trades)
    nets = [tr["Net_PnL_USD"] for tr in trades]
    win_rate = (sum(1 for x in nets if x > 0) / n_tr) if n_tr else float("nan")
    avg_net = float(np.mean(nets)) if n_tr else float("nan")
    best = max(nets) if n_tr else float("nan")
    worst = min(nets) if n_tr else float("nan")
    avg_hold = float(np.mean([tr["Hold_Time_Days"] for tr in trades])) if n_tr else float("nan")
    avg_cost = float(np.mean([tr["Costs_USD"] for tr in trades])) if n_tr else float("nan")
    total_cost = float(sum(tr["Costs_USD"] for tr in trades)) if n_tr else 0.0

    gf1 = (not np.isnan(frac)) and frac >= ADF_FRAC_MIN
    gf2 = net_total > NET_MIN
    gf3 = (not np.isnan(sharpe)) and sharpe > SHARPE_MIN
    overall = gf1 and gf2 and gf3
    verdict = "PASS" if overall else "FAIL"

    out = pd.DataFrame(trades, columns=TRADE_COLS)
    if len(out):
        out["Entry_Date"] = pd.to_datetime(out["Entry_Date"]).dt.strftime("%Y-%m-%d")
        out["Exit_Date"] = pd.to_datetime(out["Exit_Date"]).dt.strftime("%Y-%m-%d")
        out["Z_Entry"] = out["Z_Entry"].round(3)
        out["Z_Exit"] = out["Z_Exit"].round(3)
        out["Beta_Entry"] = out["Beta_Entry"].round(4)
        for col in ("Gross_PnL_USD", "Costs_USD", "Net_PnL_USD"):
            out[col] = out[col].round(2)
    out.to_csv(TRADES_CSV, index=False)

    betas = beta[L:]
    lines = []
    a = lines.append
    a("REPORT_PAIRS_FUTURES")
    a(f"Written: {datetime.now().isoformat(timespec='seconds')}")
    a("Study: pairs_futures (Hypothesis #20) - ES/YM CME continuous-futures pairs mean-reversion in a zero-swap environment")
    a("Protocol: SPEC/GATE pre-declared before the run; single configuration; no parameter search")
    a("")
    a("DATA")
    a(f"Source: Refinitiv Eikon get_timeseries, daily CLOSE, {RIC_ES} + {RIC_YM}")
    a(f"Requested range {START_DATE}..{END_DATE}; pulled per calendar year, de-duplicated, inner-joined, dropna")
    a(f"Aligned rows: {len(df)}; span {df.index[0].date()} .. {df.index[-1].date()}")
    a(f"Raw backup: {RAW_CSV}")
    a(f"Trades CSV: {TRADES_CSV}")
    a("")
    a("CONFIGURATION")
    a(f"L={L}; entry |z|>={Z_ENTRY}; exit z-cross-0 or |z|>={Z_STOP}; max one position; EOD fills at close")
    a(f"ES 1 contract x ${ES_MULT:.0f}/pt; YM contracts = round((1*P_ES*{ES_MULT:.0f})/(P_YM*{YM_MULT:.0f})*beta_t) x ${YM_MULT:.0f}/pt, fixed at entry")
    a(f"Costs per round trip: ES {ES_SPREAD_PTS} pt (${ES_SPREAD_PTS * ES_MULT:.2f}) + ${COMMISSION_RT:.2f} commission; YM {YM_SPREAD_PTS} pt (${YM_SPREAD_PTS * YM_MULT:.2f}) + ${COMMISSION_RT:.2f} commission per contract; overnight swap $0.00")
    a(f"Mean rolling beta {np.nanmean(betas):.4f} (min {np.nanmin(betas):.4f}, max {np.nanmax(betas):.4f})")
    a("")
    a("GATE VERDICTS")
    a(f"G-F1 cointegration density: ADF p<{ADF_PVAL} on {frac * 100:.1f}% of {nwin} rolling {ADF_WIN}d windows (step {ADF_STEP}) -> {'PASS' if gf1 else 'FAIL'} (threshold >= {ADF_FRAC_MIN * 100:.0f}%)")
    a(f"G-F2 net edge: total net ${net_total:,.2f} -> {'PASS' if gf2 else 'FAIL'} (threshold > ${NET_MIN:.0f})")
    a(f"G-F3 risk-adjusted: annualized net Sharpe {sharpe:.3f} -> {'PASS' if gf3 else 'FAIL'} (threshold > {SHARPE_MIN})")
    a(f"OVERALL VERDICT: {verdict}")
    a("")
    a("PERFORMANCE")
    a(f"Trades: {n_tr}; win rate {win_rate * 100:.1f}%; avg net ${avg_net:,.2f}; best ${best:,.2f}; worst ${worst:,.2f}")
    a(f"Avg hold {avg_hold:.1f} calendar days; avg round-trip cost ${avg_cost:,.2f}; total costs ${total_cost:,.2f}")
    a(f"Net total ${net_total:,.2f}; max drawdown ${maxdd:,.2f}")
    for yv in yearly.index:
        a(f"  {yv}: ${yearly[yv]:,.2f}")
    a("")
    a("ZERO-SWAP FUTURES ENVIRONMENT - EXPLICIT COMPARISON")
    a(f"- Futures book paid spreads+commissions only: ${total_cost:,.2f} total across {n_tr} round trips (avg ${avg_cost:,.2f}); overnight financing $0.00 by construction.")
    a(f"- Counterfactual: charging the same trades the Hypothesis-#19 US500 CFD swap schedule ({SWAP_LONG_PER_NIGHT:.0f}/night on a long ES leg, +{SWAP_SHORT_PER_NIGHT:.0f}/night on a short ES leg, Wednesday x{SWAP_WED_X:.0f}) would add ${swap_total:,.2f} of financing drag, turning net ${net_total:,.2f} into ${net_total + swap_total:,.2f}.")
    a("- Prior context: pairs_us500_usa30 (#19, same rule family on CFD tape) FAILED G-P1 (ADF density 22.31% vs 40%) and G-P3 (net Sharpe -0.8344; swap drag -$5,510 vs gross edge +$1,573). The futures environment removes the financing burden entirely, but it cannot repair a weak cointegration signal; G-F1 above is the binding structural test.")
    a("")
    a("HONESTY NOTES")
    a("- Every spread/z point is out-of-sample by construction (regression window ends at t-1).")
    a("- Costs charged half at the entry day, half at the exit day; flat days carry zero PnL and are included in the Sharpe denominator.")
    a("- Sharpe = mean/std(ddof=1) x sqrt(252) on daily PnL, project convention.")
    a("- The CFD-swap comparison is a reported counterfactual, not a gate.")
    a("- FAIL verdicts are final for this artifact; any re-test requires a new pre-declared SPEC under a new name.")
    with open(REPORT_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print()
    print(f"G-F1 ADF density: {frac * 100:.1f}% of {nwin} windows -> {'PASS' if gf1 else 'FAIL'}")
    print(f"G-F2 net total: ${net_total:,.2f} -> {'PASS' if gf2 else 'FAIL'}")
    print(f"G-F3 Sharpe: {sharpe:.3f} -> {'PASS' if gf3 else 'FAIL'}")
    print(f"OVERALL: {verdict} ({n_tr} trades)")
    print(f"report -> {REPORT_TXT}")
    sys.exit(0)


if __name__ == "__main__":
    main()
