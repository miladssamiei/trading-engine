# -*- coding: utf-8 -*-
"""
momentum_common.py — shared infrastructure for the multi-horizon momentum
study on US500 daily closes (Variant P: paper-faithful Lempérière et al. 2014
signal; Variant R: video retail rubric).

Step 0 decisions (probe_report.md in data/results/momentum_multihorizon/):
  * Primary data: US500.c native D1 (real broker symbol; 2014-06-23..present,
    3,146 bars; tape sane vs the verified US500-tick import: daily-return
    corr 0.969, 92.8% sign agreement, median close diff 0.059%).
    US500-tick D1 (verified import, 2021-01..2026-04) used only for the
    B&H/financing anchor reproduction.
  * P&L basis: $1 per index point per lot (contract 1.0), fixed $10,000
    account base, no compounding — same as every prior strategy here.
  * Spread cost: 1.0 index point per unit of position CHANGE (US500.c spread
    reads 100-300 symbol points = 1-3 index pts; 1.0 kept, conservative vs
    the ~0.41 pt documented on the tick import).
  * Financing: the project-documented swap convention, identical to ORB /
    VWAP MR / B&H work: swap_long -20 $/day/lot, swap_short +2 $/day/lot,
    Wednesday x5 (rollover3days=5 read as Wednesday swap x5 — this reproduces
    the documented 5.3y B&H financing of -$55,360 exactly). Charged on the
    close-of-day position for the coming night. CAVEAT (also in REPORT.md):
    strict MT5 points semantics on US500.c (point=0.01) would instead give
    -20 x 0.01 = -$0.20/day; the project convention is used for
    comparability with all prior financing figures, and ex-financing results
    are always reported alongside so conclusions can be read either way.
  * Execution: signals evaluated on the close of the last trading day of each
    ISO week; new position effective at that same close (weekly-rebalanced
    daily-close strategy; no intrabar precision needed).
"""

import datetime as dt
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

RESULTS = Path(r"C:\Users\Milad\Desktop\RR\data\results\momentum_multihorizon")
D1_US500C = RESULTS / "us500c_d1.csv"
D1_US500TICK = RESULTS / "us500tick_d1.csv"

ACCOUNT_BASE = 10000.0     # fixed $10k base, no compounding (project convention)
USD_PER_POINT = 1.0        # per lot
SPREAD_PT = 1.0            # index points charged per unit of position change
SWAP_LONG = -20.0          # $/day/lot (project-documented convention)
SWAP_SHORT = +2.0          # $/day/lot
WED_FACTOR = 5             # Wednesday swap x5 (rollover3days=5 reading)

# intraday-strategy trade logs for the daily-PnL correlation section
ORB_TRADES = Path(r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present\py_trades.csv")
VWAP_DAYS = Path(r"C:\Users\Milad\Desktop\RR\data\results\vwap_2021_2026\py_days.csv")
TP_TRADES = Path(r"C:\Users\Milad\Desktop\RR\data\results\vwap_trend_pullback_rr\wf_trades.csv")
SP2L_TRADES = Path(r"C:\Users\Milad\Desktop\RR\data\results\sp2l_2021_2026\py_trades.csv")


# ---------------------------------------------------------------- data

def load_d1(path=D1_US500C):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.date
    return df.reset_index(drop=True)


def week_last_days(df):
    """Row indices of the last trading day of each ISO week (rebalance days)."""
    dts = pd.to_datetime(df["date"])
    keys = list(zip(dts.dt.isocalendar().year, dts.dt.isocalendar().week))
    last = {}
    for i, k in enumerate(keys):
        last[k] = i           # overwritten -> ends at last bar of the week
    return sorted(last.values())


def month_end_rows(df):
    """(month_key, row_index of its last trading day, close) per calendar month."""
    dts = pd.to_datetime(df["date"])
    mk = dts.dt.strftime("%Y-%m")
    last = {}
    for i, k in enumerate(mk):
        last[k] = i
    return [(k, i, float(df["close"].iloc[i])) for k, i in sorted(last.items())]


# ---------------------------------------------------------------- costs

def swap_usd(lots, day):
    """Swap accrued for the coming night on the close-of-day position."""
    if lots == 0:
        return 0.0
    rate = SWAP_LONG if lots > 0 else SWAP_SHORT
    if day.weekday() == 2:                     # Wednesday
        rate *= WED_FACTOR
    return rate * abs(lots)


# ---------------------------------------------------------------- simulate

def simulate(df, target_at_close):
    """Daily accounting for a piecewise-constant position series.

    target_at_close[i] = desired lots AT the close of row i (signals computed
    on close[i]; held into day i+1). Price P&L of day i accrues to the
    position set at close i-1. Swap is charged on the close position for the
    coming night (project B&H-anchor convention). Spread is charged on the
    full size of every position CHANGE (including establishment).
    """
    n = len(df)
    closes = df["close"].to_numpy(dtype=float)
    dates = list(df["date"])
    pos = [0.0] * n
    for i in range(n):
        p = target_at_close[i] if target_at_close[i] is not None else pos[i - 1] if i else 0.0
        pos[i] = p
    price = [0.0] * n
    spread = [0.0] * n
    swap = [0.0] * n
    for i in range(1, n):
        price[i] = pos[i - 1] * (closes[i] - closes[i - 1]) * USD_PER_POINT
    for i in range(n):
        d_lots = abs(pos[i] - (pos[i - 1] if i else 0.0))
        spread[i] = d_lots * SPREAD_PT
        swap[i] = swap_usd(pos[i], dates[i])
    exfin = [price[i] - spread[i] for i in range(n)]
    net = [exfin[i] + swap[i] for i in range(n)]
    return {
        "pos": pos, "price": price, "spread": spread, "swap": swap,
        "exfin": exfin, "net": net,
    }


def norm_k(raw):
    """Scale so mean(|lots|)=1 over the given raw-target series."""
    s = sum(abs(x) for x in raw)
    return len(raw) / s if s > 0 else 1.0


# ---------------------------------------------------------------- stats

def stats(daily, base=ACCOUNT_BASE, first_date=None, last_date=None):
    s = pd.Series(daily, dtype=float)
    total = float(s.sum())
    years = ((pd.Timestamp(last_date) - pd.Timestamp(first_date)).days / 365.25
             if first_date is not None else None)
    cum = s.cumsum()
    maxdd = float((cum.cummax() - cum).max())
    sd = float(s.std(ddof=1))
    sharpe = float(s.mean() / sd * math.sqrt(252)) if sd > 0 else None
    cagr = (((1 + total / base) ** (1 / years) - 1) * 100) if (years and 1 + total / base > 0) else None
    return {
        "total_usd": total,
        "total_ret_pct": total / base * 100,
        "cagr_pct": cagr,
        "sharpe": sharpe,
        "maxdd_usd": maxdd,
        "maxdd_pct": maxdd / base * 100,
    }


def yearly(df, daily):
    years = pd.to_datetime(df["date"]).dt.year
    s = pd.Series(daily, dtype=float)
    out = {}
    for y in sorted(years.unique()):
        out[int(y)] = float(s[years == y].sum())
    return out


def exposure(df, pos, active_from):
    seg = pos[active_from:]
    n = len(seg)
    return {
        "active_days": n,
        "pct_long": sum(1 for x in seg if x > 0) / n * 100,
        "pct_short": sum(1 for x in seg if x < 0) / n * 100,
        "pct_flat": sum(1 for x in seg if x == 0) / n * 100,
        "avg_abs_lots": sum(abs(x) for x in seg) / n,
        "max_lots": max((abs(x) for x in seg), default=0.0),
    }


def bh_stats(df, start_idx=0, with_spread=False):
    """Buy-and-hold 1 lot from first OPEN of the active range to last close,
    with the same swap convention (charged every trading day, Wednesday x5).
    Mirrors orb_stats.bh_metrics: ex-financing = optimistic upper bound."""
    sub = df.iloc[start_idx:].reset_index(drop=True)
    entry = float(sub["open"].iloc[0])
    closes = sub["close"].to_numpy(dtype=float)
    dates = list(sub["date"])
    price_usd = (closes[-1] - entry) * USD_PER_POINT
    price = [0.0] + [(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    swap = [swap_usd(1.0, d) for d in dates]
    exfin = price if not with_spread else [p - 0.0 for p in price]
    st_ex = stats(exfin, first_date=dates[0], last_date=dates[-1])
    st_net = stats([p + s for p, s in zip(price, swap)],
                   first_date=dates[0], last_date=dates[-1])
    return {
        "first_date": dates[0], "last_date": dates[-1],
        "entry_open": entry, "last_close": float(closes[-1]),
        "price_ret_pct_index": (closes[-1] / entry - 1) * 100,
        "exfin": st_ex, "with_fin": st_net,
        "swap_total_usd": float(sum(swap)),
    }


# ---------------------------------------------------------------- anchor

def us500tick_anchor():
    """Reproduce the documented US500-tick B&H numbers from the D1 export:
    ex-financing +34.66% (first M5 open -> last close) and swap -$55,360
    (-20 x (1656 + 4x278 Wednesdays))."""
    df = load_d1(D1_US500TICK)
    bh = bh_stats(df)
    n_days = len(df)
    weds = sum(1 for d in df["date"] if d.weekday() == 2)
    expect_swap = SWAP_LONG * (n_days + (WED_FACTOR - 1) * weds)
    ok_price = abs(bh["exfin"]["total_ret_pct"] - 34.66) < 0.10
    ok_swap = abs(bh["swap_total_usd"] - expect_swap) < 1.0 and abs(expect_swap - (-55360.0)) < 100.0
    return {
        "n_days": n_days, "wednesdays": weds,
        "exfin_total_pct": bh["exfin"]["total_ret_pct"],
        "swap_total_usd": bh["swap_total_usd"],
        "expected_swap_usd": expect_swap,
        "price_anchor_ok": bool(ok_price), "swap_anchor_ok": bool(ok_swap),
    }


# ---------------------------------------------------------------- intraday correlations

def _parse_ts(s):
    s = (s or "").strip()
    return dt.datetime.strptime(s, "%Y.%m.%d %H:%M") if s else None


def daily_pnl_from_trades(path):
    """sp2l_stats convention: P&L summed by OPEN-date (YYYYMMDD)."""
    per = defaultdict(float)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        import csv
        for r in csv.DictReader(fh, delimiter=";"):
            d = _parse_ts(r["open_time"])
            pnl_col = "profit" if "profit" in r else "pnl"
            per[d.strftime("%Y%m%d")] += float(r[pnl_col])
    return per


def daily_pnl_vwap_days(path):
    out = {}
    import csv
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            out[r["ny_date"].replace("-", "")] = float(r["total_pnl"])
    return out


def pearson_zf(a, b, all_days):
    """Zero-fill both on the union calendar then Pearson over the common days
    (sp2l_stats convention; >=30 common days required)."""
    az = {k: a.get(k, 0.0) for k in all_days}
    bz = {k: b.get(k, 0.0) for k in all_days}
    ks = sorted(set(az) & set(bz))
    if len(ks) < 30:
        return None, len(ks)
    xa = [az[k] for k in ks]
    xb = [bz[k] for k in ks]
    ma, mb = sum(xa) / len(xa), sum(xb) / len(xb)
    num = sum((x - ma) * (y - mb) for x, y in zip(xa, xb))
    da = math.sqrt(sum((x - ma) ** 2 for x in xa))
    db = math.sqrt(sum((y - mb) ** 2 for y in xb))
    return (num / (da * db) if da > 0 and db > 0 else None), len(ks)


def load_momentum_daily(csv_path, col="exfin"):
    """momentum daily-PnL CSV -> {YYYYMMDD: pnl}."""
    df = pd.read_csv(csv_path)
    out = {}
    for _, r in df.iterrows():
        out[pd.Timestamp(r["date"]).strftime("%Y%m%d")] = float(r[col])
    return out


def run_correlations():
    """Daily-PnL correlations: Variant P (n=5) and Variant R (normalized)
    vs each other, vs the four intraday strategies, and vs B&H daily price
    P&L — sp2l_stats conventions (open-date keying, zero-filled union
    calendar, Pearson on common days, >=30 days required). Uses ex-financing
    P&L (net of spread, before swap) so the signal-driven component is
    correlated; financing is a slow deterministic drift."""
    p = load_momentum_daily(RESULTS / "variant_p_daily_n5.csv", "exfin")
    r = load_momentum_daily(RESULTS / "variant_r_daily.csv", "exfin")
    orb = daily_pnl_from_trades(ORB_TRADES)
    vwap = daily_pnl_vwap_days(VWAP_DAYS)
    tp = daily_pnl_from_trades(TP_TRADES)
    sp2l = daily_pnl_from_trades(SP2L_TRADES)
    df = load_d1(D1_US500C)
    closes = df["close"].to_numpy(dtype=float)
    bh = {pd.Timestamp(d).strftime("%Y%m%d"): float(closes[i] - closes[i - 1])
          for i, d in enumerate(df["date"]) if i > 0}

    series = {"VariantP_n5": p, "VariantR": r, "ORB": orb, "VWAP_MR": vwap,
              "VWAP_trend_pullback_WF": tp, "SP2L": sp2l, "BH_1lot_price": bh}
    all_days = sorted(set().union(*[set(s) for s in series.values()]))
    names = list(series)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rho, n = pearson_zf(series[a], series[b], all_days)
            rows.append({"a": a, "b": b,
                         "pearson": None if rho is None else round(rho, 3),
                         "common_days": n})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "correlations.csv", index=False)
    print(out.to_string(index=False))
    print(f"wrote {RESULTS / 'correlations.csv'}")


if __name__ == "__main__":
    run_correlations()
