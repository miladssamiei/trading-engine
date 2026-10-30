# -*- coding: utf-8 -*-
"""
sp2l_stats.py — Step 5 summary artifacts for the SP2L validation:
monthly matrix (py + mt5 + B&H), performance stats, and daily-PnL
correlations vs ORB, VWAP MR, and VWAP trend-pullback.

Outputs: data/results/sp2l_2021_2026/monthly_matrix.csv, stats.json
"""
import csv
import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\sp2l_2021_2026")
ORB = Path(r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present")
VWAP = Path(r"C:\Users\Milad\Desktop\RR\data\results\vwap_2021_2026")
TP = Path(r"C:\Users\Milad\Desktop\RR\data\results\vwap_trend_pullback_rr")
M1 = Path(r"C:\Users\Milad\Desktop\RR\data\results\vwap_2021_2026\us500tick_m1.csv")


def parse_ts(s):
    s = (s or "").strip()
    return dt.datetime.strptime(s, "%Y.%m.%d %H:%M") if s else None


def daily_pnl_from_trades(path):
    per = defaultdict(float)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            d = parse_ts(r["open_time"])
            pnl_col = "profit" if "profit" in r else "pnl"
            per[d.strftime("%Y%m%d")] += float(r[pnl_col])
    return per


def sharpe(daily):
    vals = list(daily.values())
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    sd = math.sqrt(var)
    return m / sd * math.sqrt(252) if sd > 0 else None


def pearson(a, b):
    ks = sorted(set(a) & set(b))
    if len(ks) < 30:
        return None, len(ks)
    xa = [a[k] for k in ks]
    xb = [b[k] for k in ks]
    ma, mb = sum(xa) / len(xa), sum(xb) / len(xb)
    num = sum((x - ma) * (y - mb) for x, y in zip(xa, xb))
    da = math.sqrt(sum((x - ma) ** 2 for x in xa))
    db = math.sqrt(sum((y - mb) ** 2 for y in xb))
    return (num / (da * db) if da > 0 and db > 0 else None), len(ks)


def main():
    # ---- SP2L daily PnL ------------------------------------------------
    sp2l_py = daily_pnl_from_trades(BASE / "py_trades.csv")
    sp2l_mt5 = daily_pnl_from_trades(BASE / "mt5_trades.csv")

    # zero-filled over trading dates (from the M1 export NY dates is heavy;
    # use union of strategy days across all four strategies)
    vwap_days = {}
    with open(VWAP / "py_days.csv", newline="") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            vwap_days[r["ny_date"]] = float(r["total_pnl"])
    orb_tr = daily_pnl_from_trades(ORB / "py_trades.csv")
    tp_days = daily_pnl_from_trades(TP / "wf_trades.csv")

    all_days = sorted(set(sp2l_py) | set(sp2l_mt5) | set(vwap_days)
                      | set(orb_tr) | set(tp_days))

    def zf(d):
        return {k: d.get(k, 0.0) for k in all_days}

    sp2l_py_z, sp2l_mt5_z = zf(sp2l_py), zf(sp2l_mt5)

    # ---- stats ----------------------------------------------------------
    def stats(daily, trades_path, label):
        with open(trades_path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh, delimiter=";"))
        n = len(rows)
        pnl = sum(float(r["profit"]) for r in rows)
        wins = sum(1 for r in rows if float(r["profit"]) > 0)
        cum = peak = mdd = 0.0
        for k in all_days:
            cum += daily[k]
            peak = max(peak, cum)
            mdd = max(mdd, peak - cum)
        per_year = defaultdict(lambda: [0, 0.0])
        for r in rows:
            y = int(r["open_time"][:4])
            per_year[y][0] += 1
            per_year[y][1] += float(r["profit"])
        return {
            "label": label, "trades": n,
            "winrate": round(100.0 * wins / n, 2) if n else None,
            "total_pts": round(pnl, 2),
            "expt": round(pnl / n, 3) if n else None,
            "sharpe_zerofilled": None if sharpe(daily) is None
            else round(sharpe(daily), 3),
            "maxdd_pts": round(mdd, 2),
            "per_year": {str(y): [v[0], round(v[1], 2)]
                         for y, v in sorted(per_year.items())},
        }

    s_py = stats(sp2l_py_z, BASE / "py_trades.csv", "python gross (chart tape)")
    s_mt5 = stats(sp2l_mt5_z, BASE / "mt5_trades.csv", "mt5 net (tester tape, real spread)")
    print(json.dumps(s_py, indent=2))
    print(json.dumps(s_mt5, indent=2))

    # ---- correlations ----------------------------------------------------
    cor = {}
    for name, d in (("ORB", orb_tr), ("VWAP_MR", vwap_days),
                    ("VWAP_trend_pullback_WF", tp_days)):
        r, nk = pearson(sp2l_py_z, zf(d))
        cor[name] = None if r is None else round(r, 3)
        cor[f"{name}_common_days"] = nk
    print("daily-PnL correlations (SP2L py gross):", cor)

    # ---- monthly matrix + B&H --------------------------------------------
    # B&H monthly price returns from month-end closes of the M1 export
    me_close = {}
    with open(M1, newline="") as fh:
        for r in csv.DictReader(fh):
            t = dt.datetime.fromtimestamp(int(r["time"]), tz=dt.timezone.utc)
            key = t.year * 100 + t.month
            me_close[key] = float(r["close"])     # last row wins = month end
    months = sorted(set(k // 100 for k in
                        list(me_close)) & set(
                        int(d[:6]) for d in all_days))
    prev = None
    rows = []
    for ym in sorted(set(int(d[:6]) for d in all_days)):
        py_m = round(sum(v for k, v in sp2l_py.items() if int(k[:6]) == ym), 2)
        mt_m = round(sum(v for k, v in sp2l_mt5.items() if int(k[:6]) == ym), 2)
        c = me_close.get(ym)
        bh = None
        if c is not None and prev is not None and prev > 0:
            bh = round((c / prev - 1.0) * 100.0, 2)
        if c is not None:
            prev = c
        rows.append({"month": f"{ym // 100}-{ym % 100:02d}",
                     "sp2l_py_pts": py_m, "sp2l_mt5_pts": mt_m,
                     "bh_price_ret_pct": bh})
    with open(BASE / "monthly_matrix.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["month", "sp2l_py_pts",
                                           "sp2l_mt5_pts", "bh_price_ret_pct"])
        w.writeheader()
        w.writerows(rows)
    bh_total = None
    if me_close:
        k0, k1 = min(me_close), max(me_close)
        bh_total = round((me_close[k1] / me_close[k0] - 1.0) * 100.0, 2)
    print(f"B&H price return over matrix range: {bh_total}% "
          f"(ex-financing +34.66% on the $10k/1-lot base per prior work)")

    with open(BASE / "stats.json", "w", encoding="utf-8") as fh:
        json.dump({"python": s_py, "mt5": s_mt5,
                   "correlations_daily_pnl": cor,
                   "bh_price_ret_pct_range": bh_total,
                   "bh_exfin_pct_prior_work": 34.66}, fh, indent=2)
    print(f"wrote monthly_matrix.csv ({len(rows)} months), stats.json")


if __name__ == "__main__":
    main()
