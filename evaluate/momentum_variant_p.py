# -*- coding: utf-8 -*-
"""
momentum_variant_p.py — Variant P (paper-faithful) of the multi-horizon
momentum study: daily-close trend following on US500.c D1.

Signal (Lempérière, Deremble, Seager, Potters, Bouchaud 2014, "Two Centuries
of Trend Following", arXiv:1404.3274 — monthly-closes EMA/crossover signal):

    s_n(t) = ( p(t-1) - EMA_n(t-1) ) / vol_n(t-1)

with p = last COMPLETED monthly close, EMA_n = exponential moving average of
monthly closes with decay ~n months (alpha = 1/n, seeded at the first close),
vol_n = EMA (same decay) of |month-over-month close changes|. Everything is
evaluated at t-1 (no lookahead). Position sizing: constant-risk scaling,
lots ∝ sign(s_n) x 1/vol_n (no discretization). The paper's primary horizon
is n=5 months; its Table 1 stability range {2,3,5,7,10,15,20} is run in full.

Free constant: for a single instrument the proportionality constant is a risk
budget; it is fixed (documented) so the time-average |position| over the
active span = 1.0 lot — exposure-matched to the project's 1-lot B&H
convention. Max leverage is reported.

Application: re-scored at the close of the last trading day of each ISO week
(project-wide weekly-rebalance decision). Within a month the monthly inputs
do not change, so weekly re-application is a no-op; positions turn over at
the first weekly close after each month end (documented, not hidden).

Costs: spread 1.0 pt per |Δlots|; swap per the project-documented convention
(-20/+2 $/day/lot, Wednesday x5) — see momentum_common header caveats.
Warm-up: no trading until 2n monthly closes exist (EMA/vol burn-in).

Outputs (data/results/momentum_multihorizon/):
  variant_p_summary.csv, variant_p_yearly.csv,
  variant_p_daily_n5.csv, variant_p_monthly_n5.csv, variant_p_weekly_n5.csv,
  equity_curve_variant_p.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import momentum_common as mc

N_GRID = [2, 3, 5, 7, 10, 15, 20]
PRIMARY_N = 5


def run_n(df, weeks, months, n):
    """Monthly EMA/vol recursion + weekly application; returns daily target
    series (unscaled), first-active row index, weekly decision log."""
    m = [x[2] for x in months]
    ema = [0.0] * len(m)
    vol = [0.0] * len(m)
    ema[0] = m[0]
    for j in range(1, len(m)):
        ema[j] = ema[j - 1] + (m[j] - ema[j - 1]) / n
        d = abs(m[j] - m[j - 1])
        vol[j] = d if j == 1 else vol[j - 1] + (d - vol[j - 1]) / n
    # month_end_row: last daily row of each completed month
    me_row = [x[1] for x in months]
    burn = 2 * n

    target = [None] * len(df)
    log = []
    first_active = None
    for w in weeks:
        # last completed month strictly before this weekly close
        j = -1
        for k, r in enumerate(me_row):
            if r < w:
                j = k
            else:
                break
        if j < burn or vol[j] <= 0:
            continue
        s = (m[j] - ema[j]) / vol[j]
        raw = (1.0 if s > 0 else -1.0) / vol[j]
        target[w] = raw
        log.append({"row": w, "month_idx": j, "m": m[j], "ema": ema[j],
                    "vol": vol[j], "s": s, "raw": raw})
        if first_active is None:
            first_active = w
    # forward-fill raw to daily
    daily_raw = [0.0] * len(df)
    cur = 0.0
    for i in range(len(df)):
        if target[i] is not None:
            cur = target[i]
        daily_raw[i] = cur
    return daily_raw, first_active, log


def simulate_scaled(df, daily_raw, first_active):
    k = mc.norm_k(daily_raw[first_active:])
    lots = [x * k for x in daily_raw]
    tgt = [None] * len(df)
    prev = 0.0
    for i in range(first_active, len(df)):
        if lots[i] != prev:
            tgt[i] = lots[i]
            prev = lots[i]
    # establish on the first active row explicitly (prev starts 0)
    tgt[first_active] = lots[first_active]
    res = mc.simulate(df, tgt)
    res["k"] = k
    return res


def main():
    print("=" * 78)
    print("MOMENTUM Variant P (paper-faithful) — US500.c D1 daily closes")
    print("=" * 78)

    anc = mc.us500tick_anchor()
    print(f"anchor US500-tick D1: {anc['n_days']} days, {anc['wednesdays']} Wednesdays, "
          f"ex-fin B&H {anc['exfin_total_pct']:+.2f}% (doc: +34.66) "
          f"swap {anc['swap_total_usd']:.0f} (doc: -55360) "
          f"ok={anc['price_anchor_ok'] and anc['swap_anchor_ok']}")

    df = mc.load_d1(mc.D1_US500C)
    weeks = mc.week_last_days(df)
    months = mc.month_end_rows(df)
    print(f"US500.c D1: {len(df)} bars {df['date'].iloc[0]} .. {df['date'].iloc[-1]}   "
          f"{len(weeks)} weekly closes, {len(months)} month-ends")

    summary = []
    yearly_rows = []
    detail = {}
    for n in N_GRID:
        daily_raw, fa, log = run_n(df, weeks, months, n)
        if fa is None:
            print(f"n={n}: no active weeks (insufficient history)")
            continue
        res = simulate_scaled(df, daily_raw, fa)
        sub = df.iloc[fa:].reset_index(drop=True)
        sub_res = {k: v[fa:] for k, v in res.items() if isinstance(v, list)}
        st_x = mc.stats(sub_res["exfin"], first_date=sub["date"].iloc[0],
                        last_date=sub["date"].iloc[-1])
        st_n = mc.stats(sub_res["net"], first_date=sub["date"].iloc[0],
                        last_date=sub["date"].iloc[-1])
        exp = mc.exposure(sub, sub_res["pos"], 0)
        pos_changes = sum(1 for i in range(1, len(sub_res["pos"]))
                          if sub_res["pos"][i] != sub_res["pos"][i - 1])
        traded = sum(sub_res["spread"]) / mc.SPREAD_PT
        row = {
            "n_months": n, "first_trade": str(sub["date"].iloc[0]),
            "active_days": exp["active_days"], "pos_changes": pos_changes,
            "traded_lots_abs": traded, "avg_abs_lots": exp["avg_abs_lots"],
            "max_lots": exp["max_lots"], "pct_long": exp["pct_long"],
            "pct_short": exp["pct_short"],
            "exfin_total_usd": st_x["total_usd"],
            "exfin_total_pct": st_x["total_ret_pct"],
            "exfin_cagr_pct": st_x["cagr_pct"], "exfin_sharpe": st_x["sharpe"],
            "exfin_maxdd_pct": st_x["maxdd_pct"],
            "swap_total_usd": sum(sub_res["swap"]),
            "spread_total_usd": sum(sub_res["spread"]),
            "net_total_usd": st_n["total_usd"],
            "net_total_pct": st_n["total_ret_pct"],
            "net_cagr_pct": st_n["cagr_pct"], "net_sharpe": st_n["sharpe"],
            "net_maxdd_pct": st_n["maxdd_pct"],
        }
        summary.append(row)
        for y, v in mc.yearly(sub, sub_res["net"]).items():
            yearly_rows.append({"n_months": n, "year": y, "net_usd": v})
        detail[n] = (fa, res, sub_res, sub)
        print(f"n={n:>2}: from {row['first_trade']}  days={row['active_days']}  "
              f"exfin {st_x['total_usd']:+9.1f} (Sh {st_x['sharpe']:+.2f})  "
              f"net {st_n['total_usd']:+9.1f} (Sh {st_n['sharpe']:+.2f})  "
              f"swap {row['swap_total_usd']:+9.1f}  maxlots {exp['max_lots']:.2f}")

    pd.DataFrame(summary).to_csv(mc.RESULTS / "variant_p_summary.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(mc.RESULTS / "variant_p_yearly.csv", index=False)

    # ---- primary n=5 detail ----
    fa, res, sub_res, sub = detail[PRIMARY_N]
    bh = mc.bh_stats(df, start_idx=fa)
    out = pd.DataFrame({
        "date": sub["date"], "close": sub["close"],
        "pos_lots": [round(x, 4) for x in sub_res["pos"]],
        "price_pnl": [round(x, 2) for x in sub_res["price"]],
        "spread": [round(x, 2) for x in sub_res["spread"]],
        "swap": [round(x, 2) for x in sub_res["swap"]],
        "exfin": [round(x, 2) for x in sub_res["exfin"]],
        "net": [round(x, 2) for x in sub_res["net"]],
        "cum_net": [round(x, 2) for x in pd.Series(sub_res["net"]).cumsum()],
    })
    out.to_csv(mc.RESULTS / f"variant_p_daily_n{PRIMARY_N}.csv", index=False)

    dts = pd.to_datetime(out["date"])
    mo = dts.dt.strftime("%Y-%m")
    mrows = []
    for mkey in sorted(mo.unique()):
        seg = out[mo == mkey]
        mrows.append({
            "month": mkey, "days": len(seg),
            "exfin_usd": round(seg["exfin"].sum(), 2),
            "swap_usd": round(seg["swap"].sum(), 2),
            "net_usd": round(seg["net"].sum(), 2),
            "cum_net_usd": round(seg["cum_net"].iloc[-1], 2),
            "avg_abs_pos": round(seg["pos_lots"].abs().mean(), 3),
        })
    pd.DataFrame(mrows).to_csv(mc.RESULTS / f"variant_p_monthly_n{PRIMARY_N}.csv", index=False)

    wl = []
    k = res["k"]
    prev = 0.0
    for entry in [e for e in run_n(df, weeks, months, PRIMARY_N)[2]]:
        lots = entry["raw"] * k
        dl = abs(lots - prev)
        wl.append({"date": str(df["date"].iloc[entry["row"]]),
                   "month_used": months[entry["month_idx"]][0],
                   "m": round(entry["m"], 2), "ema": round(entry["ema"], 2),
                   "vol": round(entry["vol"], 2), "s": round(entry["s"], 3),
                   "lots": round(lots, 3), "dlots": round(dl, 3),
                   "spread_cost": round(dl * mc.SPREAD_PT, 2)})
        prev = lots
    pd.DataFrame(wl).to_csv(mc.RESULTS / f"variant_p_weekly_n{PRIMARY_N}.csv", index=False)

    # ---- equity curve PNG ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(pd.to_datetime(out["date"]), out["cum_net"], label=f"Variant P net (n={PRIMARY_N})", lw=1.2)
        ax.plot(pd.to_datetime(out["date"]), pd.Series(sub_res["exfin"]).cumsum(), label=f"Variant P ex-financing", lw=1.0, alpha=0.8)
        bh_df = df.iloc[fa:].reset_index(drop=True)
        bh_price = [0.0] + list((bh_df["close"].iloc[i] - bh_df["close"].iloc[i - 1]
                                 for i in range(1, len(bh_df))))
        bh_swap = [mc.swap_usd(1.0, d) for d in bh_df["date"]]
        ax.plot(pd.to_datetime(bh_df["date"]), pd.Series(bh_price).cumsum(),
                label="B&H 1 lot ex-financing", lw=1.0, ls="--", alpha=0.8)
        ax.plot(pd.to_datetime(bh_df["date"]), pd.Series([p + s for p, s in zip(bh_price, bh_swap)]).cumsum(),
                label="B&H 1 lot with financing", lw=1.0, ls=":", alpha=0.8)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title("Momentum Variant P (paper-faithful, n=5) vs B&H — US500.c D1, $10k base / avg 1 lot")
        ax.set_ylabel("cumulative P&L, $")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(mc.RESULTS / "equity_curve_variant_p.png", dpi=130)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)

    # ---- B&H on the identical active range ----
    print(f"\nB&H same range ({bh['first_date']} .. {bh['last_date']}): "
          f"ex-fin {bh['exfin']['total_usd']:+.1f} ({bh['exfin']['total_ret_pct']:+.2f}%, "
          f"Sh {bh['exfin']['sharpe']:+.2f}, maxDD {bh['exfin']['maxdd_pct']:.1f}%)  "
          f"with-fin {bh['with_fin']['total_usd']:+.1f}  swap {bh['swap_total_usd']:+.0f}")
    print(f"wrote variant_p_summary.csv / variant_p_yearly.csv / "
          f"variant_p_daily_n{PRIMARY_N}.csv / variant_p_monthly_n{PRIMARY_N}.csv / "
          f"variant_p_weekly_n{PRIMARY_N}.csv / equity_curve_variant_p.png")


if __name__ == "__main__":
    main()
