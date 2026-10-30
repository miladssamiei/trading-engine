# -*- coding: utf-8 -*-
"""
momentum_variant_r.py — Variant R (video retail rubric) of the multi-horizon
momentum study: daily-close trend following on US500.c D1.

Rubric (TradingLab "I Stole a Trading Strategy Worth $60 Billion" —
simplified AHL attribution; UNTESTED claim, this run is its first evidence):

  * four discrete lookbacks on daily closes: 1 week / 2 weeks / 1 month /
    2 months, implemented as 5 / 10 / 21 / 42 trading days;
  * each scored +1 if close(t) > close(t-L) else -1 (start-to-end trendline
    direction); score in {-4,-2,0,+2,+4};
  * cheat sheet: |score|=4 -> 100% weight, |score|=2 -> 50%, 0 -> flat
    (weight = |score|/4); direction = sign(score);
  * volatility-scaled sizing:
        position = weight x (target_risk% / annualized_vol%) x portfolio
    with annualized_vol = mean |close-to-close % change| over the trailing
    30 daily closes x sqrt(365) (the rubric's own crypto-flavored 365-day
    annualization, kept literal), portfolio = $10,000.

SIZING CONVENTIONS (both reported, neither hidden):
  * normalized (PRIMARY, exposure-matched head-to-head vs Variant P / B&H):
    lots = sign(score) x weight / ann_vol x k with k fixed so time-average
    |lots| = 1.0 over the active span (the rubric leaves the risk target as
    a free constant for a single instrument);
  * literal: the rubric formula verbatim with target_risk in {10%, 20%, 30%}
    and portfolio $10k (lots = notional / close; US500.c = $1 per index
    point per lot, so 1 lot notional = price). Leverage reported.

Rebalance: weekly at the close of the last trading day of each ISO week;
warm-up: first 42 trading days (longest lookback; also covers the 30-day vol
window). Costs identical to Variant P (see momentum_common header).

Outputs (data/results/momentum_multihorizon/):
  variant_r_summary.csv, variant_r_yearly.csv,
  variant_r_daily.csv, variant_r_monthly.csv, variant_r_weekly.csv,
  equity_curve_variant_r.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import momentum_common as mc

LOOKBACKS = [5, 10, 21, 42]
TARGET_RISKS = [10.0, 20.0, 30.0]
SQ365 = 365.0 ** 0.5


def score_at(closes, w):
    subs = {}
    for L in LOOKBACKS:
        if w - L < 0:
            return None
        subs[L] = 1 if closes[w] > closes[w - L] else -1
    score = sum(subs.values())
    weight = abs(score) / 4.0
    if w - 30 < 0:
        return None
    rets = [abs(closes[w - i] / closes[w - i - 1] - 1.0) for i in range(30)]
    ann_vol_pct = sum(rets) / len(rets) * 100.0 * SQ365
    return {"subs": subs, "score": score, "weight": weight,
            "ann_vol_pct": ann_vol_pct}


def run_normalized(df, weeks):
    closes = df["close"].to_numpy(dtype=float)
    target = [None] * len(df)
    log = []
    first_active = None
    for w in weeks:
        sc = score_at(closes, w)
        if sc is None or sc["ann_vol_pct"] <= 0:
            continue
        raw = (1.0 if sc["score"] > 0 else -1.0 if sc["score"] < 0 else 0.0) \
            * sc["weight"] / sc["ann_vol_pct"]
        target[w] = raw
        log.append({"row": w, **sc, "raw": raw})
        if first_active is None:
            first_active = w
    daily_raw = [0.0] * len(df)
    cur = 0.0
    for i in range(len(df)):
        if target[i] is not None:
            cur = target[i]
        daily_raw[i] = cur
    return daily_raw, first_active, log


def run_literal(df, weeks, target_risk_pct):
    closes = df["close"].to_numpy(dtype=float)
    target = [None] * len(df)
    first_active = None
    for w in weeks:
        sc = score_at(closes, w)
        if sc is None or sc["ann_vol_pct"] <= 0:
            continue
        notional = sc["weight"] * (target_risk_pct / sc["ann_vol_pct"]) * mc.ACCOUNT_BASE
        lots = (1.0 if sc["score"] > 0 else -1.0 if sc["score"] < 0 else 0.0) \
            * notional / closes[w]
        target[w] = lots
        if first_active is None:
            first_active = w
    daily = [0.0] * len(df)
    cur = 0.0
    for i in range(len(df)):
        if target[i] is not None:
            cur = target[i]
        daily[i] = cur
    return daily, first_active


def stats_row(name, df, daily, first_active):
    tgt = [None] * len(df)
    prev = 0.0
    for i in range(first_active, len(df)):
        if daily[i] != prev:
            tgt[i] = daily[i]
            prev = daily[i]
    tgt[first_active] = daily[first_active]
    res = mc.simulate(df, tgt)
    sub = df.iloc[first_active:].reset_index(drop=True)
    sr = {k: v[first_active:] for k, v in res.items() if isinstance(v, list)}
    st_x = mc.stats(sr["exfin"], first_date=sub["date"].iloc[0], last_date=sub["date"].iloc[-1])
    st_n = mc.stats(sr["net"], first_date=sub["date"].iloc[0], last_date=sub["date"].iloc[-1])
    exp = mc.exposure(sub, sr["pos"], 0)
    pos_changes = sum(1 for i in range(1, len(sr["pos"])) if sr["pos"][i] != sr["pos"][i - 1])
    traded = sum(sr["spread"]) / mc.SPREAD_PT
    return {
        "config": name, "first_trade": str(sub["date"].iloc[0]),
        "active_days": exp["active_days"], "pos_changes": pos_changes,
        "traded_lots_abs": traded, "avg_abs_lots": exp["avg_abs_lots"],
        "max_lots": exp["max_lots"],
        "max_leverage_notional_x": exp["max_lots"] * float(sub["close"].max()) / mc.ACCOUNT_BASE,
        "pct_long": exp["pct_long"], "pct_short": exp["pct_short"],
        "pct_flat": exp["pct_flat"],
        "exfin_total_usd": st_x["total_usd"], "exfin_total_pct": st_x["total_ret_pct"],
        "exfin_cagr_pct": st_x["cagr_pct"], "exfin_sharpe": st_x["sharpe"],
        "exfin_maxdd_pct": st_x["maxdd_pct"],
        "swap_total_usd": sum(sr["swap"]), "spread_total_usd": sum(sr["spread"]),
        "net_total_usd": st_n["total_usd"], "net_total_pct": st_n["total_ret_pct"],
        "net_cagr_pct": st_n["cagr_pct"], "net_sharpe": st_n["sharpe"],
        "net_maxdd_pct": st_n["maxdd_pct"],
    }, sr, sub, res


def main():
    print("=" * 78)
    print("MOMENTUM Variant R (video rubric) — US500.c D1 daily closes")
    print("=" * 78)

    df = mc.load_d1(mc.D1_US500C)
    weeks = mc.week_last_days(df)
    print(f"US500.c D1: {len(df)} bars {df['date'].iloc[0]} .. {df['date'].iloc[-1]}   "
          f"{len(weeks)} weekly closes")

    summary = []
    yearly_rows = []
    keep = {}

    daily_raw, fa_norm, log = run_normalized(df, weeks)
    k = mc.norm_k(daily_raw[fa_norm:])
    for name, daily, fa in (
        ("normalized (avg 1 lot)", [x * k for x in daily_raw], fa_norm),
        *[(f"literal target={t:.0f}%", *run_literal(df, weeks, t)) for t in TARGET_RISKS],
    ):
        row, sr, sub, res = stats_row(name, df, daily, fa)
        summary.append(row)
        for y, v in mc.yearly(sub, sr["net"]).items():
            yearly_rows.append({"config": name, "year": y, "net_usd": v})
        keep[name] = (sr, sub, res, daily, fa)
        print(f"{name:26s}: from {row['first_trade']}  exfin {row['exfin_total_usd']:+9.1f} "
              f"(Sh {row['exfin_sharpe']:+.2f})  net {row['net_total_usd']:+9.1f} "
              f"(Sh {row['net_sharpe']:+.2f})  swap {row['swap_total_usd']:+9.1f}  "
              f"avg {row['avg_abs_lots']:.2f} max {row['max_lots']:.2f} lots")

    pd.DataFrame(summary).to_csv(mc.RESULTS / "variant_r_summary.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(mc.RESULTS / "variant_r_yearly.csv", index=False)

    # ---- primary (normalized) detail ----
    sr, sub, res, daily, fa = keep["normalized (avg 1 lot)"]
    out = pd.DataFrame({
        "date": sub["date"], "close": sub["close"],
        "pos_lots": [round(x, 4) for x in sr["pos"]],
        "price_pnl": [round(x, 2) for x in sr["price"]],
        "spread": [round(x, 2) for x in sr["spread"]],
        "swap": [round(x, 2) for x in sr["swap"]],
        "exfin": [round(x, 2) for x in sr["exfin"]],
        "net": [round(x, 2) for x in sr["net"]],
        "cum_net": [round(x, 2) for x in pd.Series(sr["net"]).cumsum()],
    })
    out.to_csv(mc.RESULTS / "variant_r_daily.csv", index=False)

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
    pd.DataFrame(mrows).to_csv(mc.RESULTS / "variant_r_monthly.csv", index=False)

    closes = df["close"].to_numpy(dtype=float)
    wl = []
    prev = 0.0
    for e in log:
        lots = e["raw"] * k
        dl = abs(lots - prev)
        wl.append({"date": str(df["date"].iloc[e["row"]]),
                   "c": round(closes[e["row"]], 2),
                   **{f"sub_{L}d": e["subs"][L] for L in LOOKBACKS},
                   "score": e["score"], "ann_vol_pct": round(e["ann_vol_pct"], 2),
                   "lots": round(lots, 3), "dlots": round(dl, 3),
                   "spread_cost": round(dl * mc.SPREAD_PT, 2)})
        prev = lots
    pd.DataFrame(wl).to_csv(mc.RESULTS / "variant_r_weekly.csv", index=False)

    bh = mc.bh_stats(df, start_idx=fa)
    print(f"\nB&H same range ({bh['first_date']} .. {bh['last_date']}): "
          f"ex-fin {bh['exfin']['total_usd']:+.1f} ({bh['exfin']['total_ret_pct']:+.2f}%, "
          f"Sh {bh['exfin']['sharpe']:+.2f}, maxDD {bh['exfin']['maxdd_pct']:.1f}%)  "
          f"with-fin {bh['with_fin']['total_usd']:+.1f}  swap {bh['swap_total_usd']:+.0f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(pd.to_datetime(out["date"]), out["cum_net"], label="Variant R net (normalized)", lw=1.2)
        ax.plot(pd.to_datetime(out["date"]), pd.Series(sr["exfin"]).cumsum(),
                label="Variant R ex-financing", lw=1.0, alpha=0.8)
        bh_df = df.iloc[fa:].reset_index(drop=True)
        bh_price = [0.0] + [float(bh_df["close"].iloc[i] - bh_df["close"].iloc[i - 1])
                            for i in range(1, len(bh_df))]
        bh_swap = [mc.swap_usd(1.0, d) for d in bh_df["date"]]
        ax.plot(pd.to_datetime(bh_df["date"]), pd.Series(bh_price).cumsum(),
                label="B&H 1 lot ex-financing", lw=1.0, ls="--", alpha=0.8)
        ax.plot(pd.to_datetime(bh_df["date"]),
                pd.Series([p + s for p, s in zip(bh_price, bh_swap)]).cumsum(),
                label="B&H 1 lot with financing", lw=1.0, ls=":", alpha=0.8)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title("Momentum Variant R (video rubric, normalized) vs B&H — US500.c D1, $10k base / avg 1 lot")
        ax.set_ylabel("cumulative P&L, $")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(mc.RESULTS / "equity_curve_variant_r.png", dpi=130)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)

    print("wrote variant_r_summary.csv / variant_r_yearly.csv / variant_r_daily.csv / "
          "variant_r_monthly.csv / variant_r_weekly.csv / equity_curve_variant_r.png")


if __name__ == "__main__":
    main()
