import sys
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\Milad\Desktop\RR")
STUDY = ROOT / "data" / "results" / "youtube_hybrid_scalping"

M5 = {
    "US500": ROOT / "data" / "results" / "orb_2021_present" / "us500tick_m5.csv",
    "USA30": ROOT / "data" / "results" / "usa30_onboarding" / "usa30tick_m5.csv",
    "XAUUSD": ROOT / "data" / "results" / "xauusd_onboarding" / "xauusdtick_m5.csv",
}
D1_ANCHOR = ROOT / "data" / "results" / "momentum_multihorizon" / "us500tick_d1.csv"
SPREAD = {"US500": 0.51, "USA30": 2.28, "XAUUSD": 0.41}
BAND_ATR_MULT = 3.1
TP_ATR = 2.0
SL_ATR = 1.5
WICK_BODY = 2.0
LOWER_WICK_FRAC = 0.10
ATR_N = 14
OPEN_WALL = 570
INSTS = ["US500", "USA30", "XAUUSD"]


def load_m5(path):
    df = pd.read_csv(path)
    if not df["time"].is_monotonic_increasing or df["time"].duplicated().any():
        dup = int(df["time"].duplicated().sum())
        df = df.drop_duplicates(subset="time", keep="first").sort_values("time").reset_index(drop=True)
        if dup:
            print(f"{path.name}: dropped {dup} duplicate timestamps")
    t = df["time"].to_numpy(np.int64)
    assert np.all(np.diff(t) > 0), f"non-monotonic time in {path}"
    ny = pd.DatetimeIndex(pd.to_datetime(pd.Series(t), unit="s", utc=True)).tz_convert("America/New_York")
    return {
        "t": t,
        "o": df["open"].to_numpy(np.float64),
        "h": df["high"].to_numpy(np.float64),
        "l": df["low"].to_numpy(np.float64),
        "c": df["close"].to_numpy(np.float64),
        "wall": (ny.hour * 60 + ny.minute).to_numpy(np.int64),
        "nyd": (ny.year * 10000 + ny.month * 100 + ny.day).to_numpy(np.int64),
    }


def d1_by_ny_date(ctx):
    nyd, o, h, l, c = ctx["nyd"], ctx["o"], ctx["h"], ctx["l"], ctx["c"]
    dates = [int(nyd[0])]
    opens, highs, lows, closes = {}, {}, {}, {}
    co, chh, cll, cc = o[0], h[0], l[0], c[0]
    for i in range(1, len(nyd)):
        if nyd[i] == dates[-1]:
            chh = max(chh, h[i])
            cll = min(cll, l[i])
            cc = c[i]
        else:
            opens[dates[-1]] = co
            highs[dates[-1]] = chh
            lows[dates[-1]] = cll
            closes[dates[-1]] = cc
            dates.append(int(nyd[i]))
            co, chh, cll, cc = o[i], h[i], l[i], c[i]
    opens[dates[-1]] = co
    highs[dates[-1]] = chh
    lows[dates[-1]] = cll
    closes[dates[-1]] = cc
    return dates, opens, closes


def prev_red_map(dates, opens, closes):
    red = {}
    for k in range(1, len(dates)):
        red[dates[k]] = bool(closes[dates[k - 1]] < opens[dates[k - 1]])
    return red


def atr_wilder(ctx):
    h, l, c = ctx["h"], ctx["l"], ctx["c"]
    n = len(c)
    tr = np.empty(n, np.float64)
    tr[0] = h[0] - l[0]
    tr[1:] = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])
    atr = np.full(n, np.nan, np.float64)
    if n <= ATR_N:
        return atr
    atr[ATR_N] = tr[: ATR_N + 1].mean()
    k = ATR_N
    for i in range(ATR_N + 1, n):
        atr[i] = (atr[i - 1] * (k - 1) + tr[i]) / k
    return atr


def run_long(ctx, atr):
    t, h, l, c, wall, nyd = ctx["t"], ctx["h"], ctx["l"], ctx["c"], ctx["wall"], ctx["nyd"]
    n = len(c)
    trades = []
    i = ATR_N + 1
    while i < n:
        if wall[i] == OPEN_WALL:
            i += 1
            continue
        a = atr[i - 1]
        if not np.isfinite(a) or a <= 0:
            i += 1
            continue
        band = c[i - 1] - BAND_ATR_MULT * a
        if l[i] <= band:
            entry = band
            tp = entry + TP_ATR * a
            sl = entry - SL_ATR * a
            j = i
            exit_price = None
            reason = None
            while j < n:
                if l[j] <= sl:
                    exit_price = sl
                    reason = "stop"
                    break
                if h[j] >= tp:
                    exit_price = tp
                    reason = "target"
                    break
                j += 1
            if exit_price is None:
                j = n - 1
                exit_price = c[n - 1]
                reason = "end_of_tape"
            trades.append({"sig": t[i], "ent": t[i], "ex": t[j], "ep": entry, "xp": exit_price,
                           "reason": reason, "atr": a, "nyd": int(nyd[i]), "red": ""})
            i = j + 1
        else:
            i += 1
    return trades


def run_short(ctx, red_map):
    t, o, h, l, c, nyd = ctx["t"], ctx["o"], ctx["h"], ctx["l"], ctx["c"], ctx["nyd"]
    n = len(c)
    trades = []
    for i in range(n - 1):
        if not red_map.get(int(nyd[i]), False):
            continue
        body = abs(c[i] - o[i])
        uw = h[i] - max(c[i], o[i])
        lw = min(c[i], o[i]) - l[i]
        if uw >= WICK_BODY * body and lw <= LOWER_WICK_FRAC * uw:
            trades.append({"sig": t[i], "ent": t[i + 1], "ex": t[i + 1], "ep": o[i + 1],
                           "xp": c[i + 1], "reason": "time_exit_5m", "atr": np.nan,
                           "nyd": int(nyd[i + 1]), "red": True})
    return trades


def summarize(pnls, spread):
    g = np.asarray(pnls, np.float64)
    if g.size == 0:
        return {"n": 0, "exp_gross": np.nan, "exp_net": np.nan, "wr_gross": np.nan, "wr_net": np.nan}
    net = g - spread
    return {"n": int(g.size), "exp_gross": float(g.mean()), "exp_net": float(net.mean()),
            "wr_gross": float((g > 0).mean()), "wr_net": float((net > 0).mean())}


def main():
    if not (STUDY / "SPEC_YOUTUBE_HYBRID_SCALPING.txt").exists():
        raise SystemExit("SPEC_YOUTUBE_HYBRID_SCALPING.txt missing; refusing to run")
    if not (STUDY / "GATE_YOUTUBE_HYBRID_SCALPING.txt").exists():
        raise SystemExit("GATE_YOUTUBE_HYBRID_SCALPING.txt missing; refusing to run")

    rows = []
    summaries = {}
    tape_info = {}
    us500_d1 = None
    for inst in INSTS:
        ctx = load_m5(M5[inst])
        dates, opens, closes = d1_by_ny_date(ctx)
        red_map = prev_red_map(dates, opens, closes)
        atr = atr_wilder(ctx)
        longs = run_long(ctx, atr)
        shorts = run_short(ctx, red_map)
        for tr in longs:
            pnl = tr["xp"] - tr["ep"]
            rows.append([inst, "Long", "long", tr["sig"], tr["ent"], tr["ex"], tr["ep"], tr["xp"],
                         tr["reason"], tr["atr"], tr["nyd"], pnl, pnl - SPREAD[inst], tr["red"]])
        for tr in shorts:
            pnl = tr["ep"] - tr["xp"]
            rows.append([inst, "Short", "short", tr["sig"], tr["ent"], tr["ex"], tr["ep"], tr["xp"],
                         tr["reason"], tr["atr"], tr["nyd"], pnl, pnl - SPREAD[inst], tr["red"]])
        summaries[inst] = {
            "Long": summarize([r[11] for r in rows if r[0] == inst and r[1] == "Long"], SPREAD[inst]),
            "Short": summarize([r[11] for r in rows if r[0] == inst and r[1] == "Short"], SPREAD[inst]),
            "Combined": summarize([r[11] for r in rows if r[0] == inst], SPREAD[inst]),
        }
        tape_info[inst] = (dt.datetime.fromtimestamp(int(ctx["t"][0]), dt.timezone.utc),
                           dt.datetime.fromtimestamp(int(ctx["t"][-1]), dt.timezone.utc),
                           len(ctx["t"]), len(dates))
        if inst == "US500":
            us500_d1 = (dates, closes)
        print(f"{inst}: {len(longs)} long, {len(shorts)} short "
              f"({tape_info[inst][0].date()} .. {tape_info[inst][1].date()}, "
              f"{len(ctx['t'])} M5 bars, {len(dates)} NY dates)")

    cols = ["instrument", "model", "dir", "signal_time", "entry_time", "exit_time",
            "entry_price", "exit_price", "exit_reason", "atr_ref", "ny_date",
            "pnl_gross_pts", "pnl_net_pts", "prev_day_red"]
    tdf = pd.DataFrame(rows, columns=cols)
    for tc in ["signal_time", "entry_time", "exit_time"]:
        tdf[tc + "_utc"] = pd.to_datetime(tdf[tc], unit="s", utc=True).dt.strftime("%Y.%m.%d %H:%M")
    ordered = ["instrument", "model", "dir", "signal_time_utc", "entry_time_utc", "exit_time_utc",
               "signal_time", "entry_time", "exit_time", "entry_price", "exit_price", "exit_reason",
               "atr_ref", "ny_date", "pnl_gross_pts", "pnl_net_pts", "prev_day_red"]
    tdf[ordered].to_csv(STUDY / "hybrid_scalp_trades.csv", index=False)

    try:
        d1a = pd.read_csv(D1_ANCHOR)
        a_dates = pd.to_datetime(d1a["time"], unit="s", utc=True).dt.strftime("%Y%m%d").astype(int)
        amap = dict(zip(a_dates.tolist(), d1a["close"].tolist()))
        dates5, closes5 = us500_d1
        common = [d for d in dates5 if d in amap]
        diffs = np.array([abs(closes5[d] - amap[d]) for d in common])
        anchor_note = (f"{len(common)} common dates, median |close diff| = {np.median(diffs):.3f} pts, "
                       f"max = {diffs.max():.3f} pts (descriptive; 00:00-UTC vs NY-date boundaries differ by construction)")
    except Exception as exc:
        anchor_note = f"anchor load failed: {exc}"

    gy1_by_inst = {i: summaries[i]["Combined"]["n"] >= 150 for i in INSTS}
    gy1 = all(gy1_by_inst.values())
    gy2_insts = [i for i in INSTS if summaries[i]["Combined"]["n"] > 0 and summaries[i]["Combined"]["exp_net"] > 0]
    gy2 = len(gy2_insts) > 0
    verdict = "PASS" if (gy1 and gy2) else "FAIL"

    L = []
    L.append("REPORT_YOUTUBE_HYBRID_SCALPING")
    L.append("Study: youtube_hybrid_scalping (Hypothesis #22, owner-tasked post-termination exploratory)")
    L.append("Mechanism: Long = ATR(14) lower-band limit touch (Close[i-1] - 3.1*ATR[i-1]), TP 2.0 ATR / SL 1.5 ATR,")
    L.append("           pessimistic same-bar collision (stop first), 09:30 opening candle excluded, single position;")
    L.append("           Short = shooting star (upper wick >= 2x body, lower wick <= 0.10x upper wick) after a red")
    L.append("           previous NY-date D1 candle, market short at next M5 open, time exit at that candle's close.")
    L.append("Status: descriptive spread-decay quantification; downstream_model_allowed=false, live_execution_allowed=false.")
    L.append("")
    L.append("3x3 MATRIX - EXPECTANCY PER TRADE (points): GROSS / NET (verified median RT spread subtracted)")
    L.append(f"{'Instrument':<10}{'Long-Only':>26}{'Short-Only':>26}{'Combined':>26}")
    for inst in INSTS:
        cells = []
        for model in ["Long", "Short", "Combined"]:
            s = summaries[inst][model]
            if s["n"] == 0:
                cells.append(f"{'n=0':>26}")
            else:
                cells.append(f"{s['exp_gross']:+.4f}/{s['exp_net']:+.4f} n={s['n']}".rjust(26))
        L.append(f"{inst:<10}" + "".join(cells))
    L.append("")
    L.append("DETAIL (win rates on gross / net PnL; spread decay = gross - net expectancy)")
    for inst in INSTS:
        ti = tape_info[inst]
        L.append(f"{inst} (spread {SPREAD[inst]:.2f} pt RT): tape {ti[0].date()} .. {ti[1].date()}, "
                 f"{ti[2]} M5 bars, {ti[3]} NY dates")
        for model in ["Long", "Short", "Combined"]:
            s = summaries[inst][model]
            if s["n"] == 0:
                L.append(f"  {model:<9} n=0")
                continue
            decay = s["exp_gross"] - s["exp_net"]
            frac = (decay / s["exp_gross"] * 100) if s["exp_gross"] != 0 else float("nan")
            L.append(f"  {model:<9} n={s['n']:>5}  gross E={s['exp_gross']:+.4f} WR={s['wr_gross']*100:.2f}%  "
                     f"net E={s['exp_net']:+.4f} WR={s['wr_net']*100:.2f}%  decay={decay:.2f}pt ({frac:.0f}% of gross)")
    L.append("")
    L.append("US500 D1 sanity anchor (momentum_multihorizon/us500tick_d1.csv vs M5-derived NY-date D1 closes):")
    L.append(f"  {anchor_note}")
    L.append("")
    L.append("GATES")
    L.append("G-Y1 (Combined >= 150 trades on EACH instrument): "
             + "; ".join(f"{i} n={summaries[i]['Combined']['n']}={'PASS' if v else 'FAIL'}" for i, v in gy1_by_inst.items())
             + f" -> {'PASS' if gy1 else 'FAIL'}")
    gy2_txt = "; ".join(f"{i} net E={summaries[i]['Combined']['exp_net']:+.4f}" for i in gy2_insts) if gy2_insts else "none"
    L.append(f"G-Y2 (Combined NET expectancy > 0 on >= 1 instrument): {gy2_txt} -> {'PASS' if gy2 else 'FAIL'}")
    L.append(f"OVERALL VERDICT: {verdict}")
    L.append("")
    L.append("Spread-decay reading: the fixed median RT spread is subtracted from every trade, so Net = Gross shifted")
    L.append("down by exactly the spread per trade; win rate decays accordingly. Any cell whose gross expectancy is")
    L.append("smaller than the spread flips NET-negative regardless of entry mechanics. Long trades carry no swap")
    L.append("charge in this model (declared limitation); overnight holds are charged spread only.")
    report = "\n".join(L)
    (STUDY / "REPORT_YOUTUBE_HYBRID_SCALPING.txt").write_text(report, encoding="utf-8")
    print()
    print(report)


if __name__ == "__main__":
    sys.exit(main())
