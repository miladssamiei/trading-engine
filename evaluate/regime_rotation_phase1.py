# -*- coding: utf-8 -*-
"""
regime_rotation_phase1.py - Phase 1 offline regime-rotation for
hypothesis #13, per
data/results/regime_rotation/SPEC_REGIME_ROTATION.txt sections 3-5
and GATE_REGIME_ROTATION.txt (R1-R4 pre-declared).

Order of operations (SPEC section 8): regime labels are computed and
WRITTEN FIRST (regime_labels.csv), before any child P&L is joined.
Children: ORB + SP2L MT5 net streams on the common ORB session-day
universe; reproduction gate aborts on mismatch. Arms: P complementary
(CALM->ORB, VOL->SP2L, RANGE->flat), Q vol-gated stack (CALM->both,
VOL->SP2L, RANGE->flat). R1-R4 evaluated per arm.

Outputs (data/results/regime_rotation/):
  regime_labels.csv, child_streams_regime.csv, ensemble_regime_pnl.csv,
  yearly_breakdown_regime.csv, phase1_regime_stats.json
"""

import csv
import json
import math
import time as timemod
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results\regime_rotation")
ORB = Path(r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present")
SP2L = Path(r"C:\Users\Milad\Desktop\RR\data\results\sp2l_2021_2026")
D1 = Path(r"C:\Users\Milad\Desktop\RR\data\results\momentum_multihorizon\us500c_d1.csv")
PORT = Path(r"C:\Users\Milad\Desktop\RR\data\results\portfolio_orb_sp2l")

BAND = 1.0          # +/- % around EMA200 defining RANGE
VOL_WIN = 20
MED_WIN = 252
ACCOUNT = 10000.0
SQ252 = math.sqrt(252)
CHILD_DOC = {"ORB": {"trades": 447, "pnl": 470.67, "sharpe": 0.337,
                     "dd_pct": 5.13},
             "SP2L": {"trades": 1343, "pnl": 281.84, "sharpe": 0.275,
                      "dd_pct": 4.17}}


def sharpe(v):
    a = np.asarray(v, dtype=float)
    sd = a.std(ddof=1)
    return float(a.mean() / sd * SQ252) if sd > 0 else None


def max_dd(v):
    cum = np.cumsum(np.asarray(v, dtype=float))
    return float((np.maximum.accumulate(cum) - cum).max())


def main():
    t0 = timemod.time()

    # ---------------- regime labels (written BEFORE any P&L join) ----
    d1 = pd.read_csv(D1)
    d1t = d1["time"].to_numpy(np.int64)
    d1c = d1["close"].to_numpy(float)
    didx = pd.DatetimeIndex(pd.to_datetime(d1t, unit="s", utc=True)
                            ).tz_convert("America/New_York")
    d1_date = (didx.year * 10000 + didx.month * 100 + didx.day).to_numpy()
    ema200 = pd.Series(d1c).ewm(span=200, adjust=False).mean().to_numpy(float)
    r = np.diff(np.log(d1c))
    vol20 = np.full(len(d1c), np.nan)
    vol20[1 + VOL_WIN - 1:] = (pd.Series(r).rolling(VOL_WIN).std(ddof=1)
                               .to_numpy() * math.sqrt(252) * 100.0)[VOL_WIN - 1:]

    # months covering the children's window
    months = pd.period_range("2021-01", "2026-04", freq="M")
    labels = {}
    rows = []
    for m in months:
        # Causality rule (SPEC section 3): use only D1 bars completed
        # before the first cash day of month M. US500.c D1 bars are
        # stamped 00:00 UTC of their NY trading date and complete at
        # ~17:00 ET that date, so the correct cutoff is UTC midnight of
        # the FIRST CALENDAR DAY of M: it includes the last prior-day
        # bar and excludes M's own first trading day (a NY-midnight
        # cutoff would wrongly include that bar - one-day lookahead,
        # caught by hand-check 2026-08-29).
        first_epoch = int(pd.Timestamp(str(m.start_time.date()))
                          .tz_localize("UTC").timestamp())
        prior = np.where(d1t < first_epoch)[0]
        if len(prior) == 0:
            continue
        i = prior[-1]
        med = np.median(vol20[max(0, i - MED_WIN):i])  # strictly before i
        trend_rel = (d1c[i] / ema200[i] - 1.0) * 100.0
        v20 = vol20[i]
        if abs(trend_rel) < BAND:
            lab = "RANGE"
        elif trend_rel >= BAND and v20 <= med:
            lab = "UP-CALM"
        elif trend_rel >= BAND:
            lab = "UP-VOL"
        elif trend_rel <= -BAND and v20 <= med:
            lab = "DOWN-CALM"
        elif trend_rel <= -BAND:
            lab = "DOWN-VOL"
        else:
            lab = "RANGE"
        assert np.isfinite(v20) and np.isfinite(med), f"burn-in gap at {m}"
        labels[str(m)] = lab
        rows.append({"month": str(m), "label": lab,
                     "label_bar_date": int(d1_date[i]),
                     "trend_rel_pct": round(trend_rel, 3),
                     "vol20_pct": round(v20, 2),
                     "vol_med_pct": round(med, 2)})
    with open(BASE / "regime_labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    cnt = defaultdict(int)
    for lab in labels.values():
        cnt[lab] += 1
    print(f"labels written first ({timemod.time() - t0:.0f}s); "
          f"month counts: {dict(cnt)}")

    # ---------------- children + universe + reproduction -------------
    def load_child(p):
        df = pd.read_csv(p, sep=";")
        df["d"] = df["open_time"].str[:10].str.replace(".", "").astype(int)
        df["profit"] = df["profit"].astype(float)
        return df

    orb_tr, sp2l_tr = load_child(ORB / "mt5_trades.csv"), load_child(
        SP2L / "mt5_trades.csv")
    od = pd.read_csv(ORB / "mt5_days.csv", sep=";", dtype=str)
    od["session_open"] = pd.to_numeric(od["session_open"], errors="coerce")
    od["prev_close"] = pd.to_numeric(od["prev_close"], errors="coerce")
    universe = sorted(od[(od["session_open"] > 0) | (od["prev_close"] > 0)]
                      ["ny_date"].astype(int))
    nu = len(universe)
    uidx = {d: j for j, d in enumerate(universe)}

    def stream(df):
        pnl = np.zeros(nu)
        sig = np.zeros(nu, dtype=bool)
        for d, g in df.groupby("d"):
            j = uidx.get(d)
            if j is not None:
                pnl[j] = g["profit"].sum()
                sig[j] = True
        return pnl, sig

    orb_pnl, orb_sig = stream(orb_tr)
    sp2l_pnl, sp2l_sig = stream(sp2l_tr)
    rep = {}
    ok = True
    for name, tr, pnl in (("ORB", orb_tr, orb_pnl),
                          ("SP2L", sp2l_tr, sp2l_pnl)):
        sh, dd = sharpe(pnl), max_dd(pnl) / ACCOUNT * 100
        good = (len(tr) == CHILD_DOC[name]["trades"]
                and abs(pnl.sum() - CHILD_DOC[name]["pnl"]) <= 0.05
                and abs(sh - CHILD_DOC[name]["sharpe"]) <= 0.03
                and abs(dd - CHILD_DOC[name]["dd_pct"]) <= 0.03)
        rep[name] = {"trades": len(tr), "pnl": round(float(pnl.sum()), 2),
                     "sharpe": round(sh, 3), "dd_pct": round(dd, 2),
                     "reproduced": bool(good)}
        ok = ok and good
        print(f"repro {name}: {rep[name]}")
    if not ok:
        print("REPRODUCTION GATE FAILED - abort")
        return

    # ---------------- arms -------------------------------------------
    def enabled(label, child, arm):
        if label == "RANGE":
            return False
        calm = label.endswith("CALM")
        if arm == "P":
            return ("ORB" if calm else "SP2L") == child
        # arm Q: calm -> both, vol -> SP2L only
        return True if calm else child == "SP2L"

    arms = {"P_complementary": np.zeros(nu), "Q_vol_gated_stack": np.zeros(nu)}
    day_rows = []
    disabled_tally = defaultdict(int)
    traded_days = {"P": 0, "Q": 0}
    for j, d in enumerate(universe):
        mkey = str(pd.Period(freq="M", year=int(d // 10000),
                             month=int((d // 100) % 100)))
        lab = labels[mkey]
        row = {"date": d, "month": mkey, "label": lab,
               "orb_pnl": round(orb_pnl[j], 2),
               "sp2l_pnl": round(sp2l_pnl[j], 2)}
        for arm in arms:
            a = arm[0]
            e_o = enabled(lab, "ORB", a)
            e_s = enabled(lab, "SP2L", a)
            arms[arm][j] = (orb_pnl[j] if e_o else 0.0) + (
                sp2l_pnl[j] if e_s else 0.0)
            row[f"orb_on_{a}"] = int(e_o)
            row[f"sp2l_on_{a}"] = int(e_s)
            row[f"ens_{a}"] = round(arms[arm][j], 2)
            if not e_o:
                disabled_tally[f"{a}_orb_disabled_days"] += 1
            else:
                disabled_tally[f"{a}_orb_enabled_days"] += 1
            if not e_s:
                disabled_tally[f"{a}_sp2l_disabled_days"] += 1
            else:
                disabled_tally[f"{a}_sp2l_enabled_days"] += 1
            if (e_o and orb_sig[j]) or (e_s and sp2l_sig[j]):
                traded_days[a] += 1
        day_rows.append(row)
    pd.DataFrame(day_rows).to_csv(BASE / "child_streams_regime.csv",
                                  index=False)

    port = pd.read_csv(PORT / "combined_daily_pnl.csv")
    port = port.set_index(port["date"].astype(int)).reindex(universe).fillna(0.0)

    # ---------------- metrics + gates --------------------------------
    results = {}
    yearly_rows = []
    regime_rows = []
    for arm, arr in arms.items():
        a = arm[0]
        sh = sharpe(arr)
        dd_pct = max_dd(arr) / ACCOUNT * 100
        tot = float(arr.sum())
        traded = traded_days[a]
        per_year = defaultdict(float)
        for j, d in enumerate(universe):
            per_year[d // 10000] += arr[j]
        full = {y: v for y, v in per_year.items() if y < 2026}
        n_pos = sum(1 for v in full.values() if v > 0)
        best = max(full.values()) / tot * 100 if tot > 0 and full else None
        # per-regime stats
        per_reg = defaultdict(list)
        for j, d in enumerate(universe):
            mk = str(pd.Period(freq="M", year=int(d // 10000),
                               month=int((d // 100) % 100)))
            per_reg[labels[mk]].append(arr[j])
        regime_stats = {}
        for k, v in sorted(per_reg.items()):
            s_ = sharpe(v) if len(v) > 1 else None
            regime_stats[k] = {"days": len(v),
                               "net": round(float(np.sum(v)), 2),
                               "sharpe": round(s_, 3) if s_ is not None else None}
        r1 = sh is not None and sh > 0 and sh > 0.420
        r2 = dd_pct <= 11.29
        r3 = traded >= 150
        r4 = (n_pos >= 4 and best is not None and best <= 60
              and (full.get(2025, 0.0) >= 0 or full.get(2025, 0.0) > -228.03))
        results[arm] = {"sharpe": round(sh, 3) if sh else None,
                        "max_dd_pct": round(dd_pct, 2),
                        "total_net": round(tot, 2), "traded_days": traded,
                        "per_year": {str(k): round(v, 2)
                                     for k, v in sorted(per_year.items())},
                        "pos_full_years": f"{n_pos}/{len(full)}",
                        "best_year_share_pct": round(best, 1) if best else None,
                        "net_2025": round(full.get(2025, 0.0), 2),
                        "per_regime": regime_stats,
                        "corr_vs_children": {
                            "orb": round(float(np.corrcoef(
                                arr, orb_pnl)[0, 1]), 3),
                            "sp2l": round(float(np.corrcoef(
                                arr, sp2l_pnl)[0, 1]), 3),
                            "S1": round(float(np.corrcoef(
                                arr, port['S1_equal_notional'].to_numpy())[0, 1]), 3)},
                        "gates": {"R1": bool(r1), "R2": bool(r2),
                                  "R3": bool(r3), "R4": bool(r4)}}
        results[arm]["verdict"] = ("PASS" if all(results[arm]["gates"].values())
                                   else "FAIL")
        print(f"\n{arm}: Sharpe {results[arm]['sharpe']} (need >0.420); "
              f"DD {dd_pct:.2f}%; net {tot:+.2f}; traded {traded}; "
              f"years {results[arm]['per_year']}; best-year "
              f"{results[arm]['best_year_share_pct']}%; 2025 "
              f"{results[arm]['net_2025']:+.2f}")
        print(f"  per-regime: {regime_stats}")
        print(f"  gates {results[arm]['gates']} -> {results[arm]['verdict']}")
        for y, v in sorted(per_year.items()):
            yearly_rows.append({"arm": arm, "year": y, "net": round(v, 2),
                                "share_pct": round(100 * v / tot, 1)
                                if tot > 0 else None})
        for k, s in regime_stats.items():
            regime_rows.append({"arm": arm, "regime": k, **s})

    overall = ("PASS -> Phase 2" if any(v["verdict"] == "PASS"
                                        for v in results.values())
               else "FAIL -> STOP (null result)")
    print(f"\nOVERALL: {overall}")

    pd.DataFrame({"date": universe,
                  "ens_P": arms["P_complementary"],
                  "ens_Q": arms["Q_vol_gated_stack"]}
                 ).to_csv(BASE / "ensemble_regime_pnl.csv", index=False)
    with open(BASE / "yearly_breakdown_regime.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "year", "net", "share_pct"])
        w.writeheader()
        w.writerows(yearly_rows)
    with open(BASE / "regime_breakdown.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "regime", "days",
                                           "net", "sharpe"])
        w.writeheader()
        w.writerows(regime_rows)

    stats = {"reproduction": rep, "labels": dict(cnt), "arms": results,
             "disabled_tallies": dict(disabled_tally),
             "overall_verdict": overall,
             "params": {"BAND": BAND, "VOL_WIN": VOL_WIN,
                        "MED_WIN": MED_WIN, "universe_days": nu},
             "runtime_s": round(timemod.time() - t0, 1)}
    with open(BASE / "phase1_regime_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2, default=str)
    print(f"wrote artifacts ({stats['runtime_s']}s)")


if __name__ == "__main__":
    main()
