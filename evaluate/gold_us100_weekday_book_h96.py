"""Frozen H96 gold plus US100 weekday book Phase 1; read-only proxy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from xauusd_kc_breakout_oos_phase1 import wilder_atr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/gold_us100_weekday_book_h96"
SOURCES = {
    "GOLD": (ROOT / "data/results/xauusd_onboarding/xauusdtick_m1.csv",
             "a8c4c1d6551d27f3796f8d9b88d332172c41c76c151844e759492837152e7b1f"),
    "US100": (ROOT / "data/results/virgin_assets/ustech_m1.csv",
              "152f9c995b370e5391cb7da99b64a8002ceee5e52cc68899ddca409ac7e15e39"),
}
SEED = 20261006


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def epoch(day: pd.Timestamp, hour: int, minute: int = 0) -> int:
    return int((day + pd.Timedelta(hours=hour, minutes=minute))
               .tz_localize("America/New_York").timestamp())


def holiday_thursday(day: pd.Timestamp) -> bool:
    return bool((day.month == 1 and day.day == 1) or
                (day.month == 7 and day.day == 4) or
                (day.month == 12 and day.day == 25) or
                (day.month == 11 and day.weekday() == 3 and
                 22 <= day.day <= 28))


def daily_atr(tape: pd.DataFrame, asset: str, wall: pd.DatetimeIndex):
    if asset == "GOLD":
        keys = (wall + pd.Timedelta(hours=7)).date
        day = (pd.DataFrame({"date": keys, "o": tape.open.to_numpy(float),
                             "h": tape.high.to_numpy(float),
                             "l": tape.low.to_numpy(float),
                             "c": tape.close.to_numpy(float)})
               .groupby("date", sort=True)
               .agg(o=("o", "first"), h=("h", "max"),
                    l=("l", "min"), c=("c", "last")))
        dates = pd.DatetimeIndex(day.index)
        assert (dates.weekday < 5).all()
    else:
        minutes = wall.hour * 60 + wall.minute
        mask = (wall.weekday < 5) & (minutes >= 570) & (minutes < 960)
        day = (pd.DataFrame({"date": wall[mask].normalize(),
                             "o": tape.open.to_numpy(float)[mask],
                             "h": tape.high.to_numpy(float)[mask],
                             "l": tape.low.to_numpy(float)[mask],
                             "c": tape.close.to_numpy(float)[mask],
                             "first_minute": minutes[mask]})
               .groupby("date", sort=True)
               .agg(o=("o", "first"), h=("h", "max"),
                    l=("l", "min"), c=("c", "last"),
                    first_minute=("first_minute", "first")))
        day = day.loc[day.first_minute == 570]
        dates = pd.DatetimeIndex(day.index)
    atr = wilder_atr(day.h.to_numpy(float), day.l.to_numpy(float),
                     day.c.to_numpy(float), 20)
    return dates, atr


def replay_asset(asset: str):
    path, expected = SOURCES[asset]
    assert sha(path) == expected
    tape = pd.read_csv(path, usecols=["time", "open", "high", "low", "close"])
    assert tape.time.is_monotonic_increasing and tape.time.is_unique
    assert (tape.high >= tape[["open", "low", "close"]].max(axis=1)).all()
    assert (tape.low <= tape[["open", "high", "close"]].min(axis=1)).all()
    times = tape.time.to_numpy(np.int64)
    o = tape.open.to_numpy(float)
    low = tape.low.to_numpy(float)
    wall = pd.to_datetime(times, unit="s", utc=True).tz_convert("America/New_York").tz_localize(None)
    dates, atr = daily_atr(tape, asset, wall)
    mondays = pd.date_range(wall[0].normalize(), wall[-1].normalize(), freq="W-MON")
    trades, coverage = [], []
    is_gold = asset == "GOLD"
    spread = 0.41 if is_gold else 1.45
    commission_rate = 0.000007 if is_gold else 0.0
    extra = 0.66 if is_gold else 1.70
    for monday in mondays:
        thursday = monday + pd.Timedelta(days=3)
        early = holiday_thursday(thursday)
        xday = monday + pd.Timedelta(days=2) if early else thursday
        xhour = 15 if early else (14 if is_gold else 11)
        xminute = 55 if early else 0
        x0 = epoch(xday, xhour, xminute)
        x1 = epoch(xday, 16) if early else epoch(xday, xhour, 15)
        if x1 > times[-1]:
            coverage.append({"monday": str(monday.date()), "asset": asset,
                             "status": "INCOMPLETE_TAPE_WEEK"})
            continue
        if not is_gold and monday not in dates:
            coverage.append({"monday": str(monday.date()), "asset": asset,
                             "status": "NO_MONDAY_CASH_OPEN"})
            continue
        e0 = epoch(monday, 0 if is_gold else 9, 0 if is_gold else 30)
        e1 = epoch(monday, 0 if is_gold else 9, 5 if is_gold else 35)
        k0 = int(np.searchsorted(times, e0))
        if k0 >= len(times) or times[k0] >= e1:
            coverage.append({"monday": str(monday.date()), "asset": asset,
                             "status": "NO_REAL_ENTRY"})
            continue
        j = int(dates.searchsorted(monday)) - 1
        valid_atr = (j >= 19 and np.isfinite(atr[j]) and
                     (dates[j].weekday() in (3, 4) if is_gold else
                      (monday-dates[j]).days <= 4))
        if not valid_atr:
            coverage.append({"monday": str(monday.date()), "asset": asset,
                             "status": "NO_PRIOR_ATR"})
            continue
        entry = float(o[k0] + spread)
        dist = float(2*atr[j])
        assert dist > 0 and dates[j] < monday
        stop = entry-dist
        out, reason, kx = None, None, None
        for k in range(k0, len(times)):
            if times[k] >= x0:
                if times[k] < x1:
                    out, reason, kx = float(o[k]), "HOLIDAY_FLAT" if early else "THURSDAY_FLAT", k
                else:
                    reason = "UNRESOLVED_HOLIDAY" if early else "UNRESOLVED_THURSDAY"
                break
            if o[k] <= stop:
                out, reason, kx = float(o[k]), "STOP_GAP", k
                break
            if low[k] <= stop:
                out, reason, kx = float(stop), "STOP", k
                break
        if out is None:
            coverage.append({"monday": str(monday.date()), "asset": asset,
                             "status": reason or "UNRESOLVED_END"})
            continue
        commission = commission_rate*(entry+out)
        base = out-entry-commission
        trades.append({"monday": str(monday.date()), "year": monday.year,
                       "asset": asset, "entry_epoch": int(times[k0]),
                       "exit_epoch": int(times[kx]), "entry_ny": str(wall[k0]),
                       "exit_ny": str(wall[kx]), "holiday_exit": early,
                       "prior_date": str(dates[j].date()),
                       "prior_atr20": float(atr[j]), "entry": entry,
                       "initial_stop": stop, "exit": out, "exit_reason": reason,
                       "commission": commission, "base_net": base,
                       "stress_net": base-extra,
                       "stress_r": (base-extra)/dist,
                       "hold_hours": (times[kx]-times[k0])/3600})
        coverage.append({"monday": str(monday.date()), "asset": asset,
                         "status": "EXECUTED"})
    return pd.DataFrame(trades), pd.DataFrame(coverage), len(tape), len(dates)


def bootstrap(v: np.ndarray):
    n = len(v)
    if n < 5:
        return None
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, n, size=(10000, int(np.ceil(n/5))))
    ix = (starts[:, :, None] + np.arange(5)[None, None, :]) % n
    return float((v[ix.reshape(10000, -1)[:, :n]].mean(axis=1) <= 0).mean())


def main():
    spec = OUT / "SPEC_GOLD_US100_WEEKDAY_BOOK_H96.txt"
    gate = OUT / "GATE_GOLD_US100_WEEKDAY_BOOK_H96.txt"
    report = OUT / "REPORT_GOLD_US100_WEEKDAY_BOOK_H96.txt"
    assert not report.exists()
    assert [line for line in spec.read_text(encoding="utf-8").splitlines()
            if line.startswith("G-GB")] == gate.read_text(encoding="utf-8").splitlines()
    gold, gcov, grows, gdays = replay_asset("GOLD")
    index, icov, irows, idays = replay_asset("US100")
    gold.to_csv(OUT / "gold_trades.csv", index=False)
    index.to_csv(OUT / "us100_trades.csv", index=False)
    coverage = pd.concat([gcov, icov], ignore_index=True)
    coverage.to_csv(OUT / "coverage.csv", index=False)
    book = gold.merge(index, on="monday", suffixes=("_gold", "_us100"),
                      validate="one_to_one")
    book = book.loc[book.monday <= gold.monday.max()].sort_values("monday").copy()
    book["stress_eur"] = 200*(book.stress_r_gold+book.stress_r_us100)
    book.to_csv(OUT / "book.csv", index=False)
    oos = book.loc[book.year_gold >= 2024]
    v = oos.stress_eur.to_numpy(float)
    gold_oos = gold.loc[gold.year >= 2024]
    index_oos = index.loc[(index.year >= 2024) & (index.monday <= gold.monday.max())]
    unilateral = (set(gold_oos.monday) ^ set(index_oos.monday))
    coverage_oos = coverage.loc[pd.to_datetime(coverage.monday).dt.year >= 2024]
    unresolved = int(coverage_oos.status.str.startswith("UNRESOLVED").sum())
    annual = {str(y): {"n": int((oos.year_gold == y).sum()),
                       "gold_r": float(oos.loc[oos.year_gold == y, "stress_r_gold"].sum()),
                       "us100_r": float(oos.loc[oos.year_gold == y, "stress_r_us100"].sum()),
                       "eur": float(oos.loc[oos.year_gold == y, "stress_eur"].sum())}
              for y in (2024, 2025, 2026)}
    p = bootstrap(v)
    path_ok = bool((gold.exit_epoch < gold.entry_epoch + 5*86400).all()
                   and (index.exit_epoch < index.entry_epoch + 5*86400).all()
                   and np.allclose(gold.stress_net, gold.base_net-0.66)
                   and np.allclose(index.stress_net, index.base_net-1.70)
                   and (gold.prior_atr20 > 0).all() and (index.prior_atr20 > 0).all())
    gates = {"G-GB1": len(oos) >= 120 and oos.monday.nunique() >= 120
             and unresolved == 0,
             "G-GB2": len(v) > 0 and float(v.mean()) > 0,
             "G-GB3": p is not None and p <= 0.05,
             "G-GB4": all(annual[str(y)]["eur"] > 0 for y in (2024, 2025, 2026)),
             "G-GB5": "NOT_RUN", "G-GB6": path_ok and unresolved == 0
             and len(unilateral) == 0}
    passed = all(gates[z] for z in ("G-GB1", "G-GB2", "G-GB3", "G-GB4", "G-GB6"))
    status = "PHASE1_PASS_NEEDS_PHASE2" if passed else "FINAL_FAIL"
    meta = {"status": status, "source_sha256": {z: h for z, (_, h) in SOURCES.items()},
            "spec_sha256": sha(spec), "gate_sha256": sha(gate),
            "script_sha256": sha(Path(__file__)),
            "source_rows": {"GOLD": grows, "US100": irows},
            "daily_rows": {"GOLD": gdays, "US100": idays},
            "all_leg_trades": {"GOLD": len(gold), "US100": len(index)},
            "all_common_books": len(book), "evaluation_common_books": len(oos),
            "evaluation_unresolved": unresolved,
            "evaluation_unilateral_dates": sorted(unilateral),
            "coverage": {z: coverage.loc[coverage.asset == z, "status"].value_counts().to_dict()
                         for z in ("GOLD", "US100")},
            "evaluation_gold_exits": oos.exit_reason_gold.value_counts().to_dict(),
            "evaluation_us100_exits": oos.exit_reason_us100.value_counts().to_dict(),
            "evaluation_stressed_eur_mean": float(v.mean()) if len(v) else None,
            "evaluation_stressed_eur_total": float(v.sum()),
            "evaluation_boot_p_nonpositive": p,
            "evaluation_correlation": float(oos.stress_r_gold.corr(oos.stress_r_us100)),
            "annual": annual, "gates": gates,
            "prior_component_outcome_exposure": True,
            "zero_swap_owner_assumption": True}
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with report.open("x", encoding="utf-8") as f:
        f.write("H96 GOLD+US100 WEEKDAY BOOK - " + status + "\n")
        f.write("Dukascopy BID M1 fixed-spread proxy, known-holiday early exits, no orders.\n")
        f.write("Sources/SPEC/GATE/script SHA256: " + json.dumps(meta["source_sha256"])
                + " / " + sha(spec) + " / " + sha(gate) + " / "
                + sha(Path(__file__)) + "\n")
        f.write("Leg/common/evaluation trades: " + str((len(gold), len(index),
                                                        len(book), len(oos))) + "\n")
        f.write("Evaluation unresolved/unilateral: "
                + str((unresolved, meta["evaluation_unilateral_dates"])) + "\n")
        f.write("Evaluation stressed EUR mean/total/bootstrap P/correlation: "
                + str((meta["evaluation_stressed_eur_mean"],
                       meta["evaluation_stressed_eur_total"], p,
                       meta["evaluation_correlation"])) + "\n")
        f.write("Annual: " + json.dumps(annual) + "\n")
        f.write("Exits: " + json.dumps({"GOLD": meta["evaluation_gold_exits"],
                                      "US100": meta["evaluation_us100_exits"]}) + "\n")
        f.write("Gates: " + json.dumps(gates) + "\n")
        f.write("Component tape outcomes were exposed; no unseen confirmation or native prop replay.\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
