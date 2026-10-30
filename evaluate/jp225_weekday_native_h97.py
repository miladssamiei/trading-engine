"""Frozen H97 native FTMO JP225 weekday hold Phase 1; no orders."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gold_us100_weekday_book_h96 import epoch, holiday_thursday
from xauusd_kc_breakout_oos_phase1 import wilder_atr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/jp225_weekday_native_h97"
SOURCE = ROOT / "data/results/jp225_source_qualification/jp225cash_native_tick_m1_2022_2026.csv"
SOURCE_META = ROOT / "data/results/jp225_source_qualification/native_tick_source_meta.json"
EXPECTED = "7930268680aad61134831c48a0223499990dd9a9470dd7662b383f888aac84a1"
EXTRA = 11.54
SEED = 20261007


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bootstrap(v: np.ndarray) -> float | None:
    n = len(v)
    if n < 5:
        return None
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, n, size=(10000, int(np.ceil(n/5))))
    ix = (starts[:, :, None] + np.arange(5)[None, None, :]) % n
    return float((v[ix.reshape(10000, -1)[:, :n]].mean(axis=1) <= 0).mean())


def replay(tape: pd.DataFrame):
    times = tape.minute_epoch.to_numpy(np.int64)
    bid_o = tape.bid_open.to_numpy(float)
    bid_l = tape.bid_low.to_numpy(float)
    ask_o = tape.ask_open.to_numpy(float)
    wall = pd.to_datetime(times, unit="s", utc=True).tz_convert("America/New_York").tz_localize(None)
    keys = pd.DatetimeIndex((wall + pd.Timedelta(hours=7)).date)
    weekday = keys.weekday < 5
    day = (pd.DataFrame({"date": keys[weekday], "o": bid_o[weekday],
                         "h": tape.bid_high.to_numpy(float)[weekday],
                         "l": bid_l[weekday],
                         "c": tape.bid_close.to_numpy(float)[weekday]})
           .groupby("date", sort=True)
           .agg(o=("o", "first"), h=("h", "max"),
                l=("l", "min"), c=("c", "last")))
    dates = pd.DatetimeIndex(day.index)
    assert (dates.weekday < 5).all()
    atr = wilder_atr(day.h.to_numpy(float), day.l.to_numpy(float),
                     day.c.to_numpy(float), 20)
    mondays = pd.date_range(wall[0].normalize(), wall[-1].normalize(), freq="W-MON")
    trades, coverage = [], []
    for monday in mondays:
        e0, e1 = epoch(monday, 0), epoch(monday, 0, 5)
        thursday = monday + pd.Timedelta(days=3)
        early = holiday_thursday(thursday)
        xday = monday + pd.Timedelta(days=2) if early else thursday
        x0 = epoch(xday, 11 if early else 0)
        x1 = epoch(xday, 11, 15) if early else epoch(xday, 0, 15)
        if x1 > times[-1]:
            coverage.append({"monday": str(monday.date()), "status": "INCOMPLETE_TAPE_WEEK"})
            continue
        k0 = int(np.searchsorted(times, e0))
        if k0 >= len(times) or times[k0] >= e1:
            coverage.append({"monday": str(monday.date()), "status": "NO_REAL_ENTRY"})
            continue
        j = int(dates.searchsorted(monday)) - 1
        if j < 19 or dates[j].weekday() not in (3, 4) or not np.isfinite(atr[j]):
            coverage.append({"monday": str(monday.date()), "status": "NO_PRIOR_ATR"})
            continue
        entry = float(ask_o[k0])
        distance = float(2*atr[j])
        assert distance > 0 and dates[j] < monday
        stop = entry-distance
        out, reason, kx = None, None, None
        for k in range(k0, len(times)):
            if times[k] >= x0:
                if times[k] < x1:
                    out, reason, kx = float(bid_o[k]), "HOLIDAY_FLAT" if early else "THURSDAY_FLAT", k
                else:
                    reason = "UNRESOLVED_HOLIDAY" if early else "UNRESOLVED_THURSDAY"
                break
            if bid_o[k] <= stop:
                out, reason, kx = float(bid_o[k]), "STOP_GAP", k
                break
            if bid_l[k] <= stop:
                out, reason, kx = float(stop), "STOP", k
                break
        if out is None:
            coverage.append({"monday": str(monday.date()),
                             "status": reason or "UNRESOLVED_END"})
            continue
        base = out-entry
        trades.append({"monday": str(monday.date()), "year": monday.year,
                       "entry_epoch": int(times[k0]), "exit_epoch": int(times[kx]),
                       "entry_ny": str(wall[k0]), "exit_ny": str(wall[kx]),
                       "prior_date": str(dates[j].date()),
                       "prior_atr20": float(atr[j]), "entry_bid": float(bid_o[k0]),
                       "entry_ask": entry, "entry_spread": float(entry-bid_o[k0]),
                       "initial_stop": stop, "exit_bid": out,
                       "exit_reason": reason, "holiday_exit": early,
                       "base_net": base, "stress_net": base-EXTRA,
                       "stress_r": (base-EXTRA)/distance,
                       "hold_hours": (times[kx]-times[k0])/3600})
        coverage.append({"monday": str(monday.date()), "status": "EXECUTED"})
    return pd.DataFrame(trades), pd.DataFrame(coverage), len(day)


def main():
    spec = OUT / "SPEC_JP225_WEEKDAY_NATIVE_H97.txt"
    gate = OUT / "GATE_JP225_WEEKDAY_NATIVE_H97.txt"
    report = OUT / "REPORT_JP225_WEEKDAY_NATIVE_H97.txt"
    assert not report.exists(), "Immutable H97 REPORT exists"
    assert [z for z in spec.read_text(encoding="utf-8").splitlines()
            if z.startswith("G-JP")] == gate.read_text(encoding="utf-8").splitlines()
    assert sha(SOURCE) == EXPECTED
    meta_source = json.loads(SOURCE_META.read_text(encoding="utf-8"))
    assert meta_source["sha256"] == EXPECTED
    assert sum(z["ticks"] for z in meta_source["monthly"]) == 50121955
    tape = pd.read_csv(SOURCE)
    assert len(tape) == meta_source["rows"]
    assert tape.minute_epoch.is_monotonic_increasing and tape.minute_epoch.is_unique
    assert tape.tick_count.gt(0).all()
    assert (tape.ask_open >= tape.bid_open).all()
    assert (tape.bid_high >= tape[["bid_open", "bid_low", "bid_close"]].max(axis=1)).all()
    assert (tape.bid_low <= tape[["bid_open", "bid_high", "bid_close"]].min(axis=1)).all()
    trades, coverage, ndays = replay(tape)
    trades.to_csv(OUT / "trades.csv", index=False)
    coverage.to_csv(OUT / "coverage.csv", index=False)
    oos = trades.loc[trades.year >= 2024]
    v = oos.stress_net.to_numpy(float)
    unresolved = int(coverage.loc[pd.to_datetime(coverage.monday).dt.year >= 2024,
                                  "status"].str.startswith("UNRESOLVED").sum())
    annual = {str(y): {"n": int((oos.year == y).sum()),
                       "base_sum": float(oos.loc[oos.year == y, "base_net"].sum()),
                       "stress_sum": float(oos.loc[oos.year == y, "stress_net"].sum())}
              for y in (2024, 2025, 2026)}
    p = bootstrap(v)
    arithmetic = bool(np.allclose(trades.base_net, trades.exit_bid-trades.entry_ask)
                      and np.allclose(trades.stress_net, trades.base_net-EXTRA)
                      and np.allclose(trades.stress_r,
                                      trades.stress_net/(2*trades.prior_atr20)))
    causal = bool((pd.to_datetime(trades.prior_date) < pd.to_datetime(trades.monday)).all())
    weekday_flat = bool((pd.to_datetime(trades.exit_ny).dt.dayofweek < 4).all())
    gates = {"G-JP1": len(oos) >= 120 and oos.monday.nunique() >= 120
             and unresolved == 0,
             "G-JP2": len(v) > 0 and float(v.mean()) > 0,
             "G-JP3": p is not None and p <= 0.05,
             "G-JP4": all(annual[str(y)]["stress_sum"] > 0 for y in (2024, 2025, 2026)),
             "G-JP5": "NOT_RUN", "G-JP6": arithmetic and causal and weekday_flat
             and unresolved == 0}
    passed = all(gates[z] for z in ("G-JP1", "G-JP2", "G-JP3", "G-JP4", "G-JP6"))
    status = "PHASE1_PASS_NEEDS_PHASE2" if passed else "FINAL_FAIL"
    meta = {"status": status, "source_sha256": EXPECTED, "spec_sha256": sha(spec),
            "gate_sha256": sha(gate), "script_sha256": sha(Path(__file__)),
            "m1_rows": len(tape), "daily_rows": ndays,
            "all_trades": len(trades), "evaluation_trades": len(oos),
            "evaluation_unresolved": unresolved,
            "coverage": coverage.status.value_counts().to_dict(),
            "evaluation_exits": oos.exit_reason.value_counts().to_dict(),
            "evaluation_entry_spread": {"median": float(oos.entry_spread.median()),
                                        "p90": float(oos.entry_spread.quantile(.9))},
            "evaluation_stress_mean": float(v.mean()) if len(v) else None,
            "evaluation_stress_total": float(v.sum()),
            "evaluation_mean_r": float(oos.stress_r.mean()) if len(oos) else None,
            "evaluation_boot_p_nonpositive": p,
            "annual": annual, "gates": gates,
            "zero_swap_owner_assumption": True}
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with report.open("x", encoding="utf-8") as f:
        f.write("H97 NATIVE JP225 WEEKDAY HOLD - " + status + "\n")
        f.write("Connected FTMO-Demo tick-derived native M1 variable Ask/Bid; no orders.\n")
        f.write("Source/SPEC/GATE/script SHA256: " + EXPECTED + " / " + sha(spec)
                + " / " + sha(gate) + " / " + sha(Path(__file__)) + "\n")
        f.write("All/evaluation trades/unresolved: "
                + str((len(trades), len(oos), unresolved)) + "\n")
        f.write("Coverage/exits: " + json.dumps(meta["coverage"]) + " / "
                + json.dumps(meta["evaluation_exits"]) + "\n")
        f.write("Entry spread: " + json.dumps(meta["evaluation_entry_spread"]) + "\n")
        f.write("Evaluation stressed mean/total/mean R/bootstrap P: "
                + str((meta["evaluation_stress_mean"],
                       meta["evaluation_stress_total"],
                       meta["evaluation_mean_r"], p)) + "\n")
        f.write("Annual: " + json.dumps(annual) + "\n")
        f.write("Gates: " + json.dumps(gates) + "\n")
        f.write("M1 stop paths still need exact tick-order and EURJPY/lot/equity Phase 2.\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
