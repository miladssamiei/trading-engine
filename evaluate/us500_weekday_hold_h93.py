"""Frozen H93 US500 Monday-Thursday cash-open hold Phase 1; no orders."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from xauusd_kc_breakout_oos_phase1 import wilder_atr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/us500_weekday_hold_h93"
TAPE = ROOT / "data/results/vwap_2021_2026/us500tick_m1.csv"
TAPE_SHA = "29ad8f8b25b43be43a87c4ac39575c3cd280d557661c31f8e691990949087453"
SPREAD = 0.60
EXTRA = 0.85
SEED = 20261003


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def epoch(day: pd.Timestamp, hour: int, minute: int = 0) -> int:
    return int((day + pd.Timedelta(hours=hour, minutes=minute))
               .tz_localize("America/New_York").timestamp())


def bootstrap(v: np.ndarray) -> float | None:
    n = len(v)
    if n < 5:
        return None
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, n, size=(10000, int(np.ceil(n / 5))))
    ix = (starts[:, :, None] + np.arange(5)[None, None, :]) % n
    return float((v[ix.reshape(10000, -1)[:, :n]].mean(axis=1) <= 0).mean())


def replay(tape: pd.DataFrame):
    times = tape.time.to_numpy(np.int64)
    o, h, l, c = [tape[z].to_numpy(float) for z in ("open", "high", "low", "close")]
    wall = pd.to_datetime(times, unit="s", utc=True).tz_convert("America/New_York").tz_localize(None)
    wall_min = wall.hour * 60 + wall.minute
    cash_mask = ((wall.weekday < 5) & (wall_min >= 570) & (wall_min < 960))
    cash = (pd.DataFrame({"date": wall[cash_mask].normalize(),
                          "o": o[cash_mask], "h": h[cash_mask],
                          "l": l[cash_mask], "c": c[cash_mask],
                          "minute": wall_min[cash_mask]})
            .groupby("date", sort=True)
            .agg(o=("o", "first"), h=("h", "max"), l=("l", "min"),
                 c=("c", "last"), first_minute=("minute", "first")))
    cash = cash.loc[cash.first_minute == 570]
    dates = pd.DatetimeIndex(cash.index)
    dc = cash.c.to_numpy(float)
    atr = wilder_atr(cash.h.to_numpy(float), cash.l.to_numpy(float), dc, 20)
    mondays = pd.date_range(wall[0].normalize(), wall[-1].normalize(), freq="W-MON")
    trades, coverage = [], []
    for monday in mondays:
        e0, e1 = epoch(monday, 9, 30), epoch(monday, 9, 35)
        thursday = monday + pd.Timedelta(days=3)
        x0, x1 = epoch(thursday, 11), epoch(thursday, 11, 15)
        if x1 > times[-1]:
            coverage.append({"monday": str(monday.date()), "status": "INCOMPLETE_TAPE_WEEK"})
            continue
        if monday not in dates:
            coverage.append({"monday": str(monday.date()), "status": "NO_MONDAY_CASH_OPEN"})
            continue
        k0 = int(np.searchsorted(times, e0))
        if k0 >= len(times) or times[k0] >= e1:
            coverage.append({"monday": str(monday.date()), "status": "NO_REAL_ENTRY"})
            continue
        j = int(dates.searchsorted(monday)) - 1
        if j < 19 or (monday - dates[j]).days > 4 or not np.isfinite(atr[j]):
            coverage.append({"monday": str(monday.date()), "status": "NO_PRIOR_ATR"})
            continue
        entry = float(o[k0] + SPREAD)
        distance = float(2 * atr[j])
        assert distance > 0
        stop = entry - distance
        out, reason, kx = None, None, None
        for k in range(k0, len(times)):
            if times[k] >= x0:
                if times[k] < x1:
                    out, reason, kx = float(o[k]), "THURSDAY_FLAT", k
                else:
                    reason = "UNRESOLVED_THURSDAY"
                break
            if o[k] <= stop:
                out, reason, kx = float(o[k]), "STOP_GAP", k
                break
            if l[k] <= stop:
                out, reason, kx = float(stop), "STOP", k
                break
        if out is None:
            coverage.append({"monday": str(monday.date()), "status": reason or "UNRESOLVED_END"})
            continue
        base = out - entry
        trades.append({"monday": str(monday.date()), "year": monday.year,
                       "entry_epoch": int(times[k0]), "exit_epoch": int(times[kx]),
                       "entry_ny": str(wall[k0]), "exit_ny": str(wall[kx]),
                       "prior_cash_date": str(dates[j].date()),
                       "prior_atr20": float(atr[j]), "entry": entry,
                       "initial_stop": stop, "exit": out, "exit_reason": reason,
                       "base_net": base, "stress_net": base - EXTRA,
                       "hold_hours": (times[kx] - times[k0]) / 3600})
        coverage.append({"monday": str(monday.date()), "status": "EXECUTED"})
    return pd.DataFrame(trades), pd.DataFrame(coverage), len(cash)


def main():
    spec = OUT / "SPEC_US500_WEEKDAY_HOLD_H93.txt"
    gate = OUT / "GATE_US500_WEEKDAY_HOLD_H93.txt"
    report = OUT / "REPORT_US500_WEEKDAY_HOLD_H93.txt"
    assert not report.exists(), "Immutable H93 REPORT exists"
    assert [line for line in spec.read_text(encoding="utf-8").splitlines()
            if line.startswith("G-WH")] == gate.read_text(encoding="utf-8").splitlines()
    assert sha(TAPE) == TAPE_SHA
    tape = pd.read_csv(TAPE, usecols=["time", "open", "high", "low", "close"])
    assert tape.time.is_monotonic_increasing and not tape.time.duplicated().any()
    assert (tape.high >= tape[["open", "low", "close"]].max(axis=1)).all()
    assert (tape.low <= tape[["open", "high", "close"]].min(axis=1)).all()
    trades, coverage, cash_days = replay(tape)
    trades.to_csv(OUT / "trades.csv", index=False)
    coverage.to_csv(OUT / "coverage.csv", index=False)
    oos = trades[trades.year >= 2024] if len(trades) else trades
    v = oos.stress_net.to_numpy(float) if len(oos) else np.array([])
    unresolved = int(coverage.status.str.startswith("UNRESOLVED").sum())
    annual = {str(y): {"n": int((oos.year == y).sum()),
                       "stress_sum": float(oos.loc[oos.year == y, "stress_net"].sum())}
              for y in (2024, 2025, 2026)} if len(oos) else {}
    p = bootstrap(v)
    gates = {"G-WH1": len(oos) >= 100 and oos.monday.nunique() >= 100 and unresolved == 0,
             "G-WH2": len(v) > 0 and float(v.mean()) > 0,
             "G-WH3": p is not None and p <= 0.05,
             "G-WH4": bool(annual) and all(annual[str(y)]["stress_sum"] > 0
                                           for y in (2024, 2025, 2026)),
             "G-WH5": "NOT_RUN", "G-WH6": unresolved == 0 and len(trades) > 0}
    passed = all(gates[k] for k in ("G-WH1", "G-WH2", "G-WH3", "G-WH4", "G-WH6"))
    status = "PHASE1_PASS_NEEDS_PHASE2" if passed else "FINAL_FAIL"
    meta = {"status": status, "source_sha256": sha(TAPE), "spec_sha256": sha(spec),
            "gate_sha256": sha(gate), "script_sha256": sha(Path(__file__)),
            "m1_rows": len(tape), "cash_days": cash_days, "all_trades": len(trades),
            "oos_trades": len(oos), "unresolved": unresolved,
            "coverage": coverage.status.value_counts().to_dict(),
            "oos_stress_mean": float(v.mean()) if len(v) else None,
            "oos_stress_total": float(v.sum()) if len(v) else None,
            "oos_boot_p_nonpositive": p, "annual": annual, "gates": gates,
            "zero_swap_owner_assumption": True}
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with report.open("x", encoding="utf-8") as f:
        f.write("H93 US500 WEEKDAY HOLD - " + status + "\n")
        f.write("Dukascopy BID M1 proxy; owner-stated zero swap; news holds allowed. No orders.\n")
        f.write("Source/SPEC/GATE SHA256: " + TAPE_SHA + " / " + sha(spec) + " / " + sha(gate) + "\n")
        f.write("M1/cash-day rows: " + str((len(tape), cash_days)) + "\n")
        f.write("All/OOS trades/unresolved: " + str((len(trades), len(oos), unresolved)) + "\n")
        f.write("Coverage: " + str(meta["coverage"]) + "\n")
        f.write("OOS stressed mean/total/bootstrap P: " + str((meta["oos_stress_mean"],
                                                               meta["oos_stress_total"], p)) + "\n")
        f.write("Annual: " + json.dumps(annual) + "\n")
        f.write("Gates: " + json.dumps(gates) + "\n")
        f.write("Historical exposed proxy, not native FTMO execution or unseen forward confirmation.\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
