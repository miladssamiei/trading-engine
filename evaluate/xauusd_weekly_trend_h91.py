"""H91 frozen Monday-Thursday gold time-series-momentum Phase 1; no orders."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from xauusd_kc_breakout_oos_phase1 import wilder_atr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/xauusd_weekly_trend_h91"
TAPE = ROOT / "data/results/xauusd_onboarding/xauusdtick_m1.csv"
SOURCE_SHA = "a8c4c1d6551d27f3796f8d9b88d332172c41c76c151844e759492837152e7b1f"
SPREAD = 0.41
STRESS = 0.66
COMMISSION = 0.000007
SEED = 20261002


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def epoch(ny_day: pd.Timestamp, hour: int, minute: int = 0) -> int:
    return int((ny_day + pd.Timedelta(hours=hour, minutes=minute))
               .tz_localize("America/New_York").timestamp())


def bootstrap(values: np.ndarray) -> float | None:
    n = len(values)
    if n < 5:
        return None
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, n, size=(10000, int(np.ceil(n / 5))))
    ix = (starts[:, :, None] + np.arange(5)[None, None, :]) % n
    return float((values[ix.reshape(10000, -1)[:, :n]].mean(axis=1) <= 0).mean())


def replay(tape: pd.DataFrame):
    times = tape.time.to_numpy(np.int64)
    o, h, l, c = [tape[z].to_numpy(float) for z in ("open", "high", "low", "close")]
    ny = pd.to_datetime(times, unit="s", utc=True).tz_convert("America/New_York")
    day_key = (ny + pd.Timedelta(hours=8)).date
    day = (pd.DataFrame({"date": day_key, "o": o, "h": h, "l": l, "c": c})
           .groupby("date", sort=True)
           .agg(o=("o", "first"), h=("h", "max"),
                l=("l", "min"), c=("c", "last")))
    day_dates = pd.DatetimeIndex(day.index)
    dc = day.c.to_numpy(float)
    atr = wilder_atr(day.h.to_numpy(float), day.l.to_numpy(float), dc, 20)
    mondays = pd.date_range(ny[0].tz_localize(None).normalize(),
                            ny[-1].tz_localize(None).normalize(), freq="W-MON")
    trades, coverage = [], []
    for monday in mondays:
        start, entry_end = epoch(monday, 0), epoch(monday, 0, 5)
        exit_start, exit_end = epoch(monday + pd.Timedelta(days=3), 14), epoch(monday + pd.Timedelta(days=3), 14, 15)
        if exit_end > times[-1]:
            coverage.append({"monday": str(monday.date()), "status": "INCOMPLETE_TAPE_WEEK"})
            continue
        k0 = int(np.searchsorted(times, start))
        if k0 >= len(times) or times[k0] >= entry_end:
            coverage.append({"monday": str(monday.date()), "status": "NO_REAL_ENTRY"})
            continue
        j = int(day_dates.searchsorted(monday)) - 1
        if j < 21 or day_dates[j].weekday() not in (3, 4) or not np.isfinite(atr[j]):
            coverage.append({"monday": str(monday.date()), "status": "NO_CAUSAL_SIGNAL"})
            continue
        side = int(np.sign(dc[j] - dc[j - 21]))
        if side == 0:
            coverage.append({"monday": str(monday.date()), "status": "FLAT_SIGNAL"})
            continue
        entry = float(o[k0] + SPREAD if side == 1 else o[k0])
        dist = float(2 * atr[j])
        stop = entry - side * dist
        assert dist > 0
        out, reason, kx = None, None, None
        for k in range(k0, len(times)):
            if times[k] >= exit_start:
                if times[k] < exit_end:
                    out = float(o[k] if side == 1 else o[k] + SPREAD)
                    reason, kx = "THURSDAY_FLAT", k
                else:
                    reason = "UNRESOLVED_THURSDAY"
                break
            if side == 1:
                if o[k] <= stop:
                    out, reason, kx = float(o[k]), "STOP_GAP", k
                    break
                if l[k] <= stop:
                    out, reason, kx = float(stop), "STOP", k
                    break
            else:
                ask_o = o[k] + SPREAD
                if ask_o >= stop:
                    out, reason, kx = float(ask_o), "STOP_GAP", k
                    break
                if h[k] + SPREAD >= stop:
                    out, reason, kx = float(stop), "STOP", k
                    break
        if out is None:
            coverage.append({"monday": str(monday.date()), "status": reason or "UNRESOLVED_END"})
            continue
        gross = side * (out - entry)
        commission = COMMISSION * (entry + out)
        base = gross - commission
        kl = int(np.searchsorted(times, exit_start))
        if kl < len(times) and times[kl] < exit_end:
            bentry, bexit = float(o[k0] + SPREAD), float(o[kl])
            bench = bexit - bentry - COMMISSION * (bentry + bexit) - STRESS
        else:
            bench = np.nan
        trades.append({"monday": str(monday.date()),
                       "entry_epoch": int(times[k0]), "exit_epoch": int(times[kx]),
                       "entry_ny": str(ny[k0]), "exit_ny": str(ny[kx]),
                       "year": monday.year, "side": "LONG" if side == 1 else "SHORT",
                       "prior_date": str(day_dates[j].date()),
                       "prior_close": float(dc[j]), "momentum_21": float(dc[j] - dc[j - 21]),
                       "prior_atr20": float(atr[j]), "entry": entry,
                       "exit": out, "initial_stop": stop, "exit_reason": reason,
                       "gross": gross, "commission": commission, "base_net": base,
                       "stress_net": base - STRESS, "benchmark_stress": bench,
                       "hold_hours": (times[kx] - times[k0]) / 3600})
        coverage.append({"monday": str(monday.date()), "status": "EXECUTED"})
    return pd.DataFrame(trades), pd.DataFrame(coverage), len(day)


def main():
    spec = OUT / "SPEC_XAUUSD_WEEKLY_TREND_H91.txt"
    gate = OUT / "GATE_XAUUSD_WEEKLY_TREND_H91.txt"
    report = OUT / "REPORT_XAUUSD_WEEKLY_TREND_H91.txt"
    assert not report.exists(), "Immutable H91 report exists"
    assert [x for x in spec.read_text(encoding="utf-8").splitlines()
            if x.startswith("G-WT")] == gate.read_text(encoding="utf-8").splitlines()
    assert sha(TAPE) == SOURCE_SHA
    tape = pd.read_csv(TAPE, usecols=["time", "open", "high", "low", "close"])
    assert tape.time.is_monotonic_increasing and not tape.time.duplicated().any()
    assert (tape.high >= tape[["open", "low", "close"]].max(axis=1)).all()
    assert (tape.low <= tape[["open", "high", "close"]].min(axis=1)).all()
    trades, coverage, ndays = replay(tape)
    trades.to_csv(OUT / "trades.csv", index=False)
    coverage.to_csv(OUT / "coverage.csv", index=False)
    oos = trades[trades.year >= 2024] if len(trades) else trades
    v = oos.stress_net.to_numpy(float) if len(oos) else np.array([])
    unresolved = int(coverage.status.str.startswith("UNRESOLVED").sum())
    annual = {str(y): {"n": int((oos.year == y).sum()),
                       "stress_sum": float(oos.loc[oos.year == y, "stress_net"].sum())}
              for y in (2024, 2025, 2026)} if len(oos) else {}
    p = bootstrap(v)
    gates = {"G-WT1": len(oos) >= 100 and unresolved == 0,
             "G-WT2": len(v) > 0 and float(v.mean()) > 0,
             "G-WT3": p is not None and p <= 0.05,
             "G-WT4": bool(annual) and all(annual[str(y)]["stress_sum"] > 0
                                           for y in (2024, 2025, 2026)),
             "G-WT5": "NOT_RUN", "G-WT6": unresolved == 0 and len(trades) > 0}
    passed = all(gates[k] for k in ("G-WT1", "G-WT2", "G-WT3", "G-WT4", "G-WT6"))
    status = "PHASE1_PASS_NEEDS_PHASE2" if passed else "FINAL_FAIL"
    meta = {"status": status, "source_sha256": sha(TAPE), "spec_sha256": sha(spec),
            "gate_sha256": sha(gate), "script_sha256": sha(Path(__file__)),
            "m1_rows": len(tape), "daily_bars": ndays,
            "all_trades": len(trades), "oos_trades": len(oos),
            "unresolved": unresolved, "coverage": coverage.status.value_counts().to_dict(),
            "oos_stress_mean": float(v.mean()) if len(v) else None,
            "oos_stress_total": float(v.sum()) if len(v) else None,
            "oos_boot_p_nonpositive": p, "annual": annual,
            "benchmark_stress_total": float(oos.benchmark_stress.sum()) if len(oos) else None,
            "benchmark_covered": int(oos.benchmark_stress.notna().sum()) if len(oos) else 0,
            "gates": gates, "zero_swap_owner_assumption": True}
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with report.open("x", encoding="utf-8") as f:
        f.write("H91 WEEKLY GOLD MOMENTUM - " + status + "\n")
        f.write("Dukascopy BID M1 proxy; owner-stated zero swap; news holds allowed. No orders.\n")
        f.write("Source/SPEC/GATE SHA256: " + SOURCE_SHA + " / " + sha(spec) + " / " + sha(gate) + "\n")
        f.write("M1/daily rows: " + str((len(tape), ndays)) + "\n")
        f.write("All/OOS trades/unresolved: " + str((len(trades), len(oos), unresolved)) + "\n")
        f.write("Coverage: " + str(meta["coverage"]) + "\n")
        f.write("OOS stressed mean/total/bootstrap P: " + str((meta["oos_stress_mean"],
                                                               meta["oos_stress_total"], p)) + "\n")
        f.write("OOS annual: " + json.dumps(annual) + "\n")
        f.write("Matched-week always-long benchmark: " + str((meta["benchmark_covered"],
                                                               meta["benchmark_stress_total"])) + "\n")
        f.write("Gates: " + json.dumps(gates) + "\n")
        f.write("Historical exposed tape is not independent forward confirmation.\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
