"""Frozen H95 US100 prior-Cboe-curve weekday direction; read-only Phase 1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from us100_weekday_hold_h94 import epoch
from xauusd_kc_breakout_oos_phase1 import wilder_atr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/vix_weekday_h95"
TAPE = ROOT / "data/results/virgin_assets/ustech_m1.csv"
VIX = OUT / "VIX_History.csv"
VIX9D = OUT / "VIX9D_History.csv"
HASHES = {
    TAPE: "152f9c995b370e5391cb7da99b64a8002ceee5e52cc68899ddca409ac7e15e39",
    VIX: "4c1af3176a904f3a0dd6dcc55b3762283ed0d972f55c863641cd29c03bf6efd6",
    VIX9D: "6b5ca2eb12aae4c072e21ec6ee8f7806f7e4c12e319e2bf754fdc6ee44ea97cb",
}
SPREAD = 1.45
EXTRA = 1.70
SEED = 20261005


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cboe_signal() -> pd.DataFrame:
    def read(path: Path, name: str) -> pd.DataFrame:
        x = pd.read_csv(path, parse_dates=["DATE"])
        assert x.DATE.is_unique and x.DATE.is_monotonic_increasing
        assert x.CLOSE.gt(0).all()
        return x.set_index("DATE")[["CLOSE"]].rename(columns={"CLOSE": name})

    x = read(VIX, "vix").join(read(VIX9D, "vix9d"), how="inner")
    assert x.index.is_unique and x.vix.gt(0).all() and x.vix9d.gt(0).all()
    return x


def bootstrap(v: np.ndarray) -> float | None:
    n = len(v)
    if n < 5:
        return None
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, n, size=(10000, int(np.ceil(n / 5))))
    ix = (starts[:, :, None] + np.arange(5)[None, None, :]) % n
    return float((v[ix.reshape(10000, -1)[:, :n]].mean(axis=1) <= 0).mean())


def replay(tape: pd.DataFrame, signal: pd.DataFrame):
    times = tape.time.to_numpy(np.int64)
    o, h, l, c = [tape[z].to_numpy(float) for z in ("open", "high", "low", "close")]
    wall = pd.to_datetime(times, unit="s", utc=True).tz_convert("America/New_York").tz_localize(None)
    wall_min = wall.hour * 60 + wall.minute
    cash_mask = (wall.weekday < 5) & (wall_min >= 570) & (wall_min < 960)
    cash = (pd.DataFrame({"date": wall[cash_mask].normalize(), "o": o[cash_mask],
                          "h": h[cash_mask], "l": l[cash_mask], "c": c[cash_mask],
                          "minute": wall_min[cash_mask]})
            .groupby("date", sort=True)
            .agg(o=("o", "first"), h=("h", "max"), l=("l", "min"),
                 c=("c", "last"), first_minute=("minute", "first")))
    cash = cash.loc[cash.first_minute == 570]
    dates = pd.DatetimeIndex(cash.index)
    atr = wilder_atr(cash.h.to_numpy(float), cash.l.to_numpy(float),
                     cash.c.to_numpy(float), 20)
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
        si = int(signal.index.searchsorted(monday)) - 1
        if si < 0 or (monday - signal.index[si]).days > 4:
            coverage.append({"monday": str(monday.date()), "status": "NO_PRIOR_CBOE"})
            continue
        sdate = signal.index[si]
        vix, vix9d = [float(signal.iloc[si][z]) for z in ("vix", "vix9d")]
        side = "SHORT" if vix9d >= vix else "LONG"
        entry = float(o[k0] if side == "SHORT" else o[k0] + SPREAD)
        distance = float(2 * atr[j])
        assert distance > 0 and sdate < monday
        stop = entry + distance if side == "SHORT" else entry - distance
        out, reason, kx = None, None, None
        for k in range(k0, len(times)):
            if times[k] >= x0:
                if times[k] < x1:
                    out = float(o[k] + (SPREAD if side == "SHORT" else 0))
                    reason, kx = "THURSDAY_FLAT", k
                else:
                    reason = "UNRESOLVED_THURSDAY"
                break
            if side == "LONG":
                if o[k] <= stop:
                    out, reason, kx = float(o[k]), "STOP_GAP", k
                    break
                if l[k] <= stop:
                    out, reason, kx = float(stop), "STOP", k
                    break
            else:
                if o[k] + SPREAD >= stop:
                    out, reason, kx = float(o[k] + SPREAD), "STOP_GAP", k
                    break
                if h[k] + SPREAD >= stop:
                    out, reason, kx = float(stop), "STOP", k
                    break
        if out is None:
            coverage.append({"monday": str(monday.date()),
                             "status": reason or "UNRESOLVED_END"})
            continue
        base = out - entry if side == "LONG" else entry - out
        trades.append({"monday": str(monday.date()), "year": monday.year,
                       "signal_date": str(sdate.date()), "vix": vix, "vix9d": vix9d,
                       "side": side, "entry_epoch": int(times[k0]),
                       "exit_epoch": int(times[kx]), "entry_ny": str(wall[k0]),
                       "exit_ny": str(wall[kx]),
                       "prior_cash_date": str(dates[j].date()),
                       "prior_atr20": float(atr[j]), "entry": entry,
                       "initial_stop": stop, "exit": out, "exit_reason": reason,
                       "base_net": base, "stress_net": base - EXTRA,
                       "hold_hours": (times[kx] - times[k0]) / 3600})
        coverage.append({"monday": str(monday.date()), "status": "EXECUTED"})
    return pd.DataFrame(trades), pd.DataFrame(coverage), len(cash)


def main():
    spec = OUT / "SPEC_US100_VIX_WEEKDAY_H95.txt"
    gate = OUT / "GATE_US100_VIX_WEEKDAY_H95.txt"
    report = OUT / "REPORT_US100_VIX_WEEKDAY_H95.txt"
    assert not report.exists(), "Immutable H95 REPORT exists"
    assert [z for z in spec.read_text(encoding="utf-8").splitlines()
            if z.startswith("G-VW")] == gate.read_text(encoding="utf-8").splitlines()
    for path, expected in HASHES.items():
        assert sha(path) == expected, path
    signal = cboe_signal()
    tape = pd.read_csv(TAPE, usecols=["time", "open", "high", "low", "close"])
    assert tape.time.is_monotonic_increasing and tape.time.is_unique
    assert (tape.high >= tape[["open", "low", "close"]].max(axis=1)).all()
    assert (tape.low <= tape[["open", "high", "close"]].min(axis=1)).all()
    trades, coverage, cash_days = replay(tape, signal)
    trades.to_csv(OUT / "trades.csv", index=False)
    coverage.to_csv(OUT / "coverage.csv", index=False)
    oos = trades.loc[trades.year >= 2024]
    v = oos.stress_net.to_numpy(float)
    annual = {str(y): {"n": int((oos.year == y).sum()),
                       "stress_sum": float(oos.loc[oos.year == y, "stress_net"].sum())}
              for y in (2024, 2025, 2026)}
    unresolved = int(coverage.status.str.startswith("UNRESOLVED").sum())
    missing_signal = int(coverage.loc[pd.to_datetime(coverage.monday).dt.year >= 2024,
                                      "status"].eq("NO_PRIOR_CBOE").sum())
    p = bootstrap(v)
    sided = bool((trades.side.eq("SHORT") == (trades.vix9d >= trades.vix)).all())
    causal = bool((pd.to_datetime(trades.signal_date) < pd.to_datetime(trades.monday)).all()
                  and ((pd.to_datetime(trades.monday) -
                        pd.to_datetime(trades.signal_date)).dt.days <= 4).all())
    arithmetic = bool(np.allclose(trades.base_net,
                                  np.where(trades.side.eq("LONG"),
                                           trades.exit-trades.entry,
                                           trades.entry-trades.exit))
                      and np.allclose(trades.stress_net, trades.base_net-EXTRA))
    weekend_flat = bool((pd.to_datetime(trades.exit_ny).dt.dayofweek < 4).all())
    gates = {"G-VW1": len(oos) >= 120 and oos.monday.nunique() >= 120
             and unresolved == 0 and missing_signal == 0,
             "G-VW2": len(v) > 0 and float(v.mean()) > 0,
             "G-VW3": p is not None and p <= 0.05,
             "G-VW4": all(annual[str(y)]["stress_sum"] > 0 for y in (2024, 2025, 2026)),
             "G-VW5": "NOT_RUN", "G-VW6": sided and causal and arithmetic
             and weekend_flat and unresolved == 0}
    passed = all(gates[z] for z in ("G-VW1", "G-VW2", "G-VW3", "G-VW4", "G-VW6"))
    status = "PHASE1_PASS_NEEDS_PHASE2" if passed else "FINAL_FAIL"
    meta = {"status": status, "source_sha256": {str(k.relative_to(ROOT)): v
                                               for k, v in HASHES.items()},
            "spec_sha256": sha(spec), "gate_sha256": sha(gate),
            "script_sha256": sha(Path(__file__)), "m1_rows": len(tape),
            "cash_days": cash_days, "all_trades": len(trades),
            "evaluation_trades": len(oos), "unresolved": unresolved,
            "missing_evaluation_signal": missing_signal,
            "coverage": coverage.status.value_counts().to_dict(),
            "evaluation_sides": oos.side.value_counts().to_dict(),
            "evaluation_stressed_mean": float(v.mean()) if len(v) else None,
            "evaluation_stressed_total": float(v.sum()),
            "evaluation_boot_p_nonpositive": p, "annual": annual,
            "gates": gates, "zero_swap_owner_assumption": True,
            "prior_h94_outcome_exposure": True}
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with report.open("x", encoding="utf-8") as f:
        f.write("H95 US100 VIX CURVE WEEKDAY - " + status + "\n")
        f.write("Cboe daily close signal; Dukascopy BID M1 fixed-spread proxy; no orders.\n")
        f.write("Sources/SPEC/GATE/script SHA256: " + json.dumps(meta["source_sha256"])
                + " / " + sha(spec) + " / " + sha(gate) + " / "
                + sha(Path(__file__)) + "\n")
        f.write("All/evaluation trades: " + str((len(trades), len(oos))) + "\n")
        f.write("Unresolved/missing signal: " + str((unresolved, missing_signal)) + "\n")
        f.write("Coverage/sides: " + json.dumps(meta["coverage"]) + " / "
                + json.dumps(meta["evaluation_sides"]) + "\n")
        f.write("Evaluation stressed mean/total/bootstrap P: "
                + str((meta["evaluation_stressed_mean"],
                       meta["evaluation_stressed_total"], p)) + "\n")
        f.write("Annual: " + json.dumps(annual) + "\n")
        f.write("Gates: " + json.dumps(gates) + "\n")
        f.write("H94 asset outcomes were exposed before H95; no genuinely unseen forward confirmation.\n")
        f.write("Native FTMO variable quotes, EUR conversion, floating equity and two-stage prop pace untested.\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
