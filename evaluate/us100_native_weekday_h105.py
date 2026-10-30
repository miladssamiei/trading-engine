"""H105 frozen native FTMO US100 paid-swap weekday Phase 1; no orders."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ftmo_swing_swap_transfer_2026_09_26 import rollover_weight
from us100_weekday_hold_h94 import bootstrap, epoch
from xauusd_kc_breakout_oos_phase1 import wilder_atr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/us100_native_weekday_h105"
SOURCE = ROOT / "data/results/us100_native_full_source_h94/us100cash_ftmo_native_m1_2023_2026.csv"
SOURCE_SHA = "f9552440b4af1e6d20fd9934d0c54b89a21131ccf9948d2b2b1f875dad90b9a5"
SWAP = -6.7512
EXTRA = 2.50


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def replay(tape: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    times = tape.minute_epoch_s.to_numpy(np.int64)
    bo = tape.bid_open.to_numpy(float)
    bh = tape.bid_high.to_numpy(float)
    bl = tape.bid_low.to_numpy(float)
    bc = tape.bid_close.to_numpy(float)
    ao = tape.ask_open.to_numpy(float)
    wall = pd.to_datetime(times, unit="s", utc=True).tz_convert("America/New_York").tz_localize(None)
    minute = wall.hour * 60 + wall.minute
    cash_mask = (wall.weekday < 5) & (minute >= 570) & (minute < 960)
    cash = (pd.DataFrame({"date": wall[cash_mask].normalize(),
                          "h": bh[cash_mask], "l": bl[cash_mask],
                          "c": bc[cash_mask], "minute": minute[cash_mask]})
            .groupby("date", sort=True)
            .agg(h=("h", "max"), l=("l", "min"), c=("c", "last"),
                 first_minute=("minute", "first")))
    cash = cash.loc[cash.first_minute == 570]
    dates = pd.DatetimeIndex(cash.index)
    atr = wilder_atr(cash.h.to_numpy(float), cash.l.to_numpy(float),
                     cash.c.to_numpy(float), 20)
    rows, coverage = [], []
    for monday in pd.date_range("2023-01-02", "2026-09-21", freq="W-MON"):
        thursday = monday + pd.Timedelta(days=3)
        entry_lo, entry_hi = epoch(monday, 9, 30), epoch(monday, 9, 35)
        exit_lo, exit_hi = epoch(thursday, 11), epoch(thursday, 11, 15)
        status = {"monday": str(monday.date()), "year": int(monday.year)}
        if exit_hi > times[-1]:
            status["status"] = "INCOMPLETE_TAPE_WEEK"
            coverage.append(status)
            continue
        if (thursday.month, thursday.day) in ((1, 1), (12, 25)):
            status["status"] = "PREDECLARED_THURSDAY_CLOSURE_SKIP"
            coverage.append(status)
            continue
        if monday not in dates:
            status["status"] = "NO_MONDAY_CASH_OPEN"
            coverage.append(status)
            continue
        k0 = int(np.searchsorted(times, entry_lo))
        if k0 >= len(times) or times[k0] >= entry_hi:
            status["status"] = "NO_REAL_ENTRY"
            coverage.append(status)
            continue
        j = int(dates.searchsorted(monday)) - 1
        if j < 19 or (monday - dates[j]).days > 4 or not np.isfinite(atr[j]):
            status["status"] = "NO_PRIOR_ATR"
            coverage.append(status)
            continue
        entry_ask = float(ao[k0])
        entry_bid = float(bo[k0])
        stop = entry_ask - 2 * float(atr[j])
        if stop <= 0:
            raise ValueError("Nonpositive H105 stop")
        kx = None
        exit_bid = None
        reason = None
        for k in range(k0, len(times)):
            if times[k] >= exit_lo:
                if times[k] < exit_hi:
                    kx = k
                    exit_bid = float(bo[k])
                    reason = "THURSDAY_FLAT"
                else:
                    reason = "UNRESOLVED_THURSDAY"
                break
            if bo[k] <= stop:
                kx = k
                exit_bid = float(bo[k])
                reason = "STOP_GAP"
                break
            if bl[k] <= stop:
                kx = k
                exit_bid = float(stop)
                reason = "STOP"
                break
        if kx is None or exit_bid is None:
            status["status"] = reason or "UNRESOLVED_END"
            coverage.append(status)
            continue
        rollovers, weight = rollover_weight(int(times[k0]), int(times[kx]))
        swap = SWAP * weight
        net = exit_bid - entry_ask + swap
        rows.append({"monday": str(monday.date()), "year": int(monday.year),
                     "entry_epoch": int(times[k0]), "exit_epoch": int(times[kx]),
                     "entry_ny": str(wall[k0]), "exit_ny": str(wall[kx]),
                     "prior_cash_date": str(dates[j].date()), "prior_atr20": float(atr[j]),
                     "entry_ask": entry_ask, "entry_bid": entry_bid,
                     "entry_spread": entry_ask - entry_bid, "initial_stop": stop,
                     "exit_bid": exit_bid, "exit_reason": reason,
                     "rollovers": rollovers, "weighted_swap_days": weight,
                     "snapshot_swap_units": swap, "net_units": net,
                     "stress_net_units": net - EXTRA,
                     "hold_hours": (int(times[kx]) - int(times[k0])) / 3600})
        status["status"] = "EXECUTED"
        coverage.append(status)
    return pd.DataFrame(rows), pd.DataFrame(coverage), len(cash)


def main() -> None:
    ledger = OUT / "trades.csv"
    coverage_path = OUT / "coverage.csv"
    report = OUT / "REPORT_US100_NATIVE_WEEKDAY_H105.txt"
    meta_path = OUT / "run_meta.json"
    if any(path.exists() for path in (ledger, coverage_path, report, meta_path)):
        raise FileExistsError("H105 historical artifact already exists")
    spec = OUT / "SPEC_US100_NATIVE_WEEKDAY_H105.txt"
    gate = OUT / "GATE_US100_NATIVE_WEEKDAY_H105.txt"
    lines = [line for line in spec.read_text(encoding="utf-8").splitlines()
             if line.startswith("G-H105-")]
    if lines != gate.read_text(encoding="utf-8").splitlines():
        raise ValueError("H105 gate extraction differs from SPEC")
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError("Native source hash mismatch")
    tape = pd.read_csv(SOURCE)
    if not tape.minute_epoch_s.is_monotonic_increasing or tape.minute_epoch_s.duplicated().any():
        raise ValueError("Native M1 timestamp integrity failed")
    if not ((tape.bid_high >= tape[["bid_open", "bid_low", "bid_close"]].max(axis=1)).all()
            and (tape.bid_low <= tape[["bid_open", "bid_high", "bid_close"]].min(axis=1)).all()
            and (tape.ask_open >= tape.bid_open).all()):
        raise ValueError("Native M1 Bid/Ask integrity failed")
    trades, coverage, cash_days = replay(tape)
    trades.to_csv(ledger, index=False)
    coverage.to_csv(coverage_path, index=False)
    oos = trades[trades.year >= 2024]
    v = oos.stress_net_units.to_numpy(float)
    p = bootstrap(v)
    annual = {str(y): {"n": int((oos.year == y).sum()),
                       "stress_sum": float(oos.loc[oos.year == y, "stress_net_units"].sum())}
              for y in (2024, 2025, 2026)}
    unresolved = int(coverage[(coverage.year >= 2024) &
                              coverage.status.str.startswith("UNRESOLVED")].shape[0])
    gates = {"G-H105-1": len(oos) >= 130 and oos.monday.nunique() == len(oos) and unresolved == 0,
             "G-H105-2": len(v) > 0 and float(v.mean()) > 0,
             "G-H105-3": p is not None and p <= 0.05,
             "G-H105-4": all(annual[str(y)]["stress_sum"] > 0 for y in (2024, 2025, 2026)),
             "G-H105-5": True, "G-H105-6": "NOT_RUN"}
    passed = all(gates[k] for k in ("G-H105-1", "G-H105-2", "G-H105-3", "G-H105-4", "G-H105-5"))
    verdict = "PHASE1_PASS_NEEDS_PHASE2" if passed else "FINAL_FAIL"
    result = {"verdict": verdict, "source_sha256": SOURCE_SHA, "spec_sha256": sha(spec),
              "gate_sha256": sha(gate), "source_m1_rows": len(tape), "cash_days": cash_days,
              "all_trades": len(trades), "gated_trades": len(oos), "unresolved_oos": unresolved,
              "coverage": coverage.status.value_counts().to_dict(),
              "gated_stress_mean": float(v.mean()) if len(v) else None,
              "gated_stress_total": float(v.sum()) if len(v) else None,
              "bootstrap_p_nonpositive": p,
              "entry_spread_median_oos": float(oos.entry_spread.median()) if len(oos) else None,
              "snapshot_swap_total_oos": float(oos.snapshot_swap_units.sum()),
              "annual": annual, "gates": gates,
              "ledger_sha256": sha(ledger), "coverage_sha256": sha(coverage_path),
              "phase2": "NOT_RUN"}
    meta_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report.write_text("H105 US100 NATIVE FTMO WEEKDAY HOLD - " + verdict + "\n" +
                      json.dumps(result, indent=2, allow_nan=False) + "\n" +
                      "Current Swing swap snapshot is a scenario, not historical postings.\n" +
                      "Historical US100 outcomes already exposed; no forward claim or account order.\n",
                      encoding="utf-8")
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
