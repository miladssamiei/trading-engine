"""Hypothesis 70: same-day XAUUSD adaptation of frozen Saleh breakout signal."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import xauusd_saleh_donchian_phase1 as saleh


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/xauusd_saleh_intraday_v1"
SPEC = OUT / "SPEC_SALEH_INTRADAY.txt"
GATE = OUT / "GATE_SALEH_INTRADAY.txt"
REPORT = OUT / "REPORT_SALEH_INTRADAY.txt"
SEED = 20261006
N_BOOT = 10_000


def flat_indices(ctx: dict) -> dict[int, int]:
    """Map each NY date to its real [15:00,16:00) hour."""
    mask = ctx["real1"] & (ctx["h1w"] == 15)
    indices = np.flatnonzero(mask)
    result = {int(ctx["d1"][i]): int(i) for i in indices}
    assert len(result) == len(indices)
    return result


def walk_same_day(i: int, direction: int, last: int, ctx: dict) -> dict:
    o1, h1, l1, c1 = (ctx[k] for k in ("o1", "h1", "l1", "c1"))
    atr1, real1 = ctx["atr1"], ctx["real1"]
    spread = saleh.SPREAD
    sl = float(o1[i] - direction * saleh.SL_ATR_MULT * atr1[i - 1])
    initial_sl = sl
    entry_fill = float(o1[i] + spread) if direction == 1 else float(o1[i])
    exit_fill = None
    exit_type = None
    exit_i = None
    eps = 1e-9
    for j in range(i, last + 1):
        if not real1[j]:
            continue
        if direction == 1 and l1[j] <= sl:
            exit_fill = float(min(sl, o1[j]))
            exit_type = "SL_TRAIL" if sl > initial_sl + eps else "SL_INIT"
        elif direction == -1 and h1[j] >= sl - spread:
            exit_fill = float(max(sl, o1[j] + spread))
            exit_type = "SL_TRAIL" if sl < initial_sl - eps else "SL_INIT"
        if exit_fill is not None:
            exit_i = j
            break
        if j == last:
            exit_fill = float(c1[j]) if direction == 1 else float(c1[j] + spread)
            exit_type = "DAY_FLAT"
            exit_i = j
            break
        if direction == 1:
            sl = float(max(sl, c1[j] - saleh.SL_ATR_MULT * atr1[j]))
        else:
            sl = float(min(sl, c1[j] + saleh.SL_ATR_MULT * atr1[j]))
    assert exit_i is not None
    pnl = exit_fill - entry_fill if direction == 1 else entry_fill - exit_fill
    assert int(ctx["d1"][i]) == int(ctx["d1"][exit_i])
    assert int(ctx["h1w"][exit_i]) <= 15
    # Both fills happen before the exit hour closes, so this is a strict
    # upper bound on elapsed hours, even if entry is the 00:00 NY hour.
    hold_upper = int(ctx["grid1"][exit_i] - ctx["grid1"][i] + 1)
    assert hold_upper <= 16
    return {"i": i, "v": exit_i, "dir": direction,
            "entry_fill": entry_fill, "exit_fill": exit_fill,
            "initial_sl": initial_sl, "final_sl": sl,
            "exit_type": exit_type, "pnl": float(pnl),
            "gross": float(pnl + spread), "stress": float(pnl - spread),
            "hold_hours_upper": hold_upper}


def run_machine(ctx: dict, candidates: list[tuple[int, int]]) -> tuple[list[dict], dict]:
    flats = flat_indices(ctx)
    eligible = []
    skipped_hour = skipped_no_flat = 0
    for i, direction in candidates:
        day = int(ctx["d1"][i])
        hour = int(ctx["h1w"][i])
        if hour > 15:
            skipped_hour += 1
        elif day not in flats or i > flats[day]:
            skipped_no_flat += 1
        else:
            eligible.append((i, direction, flats[day]))
    trades = []
    last_exit_i = -1
    shadowed = 0
    for i, direction, last in eligible:
        if i <= last_exit_i:
            shadowed += 1
            continue
        trade = walk_same_day(i, direction, last, ctx)
        last_exit_i = trade["v"]
        trades.append(trade)
    return trades, {"candidates": len(candidates), "skipped_entry_hour": skipped_hour,
                    "skipped_missing_real_15_hour": skipped_no_flat,
                    "eligible": len(eligible), "shadowed_by_occupancy": shadowed,
                    "executed": len(trades)}


def stats(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0}
    pnl = frame["net_units"].to_numpy(np.float64)
    equity = np.r_[0.0, np.cumsum(pnl)]
    return {"n": len(frame), "ny_dates": int(frame["ny_date"].nunique()),
            "gross_mean_units": float(frame["gross_units"].mean()),
            "net_mean_units": float(pnl.mean()), "net_total_units": float(pnl.sum()),
            "stress_mean_units": float(frame["stress_units"].mean()),
            "stress_total_units": float(frame["stress_units"].sum()),
            "win_rate": float(np.mean(pnl > 0)),
            "profit_factor": float(pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum())) if (pnl < 0).any() else float("inf"),
            "max_drawdown_units": float(np.max(np.maximum.accumulate(equity) - equity)),
            "median_hold_hours_upper": float(frame["hold_hours_upper"].median()),
            "p90_hold_hours_upper": float(frame["hold_hours_upper"].quantile(0.9)),
            "side": {str(k): {"n": len(g), "net_total_units": float(g["net_units"].sum())}
                     for k, g in frame.groupby("side")},
            "exit_type": {str(k): int(v) for k, v in frame["exit_type"].value_counts().items()}}


def bootstrap(frame: pd.DataFrame) -> dict:
    grouped = frame.groupby("ny_date", sort=True)["net_units"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(np.float64)
    counts = grouped["count"].to_numpy(np.int32)
    rng = np.random.default_rng(SEED)
    means = np.empty(N_BOOT, np.float64)
    for start in range(0, N_BOOT, 250):
        end = min(start + 250, N_BOOT)
        picks = rng.integers(0, len(sums), size=(end - start, len(sums)))
        means[start:end] = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return {"date_blocks": len(sums), "p_mean_le_zero": float(np.mean(means <= 0)),
            "ci_95_low_units": float(np.quantile(means, 0.025)),
            "ci_95_high_units": float(np.quantile(means, 0.975))}


def main() -> None:
    assert not REPORT.exists(), "Frozen REPORT exists; never overwrite"
    spec = SPEC.read_text(encoding="utf-8")
    gate_lines = GATE.read_text(encoding="utf-8").splitlines()
    assert len(gate_lines) == 5 and all(line in spec for line in gate_lines)
    assert saleh.SPREAD == 0.41
    previous_meta = json.loads((ROOT / "data/results/xauusd_saleh_donchian/run_meta.json").read_text(encoding="utf-8"))
    tape_hash = saleh.sha256_file(saleh.M1_CSV)
    assert tape_hash == previous_meta["tape_sha256"]
    ctx = saleh.load_ctx()
    candidates, signal_funnel = saleh.detect_signals(ctx)
    trades, machine_funnel = run_machine(ctx, candidates)
    rows = []
    for trade in trades:
        i, v = trade["i"], trade["v"]
        day = int(ctx["d1"][i])
        rows.append({"ny_date": str(day), "year": day // 10000,
                     "side": "BUY" if trade["dir"] == 1 else "SELL",
                     "entry_epoch_hour": int(ctx["grid1"][i]),
                     "exit_epoch_hour": int(ctx["grid1"][v]),
                     "entry_hour_ny": int(ctx["h1w"][i]),
                     "exit_hour_ny": int(ctx["h1w"][v]),
                     "entry_bid_open": float(ctx["o1"][i]),
                     "entry_exec": trade["entry_fill"],
                     "exit_exec": trade["exit_fill"],
                     "initial_sl": trade["initial_sl"],
                     "final_sl": trade["final_sl"],
                     "exit_type": trade["exit_type"],
                     "net_units": trade["pnl"],
                     "gross_units": trade["gross"],
                     "stress_units": trade["stress"],
                     "hold_hours_upper": trade["hold_hours_upper"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "intraday_trades.csv", index=False, float_format="%.6f")
    test = frame[frame["year"] >= 2024]
    cal = frame[frame["year"] < 2024]
    yearly = {str(y): stats(g) for y, g in frame.groupby("year", sort=True)}
    boot = bootstrap(test)
    same_day = bool(np.all(frame["entry_hour_ny"] <= frame["exit_hour_ny"]))
    within_16h = bool(np.all(frame["hold_hours_upper"] <= 16))
    gates = {"G-SID1": len(test) >= 150 and test["ny_date"].nunique() >= 150,
             "G-SID2": float(test["stress_units"].mean()) > 0,
             "G-SID3": boot["p_mean_le_zero"] <= 0.05,
             "G-SID4": all(yearly.get(str(y), {}).get("net_total_units", 0) > 0 for y in (2024, 2025)),
             "G-SID5": same_day and within_16h}
    verdict = "PASS" if all(gates.values()) else "FAIL"
    result = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "spec_sha256": saleh.sha256_file(SPEC), "gate_sha256": saleh.sha256_file(GATE),
              "source_m1_sha256": tape_hash,
              "signal_funnel": signal_funnel, "machine_funnel": machine_funnel,
              "calibration": stats(cal), "test": stats(test),
              "bootstrap": boot, "yearly": yearly, "gates": gates,
              "verdict": verdict}
    (OUT / "run_meta.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["REPORT: SAME-DAY XAUUSD DONCHIAN/SUPERTREND, HYPOTHESIS 70",
             f"Generated UTC: {result['generated_utc']}   PHASE-1 VERDICT: {verdict}",
             "Frozen #57 signal/stops; eligible entry NY hours 00..15; flat by 16:00 same date.",
             "M1 BID bar execution proxy: fixed 0.41 spread plus extra 0.41-unit stress.",
             "Full gold tape already exposed; historical PASS would not prove live edge.", "",
             f"SIGNAL FUNNEL: {json.dumps(signal_funnel, sort_keys=True)}",
             f"MACHINE FUNNEL: {json.dumps(machine_funnel, sort_keys=True)}",
             f"CALIBRATION 2020-23: {json.dumps(result['calibration'], sort_keys=True)}",
             f"GATED TEST 2024+: {json.dumps(result['test'], sort_keys=True)}",
             f"DATE BOOTSTRAP: {json.dumps(boot, sort_keys=True)}",
             f"GATES: {json.dumps(gates, sort_keys=True)}", "YEARLY:"]
    lines += [f"  {year}: {json.dumps(value, sort_keys=True)}" for year, value in yearly.items()]
    lines += ["", f"OVERALL PHASE-1 VERDICT: {verdict}",
              "PASS requires separately frozen real-tick Model=4 execution and later forward observation.",
              "FAIL is final; no post-hoc hour, side, stop or cost rescue."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "test": result["test"],
                      "bootstrap": boot, "gates": gates,
                      "machine_funnel": machine_funnel}, indent=2))


if __name__ == "__main__":
    main()
