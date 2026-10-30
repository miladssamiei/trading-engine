"""Hypothesis 62: frozen EURUSD transfer of the ATR-band M1 proxy."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from atr_band_intraday_m1_phase1 import load, sha256, source_atr, walk_exit


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/atr_band_intraday_eurusd_v1"
SPEC = OUT / "SPEC_ATR_BAND_INTRADAY_EURUSD.txt"
GATE = OUT / "GATE_ATR_BAND_INTRADAY_EURUSD.txt"
M5 = ROOT / "data/results/eurusd_onboarding/eurusdtick_m5.csv"
M1 = ROOT / "data/results/eurusd_onboarding/eurusdtick_m1.csv"
SPREAD = 0.00003
PIP = 0.00010
SEED = 20260928
N_BOOT = 10_000


def metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0}
    pnl = frame["net_pips"].to_numpy(np.float64)
    equity = np.r_[0.0, np.cumsum(pnl)]
    return {
        "n": int(len(frame)),
        "gross_mean_pips": float(frame["gross_pips"].mean()),
        "net_mean_pips": float(pnl.mean()),
        "double_cost_mean_pips": float(frame["double_cost_net_pips"].mean()),
        "net_total_pips": float(pnl.sum()),
        "win_rate": float(np.mean(pnl > 0)),
        "max_drawdown_pips": float(np.max(np.maximum.accumulate(equity) - equity)),
        "median_hold_wall_minutes": float(frame["hold_wall_minutes"].median()),
        "same_entry_m1_exit_fraction": float(frame["entry_same_minute_exit"].mean()),
        "exit_counts": {str(k): int(v) for k, v in frame["exit_reason"].value_counts().items()},
    }


def date_bootstrap(frame: pd.DataFrame) -> dict:
    by_date = frame.groupby("ny_date", sort=True)["net_pips"].agg(["sum", "count"])
    sums = by_date["sum"].to_numpy(np.float64)
    counts = by_date["count"].to_numpy(np.int32)
    rng = np.random.default_rng(SEED)
    means = np.empty(N_BOOT, np.float64)
    for start in range(0, N_BOOT, 250):
        end = min(start + 250, N_BOOT)
        draws = rng.integers(0, len(sums), size=(end - start, len(sums)))
        means[start:end] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "date_blocks": int(len(sums)),
        "p_mean_le_zero": float(np.mean(means <= 0)),
        "ci_95_low_pips": float(np.quantile(means, 0.025)),
        "ci_95_high_pips": float(np.quantile(means, 0.975)),
    }


def main() -> None:
    spec_text = SPEC.read_text(encoding="utf-8")
    gate_lines = GATE.read_text(encoding="utf-8").splitlines()
    assert len(gate_lines) == 4 and all(line in spec_text for line in gate_lines)
    m5, m1 = load(M5), load(M1)
    atr = source_atr(m5)
    t5, d5, w5 = m5["time"], m5["date"], m5["wall"]
    t1, d1, w1 = m1["time"], m1["date"], m1["wall"]
    session_days = set(map(int, d5[w5 == 480]))
    end_candidates = np.flatnonzero(w1 == 659)
    end_map = {int(d1[j]): int(j) for j in end_candidates}
    assert len(end_map) == len(end_candidates)
    valid_days = session_days & end_map.keys()
    rows: list[dict] = []
    funnel = {
        "m5_rows": int(len(t5)), "m1_rows": int(len(t1)),
        "session_days": len(session_days), "valid_days": len(valid_days),
        "eligible_bars": 0, "orders_unfilled": 0, "orders_filled": 0,
        "no_printed_m1_in_order_window": 0,
        "m5_sha256": sha256(M5), "m1_sha256": sha256(M1),
    }
    i = 15
    while i < len(t5):
        date, wall = int(d5[i]), int(w5[i])
        if date not in valid_days or not 480 <= wall <= 655:
            i += 1
            continue
        a = float(atr[i - 1])
        if not np.isfinite(a) or a <= 0:
            i += 1
            continue
        funnel["eligible_bars"] += 1
        band = float(m5["close"][i - 1] - 3.1 * a)
        left = int(np.searchsorted(t1, t5[i], side="left"))
        right = int(np.searchsorted(t1, t5[i] + 300, side="left"))
        if left == right:
            funnel["no_printed_m1_in_order_window"] += 1
            funnel["orders_unfilled"] += 1
            i += 1
            continue
        assert np.all(d1[left:right] == date)
        touch = np.flatnonzero(m1["low"][left:right] + SPREAD <= band)
        if len(touch) == 0:
            funnel["orders_unfilled"] += 1
            i += 1
            continue
        fill_j = left + int(touch[0])
        last_j = end_map[date]
        assert fill_j <= last_j and int(d1[fill_j]) == date
        exit_j, exit_bid, reason = walk_exit(
            m1["open"], m1["high"], m1["low"], m1["close"],
            fill_j, last_j, band, a, SPREAD,
        )
        net = (exit_bid - band) / PIP
        rows.append({
            "instrument": "EURUSD", "ny_date": date, "year": date // 10000,
            "order_m5_epoch": int(t5[i]), "entry_m1_epoch": int(t1[fill_j]),
            "exit_m1_epoch": int(t1[exit_j]),
            "hold_printed_m1_bars": int(exit_j - fill_j + 1),
            "hold_wall_minutes": int((t1[exit_j] - t1[fill_j]) // 60),
            "entry_same_minute_exit": int(exit_j == fill_j),
            "band_ask_entry": band,
            "entry_at_m1_open": int(m1["open"][fill_j] + SPREAD <= band),
            "atr_ref": a, "stop_bid": band - 1.5 * a,
            "target_bid": band + 2.0 * a, "exit_bid": exit_bid,
            "exit_reason": reason, "gross_pips": net + 0.3,
            "net_pips": net, "double_cost_net_pips": net - 0.3,
        })
        funnel["orders_filled"] += 1
        i = int(np.searchsorted(t5, t1[exit_j], side="right"))

    trades = pd.DataFrame(rows)
    assert len(trades) == funnel["orders_filled"]
    assert funnel["eligible_bars"] == funnel["orders_filled"] + funnel["orders_unfilled"]
    assert (trades["entry_m1_epoch"] <= trades["exit_m1_epoch"]).all()
    assert (trades["entry_m1_epoch"].diff().dropna() > 0).all()
    assert (trades["exit_m1_epoch"].iloc[:-1].to_numpy() < trades["entry_m1_epoch"].iloc[1:].to_numpy()).all()
    assert np.allclose(trades["gross_pips"] - trades["net_pips"], 0.3)
    assert np.allclose(trades["net_pips"] - trades["double_cost_net_pips"], 0.3)
    trades.to_csv(OUT / "eurusd_trades.csv", index=False, float_format="%.8f")

    cal = trades[trades["year"] < 2024]
    test = trades[trades["year"] >= 2024]
    years = {str(y): metrics(part) for y, part in trades.groupby("year")}
    boot = date_bootstrap(test)
    gates = {
        "G-EAB1": len(test) >= 150,
        "G-EAB2": float(test["double_cost_net_pips"].mean()) > 0,
        "G-EAB3": boot["p_mean_le_zero"] <= 0.05,
        "G-EAB4": years.get("2024", {}).get("net_total_pips", 0) > 0
        and years.get("2025", {}).get("net_total_pips", 0) > 0,
    }
    verdict = "PASS" if all(gates.values()) else "FAIL"
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": sha256(SPEC), "gate_sha256": sha256(GATE),
        "funnel": funnel, "calibration": metrics(cal), "test": metrics(test),
        "bootstrap": boot, "yearly": years, "gates": gates, "verdict": verdict,
    }
    (OUT / "run_meta.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "REPORT: EURUSD INTRADAY ATR LOWER-BAND LIMIT TRANSFER, HYPOTHESIS 62",
        f"Generated UTC: {result['generated_utc']}   PHASE-1 VERDICT: {verdict}",
        "Fixed H22/#61 long rule; EURUSD 08:00-11:00 NY; M1 Ask proxy; 0.3-pip median spread.",
        "Prior exposure means historical replication, not virgin out-of-sample evidence.", "",
        f"FUNNEL: {json.dumps(funnel, sort_keys=True)}",
        f"CALIBRATION (diagnostic): {json.dumps(result['calibration'], sort_keys=True)}",
        f"TEST 2024+ (gated): {json.dumps(result['test'], sort_keys=True)}",
        f"BOOTSTRAP (date blocks): {json.dumps(boot, sort_keys=True)}",
        f"GATES: {json.dumps(gates, sort_keys=True)}", "YEARLY:",
    ]
    lines += [f"  {year}: {json.dumps(value, sort_keys=True)}" for year, value in years.items()]
    lines += [
        "", f"OVERALL PHASE-1 VERDICT: {verdict}",
        "PASS only permits a new predeclared Model=4 real-tick execution study.",
        "FAIL is final; no post-hoc subset or tuning rescues this artifact.",
        "Fixed-spread Bid M1 cannot establish real Ask fills or exact minute path.",
    ]
    (OUT / "REPORT_ATR_BAND_INTRADAY_EURUSD.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "test": result["test"], "bootstrap": boot, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
