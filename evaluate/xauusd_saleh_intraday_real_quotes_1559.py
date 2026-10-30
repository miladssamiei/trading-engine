"""Hypothesis 70: separately frozen 15:59 NY real-quote exit remeasurement."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import xauusd_saleh_donchian_phase1 as saleh
from xauusd_saleh_intraday_phase1 import flat_indices
from xauusd_saleh_intraday_real_quotes import (
    bootstrap, metrics, replay_one, source_files, source_for,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/xauusd_saleh_intraday_real_quotes_close1559_v1"
SPEC = OUT / "SPEC_SALEH_INTRADAY_REAL_QUOTES_1559.txt"
GATE = OUT / "GATE_SALEH_INTRADAY_REAL_QUOTES_1559.txt"
REPORT = OUT / "REPORT_SALEH_INTRADAY_REAL_QUOTES_1559.txt"
BAR_META = ROOT / "data/results/xauusd_saleh_intraday_v1/run_meta.json"
PRIOR_REPORT = ROOT / "data/results/xauusd_saleh_intraday_real_quotes_v1/REPORT_SALEH_INTRADAY_REAL_QUOTES.txt"
NY = ZoneInfo("America/New_York")


def end_ms_for_nydate_1559(day: int) -> int:
    year, month, dom = day // 10000, (day // 100) % 100, day % 100
    wall = datetime(year, month, dom, 15, 59, tzinfo=NY)
    return int(wall.timestamp() * 1000)


def run_machine(ctx: dict, candidates: list[tuple[int, int]], sources: list[tuple[int, int, Path]]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    flats = flat_indices(ctx)
    trades, skipped = [], []
    funnel = {"signal_candidates": len(candidates), "skipped_entry_hour": 0,
              "skipped_missing_real_15_hour": 0, "shadowed_by_occupancy": 0,
              "missing_source_file": 0, "entry_quote_missing": 0,
              "unresolved_exits": 0, "invalid_sampled_quotes": 0,
              "quotes_in_fill_hours": 0}
    last_exit_hour = -1
    end_by_day: dict[int, int] = {}
    for i, direction in candidates:
        day = int(ctx["d1"][i])
        if int(ctx["h1w"][i]) > 15:
            funnel["skipped_entry_hour"] += 1
            continue
        if day not in flats or i > flats[day]:
            funnel["skipped_missing_real_15_hour"] += 1
            continue
        if i <= last_exit_hour:
            funnel["shadowed_by_occupancy"] += 1
            continue
        target_ms = int(ctx["grid1"][i]) * 3_600_000
        source = source_for(target_ms, sources)
        if source is None:
            funnel["missing_source_file"] += 1
            skipped.append({"ny_date": str(day), "signal_epoch_hour": int(ctx["grid1"][i]),
                            "reason": "SOURCE_FILE_MISSING"})
            continue
        end_ms = end_by_day.setdefault(day, end_ms_for_nydate_1559(day))
        trade, diag = replay_one(i, direction, ctx, source, end_ms)
        funnel["invalid_sampled_quotes"] += diag["invalid_quotes"]
        funnel["quotes_in_fill_hours"] += diag["fill_hour_quotes"]
        if trade is None:
            funnel["entry_quote_missing"] += 1
            skipped.append({"ny_date": str(day), "signal_epoch_hour": int(ctx["grid1"][i]),
                            "reason": diag["reason"]})
            continue
        if trade["unresolved"]:
            funnel["unresolved_exits"] += 1
        trades.append(trade)
        last_exit_hour = int(trade["exit_epoch_hour"] - int(ctx["grid1"][0]))
    funnel["executed_or_marked"] = len(trades)
    return pd.DataFrame(trades), pd.DataFrame(skipped), funnel


def main() -> None:
    assert not REPORT.exists(), "Frozen REPORT exists; never overwrite"
    assert PRIOR_REPORT.is_file()
    spec = SPEC.read_text(encoding="utf-8")
    gate_lines = GATE.read_text(encoding="utf-8").splitlines()
    assert len(gate_lines) == 5 and all(line in spec for line in gate_lines)
    bar_meta = json.loads(BAR_META.read_text(encoding="utf-8"))
    assert bar_meta["verdict"] == "PASS"
    assert saleh.sha256_file(saleh.M1_CSV) == bar_meta["source_m1_sha256"]
    sources, hashed = source_files()
    ctx = saleh.load_ctx()
    candidates, signal_funnel = saleh.detect_signals(ctx)
    trades, skipped, funnel = run_machine(ctx, candidates, sources)
    trades.to_csv(OUT / "native_quote_trades.csv", index=False, float_format="%.6f")
    skipped.to_csv(OUT / "skipped_signals.csv", index=False)
    cal, test = trades[trades["year"] < 2024], trades[trades["year"] >= 2024]
    yearly = {str(y): metrics(group) for y, group in trades.groupby("year", sort=True)}
    boot = bootstrap(test)
    entry_dates = pd.to_datetime(trades["entry_ms"], unit="ms", utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y%m%d")
    exit_dates = pd.to_datetime(trades["exit_ms"], unit="ms", utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y%m%d")
    same_ny_date = bool(np.all(entry_dates.to_numpy() == exit_dates.to_numpy()))
    gates = {"G-SRC1": len(test) >= 150 and test["ny_date"].nunique() >= 150,
             "G-SRC2": float(test["stress_units"].mean()) > 0,
             "G-SRC3": boot["p_mean_le_zero"] <= 0.05,
             "G-SRC4": all(yearly.get(str(y), {}).get("quote_net_total_units", 0) > 0 for y in (2024, 2025)),
             "G-SRC5": funnel["unresolved_exits"] == 0 and same_ny_date}
    verdict = "PASS" if all(gates.values()) else "FAIL"
    result = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "spec_sha256": saleh.sha256_file(SPEC), "gate_sha256": saleh.sha256_file(GATE),
              "source_hashes": hashed, "m1_hash": bar_meta["source_m1_sha256"],
              "bar_model_test": bar_meta["test"], "signal_funnel": signal_funnel,
              "execution_funnel": funnel, "calibration": metrics(cal),
              "test": metrics(test), "bootstrap": boot, "yearly": yearly,
              "gates": gates, "verdict": verdict,
              "prior_1600_order_report_sha256": saleh.sha256_file(PRIOR_REPORT)}
    (OUT / "run_meta.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["REPORT: XAUUSD SAME-DAY SALEH NATIVE BID/ASK 15:59 EXIT REMEASUREMENT",
             f"Generated UTC: {result['generated_utc']}   CORRECTED QUOTE VERDICT: {verdict}",
             "#70 signal and stops unchanged; real quote flat order at 15:59 NY inside final H1 bar.",
             "Prior exact-16:00 order artifact remains a frozen FAIL from the broker session break.",
             "Native variable Bid/Ask charged plus separate extra 0.41-unit stress.",
             "First-quote fills remain optimistic about live order latency and rejection.", "",
             f"SOURCE HASHES: {json.dumps(hashed, sort_keys=True)}",
             f"SIGNAL FUNNEL: {json.dumps(signal_funnel, sort_keys=True)}",
             f"EXECUTION FUNNEL: {json.dumps(funnel, sort_keys=True)}",
             f"BAR MODEL 2024+: {json.dumps(bar_meta['test'], sort_keys=True)}",
             f"QUOTE CALIBRATION 2020-23: {json.dumps(result['calibration'], sort_keys=True)}",
             f"QUOTE GATED TEST 2024+: {json.dumps(result['test'], sort_keys=True)}",
             f"DATE BOOTSTRAP: {json.dumps(boot, sort_keys=True)}",
             f"GATES: {json.dumps(gates, sort_keys=True)}", "YEARLY:"]
    lines += [f"  {year}: {json.dumps(value, sort_keys=True)}" for year, value in yearly.items()]
    lines += ["", f"OVERALL CORRECTED NATIVE QUOTE VERDICT: {verdict}",
              "PASS permits a separately frozen native gold Model=4 MT5 tester, not a live edge claim.",
              "FAIL is final for this new artifact; no further exit-time tuning."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "test": result["test"],
                      "bootstrap": boot, "gates": gates,
                      "execution_funnel": funnel}, indent=2))


if __name__ == "__main__":
    main()
