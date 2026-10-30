"""Hypothesis 69: published post-Tokyo-fix EURUSD reversal on actual quotes."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from eurusd_london_fix_drift_phase1 import first_valid_quote, metrics
from xauusd_macro_reaction_follow_phase1 import sha256


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/eurusd_post_tokyo_fix_v1"
SPEC = OUT / "SPEC_EURUSD_POST_TOKYO_FIX.txt"
GATE = OUT / "GATE_EURUSD_POST_TOKYO_FIX.txt"
REPORT = OUT / "REPORT_EURUSD_POST_TOKYO_FIX.txt"
ONBOARD = ROOT / "data/results/eurusd_onboarding/onboarding_stats.json"
SOURCE_HASH = "23376d5ffd9d5f287f92875faa4df303032162158fff20387c4c284894a8e3b1"
TOKYO = ZoneInfo("Asia/Tokyo")
NEW_YORK = ZoneInfo("America/New_York")
START = date(2020, 1, 2)
END = date(2026, 9, 18)
SEED = 20261005
N_BOOT = 10_000
PIP = 0.0001


def roster() -> pd.DataFrame:
    rows = []
    day = START
    while day <= END:
        if day.weekday() < 5:
            entry = datetime(day.year, day.month, day.day, 9, 55, 5, tzinfo=TOKYO)
            exit_at = datetime(day.year, day.month, day.day, 2, 0, tzinfo=NEW_YORK)
            entry_ms = int(entry.timestamp() * 1000)
            exit_ms = int(exit_at.timestamp() * 1000)
            assert 4 * 3600_000 <= exit_ms - entry_ms <= 7 * 3600_000
            rows.append({"tokyo_date": day.isoformat(), "year": day.year,
                         "entry_target_ms": entry_ms, "exit_target_ms": exit_ms,
                         "entry_target_utc": entry.astimezone(timezone.utc).isoformat(),
                         "exit_target_utc": exit_at.astimezone(timezone.utc).isoformat()})
        day += timedelta(days=1)
    frame = pd.DataFrame(rows)
    assert frame["tokyo_date"].is_unique
    return frame


def evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    onboard = json.loads(ONBOARD.read_text(encoding="utf-8"))
    source = onboard["source"]
    assert source["sha256"] == SOURCE_HASH
    path = Path(source["file"])
    assert path.is_file() and path.stat().st_size == source["size_bytes"]
    assert sha256(path) == SOURCE_HASH
    trades, skipped = [], []
    invalid = 0
    with path.open("rb") as stream:
        assert stream.readline().strip() == b"timestamp,askPrice,bidPrice"
        header_end = stream.tell()
        for row in frame.itertuples(index=False):
            entry, n1 = first_valid_quote(stream, row.entry_target_ms, header_end)
            exit_at, n2 = first_valid_quote(stream, row.exit_target_ms, header_end)
            invalid += n1 + n2
            if entry is None or exit_at is None:
                skipped.append({"tokyo_date": row.tokyo_date,
                                "reason": "ENTRY_AND_EXIT_MISSING" if entry is None and exit_at is None else
                                          "ENTRY_MISSING" if entry is None else "EXIT_MISSING"})
                continue
            entry_mid = (entry[1] + entry[2]) / 2
            exit_mid = (exit_at[1] + exit_at[2]) / 2
            gross = (exit_mid - entry_mid) / PIP
            net = (exit_at[2] - entry[1]) / PIP
            trades.append({"tokyo_date": row.tokyo_date, "year": row.year,
                           "entry_target_ms": row.entry_target_ms, "exit_target_ms": row.exit_target_ms,
                           "entry_ms": entry[0], "exit_ms": exit_at[0],
                           "entry_ask": entry[1], "entry_bid": entry[2],
                           "exit_ask": exit_at[1], "exit_bid": exit_at[2],
                           "entry_spread_pips": (entry[1] - entry[2]) / PIP,
                           "exit_spread_pips": (exit_at[1] - exit_at[2]) / PIP,
                           "hold_hours": (exit_at[0] - entry[0]) / 3_600_000,
                           "gross_mid_pips": gross, "quote_net_pips": net,
                           "spread_drag_pips": gross - net, "stress_net_pips": net - 0.3})
    trade_frame = pd.DataFrame(trades)
    skip_frame = pd.DataFrame(skipped)
    assert len(trade_frame) + len(skip_frame) == len(frame)
    assert trade_frame["tokyo_date"].is_unique
    assert np.allclose(trade_frame["spread_drag_pips"],
                       (trade_frame["entry_spread_pips"] + trade_frame["exit_spread_pips"]) / 2)
    return trade_frame, skip_frame, {"roster_dates": len(frame), "executed": len(trade_frame),
                                     "skip_reasons": skip_frame["reason"].value_counts().to_dict(),
                                     "invalid_sampled_quotes": invalid}


def bootstrap(frame: pd.DataFrame) -> dict:
    pnl = frame.sort_values("tokyo_date")["quote_net_pips"].to_numpy(np.float64)
    rng = np.random.default_rng(SEED)
    means = np.empty(N_BOOT, np.float64)
    for start in range(0, N_BOOT, 250):
        end = min(start + 250, N_BOOT)
        picks = rng.integers(0, len(pnl), size=(end - start, len(pnl)))
        means[start:end] = pnl[picks].mean(axis=1)
    return {"date_blocks": len(pnl), "p_mean_le_zero": float(np.mean(means <= 0)),
            "ci_95_low_pips": float(np.quantile(means, 0.025)),
            "ci_95_high_pips": float(np.quantile(means, 0.975))}


def main() -> None:
    assert not REPORT.exists(), "Frozen REPORT exists; never overwrite"
    spec = SPEC.read_text(encoding="utf-8")
    gate_lines = GATE.read_text(encoding="utf-8").splitlines()
    assert len(gate_lines) == 4 and all(line in spec for line in gate_lines)
    days = roster()
    days.to_csv(OUT / "date_roster.csv", index=False)
    trades, skipped, funnel = evaluate(days)
    trades.to_csv(OUT / "quote_trades.csv", index=False, float_format="%.9f")
    skipped.to_csv(OUT / "skipped_dates.csv", index=False)
    calibration = trades[trades["year"] < 2024]
    test = trades[trades["year"] >= 2024]
    yearly = {str(y): metrics(group) for y, group in trades.groupby("year", sort=True)}
    boot = bootstrap(test)
    gates = {
        "G-PTF1": len(test) >= 150 and test["tokyo_date"].is_unique,
        "G-PTF2": float(test["stress_net_pips"].mean()) > 0,
        "G-PTF3": boot["p_mean_le_zero"] <= 0.0166667,
        "G-PTF4": all(yearly.get(str(y), {}).get("quote_net_total_pips", 0) > 0 for y in (2024, 2025)),
    }
    verdict = "PASS" if all(gates.values()) else "FAIL"
    result = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "spec_sha256": sha256(SPEC), "gate_sha256": sha256(GATE),
              "source_sha256": SOURCE_HASH, "funnel": funnel,
              "calibration": metrics(calibration), "test": metrics(test),
              "bootstrap": boot, "yearly": yearly, "gates": gates,
              "verdict": verdict}
    (OUT / "run_meta.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["REPORT: EURUSD POST-TOKYO-FIX DOLLAR REVERSAL, HYPOTHESIS 69",
             f"Generated UTC: {result['generated_utc']}   PHASE-1 VERDICT: {verdict}",
             "Fixed BUY EURUSD 09:55:05 Tokyo to 02:00 New York, each Tokyo weekday.",
             "Native Bid/Ask quote net plus extra 0.3-pip execution stress.",
             "1999-2019 published mechanism; 2024+ historical replication, not virgin OOS.",
             "First-quote economics omit live latency, rejection and market impact.", "",
             f"FUNNEL: {json.dumps(funnel, sort_keys=True)}",
             f"CALIBRATION 2020-23: {json.dumps(result['calibration'], sort_keys=True)}",
             f"GATED TEST 2024+: {json.dumps(result['test'], sort_keys=True)}",
             f"BOOTSTRAP: {json.dumps(boot, sort_keys=True)}",
             f"GATES: {json.dumps(gates, sort_keys=True)}", "YEARLY:"]
    lines += [f"  {year}: {json.dumps(value, sort_keys=True)}" for year, value in yearly.items()]
    lines += ["", f"OVERALL PHASE-1 VERDICT: {verdict}",
              "PASS requires separately frozen Model=4 execution and later forward observation.",
              "FAIL is final for this artifact; no post-hoc holiday or timing rescue."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "test": result["test"],
                      "bootstrap": boot, "gates": gates, "funnel": funnel}, indent=2))


if __name__ == "__main__":
    main()
