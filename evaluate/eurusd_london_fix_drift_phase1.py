"""Hypothesis 67: published post-London-fix EURUSD drift, real quote test."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from xauusd_macro_reaction_follow_phase1 import first_quote_at_or_after, sha256


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/eurusd_london_fix_drift_v1"
SPEC = OUT / "SPEC_EURUSD_LONDON_FIX_DRIFT.txt"
GATE = OUT / "GATE_EURUSD_LONDON_FIX_DRIFT.txt"
REPORT = OUT / "REPORT_EURUSD_LONDON_FIX_DRIFT.txt"
ONBOARD = ROOT / "data/results/eurusd_onboarding/onboarding_stats.json"
SOURCE_HASH = "23376d5ffd9d5f287f92875faa4df303032162158fff20387c4c284894a8e3b1"
LONDON = ZoneInfo("Europe/London")
NEW_YORK = ZoneInfo("America/New_York")
START = date(2020, 1, 2)
END = date(2026, 9, 18)
SEED = 20261003
N_BOOT = 10_000
PIP = 0.0001


def roster() -> pd.DataFrame:
    rows = []
    day = START
    while day <= END:
        if day.weekday() < 5:
            london_start = datetime(day.year, day.month, day.day, 16, 5, tzinfo=LONDON)
            ny_end = datetime(day.year, day.month, day.day, 16, 0, tzinfo=NEW_YORK)
            entry_ms = int(london_start.timestamp() * 1000)
            exit_ms = int(ny_end.timestamp() * 1000)
            assert 3.0 * 3600_000 <= exit_ms - entry_ms <= 5.0 * 3600_000
            rows.append({"london_date": day.isoformat(), "year": day.year,
                         "entry_target_ms": entry_ms, "exit_target_ms": exit_ms,
                         "entry_target_utc": london_start.astimezone(timezone.utc).isoformat(),
                         "exit_target_utc": ny_end.astimezone(timezone.utc).isoformat()})
        day += timedelta(days=1)
    frame = pd.DataFrame(rows)
    assert frame["london_date"].is_unique
    return frame


def first_valid_quote(stream, target_ms: int, header_end: int) -> tuple[tuple[int, float, float] | None, int]:
    first = None
    invalid = 0
    first_quote_at_or_after(stream, target_ms, header_end)
    while line := stream.readline():
        parts = line.rstrip(b"\r\n").split(b",")
        if len(parts) != 3:
            raise ValueError(f"Malformed EURUSD quote: {line[:100]!r}")
        stamp = int(parts[0])
        if stamp > target_ms + 10_000:
            break
        ask, bid = float(parts[1]), float(parts[2])
        if not (math.isfinite(ask) and math.isfinite(bid) and ask > 0 and bid > 0 and ask >= bid):
            invalid += 1
            continue
        if first is None:
            first = (stamp, ask, bid)
    return first, invalid


def evaluate(roster_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    onboarding = json.loads(ONBOARD.read_text(encoding="utf-8"))
    source = onboarding["source"]
    assert source["sha256"] == SOURCE_HASH
    path = Path(source["file"])
    assert path.is_file() and path.stat().st_size == source["size_bytes"]
    assert sha256(path) == SOURCE_HASH
    trades, skipped = [], []
    invalid_count = 0
    with path.open("rb") as stream:
        assert stream.readline().strip() == b"timestamp,askPrice,bidPrice"
        header_end = stream.tell()
        for row in roster_frame.itertuples(index=False):
            entry, n1 = first_valid_quote(stream, row.entry_target_ms, header_end)
            exit_quote, n2 = first_valid_quote(stream, row.exit_target_ms, header_end)
            invalid_count += n1 + n2
            if entry is None or exit_quote is None:
                skipped.append({"london_date": row.london_date,
                                "reason": "ENTRY_AND_EXIT_MISSING" if entry is None and exit_quote is None else
                                          "ENTRY_MISSING" if entry is None else "EXIT_MISSING"})
                continue
            entry_mid = (entry[1] + entry[2]) / 2
            exit_mid = (exit_quote[1] + exit_quote[2]) / 2
            gross = (exit_mid - entry_mid) / PIP
            net = (exit_quote[2] - entry[1]) / PIP
            trades.append({"london_date": row.london_date, "year": row.year,
                           "entry_target_ms": row.entry_target_ms, "exit_target_ms": row.exit_target_ms,
                           "entry_ms": entry[0], "exit_ms": exit_quote[0],
                           "entry_ask": entry[1], "entry_bid": entry[2],
                           "exit_ask": exit_quote[1], "exit_bid": exit_quote[2],
                           "entry_spread_pips": (entry[1] - entry[2]) / PIP,
                           "exit_spread_pips": (exit_quote[1] - exit_quote[2]) / PIP,
                           "hold_hours": (exit_quote[0] - entry[0]) / 3_600_000,
                           "gross_mid_pips": gross, "quote_net_pips": net,
                           "spread_drag_pips": gross - net,
                           "stress_net_pips": net - 0.3})
    trade_frame = pd.DataFrame(trades)
    skipped_frame = pd.DataFrame(skipped)
    assert len(trade_frame) + len(skipped_frame) == len(roster_frame)
    assert trade_frame["london_date"].is_unique
    assert np.allclose(trade_frame["spread_drag_pips"],
                       (trade_frame["entry_spread_pips"] + trade_frame["exit_spread_pips"]) / 2)
    return trade_frame, skipped_frame, {
        "roster_dates": len(roster_frame), "executed": len(trade_frame),
        "skip_reasons": skipped_frame["reason"].value_counts().to_dict(),
        "invalid_sampled_quotes": invalid_count,
    }


def metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0}
    pnl = frame["quote_net_pips"].to_numpy(np.float64)
    equity = np.r_[0, np.cumsum(pnl)]
    return {
        "n": len(frame), "gross_mid_mean_pips": float(frame["gross_mid_pips"].mean()),
        "gross_mid_total_pips": float(frame["gross_mid_pips"].sum()),
        "quote_net_mean_pips": float(pnl.mean()), "quote_net_total_pips": float(pnl.sum()),
        "stress_net_mean_pips": float(frame["stress_net_pips"].mean()),
        "stress_net_total_pips": float(frame["stress_net_pips"].sum()),
        "win_rate": float(np.mean(pnl > 0)),
        "max_drawdown_pips": float(np.max(np.maximum.accumulate(equity) - equity)),
        "median_entry_spread_pips": float(frame["entry_spread_pips"].median()),
        "p90_entry_spread_pips": float(frame["entry_spread_pips"].quantile(0.9)),
        "median_exit_spread_pips": float(frame["exit_spread_pips"].median()),
        "p90_exit_spread_pips": float(frame["exit_spread_pips"].quantile(0.9)),
        "mean_spread_drag_pips": float(frame["spread_drag_pips"].mean()),
        "median_hold_hours": float(frame["hold_hours"].median()),
    }


def bootstrap(frame: pd.DataFrame) -> dict:
    pnl = frame.sort_values("london_date")["quote_net_pips"].to_numpy(np.float64)
    rng = np.random.default_rng(SEED)
    samples = np.empty(N_BOOT, dtype=np.float64)
    for start in range(0, N_BOOT, 250):
        end = min(start + 250, N_BOOT)
        picks = rng.integers(0, len(pnl), size=(end - start, len(pnl)))
        samples[start:end] = pnl[picks].mean(axis=1)
    return {"date_blocks": len(pnl), "p_mean_le_zero": float(np.mean(samples <= 0)),
            "ci_95_low_pips": float(np.quantile(samples, 0.025)),
            "ci_95_high_pips": float(np.quantile(samples, 0.975))}


def main() -> None:
    assert not REPORT.exists(), "Frozen REPORT exists; never overwrite"
    spec = SPEC.read_text(encoding="utf-8")
    gate_lines = GATE.read_text(encoding="utf-8").splitlines()
    assert len(gate_lines) == 4 and all(line in spec for line in gate_lines)
    roster_frame = roster()
    roster_frame.to_csv(OUT / "date_roster.csv", index=False)
    trades, skipped, funnel = evaluate(roster_frame)
    trades.to_csv(OUT / "quote_trades.csv", index=False, float_format="%.9f")
    skipped.to_csv(OUT / "skipped_dates.csv", index=False)
    calibration = trades[trades["year"] < 2024]
    test = trades[trades["year"] >= 2024]
    yearly = {str(year): metrics(group) for year, group in trades.groupby("year", sort=True)}
    boot = bootstrap(test)
    gates = {
        "G-LFD1": len(test) >= 150 and test["london_date"].is_unique,
        "G-LFD2": float(test["stress_net_pips"].mean()) > 0,
        "G-LFD3": boot["p_mean_le_zero"] <= 0.05,
        "G-LFD4": all(yearly.get(str(y), {}).get("quote_net_total_pips", 0) > 0 for y in (2024, 2025)),
    }
    verdict = "PASS" if all(gates.values()) else "FAIL"
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": sha256(SPEC), "gate_sha256": sha256(GATE),
        "source_sha256": SOURCE_HASH, "funnel": funnel,
        "calibration": metrics(calibration), "test": metrics(test),
        "bootstrap": boot, "yearly": yearly, "gates": gates, "verdict": verdict,
    }
    (OUT / "run_meta.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["REPORT: EURUSD POST-LONDON-FIX DOLLAR REVERSAL, HYPOTHESIS 67",
             f"Generated UTC: {result['generated_utc']}   PHASE-1 VERDICT: {verdict}",
             "Fixed BUY EURUSD 16:05 London to 16:00 New York, one trade per weekday.",
             "Native Bid/Ask quote net plus separate extra 0.3-pip execution stress.",
             "Published 1999-2019 mechanism; 2024+ predeclared historical replication, not virgin OOS.",
             "First-quote quote economics omit live latency, rejection, financing and market impact.", "",
             f"FUNNEL: {json.dumps(funnel, sort_keys=True)}",
             f"CALIBRATION 2020-23: {json.dumps(result['calibration'], sort_keys=True)}",
             f"GATED TEST 2024+: {json.dumps(result['test'], sort_keys=True)}",
             f"BOOTSTRAP: {json.dumps(boot, sort_keys=True)}",
             f"GATES: {json.dumps(gates, sort_keys=True)}", "YEARLY:"]
    lines += [f"  {year}: {json.dumps(value, sort_keys=True)}" for year, value in yearly.items()]
    lines += ["", f"OVERALL PHASE-1 VERDICT: {verdict}",
              "PASS requires a separately frozen Model=4 order-execution study before any validated claim.",
              "FAIL is final for this artifact; no post-hoc timing, direction, or date-filter rescue."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "test": result["test"], "bootstrap": boot,
                      "gates": gates, "funnel": funnel}, indent=2))


if __name__ == "__main__":
    main()
