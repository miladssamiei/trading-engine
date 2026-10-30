"""Hypothesis 64: delayed XAUUSD response to signed growth surprises."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from xauusd_macro_reaction_follow_phase1 import quote_window, sha256


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/xauusd_macro_surprise_delayed_v1"
SPEC = OUT / "SPEC_XAUUSD_MACRO_SURPRISE_DELAYED.txt"
GATE = OUT / "GATE_XAUUSD_MACRO_SURPRISE_DELAYED.txt"
REPORT = OUT / "REPORT_XAUUSD_MACRO_SURPRISE_DELAYED.txt"
SOURCE = ROOT / "data/results/xauusd_macro_reaction_follow_v1/fred-us-macro-events-source.json"
ONBOARDING = ROOT / "data/results/xauusd_onboarding/onboarding_stats.json"
SOURCE_SHA256 = "b9d8a2256ecc61fda3f68efbc243b407db5b19b0f2288c44f827d03854861201"
NY = ZoneInfo("America/New_York")
FAMILIES = {
    "PAYEMS": (8, 30), "RSAFS": (8, 30),
    "ADPMNUSNERSA": (8, 15), "JTSJOL": (10, 0),
}
N_BOOT = 10_000
SEED = 20260930


def event_roster() -> pd.DataFrame:
    assert sha256(SOURCE) == SOURCE_SHA256
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    by_date: dict[str, list[dict]] = {}
    for event in source["events"]:
        meta = event["metadata"]
        family = meta.get("seriesId")
        if family not in FAMILIES or meta.get("releaseDateApproximate"):
            continue
        instant = datetime.fromisoformat(event["createdAt"].replace("Z", "+00:00")).astimezone(NY)
        if instant.year < 2020 or (instant.hour, instant.minute, instant.second) != (*FAMILIES[family], 0):
            continue
        consensus = meta.get("consensus") or {}
        try:
            actual = float(consensus.get("actual"))
            forecast = float(consensus.get("forecast"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(actual) or not np.isfinite(forecast) or actual == forecast:
            continue
        date = instant.date().isoformat()
        by_date.setdefault(date, []).append({
            "ny_date": date, "family": family,
            "release_ms": int(instant.timestamp() * 1000),
            "actual": actual, "forecast": forecast,
            "surprise": actual - forecast,
            "calendar_id": consensus.get("occurrenceId"),
            "source_url": consensus.get("sourceUrl"),
        })
    rows = []
    for date, events in sorted(by_date.items()):
        first_ms = min(row["release_ms"] for row in events)
        earliest = [row for row in events if row["release_ms"] == first_ms]
        selected = dict(earliest[0])
        selected["roster_reason"] = "READY" if len(earliest) == 1 else "AMBIGUOUS_EARLIEST"
        selected["same_date_event_count"] = len(events)
        selected["same_instant_event_count"] = len(earliest)
        rows.append(selected)
    roster = pd.DataFrame(rows)
    assert roster["ny_date"].is_unique
    assert len(roster[roster["ny_date"] >= "2022-01-01"]) == 184
    assert len(roster[(roster["ny_date"] >= "2022-01-01") & (roster["roster_reason"] == "AMBIGUOUS_EARLIEST")]) == 2
    return roster


def evaluate(roster: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    onboarding = json.loads(ONBOARDING.read_text(encoding="utf-8"))
    sources = [(int(x["first_ts"]), int(x["last_ts"]), Path(x["file"])) for x in onboarding["files"]]
    for _, _, path in sources:
        assert path.is_file(), path
    trades, skipped = [], []
    reasons: dict[str, int] = {}
    for event in roster.itertuples(index=False):
        release = int(event.release_ms)
        reason = None
        if event.roster_reason != "READY":
            reason = event.roster_reason
        else:
            source = next((path for first, last, path in sources if first <= release <= last), None)
            if source is None:
                reason = "SOURCE_RANGE_MISSING"
            else:
                ticks = quote_window(source, release + 60_000, release + 910_000)
                entry = next((q for q in ticks if release + 60_000 <= q[0] <= release + 65_000), None)
                exit_quote = next((q for q in ticks if release + 900_000 <= q[0] <= release + 910_000), None)
                if entry is None:
                    reason = "ENTRY_QUOTE_MISSING"
                elif exit_quote is None:
                    reason = "EXIT_QUOTE_MISSING"
                else:
                    direction = -1 if event.surprise > 0 else 1
                    entry_mid = (entry[1] + entry[2]) / 2
                    exit_mid = (exit_quote[1] + exit_quote[2]) / 2
                    gross = direction * (exit_mid - entry_mid)
                    net = exit_quote[2] - entry[1] if direction > 0 else entry[2] - exit_quote[1]
                    trades.append({
                        "ny_date": event.ny_date, "year": int(event.ny_date[:4]),
                        "family": event.family, "calendar_id": event.calendar_id,
                        "release_ms": release, "actual": event.actual,
                        "forecast": event.forecast, "surprise": event.surprise,
                        "direction": "BUY" if direction > 0 else "SELL",
                        "entry_ms": entry[0], "exit_ms": exit_quote[0],
                        "entry_bid": entry[2], "entry_ask": entry[1],
                        "exit_bid": exit_quote[2], "exit_ask": exit_quote[1],
                        "entry_spread": entry[1] - entry[2],
                        "exit_spread": exit_quote[1] - exit_quote[2],
                        "hold_seconds": (exit_quote[0] - entry[0]) / 1000.0,
                        "gross_mid_move": gross, "quote_net": net,
                        "spread_drag": gross - net, "extra_cost_stress_net": net - 0.41,
                    })
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
            skipped.append({"ny_date": event.ny_date, "family": event.family, "reason": reason})
    trade_df, skipped_df = pd.DataFrame(trades), pd.DataFrame(skipped)
    assert len(trade_df) + len(skipped_df) == len(roster)
    assert trade_df["ny_date"].is_unique
    assert np.allclose(trade_df["spread_drag"], (trade_df["entry_spread"] + trade_df["exit_spread"]) / 2)
    assert np.allclose(trade_df["quote_net"] - trade_df["extra_cost_stress_net"], 0.41)
    return trade_df, skipped_df, {"roster_dates": len(roster), "executed": len(trade_df), "skipped": reasons}


def metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0}
    pnl = frame["quote_net"].to_numpy(np.float64)
    equity = np.r_[0.0, np.cumsum(pnl)]
    return {
        "n": len(frame), "gross_mid_mean": float(frame["gross_mid_move"].mean()),
        "quote_net_mean": float(pnl.mean()),
        "stress_net_mean": float(frame["extra_cost_stress_net"].mean()),
        "quote_net_total": float(pnl.sum()),
        "win_rate": float(np.mean(pnl > 0)),
        "max_drawdown": float(np.max(np.maximum.accumulate(equity) - equity)),
        "median_entry_spread": float(frame["entry_spread"].median()),
        "p90_entry_spread": float(frame["entry_spread"].quantile(0.9)),
        "median_exit_spread": float(frame["exit_spread"].median()),
        "p90_exit_spread": float(frame["exit_spread"].quantile(0.9)),
        "mean_spread_drag": float(frame["spread_drag"].mean()),
        "median_hold_seconds": float(frame["hold_seconds"].median()),
        "directions": {str(k): int(v) for k, v in frame["direction"].value_counts().items()},
    }


def bootstrap(frame: pd.DataFrame) -> dict:
    grouped = frame.groupby("ny_date", sort=True)["quote_net"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(np.float64)
    counts = grouped["count"].to_numpy(np.int32)
    rng = np.random.default_rng(SEED)
    means = np.empty(N_BOOT, np.float64)
    for start in range(0, N_BOOT, 250):
        end = min(start + 250, N_BOOT)
        picks = rng.integers(0, len(sums), size=(end - start, len(sums)))
        means[start:end] = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return {
        "date_blocks": len(sums), "p_mean_le_zero": float(np.mean(means <= 0)),
        "ci_95_low": float(np.quantile(means, 0.025)),
        "ci_95_high": float(np.quantile(means, 0.975)),
    }


def main() -> None:
    assert not REPORT.exists(), "Frozen REPORT exists; do not overwrite it"
    spec_text = SPEC.read_text(encoding="utf-8")
    gates_text = GATE.read_text(encoding="utf-8").splitlines()
    assert len(gates_text) == 4 and all(line in spec_text for line in gates_text)
    roster = event_roster()
    roster.to_csv(OUT / "event_roster.csv", index=False)
    trades, skipped, funnel = evaluate(roster)
    trades.to_csv(OUT / "quote_trades.csv", index=False, float_format="%.6f")
    skipped.to_csv(OUT / "skipped_events.csv", index=False)
    cal, test = trades[trades["year"] < 2022], trades[trades["year"] >= 2022]
    yearly = {str(year): metrics(group) for year, group in trades.groupby("year")}
    family = {name: metrics(trades[trades["family"] == name]) for name in sorted(FAMILIES)}
    boot = bootstrap(test)
    gates = {
        "G-MSD1": len(test) >= 150 and test["ny_date"].is_unique,
        "G-MSD2": float(test["extra_cost_stress_net"].mean()) > 0,
        "G-MSD3": boot["p_mean_le_zero"] <= 0.05,
        "G-MSD4": all(yearly.get(str(y), {}).get("quote_net_total", 0) > 0 for y in range(2022, 2026)),
    }
    verdict = "PASS" if all(gates.values()) else "FAIL"
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": sha256(SPEC), "gate_sha256": sha256(GATE),
        "source_sha256": sha256(SOURCE), "onboarding_sha256": sha256(ONBOARDING),
        "funnel": funnel, "calibration": metrics(cal), "test": metrics(test),
        "bootstrap": boot, "yearly": yearly, "family": family,
        "gates": gates, "verdict": verdict,
    }
    (OUT / "run_meta.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "REPORT: DELAYED GOLD RESPONSE TO U.S. GROWTH SURPRISES, HYPOTHESIS 64",
        f"Generated UTC: {result['generated_utc']}   ECONOMIC-PROXY VERDICT: {verdict}",
        "Four frozen growth releases; sign(actual-minus-forecast) trades gold in the opposite direction.",
        "Historical Investing.com forecasts lack proved pre-release capture and actual receipt timestamps.",
        "Even an economic-proxy PASS would be PROVISIONAL; no validated or deployable system follows from it.", "",
        f"FUNNEL: {json.dumps(funnel, sort_keys=True)}",
        f"CALIBRATION 2020-2021 (descriptive): {json.dumps(result['calibration'], sort_keys=True)}",
        f"TEST 2022+ (gated): {json.dumps(result['test'], sort_keys=True)}",
        f"BOOTSTRAP (date blocks): {json.dumps(boot, sort_keys=True)}",
        f"GATES: {json.dumps(gates, sort_keys=True)}", "YEARLY:",
    ]
    lines += [f"  {year}: {json.dumps(value, sort_keys=True)}" for year, value in yearly.items()]
    lines += ["FAMILY:"]
    lines += [f"  {name}: {json.dumps(value, sort_keys=True)}" for name, value in family.items()]
    lines += [
        "", f"OVERALL ECONOMIC-PROXY VERDICT: {verdict}",
        "PASS would require a new source-vintage qualification before any Model=4 execution test.",
        "FAIL is final; no post-hoc event, threshold, latency, or exit-time subset rescues it.",
        "First-quote fills omit live latency, slippage, rejection, and market impact.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "funnel": funnel, "test": result["test"], "bootstrap": boot, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
