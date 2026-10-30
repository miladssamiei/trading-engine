"""Hypothesis 66: native EURUSD transfer of delayed macro-surprise rule."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from eurusd_leads_xauusd_macro_phase1 import valid_quotes
from xauusd_macro_reaction_follow_phase1 import sha256
from xauusd_macro_surprise_delayed_phase1 import event_roster


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/eurusd_macro_surprise_delayed_v1"
SPEC = OUT / "SPEC_EURUSD_MACRO_SURPRISE_DELAYED.txt"
GATE = OUT / "GATE_EURUSD_MACRO_SURPRISE_DELAYED.txt"
REPORT = OUT / "REPORT_EURUSD_MACRO_SURPRISE_DELAYED.txt"
ONBOARD = ROOT / "data/results/eurusd_onboarding/onboarding_stats.json"
EVENT_SOURCE = ROOT / "data/results/xauusd_macro_reaction_follow_v1/fred-us-macro-events-source.json"
EVENT_SHA = "b9d8a2256ecc61fda3f68efbc243b407db5b19b0f2288c44f827d03854861201"
PIP = 0.00010
SEED = 20261002
N_BOOT = 10_000
FAMILIES = ("PAYEMS", "RSAFS", "ADPMNUSNERSA", "JTSJOL")


def evaluate(roster: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    onboarding = json.loads(ONBOARD.read_text(encoding="utf-8"))
    assert onboarding["source"]["sha256"] == "23376d5ffd9d5f287f92875faa4df303032162158fff20387c4c284894a8e3b1"
    path = Path(onboarding["source"]["file"])
    assert path.is_file()
    trades, skipped = [], []
    reasons: dict[str, int] = {}
    invalid_quotes = 0
    for event in roster.itertuples(index=False):
        release = int(event.release_ms)
        reason = None
        if event.roster_reason != "READY":
            reason = event.roster_reason
        else:
            quotes, bad = valid_quotes(path, release + 60_000, release + 910_000)
            invalid_quotes += bad
            entry = next((q for q in quotes if release + 60_000 <= q[0] <= release + 65_000), None)
            exit_quote = next((q for q in quotes if release + 900_000 <= q[0] <= release + 910_000), None)
            if entry is None:
                reason = "ENTRY_QUOTE_MISSING"
            elif exit_quote is None:
                reason = "EXIT_QUOTE_MISSING"
            else:
                direction = -1 if event.surprise > 0 else 1
                entry_mid = (entry[1] + entry[2]) / 2
                exit_mid = (exit_quote[1] + exit_quote[2]) / 2
                gross = direction * (exit_mid - entry_mid) / PIP
                net = (exit_quote[2] - entry[1]) / PIP if direction > 0 else (entry[2] - exit_quote[1]) / PIP
                trades.append({
                    "ny_date": event.ny_date, "year": int(event.ny_date[:4]),
                    "family": event.family, "calendar_id": event.calendar_id,
                    "release_ms": release, "actual": event.actual,
                    "forecast": event.forecast, "surprise": event.surprise,
                    "direction": "BUY" if direction > 0 else "SELL",
                    "entry_ms": entry[0], "exit_ms": exit_quote[0],
                    "entry_bid": entry[2], "entry_ask": entry[1],
                    "exit_bid": exit_quote[2], "exit_ask": exit_quote[1],
                    "entry_spread_pips": (entry[1] - entry[2]) / PIP,
                    "exit_spread_pips": (exit_quote[1] - exit_quote[2]) / PIP,
                    "hold_seconds": (exit_quote[0] - entry[0]) / 1000.0,
                    "gross_mid_pips": gross, "quote_net_pips": net,
                    "spread_drag_pips": gross - net,
                    "extra_cost_stress_net_pips": net - 0.3,
                })
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
            skipped.append({"ny_date": event.ny_date, "family": event.family, "reason": reason})
    trade_df, skipped_df = pd.DataFrame(trades), pd.DataFrame(skipped)
    assert len(trade_df) + len(skipped_df) == len(roster)
    assert trade_df["ny_date"].is_unique
    assert np.allclose(trade_df["spread_drag_pips"],
                       (trade_df["entry_spread_pips"] + trade_df["exit_spread_pips"]) / 2)
    assert np.allclose(trade_df["quote_net_pips"] - trade_df["extra_cost_stress_net_pips"], 0.3)
    return trade_df, skipped_df, {
        "roster_dates": len(roster), "executed": len(trade_df),
        "skipped": reasons, "invalid_quotes_excluded": invalid_quotes,
    }


def metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0}
    pnl = frame["quote_net_pips"].to_numpy(np.float64)
    equity = np.r_[0.0, np.cumsum(pnl)]
    return {
        "n": len(frame), "gross_mid_mean_pips": float(frame["gross_mid_pips"].mean()),
        "quote_net_mean_pips": float(pnl.mean()),
        "stress_net_mean_pips": float(frame["extra_cost_stress_net_pips"].mean()),
        "quote_net_total_pips": float(pnl.sum()),
        "win_rate": float(np.mean(pnl > 0)),
        "max_drawdown_pips": float(np.max(np.maximum.accumulate(equity) - equity)),
        "median_entry_spread_pips": float(frame["entry_spread_pips"].median()),
        "p90_entry_spread_pips": float(frame["entry_spread_pips"].quantile(0.9)),
        "median_exit_spread_pips": float(frame["exit_spread_pips"].median()),
        "p90_exit_spread_pips": float(frame["exit_spread_pips"].quantile(0.9)),
        "mean_spread_drag_pips": float(frame["spread_drag_pips"].mean()),
        "median_hold_seconds": float(frame["hold_seconds"].median()),
        "directions": {str(k): int(v) for k, v in frame["direction"].value_counts().items()},
    }


def bootstrap(frame: pd.DataFrame) -> dict:
    grouped = frame.groupby("ny_date", sort=True)["quote_net_pips"].agg(["sum", "count"])
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
        "ci_95_low_pips": float(np.quantile(means, 0.025)),
        "ci_95_high_pips": float(np.quantile(means, 0.975)),
    }


def main() -> None:
    assert not REPORT.exists(), "Frozen REPORT exists; do not overwrite it"
    spec_text = SPEC.read_text(encoding="utf-8")
    gate_lines = GATE.read_text(encoding="utf-8").splitlines()
    assert len(gate_lines) == 4 and all(line in spec_text for line in gate_lines)
    assert sha256(EVENT_SOURCE) == EVENT_SHA
    roster = event_roster()
    roster.to_csv(OUT / "event_roster.csv", index=False)
    trades, skipped, funnel = evaluate(roster)
    trades.to_csv(OUT / "quote_trades.csv", index=False, float_format="%.8f")
    skipped.to_csv(OUT / "skipped_events.csv", index=False)
    cal, test = trades[trades["year"] < 2022], trades[trades["year"] >= 2022]
    yearly = {str(year): metrics(group) for year, group in trades.groupby("year")}
    family = {name: metrics(trades[trades["family"] == name]) for name in FAMILIES}
    boot = bootstrap(test)
    gates = {
        "G-EMS1": len(test) >= 150 and test["ny_date"].is_unique,
        "G-EMS2": float(test["extra_cost_stress_net_pips"].mean()) > 0,
        "G-EMS3": boot["p_mean_le_zero"] <= 0.05,
        "G-EMS4": all(yearly.get(str(y), {}).get("quote_net_total_pips", 0) > 0 for y in range(2022, 2026)),
    }
    verdict = "PASS" if all(gates.values()) else "FAIL"
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": sha256(SPEC), "gate_sha256": sha256(GATE),
        "event_source_sha256": sha256(EVENT_SOURCE), "eur_onboarding_sha256": sha256(ONBOARD),
        "funnel": funnel, "calibration": metrics(cal), "test": metrics(test),
        "bootstrap": boot, "yearly": yearly, "family": family,
        "gates": gates, "verdict": verdict,
    }
    (OUT / "run_meta.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "REPORT: DELAYED EURUSD RESPONSE TO U.S. GROWTH SURPRISES, HYPOTHESIS 66",
        f"Generated UTC: {result['generated_utc']}   ECONOMIC-PROXY VERDICT: {verdict}",
        "Exact #64 four-family signal transferred natively to EURUSD, 60-second entry and 15-minute exit.",
        "Historical forecast vintages and first actual receipt times are unproven; even PASS is PROVISIONAL.",
        "Actual EURUSD Bid/Ask charged; extra 0.3-pip cost stress applied to the same trades.", "",
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
        "PASS requires a new source-vintage qualification before any Model=4 tester.",
        "FAIL is final; no post-hoc side, event, threshold, or time subset rescues it.",
        "First-quote fills omit live latency, rejection, slippage, and market impact.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "funnel": funnel, "test": result["test"], "bootstrap": boot, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
