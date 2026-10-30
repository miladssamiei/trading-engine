"""Frozen gold Phase-1 test for hypothesis 60; see its SPEC and GATE."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from closing_hedge_momentum_phase1 import metrics, sha256


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/closing_hedge_momentum_xauusd"
SOURCE = ROOT / "data/results/xauusd_onboarding/xauusdtick_m5.csv"
SPEC = OUT / "SPEC_CLOSING_HEDGE_MOMENTUM_XAUUSD.txt"
GATE = OUT / "GATE_CLOSING_HEDGE_MOMENTUM_XAUUSD.txt"
NEEDED_WALLS = [500, 775, 780, 785, 790, 795, 800, 805]
SPREAD = 0.41
N_BOOT = 10_000
SEED = 20260926


def bootstrap(values: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    means = np.empty(N_BOOT)
    for start in range(0, N_BOOT, 250):
        end = min(start + 250, N_BOOT)
        draws = rng.integers(0, len(values), size=(end - start, len(values)))
        means[start:end] = values[draws].mean(axis=1)
    return {
        "p_mean_le_zero": float(np.mean(means <= 0)),
        "ci_95_low": float(np.quantile(means, 0.025)),
        "ci_95_high": float(np.quantile(means, 0.975)),
    }


def main() -> None:
    gate_lines = [line for line in GATE.read_text(encoding="utf-8").splitlines() if line]
    spec_text = SPEC.read_text(encoding="utf-8")
    assert len(gate_lines) == 4 and all(line in spec_text for line in gate_lines)

    bars = pd.read_csv(SOURCE, dtype={"time": "int64"})
    assert list(bars.columns) == ["time", "open", "high", "low", "close"]
    assert bars["time"].is_monotonic_increasing and bars["time"].is_unique
    assert (bars[["open", "high", "low", "close"]] > 0).all().all()
    ny = pd.to_datetime(bars["time"], unit="s", utc=True).dt.tz_convert("America/New_York")
    bars["ny_date"] = ny.dt.strftime("%Y-%m-%d")
    bars["wall"] = ny.dt.hour * 60 + ny.dt.minute
    narrow = bars.loc[bars["wall"].isin(NEEDED_WALLS), ["ny_date", "wall", "open", "close"]]
    assert not narrow.duplicated(["ny_date", "wall"]).any()
    open_panel = narrow.pivot(index="ny_date", columns="wall", values="open")
    close_panel = narrow.pivot(index="ny_date", columns="wall", values="close")
    session_dates = open_panel.index[open_panel[500].notna()]
    records = []
    skips = {"no_prior_session_close": 0, "missing_required_bar": 0, "signal_tie": 0}

    for i, date in enumerate(session_dates):
        if i == 0 or pd.isna(close_panel.at[session_dates[i - 1], 805]):
            skips["no_prior_session_close"] += 1
            continue
        if any(pd.isna(open_panel.at[date, wall]) for wall in NEEDED_WALLS) or any(
            pd.isna(close_panel.at[date, wall]) for wall in NEEDED_WALLS
        ):
            skips["missing_required_bar"] += 1
            continue
        prior_close = float(close_panel.at[session_dates[i - 1], 805])
        signal_close = float(close_panel.at[date, 775])
        direction = int(np.sign(signal_close - prior_close))
        if direction == 0:
            skips["signal_tie"] += 1
            continue
        entry = float(open_panel.at[date, 780])
        exit_bid = float(close_panel.at[date, 805])
        gross = direction * (exit_bid - entry)
        records.append(
            {
                "ny_date": date,
                "year": int(date[:4]),
                "direction": "LONG" if direction > 0 else "SHORT",
                "prior_gold_close_bid": prior_close,
                "signal_1255_close_bid": signal_close,
                "entry_1300_open_bid": entry,
                "exit_1325_close_bid": exit_bid,
                "gross_pt": gross,
                "net_pt": gross - SPREAD,
                "double_cost_net_pt": gross - 2 * SPREAD,
            }
        )

    trades = pd.DataFrame.from_records(records)
    assert not trades.empty and trades["ny_date"].is_unique
    assert len(session_dates) == len(trades) + sum(skips.values())
    assert np.allclose(trades["gross_pt"] - trades["net_pt"], SPREAD)
    assert np.allclose(trades["gross_pt"] - trades["double_cost_net_pt"], 2 * SPREAD)
    trades.to_csv(OUT / "closing_hedge_gold_trades.csv", index=False, float_format="%.6f")

    cal = trades[(trades["year"] >= 2020) & (trades["year"] <= 2023)]
    test = trades[(trades["year"] >= 2024) & (trades["year"] <= 2026)]
    boot = bootstrap(test["net_pt"].to_numpy())
    yearly = {str(year): metrics(part) for year, part in trades.groupby("year")}
    gates = {
        "G-GH1": len(test) >= 150,
        "G-GH2": float(test["double_cost_net_pt"].mean()) > 0,
        "G-GH3": boot["p_mean_le_zero"] <= 0.05,
        "G-GH4": yearly.get("2024", {}).get("net_total_pt", 0) > 0
        and yearly.get("2025", {}).get("net_total_pt", 0) > 0,
    }
    passed = all(gates.values())
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": 60,
        "source_m5_sha256": sha256(SOURCE),
        "spec_sha256": sha256(SPEC),
        "gate_sha256": sha256(GATE),
        "source_rows": int(len(bars)),
        "session_days": int(len(session_dates)),
        "skips": skips,
        "calibration": metrics(cal),
        "test": metrics(test),
        "bootstrap": boot,
        "yearly": yearly,
        "directional_test": {side: metrics(part) for side, part in test.groupby("direction")},
        "gates": gates,
        "verdict": "PASS" if passed else "FAIL",
    }
    (OUT / "run_meta.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = [
        "REPORT: XAUUSD COMEX-CLOSE HEDGING MOMENTUM, HYPOTHESIS 60",
        f"Generated UTC: {result['generated_utc']}    PHASE-1 VERDICT: {result['verdict']}",
        "Single predeclared rule: sign(prior 13:25 close to current 12:55 close), enter 13:00 open, exit 13:25 close.",
        "Source: Baltussen et al. (2021), https://academicweb.nd.edu/~zda/intramom.pdf",
        "Source paper studies GC futures through 2020; this is a transfer test on an XAUUSD CFD Bid-bar export.",
        "",
        f"DATA: {len(bars)} M5 bars; {len(session_dates)} session days; {len(trades)} executed; skips {skips}.",
        f"CALIBRATION 2020-2023 (diagnostic): {json.dumps(result['calibration'], sort_keys=True)}",
        f"TEST 2024-2026 (gated): {json.dumps(result['test'], sort_keys=True)}",
        f"TEST BOOTSTRAP: {json.dumps(boot, sort_keys=True)}",
        "YEARLY:",
    ]
    for year, value in yearly.items():
        report.append(f"  {year}: {json.dumps(value, sort_keys=True)}")
    report.append("TEST BY DIRECTION (descriptive, no side selection):")
    for side, value in result["directional_test"].items():
        report.append(f"  {side}: {json.dumps(value, sort_keys=True)}")
    report.extend(["GATES:"] + [f"  {name}: {'PASS' if value else 'FAIL'}" for name, value in gates.items()])
    report.extend(
        [
            f"OVERALL PHASE-1: {result['verdict']}",
            "",
            "HONESTY: The 2024-2026 test is a fixed historical partition, but other project studies",
            "have exposed this tape. M5 Bid opens/closes are fill proxies; only a median spread",
            "is charged, so variable spread, slippage, market impact, and MT5 order execution",
            "remain unmeasured. No swap applies because each position ends before 13:30 ET.",
            "A PASS here alone would not validate a deployable system. A FAIL is final for",
            "this artifact; no post-hoc side, threshold, or time-window adjustment is allowed.",
            "SPEC/GATE were frozen before code and before this run.",
        ]
    )
    (OUT / "REPORT_CLOSING_HEDGE_MOMENTUM_XAUUSD_PHASE1.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "test": result["test"], "bootstrap": boot, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
