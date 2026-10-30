"""Hypothesis 65: first-minute EURUSD response as delayed XAUUSD signal."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from xauusd_macro_reaction_follow_phase1 import first_quote_at_or_after, sha256


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/eurusd_leads_xauusd_macro_v1"
SPEC = OUT / "SPEC_EURUSD_LEADS_XAUUSD_MACRO.txt"
GATE = OUT / "GATE_EURUSD_LEADS_XAUUSD_MACRO.txt"
REPORT = OUT / "REPORT_EURUSD_LEADS_XAUUSD_MACRO.txt"
EVENT_SOURCE = ROOT / "data/results/xauusd_macro_reaction_follow_v1/fred-us-macro-events-source.json"
EVENT_SHA = "b9d8a2256ecc61fda3f68efbc243b407db5b19b0f2288c44f827d03854861201"
EUR_ONBOARD = ROOT / "data/results/eurusd_onboarding/onboarding_stats.json"
XAU_ONBOARD = ROOT / "data/results/xauusd_onboarding/onboarding_stats.json"
FAMILIES = {"PAYEMS", "CPIAUCNS", "PPIACO", "RSAFS"}
NY = ZoneInfo("America/New_York")
N_BOOT = 10_000
SEED = 20261001


def event_roster() -> pd.DataFrame:
    assert sha256(EVENT_SOURCE) == EVENT_SHA
    source = json.loads(EVENT_SOURCE.read_text(encoding="utf-8"))
    dates: dict[str, set[str]] = {}
    for event in source["events"]:
        meta = event["metadata"]
        family = meta.get("seriesId")
        if family not in FAMILIES or meta.get("releaseDateApproximate"):
            continue
        instant = datetime.fromisoformat(event["createdAt"].replace("Z", "+00:00")).astimezone(NY)
        if instant.year < 2020 or (instant.hour, instant.minute, instant.second) != (8, 30, 0):
            continue
        date = instant.date().isoformat()
        dates.setdefault(date, set()).add(family)
    rows = []
    for date, families in sorted(dates.items()):
        release = datetime.fromisoformat(date).replace(hour=8, minute=30, tzinfo=NY)
        rows.append({
            "ny_date": date, "release_ms": int(release.timestamp() * 1000),
            "families": "+".join(sorted(families)),
        })
    roster = pd.DataFrame(rows)
    assert roster["ny_date"].is_unique
    return roster


def valid_quotes(path: Path, start_ms: int, end_ms: int) -> tuple[list[tuple[int, float, float]], int]:
    quotes = []
    invalid = 0
    with path.open("rb") as stream:
        header = stream.readline().strip()
        assert header == b"timestamp,askPrice,bidPrice", (path, header)
        first_quote_at_or_after(stream, start_ms, stream.tell())
        while True:
            line = stream.readline()
            if not line:
                break
            raw = line.rstrip(b"\r\n").split(b",")
            if len(raw) != 3:
                raise ValueError(f"Malformed tick in {path}: {line[:100]!r}")
            stamp = int(raw[0])
            if stamp > end_ms:
                break
            ask, bid = float(raw[1]), float(raw[2])
            if not np.isfinite(ask) or not np.isfinite(bid) or ask <= 0 or bid <= 0 or ask < bid:
                invalid += 1
                continue
            quotes.append((stamp, ask, bid))
    return quotes, invalid


def evaluate(roster: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    eur_meta = json.loads(EUR_ONBOARD.read_text(encoding="utf-8"))
    xau_meta = json.loads(XAU_ONBOARD.read_text(encoding="utf-8"))
    eur_path = Path(eur_meta["source"]["file"])
    assert eur_meta["source"]["sha256"] == "23376d5ffd9d5f287f92875faa4df303032162158fff20387c4c284894a8e3b1"
    assert eur_path.is_file()
    xau_sources = [(int(x["first_ts"]), int(x["last_ts"]), Path(x["file"])) for x in xau_meta["files"]]
    for _, _, path in xau_sources:
        assert path.is_file(), path
    trades, skipped = [], []
    reasons: dict[str, int] = {}
    invalid_eur, invalid_xau = 0, 0
    for event in roster.itertuples(index=False):
        release = int(event.release_ms)
        eur, bad = valid_quotes(eur_path, release - 10_000, release + 59_999)
        invalid_eur += bad
        before = [q for q in eur if q[0] < release]
        after = [q for q in eur if q[0] < release + 60_000]
        reason = None
        if not before:
            reason = "EUR_PRE_RELEASE_MISSING"
        elif not after or after[-1][0] < release + 50_000:
            reason = "EUR_REACTION_MISSING"
        else:
            pre = before[-1]
            reaction = after[-1]
            pre_mid = (pre[1] + pre[2]) / 2
            reaction_mid = (reaction[1] + reaction[2]) / 2
            direction = int(np.sign(reaction_mid - pre_mid))
            if direction == 0:
                reason = "EUR_ZERO_REACTION"
            else:
                xau_path = next((path for first, last, path in xau_sources if first <= release <= last), None)
                if xau_path is None:
                    reason = "XAU_SOURCE_RANGE_MISSING"
                else:
                    xau, bad = valid_quotes(xau_path, release + 65_000, release + 910_000)
                    invalid_xau += bad
                    entry = next((q for q in xau if release + 65_000 <= q[0] <= release + 70_000), None)
                    exit_quote = next((q for q in xau if release + 900_000 <= q[0] <= release + 910_000), None)
                    if entry is None:
                        reason = "XAU_ENTRY_MISSING"
                    elif exit_quote is None:
                        reason = "XAU_EXIT_MISSING"
                    else:
                        entry_mid = (entry[1] + entry[2]) / 2
                        exit_mid = (exit_quote[1] + exit_quote[2]) / 2
                        gross = direction * (exit_mid - entry_mid)
                        net = exit_quote[2] - entry[1] if direction > 0 else entry[2] - exit_quote[1]
                        trades.append({
                            "ny_date": event.ny_date, "year": int(event.ny_date[:4]),
                            "families": event.families, "release_ms": release,
                            "eur_pre_ms": pre[0], "eur_reaction_ms": reaction[0],
                            "eur_pre_mid": pre_mid, "eur_reaction_mid": reaction_mid,
                            "eur_reaction_move": reaction_mid - pre_mid,
                            "direction": "BUY" if direction > 0 else "SELL",
                            "xau_entry_ms": entry[0], "xau_exit_ms": exit_quote[0],
                            "xau_entry_bid": entry[2], "xau_entry_ask": entry[1],
                            "xau_exit_bid": exit_quote[2], "xau_exit_ask": exit_quote[1],
                            "entry_spread": entry[1] - entry[2],
                            "exit_spread": exit_quote[1] - exit_quote[2],
                            "hold_seconds": (exit_quote[0] - entry[0]) / 1000.0,
                            "gross_mid_move": gross, "quote_net": net,
                            "spread_drag": gross - net, "extra_cost_stress_net": net - 0.41,
                        })
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
            skipped.append({"ny_date": event.ny_date, "families": event.families, "reason": reason})
    trade_df, skipped_df = pd.DataFrame(trades), pd.DataFrame(skipped)
    assert len(trade_df) + len(skipped_df) == len(roster)
    assert trade_df["ny_date"].is_unique
    assert np.allclose(trade_df["spread_drag"], (trade_df["entry_spread"] + trade_df["exit_spread"]) / 2)
    assert np.allclose(trade_df["quote_net"] - trade_df["extra_cost_stress_net"], 0.41)
    return trade_df, skipped_df, {
        "roster_dates": len(roster), "executed": len(trade_df),
        "skipped": reasons, "invalid_eur_quotes_excluded": invalid_eur,
        "invalid_xau_quotes_excluded": invalid_xau,
    }


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
    gate_lines = GATE.read_text(encoding="utf-8").splitlines()
    assert len(gate_lines) == 4 and all(line in spec_text for line in gate_lines)
    roster = event_roster()
    roster.to_csv(OUT / "event_roster.csv", index=False)
    trades, skipped, funnel = evaluate(roster)
    trades.to_csv(OUT / "quote_trades.csv", index=False, float_format="%.6f")
    skipped.to_csv(OUT / "skipped_events.csv", index=False)
    cal, test = trades[trades["year"] < 2022], trades[trades["year"] >= 2022]
    yearly = {str(year): metrics(group) for year, group in trades.groupby("year")}
    family = {name: metrics(trades[trades["families"].str.contains(name)]) for name in sorted(FAMILIES)}
    boot = bootstrap(test)
    gates = {
        "G-EXL1": len(test) >= 150 and test["ny_date"].is_unique,
        "G-EXL2": float(test["extra_cost_stress_net"].mean()) > 0,
        "G-EXL3": boot["p_mean_le_zero"] <= 0.05,
        "G-EXL4": all(yearly.get(str(y), {}).get("quote_net_total", 0) > 0 for y in range(2022, 2026)),
    }
    verdict = "PASS" if all(gates.values()) else "FAIL"
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": sha256(SPEC), "gate_sha256": sha256(GATE),
        "event_source_sha256": sha256(EVENT_SOURCE),
        "eur_onboarding_sha256": sha256(EUR_ONBOARD),
        "xau_onboarding_sha256": sha256(XAU_ONBOARD),
        "funnel": funnel, "calibration": metrics(cal), "test": metrics(test),
        "bootstrap": boot, "yearly": yearly, "family": family,
        "gates": gates, "verdict": verdict,
    }
    (OUT / "run_meta.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "REPORT: EURUSD FIRST-MINUTE U.S.-NEWS RESPONSE LEADS GOLD, HYPOTHESIS 65",
        f"Generated UTC: {result['generated_utc']}   PHASE-1 VERDICT: {verdict}",
        "EURUSD raw midpoint direction through 08:31; gold real Bid/Ask 08:31:05 to 08:45.",
        "Third-party release dates not reconciled date by date to official calendars; cross-feed clock unqualified.",
        "Historical inputs have been exposed; a PASS would be a provisional screen, not a validated system.", "",
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
        "", f"OVERALL PHASE-1 VERDICT: {verdict}",
        "PASS only permits a separate predeclared source/clock qualification and Model=4 execution test.",
        "FAIL is final; no post-hoc side, event, time, or magnitude subset rescues it.",
        "First-quote fills omit live order latency, slippage, rejection, and market impact.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "funnel": funnel, "test": result["test"], "bootstrap": boot, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
