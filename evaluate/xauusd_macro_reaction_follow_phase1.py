"""Hypothesis 63: post-release gold reaction follow on actual Bid/Ask ticks."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/xauusd_macro_reaction_follow_v1"
SPEC = OUT / "SPEC_XAUUSD_MACRO_REACTION_FOLLOW.txt"
GATE = OUT / "GATE_XAUUSD_MACRO_REACTION_FOLLOW.txt"
SOURCE = OUT / "fred-us-macro-events-source.json"
REPORT = OUT / "REPORT_XAUUSD_MACRO_REACTION_FOLLOW.txt"
ONBOARDING = ROOT / "data/results/xauusd_onboarding/onboarding_stats.json"
SOURCE_SHA256 = "b9d8a2256ecc61fda3f68efbc243b407db5b19b0f2288c44f827d03854861201"
FAMILIES = {"PAYEMS", "CPIAUCNS", "PPIACO"}
NY = ZoneInfo("America/New_York")
SEED = 20260929
N_BOOT = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_roster() -> pd.DataFrame:
    assert sha256(SOURCE) == SOURCE_SHA256
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    by_date: dict[str, set[str]] = {}
    for event in source["events"]:
        meta = event["metadata"]
        family = meta.get("seriesId")
        if family not in FAMILIES or meta.get("releaseDateApproximate"):
            continue
        instant = datetime.fromisoformat(event["createdAt"].replace("Z", "+00:00")).astimezone(NY)
        if instant.year < 2020 or (instant.hour, instant.minute, instant.second) != (8, 30, 0):
            continue
        key = instant.date().isoformat()
        by_date.setdefault(key, set()).add(family)
    rows = [{"ny_date": date, "families": "+".join(sorted(families))} for date, families in sorted(by_date.items())]
    roster = pd.DataFrame(rows)
    assert roster["ny_date"].is_unique
    assert len(roster[roster["ny_date"] >= "2022-01-01"]) == 153
    return roster


def first_quote_at_or_after(stream, target_ms: int, header_end: int) -> None:
    """Seek a monotonic CSV to the first full line with timestamp >= target."""
    stream.seek(0, 2)
    lo, hi = header_end, stream.tell()
    while hi - lo > 8192:
        mid = (lo + hi) // 2
        stream.seek(mid)
        stream.readline()  # discard possibly partial line
        start = stream.tell()
        line = stream.readline()
        if not line:
            hi = mid
            continue
        timestamp = int(line.split(b",", 1)[0])
        if timestamp < target_ms:
            lo = stream.tell()
        else:
            hi = start
    stream.seek(lo)
    while True:
        pos = stream.tell()
        line = stream.readline()
        if not line:
            return
        if int(line.split(b",", 1)[0]) >= target_ms:
            stream.seek(pos)
            return


def quote_window(path: Path, start_ms: int, end_ms: int) -> list[tuple[int, float, float]]:
    rows = []
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
            if ask <= 0 or bid <= 0 or ask < bid:
                raise ValueError(f"Invalid Bid/Ask quote at {stamp} in {path}")
            rows.append((stamp, ask, bid))
    return rows


def minutes_on_date(date_str: str) -> tuple[int, int, int, int]:
    date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=NY)
    release = date.replace(hour=8, minute=30).astimezone(timezone.utc)
    base = int(release.timestamp() * 1000)
    return base, base + 60_000, base + 65_000, base + 900_000


def evaluate(roster: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    onboarding = json.loads(ONBOARDING.read_text(encoding="utf-8"))
    sources = [(int(x["first_ts"]), int(x["last_ts"]), Path(x["file"])) for x in onboarding["files"]]
    for _, _, path in sources:
        assert path.is_file(), path
    trades = []
    missing = []
    reasons = {}
    for event in roster.itertuples(index=False):
        release, reaction_end, entry_start, exit_start = minutes_on_date(event.ny_date)
        source = next((path for first, last, path in sources if first <= release <= last), None)
        reason = None
        if source is None:
            reason = "SOURCE_RANGE_MISSING"
        else:
            ticks = quote_window(source, release - 10_000, exit_start + 10_000)
            pre = [q for q in ticks if q[0] < release]
            reaction = [q for q in ticks if q[0] < reaction_end]
            entry = next((q for q in ticks if entry_start <= q[0] <= entry_start + 5_000), None)
            exit_quote = next((q for q in ticks if exit_start <= q[0] <= exit_start + 10_000), None)
            if not pre:
                reason = "PRE_RELEASE_QUOTE_MISSING"
            elif not reaction or reaction[-1][0] < reaction_end - 10_000:
                reason = "REACTION_QUOTE_MISSING"
            elif entry is None:
                reason = "ENTRY_QUOTE_MISSING"
            elif exit_quote is None:
                reason = "EXIT_QUOTE_MISSING"
            else:
                before, after = pre[-1], reaction[-1]
                pre_mid = 0.5 * (before[1] + before[2])
                reaction_mid = 0.5 * (after[1] + after[2])
                direction = int(np.sign(reaction_mid - pre_mid))
                if direction == 0:
                    reason = "ZERO_REACTION"
                else:
                    entry_mid = 0.5 * (entry[1] + entry[2])
                    exit_mid = 0.5 * (exit_quote[1] + exit_quote[2])
                    gross = direction * (exit_mid - entry_mid)
                    net = exit_quote[2] - entry[1] if direction > 0 else entry[2] - exit_quote[1]
                    trades.append({
                        "ny_date": event.ny_date, "year": int(event.ny_date[:4]),
                        "families": event.families, "direction": "BUY" if direction > 0 else "SELL",
                        "pre_release_ms": before[0], "reaction_ms": after[0],
                        "entry_ms": entry[0], "exit_ms": exit_quote[0],
                        "pre_release_mid": pre_mid, "reaction_mid": reaction_mid,
                        "reaction_move": reaction_mid - pre_mid,
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
            missing.append({"ny_date": event.ny_date, "families": event.families, "reason": reason})
    trade_df, missing_df = pd.DataFrame(trades), pd.DataFrame(missing)
    assert len(trade_df) + len(missing_df) == len(roster)
    assert trade_df["ny_date"].is_unique
    assert np.allclose(trade_df["spread_drag"], (trade_df["entry_spread"] + trade_df["exit_spread"]) / 2)
    assert np.allclose(trade_df["quote_net"] - trade_df["extra_cost_stress_net"], 0.41)
    return trade_df, missing_df, {"roster_dates": len(roster), "executed": len(trade_df), "skipped": reasons}


def metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0}
    pnl = frame["quote_net"].to_numpy(np.float64)
    equity = np.r_[0.0, np.cumsum(pnl)]
    return {
        "n": len(frame),
        "gross_mid_mean": float(frame["gross_mid_move"].mean()),
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
    by_date = frame.groupby("ny_date", sort=True)["quote_net"].agg(["sum", "count"])
    sums = by_date["sum"].to_numpy(np.float64)
    counts = by_date["count"].to_numpy(np.int32)
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
    cal = trades[trades["year"] < 2022]
    test = trades[trades["year"] >= 2022]
    yearly = {str(year): metrics(group) for year, group in trades.groupby("year")}
    family = {}
    for name in sorted(FAMILIES):
        family[name] = metrics(trades[trades["families"].str.contains(name)])
    boot = bootstrap(test)
    gates = {
        "G-MRF1": len(test) >= 150 and test["ny_date"].is_unique,
        "G-MRF2": float(test["extra_cost_stress_net"].mean()) > 0,
        "G-MRF3": boot["p_mean_le_zero"] <= 0.05,
        "G-MRF4": all(yearly.get(str(y), {}).get("quote_net_total", 0) > 0 for y in range(2022, 2026)),
    }
    verdict = "PASS" if all(gates.values()) else "FAIL"
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": sha256(SPEC), "gate_sha256": sha256(GATE),
        "source_sha256": sha256(SOURCE), "funnel": funnel,
        "calibration": metrics(cal), "test": metrics(test),
        "bootstrap": boot, "yearly": yearly, "family": family,
        "gates": gates, "verdict": verdict,
    }
    (OUT / "run_meta.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "REPORT: XAUUSD POST-RELEASE FIRST-MINUTE REACTION FOLLOW, HYPOTHESIS 63",
        f"Generated UTC: {result['generated_utc']}   PHASE-1 VERDICT: {verdict}",
        "BLS PAYEMS/CPI/PPI roster; direction from first-minute XAUUSD midquote; actual Bid/Ask entry and exit.",
        "Third-party FRED-derived dates were not independently reconciled date by date to BLS archives.",
        "Historical data are exposed; even PASS would not establish a prospective trading system.", "",
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
        "A PASS only permits a separate predeclared Model=4 tester and official BLS calendar reconciliation.",
        "A FAIL is final; no post-hoc event, direction, threshold, latency, or exit-time subset rescues it.",
        "Actual quotes omit additional live order latency, slippage, rejection, and market impact.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "funnel": funnel, "test": result["test"], "bootstrap": boot, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
