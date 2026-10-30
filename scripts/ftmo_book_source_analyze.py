"""Apply the frozen FTMO market-depth source gate without using returns."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path


SYMBOLS = ("US500.cash", "XAUUSD")


def receipt_time(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Receipt timestamp lacks timezone")
    return result.astimezone(timezone.utc)


def positive_finite(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def new_stats() -> dict:
    return {"samples": 0, "nonempty": 0, "distinct_states": 0,
            "two_sided": 0, "invalid_levels_in_stream": 0,
            "first_nonempty_utc": None, "last_nonempty_utc": None,
            "bid_differences": [], "ask_differences": [], "quote_ages": [],
            "states": set()}


def analyze(folder: Path) -> dict:
    summary_path = folder / "capture_summary.json"
    snapshot_path = folder / "snapshots.jsonl"
    if not summary_path.is_file():
        raise ValueError("Missing capture_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETE":
        raise ValueError("Capture did not complete")
    if not snapshot_path.is_file():
        raise ValueError("Missing snapshots.jsonl")
    digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    if digest != summary.get("snapshot_sha256"):
        raise ValueError("Snapshot SHA256 differs from capture summary")
    stats = {symbol: new_stats() for symbol in SYMBOLS}
    with snapshot_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                symbol = row["symbol"]
                if symbol not in stats:
                    raise ValueError("Unexpected symbol")
                received = receipt_time(row["received_at_utc"])
                levels = row["levels"]
                if not isinstance(levels, list):
                    raise ValueError("Invalid levels list")
                item = stats[symbol]
                item["samples"] += 1
                for level in levels:
                    if not positive_finite(level["price"]) or not positive_finite(level["volume"]):
                        item["invalid_levels_in_stream"] += 1
                if not levels:
                    continue
                item["nonempty"] += 1
                if item["first_nonempty_utc"] is None:
                    item["first_nonempty_utc"] = received
                item["last_nonempty_utc"] = received
                item["states"].add(json.dumps(levels, sort_keys=True, separators=(",", ":")))
                buys = [float(level["price"]) for level in levels
                        if level["type"] == 2 and positive_finite(level["price"])]
                sells = [float(level["price"]) for level in levels
                         if level["type"] == 1 and positive_finite(level["price"])]
                if buys and sells:
                    item["two_sided"] += 1
                if buys and positive_finite(row.get("bid")):
                    item["bid_differences"].append(max(buys) - float(row["bid"]))
                if sells and positive_finite(row.get("ask")):
                    item["ask_differences"].append(min(sells) - float(row["ask"]))
                raw_msc = row.get("raw_broker_tick_time_msc")
                if raw_msc is not None:
                    raw = datetime.fromtimestamp(int(raw_msc) / 1000, tz=timezone.utc)
                    age = (received - (raw - timedelta(hours=3))).total_seconds()
                    item["quote_ages"].append(age)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Corrupt snapshot row {line_number}: {exc}") from exc
    result = {"verdict": "FAIL", "source": summary.get("source"),
              "snapshot_sha256": digest, "symbols": {}}
    for symbol, item in stats.items():
        declared = summary.get("symbols", {}).get(symbol)
        if not isinstance(declared, dict) or item["samples"] != declared.get("samples"):
            raise ValueError(f"Capture summary/sample mismatch for {symbol}")
        span = 0.0
        if item["first_nonempty_utc"] is not None:
            span = (item["last_nonempty_utc"] - item["first_nonempty_utc"]).total_seconds()
        invalid = item["invalid_levels_in_stream"] + int(declared.get("invalid_levels", 0))
        two_sided_share = item["two_sided"] / item["nonempty"] if item["nonempty"] else 0.0
        gates = {"nonempty_at_least_900": item["nonempty"] >= 900,
                 "span_at_least_900_seconds": span >= 900,
                 "distinct_states_at_least_50": len(item["states"]) >= 50,
                 "two_sided_share_at_least_90pct": two_sided_share >= 0.9,
                 "zero_invalid_levels": invalid == 0}
        passed = all(gates.values())
        if passed:
            result["verdict"] = "PASS"
        ages = item["quote_ages"]
        result["symbols"][symbol] = {
            "verdict": "PASS" if passed else "FAIL", "gates": gates,
            "samples": item["samples"], "nonempty": item["nonempty"],
            "span_seconds": span, "distinct_states": len(item["states"]),
            "two_sided": item["two_sided"], "two_sided_share": two_sided_share,
            "invalid_levels": invalid,
            "median_book_bid_minus_quote_bid": statistics.median(item["bid_differences"]) if item["bid_differences"] else None,
            "median_book_ask_minus_quote_ask": statistics.median(item["ask_differences"]) if item["ask_differences"] else None,
            "median_quote_age_seconds_3h_corrected": statistics.median(ages) if ages else None,
            "quote_age_gt_10s_share": sum(age > 10 for age in ages) / len(ages) if ages else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    try:
        verdict = analyze(args.input)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        verdict = {"verdict": "INCOMPLETE", "reason": str(exc)}
    args.input.mkdir(parents=True, exist_ok=True)
    (args.input / "SOURCE_VERDICT.json").write_text(
        json.dumps(verdict, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [f"FTMO MARKET DEPTH SOURCE VERDICT: {verdict['verdict']}"]
    if "reason" in verdict:
        lines.append(f"Reason: {verdict['reason']}")
    for symbol, detail in verdict.get("symbols", {}).items():
        lines.append(f"{symbol}: {detail['verdict']}; nonempty={detail['nonempty']}; "
                     f"span={detail['span_seconds']:.1f}s; distinct={detail['distinct_states']}; "
                     f"two-sided={detail['two_sided_share']:.3f}; invalid={detail['invalid_levels']}")
        lines.append("Gates: " + ", ".join(f"{name}={value}" for name, value in detail["gates"].items()))
    lines.append("Source qualification only; no trading returns or orders assessed.")
    (args.input / "SOURCE_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if verdict["verdict"] == "INCOMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
