"""Hypothesis 70 phase 2A: native XAUUSD Bid/Ask replay of the same-day rule."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import xauusd_saleh_donchian_phase1 as saleh
from xauusd_macro_reaction_follow_phase1 import first_quote_at_or_after
from xauusd_saleh_intraday_phase1 import flat_indices


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/xauusd_saleh_intraday_real_quotes_v1"
SPEC = OUT / "SPEC_SALEH_INTRADAY_REAL_QUOTES.txt"
GATE = OUT / "GATE_SALEH_INTRADAY_REAL_QUOTES.txt"
REPORT = OUT / "REPORT_SALEH_INTRADAY_REAL_QUOTES.txt"
ONBOARD = ROOT / "data/results/xauusd_onboarding/onboarding_stats.json"
BAR_META = ROOT / "data/results/xauusd_saleh_intraday_v1/run_meta.json"
NY = ZoneInfo("America/New_York")
SEED = 20261007
N_BOOT = 10_000
WINDOW_MS = 10_000


def parse_quote(line: bytes) -> tuple[int, float, float, bool]:
    raw = line.rstrip(b"\r\n").split(b",")
    if len(raw) != 3:
        raise ValueError(f"Malformed XAUUSD tick: {line[:100]!r}")
    stamp = int(raw[0])
    ask, bid = float(raw[1]), float(raw[2])
    valid = math.isfinite(ask) and math.isfinite(bid) and ask > 0 and bid > 0 and ask >= bid
    return stamp, ask, bid, valid


def source_files() -> tuple[list[tuple[int, int, Path]], list[dict]]:
    onboard = json.loads(ONBOARD.read_text(encoding="utf-8"))
    paths = []
    hashed = []
    for item in onboard["files"]:
        path = Path(item["file"])
        assert path.is_file()
        digest = saleh.sha256_file(path)
        assert digest == item["sha256"], (path, digest, item["sha256"])
        paths.append((int(item["first_ts"]), int(item["last_ts"]), path))
        hashed.append({"file": str(path), "sha256": digest, "rows": int(item["rows"])})
    assert paths == sorted(paths, key=lambda x: x[0])
    return paths, hashed


def source_for(target_ms: int, sources: list[tuple[int, int, Path]]) -> Path | None:
    for first, last, path in sources:
        if first <= target_ms + WINDOW_MS and last >= target_ms:
            return path
    return None


def end_ms_for_nydate(day: int) -> int:
    year, month, dom = day // 10000, (day // 100) % 100, day % 100
    wall = datetime(year, month, dom, 16, 0, tzinfo=NY)
    return int(wall.timestamp() * 1000)


def replay_one(i: int, direction: int, ctx: dict, source: Path, end_target_ms: int) -> tuple[dict | None, dict]:
    """A single trade; returns None only when the entry quote is absent."""
    entry_target_ms = int(ctx["grid1"][i]) * 3_600_000
    invalid = 0
    fill_hour_quotes = 0
    with source.open("rb") as stream:
        assert stream.readline().strip() == b"timestamp,askPrice,bidPrice"
        header_end = stream.tell()
        first_quote_at_or_after(stream, entry_target_ms, header_end)
        entry = None
        while line := stream.readline():
            quote = parse_quote(line)
            if quote[0] > entry_target_ms + WINDOW_MS:
                break
            if not quote[3]:
                invalid += 1
                continue
            entry = quote
            break
        if entry is None:
            return None, {"invalid_quotes": invalid, "fill_hour_quotes": 0,
                          "reason": "ENTRY_QUOTE_MISSING"}

        structural_open = float(ctx["o1"][i])
        atr = float(ctx["atr1"][i - 1])
        sl = structural_open - direction * saleh.SL_ATR_MULT * atr
        initial_sl = sl
        entry_mid = (entry[1] + entry[2]) / 2
        last_valid = entry
        processed_hour = i
        exit_quote = None
        exit_type = None
        unresolved = False
        # Process entry quote as a possible immediate stop, then stream later quotes.
        pending = entry
        while True:
            if pending is None:
                line = stream.readline()
                if not line:
                    unresolved = True
                    exit_quote = last_valid
                    exit_type = "UNRESOLVED_EXIT_MARK"
                    break
                quote = parse_quote(line)
            else:
                quote = pending
                pending = None
            stamp, ask, bid, valid = quote
            if stamp > end_target_ms + WINDOW_MS:
                unresolved = True
                exit_quote = last_valid
                exit_type = "UNRESOLVED_EXIT_MARK"
                break
            if not valid:
                invalid += 1
                continue
            if stamp >= end_target_ms:
                exit_quote = quote
                exit_type = "DAY_FLAT"
                break
            last_valid = quote
            while stamp >= (int(ctx["grid1"][processed_hour]) + 1) * 3_600_000:
                if ctx["real1"][processed_hour]:
                    close_bid = float(ctx["c1"][processed_hour])
                    bar_atr = float(ctx["atr1"][processed_hour])
                    if direction == 1:
                        sl = max(sl, close_bid - saleh.SL_ATR_MULT * bar_atr)
                    else:
                        sl = min(sl, close_bid + saleh.SL_ATR_MULT * bar_atr)
                processed_hour += 1
                assert processed_hour < ctx["n1"]
            if not ctx["real1"][processed_hour]:
                fill_hour_quotes += 1
            if direction == 1 and bid <= sl:
                exit_quote = quote
                exit_type = "SL_TRAIL" if sl > initial_sl + 1e-9 else "SL_INIT"
                break
            if direction == -1 and ask >= sl:
                exit_quote = quote
                exit_type = "SL_TRAIL" if sl < initial_sl - 1e-9 else "SL_INIT"
                break

    assert exit_quote is not None
    exit_mid = (exit_quote[1] + exit_quote[2]) / 2
    net = (exit_quote[2] - entry[1]) if direction == 1 else (entry[2] - exit_quote[1])
    gross = direction * (exit_mid - entry_mid)
    exit_hour_index = int(exit_quote[0] // 3_600_000 - int(ctx["grid1"][0]))
    assert 0 <= exit_hour_index < ctx["n1"]
    assert int(ctx["d1"][exit_hour_index]) == int(ctx["d1"][i])
    trade = {"ny_date": str(int(ctx["d1"][i])), "year": int(ctx["d1"][i]) // 10000,
             "side": "BUY" if direction == 1 else "SELL",
             "signal_epoch_hour": int(ctx["grid1"][i]),
             "exit_epoch_hour": int(ctx["grid1"][exit_hour_index]),
             "entry_ms": entry[0], "exit_ms": exit_quote[0],
             "entry_bid": entry[2], "entry_ask": entry[1],
             "exit_bid": exit_quote[2], "exit_ask": exit_quote[1],
             "entry_spread_units": entry[1] - entry[2],
             "exit_spread_units": exit_quote[1] - exit_quote[2],
             "initial_sl": initial_sl, "final_sl": sl,
             "exit_type": exit_type, "unresolved": unresolved,
             "gross_mid_units": gross, "quote_net_units": net,
             "spread_drag_units": gross - net,
             "stress_units": net - 0.41,
             "hold_hours": (exit_quote[0] - entry[0]) / 3_600_000}
    return trade, {"invalid_quotes": invalid, "fill_hour_quotes": fill_hour_quotes,
                   "reason": "UNRESOLVED_EXIT" if unresolved else None}


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
        end_ms = end_by_day.setdefault(day, end_ms_for_nydate(day))
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


def metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0}
    net = frame["quote_net_units"].to_numpy(np.float64)
    equity = np.r_[0.0, np.cumsum(net)]
    return {"n": len(frame), "ny_dates": int(frame["ny_date"].nunique()),
            "gross_mid_mean_units": float(frame["gross_mid_units"].mean()),
            "quote_net_mean_units": float(net.mean()),
            "quote_net_total_units": float(net.sum()),
            "stress_mean_units": float(frame["stress_units"].mean()),
            "stress_total_units": float(frame["stress_units"].sum()),
            "win_rate": float(np.mean(net > 0)),
            "max_drawdown_units": float(np.max(np.maximum.accumulate(equity) - equity)),
            "median_entry_spread_units": float(frame["entry_spread_units"].median()),
            "p90_entry_spread_units": float(frame["entry_spread_units"].quantile(0.9)),
            "median_exit_spread_units": float(frame["exit_spread_units"].median()),
            "p90_exit_spread_units": float(frame["exit_spread_units"].quantile(0.9)),
            "mean_spread_drag_units": float(frame["spread_drag_units"].mean()),
            "median_hold_hours": float(frame["hold_hours"].median()),
            "unresolved": int(frame["unresolved"].sum()),
            "exit_type": {str(k): int(v) for k, v in frame["exit_type"].value_counts().items()},
            "side": {str(k): {"n": len(g), "quote_net_total_units": float(g["quote_net_units"].sum())}
                     for k, g in frame.groupby("side")}}


def bootstrap(frame: pd.DataFrame) -> dict:
    grouped = frame.groupby("ny_date", sort=True)["quote_net_units"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(np.float64)
    counts = grouped["count"].to_numpy(np.int32)
    rng = np.random.default_rng(SEED)
    means = np.empty(N_BOOT, np.float64)
    for start in range(0, N_BOOT, 250):
        end = min(start + 250, N_BOOT)
        picks = rng.integers(0, len(sums), size=(end - start, len(sums)))
        means[start:end] = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return {"date_blocks": len(sums), "p_mean_le_zero": float(np.mean(means <= 0)),
            "ci_95_low_units": float(np.quantile(means, 0.025)),
            "ci_95_high_units": float(np.quantile(means, 0.975))}


def main() -> None:
    assert not REPORT.exists(), "Frozen REPORT exists; never overwrite"
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
    gates = {"G-SRQ1": len(test) >= 150 and test["ny_date"].nunique() >= 150,
             "G-SRQ2": float(test["stress_units"].mean()) > 0,
             "G-SRQ3": boot["p_mean_le_zero"] <= 0.05,
             "G-SRQ4": all(yearly.get(str(y), {}).get("quote_net_total_units", 0) > 0 for y in (2024, 2025)),
             "G-SRQ5": funnel["unresolved_exits"] == 0 and same_ny_date}
    verdict = "PASS" if all(gates.values()) else "FAIL"
    result = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "spec_sha256": saleh.sha256_file(SPEC), "gate_sha256": saleh.sha256_file(GATE),
              "source_hashes": hashed, "m1_hash": bar_meta["source_m1_sha256"],
              "bar_model_test": bar_meta["test"], "signal_funnel": signal_funnel,
              "execution_funnel": funnel, "calibration": metrics(cal),
              "test": metrics(test), "bootstrap": boot, "yearly": yearly,
              "gates": gates, "verdict": verdict}
    (OUT / "run_meta.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["REPORT: XAUUSD SAME-DAY SALEH NATIVE BID/ASK REPLAY, HYPOTHESIS 70 PHASE 2A",
             f"Generated UTC: {result['generated_utc']}   NATIVE QUOTE VERDICT: {verdict}",
             "Same frozen #70 signal; real quote entries, stop crossings and 16:00 NY market flat.",
             "Native variable Bid/Ask charged plus separate extra 0.41-unit stress.",
             "Quote replay still assumes first quote fills and does not prove MT5 order execution.", "",
             f"SOURCE HASHES: {json.dumps(hashed, sort_keys=True)}",
             f"SIGNAL FUNNEL: {json.dumps(signal_funnel, sort_keys=True)}",
             f"EXECUTION FUNNEL: {json.dumps(funnel, sort_keys=True)}",
             f"BAR MODEL 2024+: {json.dumps(bar_meta['test'], sort_keys=True)}",
             f"QUOTE CALIBRATION 2020-23: {json.dumps(result['calibration'], sort_keys=True)}",
             f"QUOTE GATED TEST 2024+: {json.dumps(result['test'], sort_keys=True)}",
             f"DATE BOOTSTRAP: {json.dumps(boot, sort_keys=True)}",
             f"GATES: {json.dumps(gates, sort_keys=True)}", "YEARLY:"]
    lines += [f"  {year}: {json.dumps(value, sort_keys=True)}" for year, value in yearly.items()]
    lines += ["", f"OVERALL NATIVE QUOTE VERDICT: {verdict}",
              "PASS permits a separately frozen Model=4 MT5 order execution study, not a live edge claim.",
              "FAIL is final for this native-quote artifact; #70 M1 PASS remains only a bar-model screen."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "test": result["test"],
                      "bootstrap": boot, "gates": gates,
                      "execution_funnel": funnel}, indent=2))


if __name__ == "__main__":
    main()
