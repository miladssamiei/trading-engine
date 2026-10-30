"""Hypothesis 61: M1 path and spread-aware proxy for the frozen ATR limit rule."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/atr_band_intraday_m1_v1"
SPEC = OUT / "SPEC_ATR_BAND_INTRADAY_M1.txt"
GATE = OUT / "GATE_ATR_BAND_INTRADAY_M1.txt"
ASSETS = {
    "US500": {
        "m5": ROOT / "data/results/orb_2021_present/us500tick_m5.csv",
        "m1": ROOT / "data/results/vwap_2021_2026/us500tick_m1.csv",
        "spread": 0.51,
        "session_open": 570,
        "first_entry": 575,
        "last_entry": 950,
        "exit_wall": 959,
    },
    "XAUUSD": {
        "m5": ROOT / "data/results/xauusd_onboarding/xauusdtick_m5.csv",
        "m1": ROOT / "data/results/xauusd_onboarding/xauusdtick_m1.csv",
        "spread": 0.41,
        "session_open": 500,
        "first_entry": 500,
        "last_entry": 800,
        "exit_wall": 809,
    },
}
SEED = 20260927
N_BOOT = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, np.ndarray]:
    df = pd.read_csv(path, usecols=["time", "open", "high", "low", "close"])
    assert df["time"].is_monotonic_increasing and df["time"].is_unique
    assert (df[["open", "high", "low", "close"]] > 0).all().all()
    time = df["time"].to_numpy(np.int64)
    ny = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("America/New_York")
    return {
        "time": time,
        "open": df["open"].to_numpy(np.float64),
        "high": df["high"].to_numpy(np.float64),
        "low": df["low"].to_numpy(np.float64),
        "close": df["close"].to_numpy(np.float64),
        "wall": (ny.dt.hour * 60 + ny.dt.minute).to_numpy(np.int16),
        "date": (ny.dt.year * 10000 + ny.dt.month * 100 + ny.dt.day).to_numpy(np.int32),
    }


def source_atr(m5: dict[str, np.ndarray]) -> np.ndarray:
    high, low, close = m5["high"], m5["low"], m5["close"]
    tr = np.empty(len(close), np.float64)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum.reduce(
        [high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])]
    )
    atr = np.full(len(close), np.nan, np.float64)
    atr[14] = tr[:15].mean()
    for i in range(15, len(close)):
        atr[i] = (13.0 * atr[i - 1] + tr[i]) / 14.0
    return atr


def walk_exit(
    minute_open: np.ndarray,
    minute_high: np.ndarray,
    minute_low: np.ndarray,
    minute_close: np.ndarray,
    fill_index: int,
    last_index: int,
    band: float,
    atr: float,
    spread: float,
) -> tuple[int, float, str]:
    stop = band - 1.5 * atr
    target = band + 2.0 * atr
    open_fill = minute_open[fill_index] + spread <= band
    for j in range(fill_index, last_index + 1):
        if minute_low[j] <= stop:
            price = stop if j == fill_index else min(stop, minute_open[j])
            return j, float(price), "STOP"
        if minute_high[j] >= target and (j > fill_index or open_fill):
            return j, float(target), "TARGET"
    return last_index, float(minute_close[last_index]), "TIME"


def evaluate_asset(name: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    m5 = load(cfg["m5"])
    m1 = load(cfg["m1"])
    atr = source_atr(m5)
    spread = cfg["spread"]
    t5, d5, w5 = m5["time"], m5["date"], m5["wall"]
    t1, d1, w1 = m1["time"], m1["date"], m1["wall"]
    session_days = set(map(int, d5[w5 == cfg["session_open"]]))
    end_candidates = np.where(w1 == cfg["exit_wall"])[0]
    end_map = {int(d1[j]): int(j) for j in end_candidates}
    valid_days = session_days & end_map.keys()
    assert len(end_map) == len(end_candidates)
    rows: list[dict] = []
    stats = {
        "m5_rows": int(len(t5)),
        "m1_rows": int(len(t1)),
        "session_days": len(session_days),
        "valid_days": len(valid_days),
        "eligible_bars": 0,
        "orders_unfilled": 0,
        "orders_filled": 0,
        "no_printed_m1_in_order_window": 0,
    }
    i = 15
    while i < len(t5):
        wall, date = int(w5[i]), int(d5[i])
        if date not in valid_days or not cfg["first_entry"] <= wall <= cfg["last_entry"]:
            i += 1
            continue
        a = float(atr[i - 1])
        if not np.isfinite(a) or a <= 0:
            i += 1
            continue
        stats["eligible_bars"] += 1
        band = float(m5["close"][i - 1] - 3.1 * a)
        left = int(np.searchsorted(t1, t5[i], side="left"))
        right = int(np.searchsorted(t1, t5[i] + 300, side="left"))
        if left == right:
            stats["no_printed_m1_in_order_window"] += 1
            stats["orders_unfilled"] += 1
            i += 1
            continue
        assert np.all(d1[left:right] == date)
        touch = np.flatnonzero(m1["low"][left:right] + spread <= band)
        if len(touch) == 0:
            stats["orders_unfilled"] += 1
            i += 1
            continue
        fill_j = left + int(touch[0])
        last_j = end_map[date]
        assert fill_j <= last_j and int(d1[fill_j]) == date
        exit_j, exit_bid, reason = walk_exit(
            m1["open"], m1["high"], m1["low"], m1["close"], fill_j, last_j, band, a, spread
        )
        stats["orders_filled"] += 1
        net = exit_bid - band
        rows.append(
            {
                "instrument": name,
                "ny_date": date,
                "year": date // 10000,
                "order_m5_epoch": int(t5[i]),
                "entry_m1_epoch": int(t1[fill_j]),
                "exit_m1_epoch": int(t1[exit_j]),
                "hold_printed_m1_bars": exit_j - fill_j + 1,
                "entry_same_minute_exit": int(exit_j == fill_j),
                "band_ask_entry": band,
                "entry_at_m1_open": int(m1["open"][fill_j] + spread <= band),
                "atr_ref": a,
                "stop_bid": band - 1.5 * a,
                "target_bid": band + 2.0 * a,
                "exit_bid": exit_bid,
                "exit_reason": reason,
                "gross_pt": net + spread,
                "net_pt": net,
                "double_cost_net_pt": net - spread,
            }
        )
        i = int(np.searchsorted(t5, t1[exit_j], side="right"))

    trades = pd.DataFrame(rows)
    assert len(trades) == stats["orders_filled"]
    assert stats["eligible_bars"] == stats["orders_filled"] + stats["orders_unfilled"]
    assert (trades["entry_m1_epoch"] <= trades["exit_m1_epoch"]).all()
    assert (trades["entry_m1_epoch"].diff().dropna() > 0).all()
    assert (trades["exit_m1_epoch"].iloc[:-1].to_numpy() < trades["entry_m1_epoch"].iloc[1:].to_numpy()).all()
    assert np.allclose(trades["gross_pt"] - trades["net_pt"], spread)
    assert np.allclose(trades["net_pt"] - trades["double_cost_net_pt"], spread)
    stats["m5_sha256"] = sha256(cfg["m5"])
    stats["m1_sha256"] = sha256(cfg["m1"])
    return trades, stats


def metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0}
    pnl = frame["net_pt"].to_numpy(np.float64)
    equity = np.r_[0.0, np.cumsum(pnl)]
    return {
        "n": int(len(frame)),
        "gross_mean": float(frame["gross_pt"].mean()),
        "net_mean": float(pnl.mean()),
        "double_cost_mean": float(frame["double_cost_net_pt"].mean()),
        "net_total": float(pnl.sum()),
        "net_win_rate": float(np.mean(pnl > 0)),
        "max_drawdown": float(np.max(np.maximum.accumulate(equity) - equity)),
        "same_entry_m1_exit_fraction": float(frame["entry_same_minute_exit"].mean()),
        "exit_counts": {str(k): int(v) for k, v in frame["exit_reason"].value_counts().items()},
    }


def date_bootstrap(frame: pd.DataFrame) -> dict:
    by_date = frame.groupby("ny_date", sort=True)["net_pt"].agg(["sum", "count"])
    sums = by_date["sum"].to_numpy(np.float64)
    counts = by_date["count"].to_numpy(np.int32)
    rng = np.random.default_rng(SEED)
    means = np.empty(N_BOOT)
    for start in range(0, N_BOOT, 250):
        end = min(start + 250, N_BOOT)
        draws = rng.integers(0, len(sums), size=(end - start, len(sums)))
        means[start:end] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "date_blocks": int(len(sums)),
        "p_mean_le_zero": float(np.mean(means <= 0)),
        "ci_95_low": float(np.quantile(means, 0.025)),
        "ci_95_high": float(np.quantile(means, 0.975)),
    }


def main() -> None:
    spec_text = SPEC.read_text(encoding="utf-8")
    gate_lines = [line for line in GATE.read_text(encoding="utf-8").splitlines() if line]
    assert len(gate_lines) == 4 and all(line in spec_text for line in gate_lines)
    results = {}
    for name, cfg in ASSETS.items():
        trades, funnel = evaluate_asset(name, cfg)
        trades.to_csv(OUT / f"atr_band_{name.lower()}_trades.csv", index=False, float_format="%.6f")
        cal = trades[trades["year"] < 2024]
        test = trades[trades["year"] >= 2024]
        yearly = {str(y): metrics(part) for y, part in trades.groupby("year")}
        boot = date_bootstrap(test)
        g1 = len(test) >= 150
        g2 = g1 and float(test["double_cost_net_pt"].mean()) > 0
        g3 = g2 and boot["p_mean_le_zero"] <= 0.025
        g4 = g3 and yearly.get("2024", {}).get("net_total", 0) > 0 and yearly.get("2025", {}).get("net_total", 0) > 0
        results[name] = {
            "funnel": funnel,
            "calibration": metrics(cal),
            "test": metrics(test),
            "bootstrap": boot,
            "yearly": yearly,
            "gates": {"G-AB1": g1, "G-AB2": g2, "G-AB3": g3, "G-AB4": g4},
        }
    gates = {gate: any(asset["gates"][gate] for asset in results.values()) for gate in ["G-AB1", "G-AB2", "G-AB3", "G-AB4"]}
    passed = gates["G-AB4"]
    report_data = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": sha256(SPEC),
        "gate_sha256": sha256(GATE),
        "assets": results,
        "gates": gates,
        "verdict": "PASS" if passed else "FAIL",
    }
    (OUT / "run_meta.json").write_text(json.dumps(report_data, indent=2) + "\n", encoding="utf-8")
    lines = [
        "REPORT: INTRADAY ATR LOWER-BAND M1 PATH AUDIT, HYPOTHESIS 61",
        f"Generated UTC: {report_data['generated_utc']}   PHASE-1 VERDICT: {report_data['verdict']}",
        "Frozen M5 ATR-band buy limit; fixed Ask proxy for touches; pessimistic M1 entry-minute path; same-day forced exit.",
        "Historical data were exposed in H22; this study is neither a virgin holdout nor real Bid/Ask tick execution.",
        "",
    ]
    for name, item in results.items():
        lines.extend(
            [
                name,
                f"FUNNEL: {json.dumps(item['funnel'], sort_keys=True)}",
                f"CALIBRATION (descriptive): {json.dumps(item['calibration'], sort_keys=True)}",
                f"TEST 2024+ (gated): {json.dumps(item['test'], sort_keys=True)}",
                f"BOOTSTRAP (date blocks): {json.dumps(item['bootstrap'], sort_keys=True)}",
                f"ASSET GATES: {json.dumps(item['gates'], sort_keys=True)}",
                "YEARLY:",
            ]
        )
        for year, value in item["yearly"].items():
            lines.append(f"  {year}: {json.dumps(value, sort_keys=True)}")
        lines.append("")
    lines.extend(
        [
            f"OVERALL GATES: {json.dumps(gates, sort_keys=True)}",
            f"OVERALL PHASE-1 VERDICT: {report_data['verdict']}",
            "A PASS only permits a new predeclared Model=4 real-tick execution study.",
            "A FAIL is final; no post-hoc time, instrument, year, hold, or exit subset may rescue this artifact.",
            "Bid M1 bars plus fixed spread cannot reveal the exact Ask order trigger or within-minute order path.",
        ]
    )
    (OUT / "REPORT_ATR_BAND_INTRADAY_M1.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": report_data["verdict"], "assets": {name: {"test": item["test"], "bootstrap": item["bootstrap"], "gates": item["gates"]} for name, item in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
