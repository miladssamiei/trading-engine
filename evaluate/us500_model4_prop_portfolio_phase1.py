"""Frozen H73: corrected real-tick ORB+SP2L portfolio and FTMO pace screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/us500_model4_prop_portfolio_v1"
SRC = ROOT / "data/results/execution_safe_model4_v1/stitched"
SCALES = (1, 2, 4, 8, 16, 32)
EXTRA_COST = 0.51
TARGET = 1000.0
FLOOR = -1000.0
DAILY_LIMIT = -500.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_leg(path: Path, name: str) -> tuple[pd.DataFrame, dict]:
    t = pd.read_csv(path, sep=";")
    t["open_ny"] = pd.to_datetime(t.open_time, format="%Y.%m.%d %H:%M", utc=True).dt.tz_convert("America/New_York").dt.date
    t["close_ny"] = pd.to_datetime(t.close_time, format="%Y.%m.%d %H:%M", utc=True).dt.tz_convert("America/New_York").dt.date
    t["profit"] = pd.to_numeric(t.profit, errors="raise")
    detail = {
        "name": name,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "trades": len(t),
        "profit_usd": round(float(t.profit.sum()), 2),
        "cross_ny_date": int((t.open_ny != t.close_ny).sum()),
        "duplicate_tickets": int(t.ticket.duplicated().sum()),
    }
    return t, detail


def bootstrap_p(stressed_daily: np.ndarray) -> float:
    rng = np.random.default_rng(20261009)
    n = len(stressed_daily)
    nonpositive = 0
    for _ in range(10000):
        nonpositive += float(stressed_daily[rng.integers(n, size=n)].mean() <= 0.0)
    return nonpositive / 10000.0


def replay(starts: list, daily: pd.DataFrame, k: int) -> dict:
    totals = {"pass": 0, "breach": 0, "timeout": 0}
    pass_days = []
    for start in starts:
        end = start + pd.Timedelta(days=180)
        window = daily.loc[(daily.index >= start) & (daily.index <= end)]
        cum = 0.0
        traded_dates = 0
        outcome = "timeout"
        for day, row in window.iterrows():
            delta = k * float(row.stressed_usd)
            cum += delta
            traded_dates += int(row.trade_count > 0)
            if delta <= DAILY_LIMIT or cum <= FLOOR:
                outcome = "breach"
                break
            if cum >= TARGET and traded_dates >= 4:
                outcome = "pass"
                pass_days.append((day - start).days)
                break
        totals[outcome] += 1
    n = len(starts)
    return {
        "k": k,
        "starts": n,
        **totals,
        "pass_rate": totals["pass"] / n,
        "breach_rate": totals["breach"] / n,
        "median_days_to_pass": float(np.median(pass_days)) if pass_days else None,
    }


def main() -> None:
    report_path = OUT / "REPORT_MODEL4_PROP_PORTFOLIO.txt"
    if report_path.exists():
        raise FileExistsError(f"Frozen report exists: {report_path}")
    orb, od = load_leg(SRC / "orb_execution_safe_trades.csv", "ORB")
    sp2l, sd = load_leg(SRC / "sp2l_execution_safe_trades.csv", "SP2L")
    universe_path = SRC / "orb_execution_safe_days.csv"
    days = pd.read_csv(universe_path, sep=";")
    valid_days = days.loc[(days.session_open > 0) | (days.prev_close > 0), "ny_date"]
    universe = pd.DatetimeIndex(pd.to_datetime(valid_days.astype(str), format="%Y%m%d")).sort_values()
    assert universe.is_unique and universe.is_monotonic_increasing

    expected = ((od, 450, 383.14), (sd, 1342, 182.75))
    reproduces = all(d["trades"] == count and abs(d["profit_usd"] - pnl) < 0.005 for d, count, pnl in expected)
    valid_dates = {d.date() for d in universe}
    out_of_universe = sum(int((~t.close_ny.isin(valid_dates)).sum()) for t in (orb, sp2l))
    valid_execution = all(d["cross_ny_date"] == 0 and d["duplicate_tickets"] == 0 for d in (od, sd)) and out_of_universe == 0
    g1 = reproduces and valid_execution
    if not g1:
        raise ValueError(f"Input integrity G-MP1 failed: {od} {sd} out_of_universe={out_of_universe}")

    combined = pd.concat([orb, sp2l], ignore_index=True)
    per_day = combined.groupby("close_ny", sort=True).agg(base_usd=("profit", "sum"), trade_count=("profit", "size"))
    per_day.index = pd.to_datetime(per_day.index)
    daily = per_day.reindex(universe, fill_value=0).rename_axis("ny_date")
    daily["stressed_usd"] = daily.base_usd - EXTRA_COST * daily.trade_count
    daily["year"] = daily.index.year
    gated = daily.loc[daily.index >= "2024-01-01"]
    calibration = daily.loc[daily.index < "2024-01-01"]
    yearly = daily.groupby("year")[["base_usd", "stressed_usd", "trade_count"]].sum()
    p = bootstrap_p(gated.stressed_usd.to_numpy(dtype=float))
    g2 = bool(gated.stressed_usd.mean() > 0 and p <= 0.05)
    g3 = bool((yearly.loc[[2024, 2025], "stressed_usd"] > 0).all())
    starts = [d for d in gated.index if d + pd.Timedelta(days=180) <= daily.index[-1]]
    pace = pd.DataFrame([replay(starts, daily, k) for k in SCALES])
    g4 = bool(((pace.pass_rate >= 0.70) & (pace.breach_rate <= 0.10)).any())
    gates = {"G-MP1": g1, "G-MP2": g2, "G-MP3": g3, "G-MP4": g4}

    def stats(frame: pd.DataFrame) -> dict:
        pnl = frame.stressed_usd.to_numpy(dtype=float)
        equity = np.cumsum(pnl)
        peak = np.maximum.accumulate(np.r_[0.0, equity])[1:]
        return {
            "days": len(frame),
            "trades": int(frame.trade_count.sum()),
            "base_total_usd": round(float(frame.base_usd.sum()), 2),
            "stressed_total_usd": round(float(frame.stressed_usd.sum()), 2),
            "stressed_daily_mean_usd": float(frame.stressed_usd.mean()),
            "stressed_daily_sharpe": float(np.sqrt(252) * pnl.mean() / pnl.std(ddof=1)) if pnl.std(ddof=1) > 0 else None,
            "stressed_max_close_drawdown_usd": float((peak - equity).max()),
        }

    metadata = {
        "source": [od, sd],
        "universe": {"path": str(universe_path.relative_to(ROOT)), "sha256": sha256(universe_path), "days": len(universe)},
        "frozen_spec_sha256": sha256(OUT / "SPEC_MODEL4_PROP_PORTFOLIO.txt"),
        "frozen_gate_sha256": sha256(OUT / "GATE_MODEL4_PROP_PORTFOLIO.txt"),
        "out_of_universe": out_of_universe,
        "calibration": stats(calibration),
        "gated_2024_plus": stats(gated),
        "bootstrap_p_nonpositive": p,
        "yearly": {str(y): {k: float(v) for k, v in row.items()} for y, row in yearly.to_dict("index").items()},
        "pace": json.loads(pace.to_json(orient="records")),
        "gates": gates,
        "verdict": "PASS" if all(gates.values()) else "FAIL",
        "caveat": "Optimistic close-based historical replay; no intraday floating-loss, FTMO feed, simultaneous MT5 portfolio execution, or forward confirmation.",
    }
    daily.to_csv(OUT / "model4_portfolio_daily.csv", float_format="%.6f")
    pace.to_csv(OUT / "challenge_180day_pace.csv", index=False)
    report_path.write_text(
        "REPORT: HYPOTHESIS 73 MODEL=4 ORB+SP2L PROP PORTFOLIO\n"
        + json.dumps(metadata, indent=2, allow_nan=False)
        + "\nFINAL PHASE-1 VERDICT: " + metadata["verdict"] + "\n",
        encoding="utf-8",
    )
    print("FINAL PHASE-1 VERDICT", metadata["verdict"])
    print("GATES", gates)
    print("GATED", metadata["gated_2024_plus"])
    print(pace.to_string(index=False))


if __name__ == "__main__":
    main()
