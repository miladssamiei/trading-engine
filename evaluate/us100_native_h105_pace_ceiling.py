"""Frozen-plan H105 necessary 180-day profit ceiling; no orders."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/us100_native_weekday_h105"
INPUT = OUT / "trades.csv"
EXPECTED = "ed74f84abe8d8a974a3d0434c2b20eb55bdaa169763a2abbfb0a498a050391d3"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def bounds(monday: str) -> tuple[int, int]:
    start_naive = pd.Timestamp(monday + " 09:30")
    end_naive = start_naive + pd.DateOffset(days=180)
    start = start_naive.tz_localize("America/New_York").timestamp()
    end = end_naive.tz_localize("America/New_York").timestamp()
    return int(start), int(end)


def main() -> None:
    csv_path = OUT / "pace_ceiling_windows.csv"
    report_path = OUT / "PACE_CEILING_REPORT_2026-09-26.txt"
    if csv_path.exists() or report_path.exists():
        raise FileExistsError("H105 pace ceiling artifact exists")
    if sha(INPUT) != EXPECTED:
        raise ValueError("Frozen H105 ledger hash changed")
    trades = pd.read_csv(INPUT)
    trades = trades[trades.year >= 2024].sort_values("entry_epoch").copy()
    if trades.monday.duplicated().any() or (trades.prior_atr20 <= 0).any():
        raise ValueError("Invalid H105 Monday uniqueness or ATR")
    trades["R"] = trades.stress_net_units / (2 * trades.prior_atr20)
    last_exit = int(trades.exit_epoch.max())
    windows = []
    for monday in trades.monday:
        start, end = bounds(monday)
        if end > last_exit:
            continue
        group = trades[(trades.entry_epoch >= start) & (trades.exit_epoch <= end)]
        windows.append({"start_monday": monday, "start_year": int(monday[:4]),
                        "start_epoch": start, "end_epoch": end,
                        "trades": len(group), "sum_R": float(group.R.sum()),
                        "necessary_1pct": int(group.R.sum() >= 15),
                        "necessary_2pct": int(group.R.sum() >= 7.5),
                        "necessary_3pct": int(group.R.sum() >= 5)})
    if not windows:
        raise ValueError("No complete 180-day H105 windows")
    frame = pd.DataFrame(windows)
    frame.to_csv(csv_path, index=False)
    summary = {}
    for risk in (1, 2, 3):
        col = f"necessary_{risk}pct"
        summary[f"{risk}pct"] = {"necessary_count": int(frame[col].sum()),
                                 "starts": len(frame), "necessary_share": float(frame[col].mean()),
                                 "by_start_year": {str(y): {"starts": len(g),
                                                            "necessary_count": int(g[col].sum()),
                                                            "necessary_share": float(g[col].mean())}
                                                   for y, g in frame.groupby("start_year")}}
    best = frame.loc[frame.sum_R.idxmax()]
    median_stop = float(np.median(2 * trades.prior_atr20))
    outcome = {"verdict": "NECESSARY_CEILING_ONLY", "ledger_sha256": EXPECTED,
               "window_csv_sha256": sha(csv_path), "oos_trades": len(trades),
               "last_exit_epoch": last_exit, "fully_observed_starts": len(frame),
               "max_sum_R": float(best.sum_R), "max_sum_R_start": str(best.start_monday),
               "median_sum_R": float(frame.sum_R.median()),
               "risk_rows": summary,
               "annual_mean_R": {str(y): float(g.R.mean()) for y, g in trades.groupby("year")},
               "median_initial_stop_distance_index_units": median_stop,
               "illustrative_lots_for_USD_stop_risk_at_median_stop":
                   {str(100 * risk): float((100 * risk) / median_stop) for risk in (1, 2, 3)},
               "phase2": "NOT_RUN", "h105_original_verdict": "FINAL_FAIL"}
    report_path.write_text("H105 US100 NATIVE 180-DAY NECESSARY PROFIT CEILING\n" +
                           json.dumps(outcome, indent=2, allow_nan=False) + "\n" +
                           "Necessary total-R only: no stage timing, floating equity, FX, real fills or forward proof.\n",
                           encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
