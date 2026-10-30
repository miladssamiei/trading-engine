"""H105 frozen-ledger 360/540-day necessary-profit sensitivity only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/us100_native_weekday_h105"
LEDGER = OUT / "trades.csv"
EXPECTED = "ed74f84abe8d8a974a3d0434c2b20eb55bdaa169763a2abbfb0a498a050391d3"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def bound(monday: str, horizon_days: int) -> tuple[int, int]:
    wall = pd.Timestamp(monday + " 09:30")
    return (int(wall.tz_localize("America/New_York").timestamp()),
            int((wall + pd.DateOffset(days=horizon_days)).tz_localize("America/New_York").timestamp()))


def main() -> None:
    csv_path = OUT / "longer_horizon_ceiling_windows.csv"
    report_path = OUT / "LONGER_HORIZON_CEILING_REPORT_2026-09-26.txt"
    if csv_path.exists() or report_path.exists():
        raise FileExistsError("H105 longer-horizon artifact exists")
    if sha(LEDGER) != EXPECTED:
        raise ValueError("Frozen H105 ledger hash mismatch")
    trades = pd.read_csv(LEDGER)
    trades = trades[trades.year >= 2024].sort_values("entry_epoch").copy()
    if trades.monday.duplicated().any() or (trades.prior_atr20 <= 0).any():
        raise ValueError("Invalid H105 trade chronology or ATR")
    r = trades.stress_net_units.to_numpy(float) / (2 * trades.prior_atr20.to_numpy(float))
    entries = trades.entry_epoch.to_numpy(np.int64)
    exits = trades.exit_epoch.to_numpy(np.int64)
    last_exit = int(exits.max())
    rows = []
    for horizon in (360, 540):
        for monday in trades.monday:
            start, end = bound(monday, horizon)
            if end > last_exit:
                continue
            mask = (entries >= start) & (exits <= end)
            total = float(r[mask].sum())
            rows.append({"horizon_days": horizon, "start_monday": monday,
                         "start_year": int(monday[:4]), "start_epoch": start,
                         "end_epoch": end, "trades": int(mask.sum()), "sum_R": total,
                         "necessary_1pct": int(total >= 15),
                         "necessary_2pct": int(total >= 7.5),
                         "necessary_3pct": int(total >= 5)})
    frame = pd.DataFrame(rows)
    if frame.empty or set(frame.horizon_days) != {360, 540}:
        raise ValueError("Insufficient complete horizons")
    frame.to_csv(csv_path, index=False)
    horizons = {}
    for days, group in frame.groupby("horizon_days"):
        best = group.loc[group.sum_R.idxmax()]
        entry = {"starts": int(len(group)), "median_sum_R": float(group.sum_R.median()),
                 "max_sum_R": float(best.sum_R), "max_start": str(best.start_monday),
                 "rows": {}}
        for risk in (1, 2, 3):
            col = f"necessary_{risk}pct"
            entry["rows"][f"{risk}pct"] = {
                "necessary_count": int(group[col].sum()),
                "necessary_share": float(group[col].mean()),
                "by_start_year": {str(y): {"starts": int(len(cohort)),
                                           "necessary_count": int(cohort[col].sum())}
                                  for y, cohort in group.groupby("start_year")}}
        horizons[str(days)] = entry
    result = {"input_sha256": EXPECTED, "window_csv_sha256": sha(csv_path),
              "horizons": horizons, "h105_original_verdict": "FINAL_FAIL",
              "phase2": "NOT_RUN"}
    report_path.write_text("H105 NATIVE US100 360/540-DAY NECESSARY PROFIT SENSITIVITY\n" +
                           json.dumps(result, indent=2, allow_nan=False) + "\n" +
                           "Necessary profit only; no separate stage/risk replay or paid-account proof.\n",
                           encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
