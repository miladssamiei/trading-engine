"""Frozen H80: FirstRate VXX first cash minutes to next-hour US500."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/us500_vxx_frd_morning_lead_v1"
VXX_ZIP = ROOT / "data/results/project_status/vxx_frd_source_probe/VXX_1min_sample_firstratedata.zip"
US500 = ROOT / "data/results/orb_2021_present/us500tick_m5.csv"
EXPECTED_VXX_SHA = "ce49f06c3fb0cae20457df6a65bba209cd1f675819123816f46af84a90748da5"
EXPECTED_US500_SHA = "a8ee68f1a0010a1be9704c0d59fd5382aa798846cdae4d519b799ae21ec66c2d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_vxx() -> tuple[pd.DataFrame, dict]:
    digest = sha256(VXX_ZIP)
    if digest != EXPECTED_VXX_SHA:
        raise ValueError("Frozen FirstRate VXX archive hash mismatch")
    with zipfile.ZipFile(VXX_ZIP) as archive:
        if archive.namelist() != ["VXX_1min_firstratedata.csv"]:
            raise ValueError("Frozen FirstRate archive shape changed")
        vxx = pd.read_csv(archive.open("VXX_1min_firstratedata.csv"), header=0,
                          names=["wall_time", "open", "high", "low", "close", "volume"])
    if len(vxx) != 121604:
        raise ValueError("Frozen VXX archive row count changed")
    wall = pd.to_datetime(vxx.wall_time, format="%Y-%m-%d %H:%M:%S", errors="raise")
    if wall.duplicated().any() or not wall.is_monotonic_increasing:
        raise ValueError("VXX NY-wall timestamps are duplicated or unsorted")
    for field in ("open", "high", "low", "close", "volume"):
        vxx[field] = pd.to_numeric(vxx[field], errors="raise")
    fields = ["open", "high", "low", "close", "volume"]
    vxx["valid_ohlc"] = np.isfinite(vxx[fields].to_numpy(dtype=float)).all(axis=1) & (vxx[fields] > 0).all(axis=1)
    vxx["date"] = wall.dt.date
    vxx["wall_minute"] = wall.dt.hour * 60 + wall.dt.minute
    vxx = vxx.loc[(vxx.wall_minute >= 570) & (vxx.wall_minute <= 584)].copy()
    grouped = vxx.groupby("date", sort=True)
    rows = []
    for day, bars in grouped:
        complete = len(bars) == 15 and bars.wall_minute.nunique() == 15 and bool(bars.valid_ohlc.all())
        first = bars.loc[bars.wall_minute.eq(570)]
        last = bars.loc[bars.wall_minute.eq(584)]
        rows.append({"date": day, "vxx_complete": bool(complete and len(first) == 1 and len(last) == 1),
                     "vxx_open": float(first.open.iloc[0]) if len(first) == 1 else np.nan,
                     "vxx_close": float(last.close.iloc[0]) if len(last) == 1 else np.nan,
                     "vxx_m1_count": len(bars)})
    day_frame = pd.DataFrame(rows).set_index("date")
    return day_frame, {"sha256": digest, "rows": 121604, "first_wall": str(wall.iloc[0]),
                       "last_wall": str(wall.iloc[-1]), "dates_with_any_signal_bar": len(day_frame),
                       "dates_with_15_valid_bars": int(day_frame.vxx_complete.sum())}


def load_us500() -> tuple[pd.DataFrame, str]:
    digest = sha256(US500)
    if digest != EXPECTED_US500_SHA:
        raise ValueError("Frozen US500 BID M5 hash mismatch")
    tape = pd.read_csv(US500, usecols=["time", "open"])
    ny = pd.to_datetime(tape.time, unit="s", utc=True).dt.tz_convert("America/New_York")
    minute = ny.dt.hour * 60 + ny.dt.minute
    keep = minute.isin([570, 585, 645])
    tape = tape.loc[keep, ["open"]].copy()
    tape["date"] = ny.loc[keep].dt.date
    tape["wall_minute"] = minute.loc[keep]
    if tape.duplicated(["date", "wall_minute"]).any():
        raise ValueError("Duplicate US500 M5 endpoint")
    wide = tape.pivot(index="date", columns="wall_minute", values="open").rename(
        columns={570: "cash_open_bid", 585: "entry_bid", 645: "exit_bid"})
    return wide, digest


def stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0, "base_total_points": 0.0}
    return {"trades": int(len(trades)), "gross_mean_points": float(trades.gross_points.mean()),
            "base_mean_points": float(trades.base_net_points.mean()),
            "base_total_points": float(trades.base_net_points.sum()),
            "stress_mean_points": float(trades.stress_net_points.mean()),
            "stress_total_points": float(trades.stress_net_points.sum()),
            "base_win_rate": float((trades.base_net_points > 0).mean()),
            "longs": int((trades.direction == 1).sum()), "shorts": int((trades.direction == -1).sum())}


def bootstrap_p(values: np.ndarray) -> float:
    rng = np.random.default_rng(20261016)
    nonpositive = 0
    for _ in range(10000):
        nonpositive += int(values[rng.integers(len(values), size=len(values))].mean() <= 0)
    return nonpositive / 10000


def main() -> None:
    report_path = OUT / "REPORT_VXX_FRD_MORNING_LEAD.txt"
    if report_path.exists():
        raise FileExistsError(f"Frozen report already exists: {report_path}")
    vxx, vxx_source = load_vxx()
    us500, us500_sha = load_us500()
    days = [stamp.date() for stamp in pd.bdate_range("2022-09-30", "2023-09-29")]
    records = []
    for day in days:
        rec = {"ny_date": day.isoformat(), "quarter": f"{day.year}Q{(day.month - 1) // 3 + 1}"}
        if day not in vxx.index or not bool(vxx.loc[day, "vxx_complete"]):
            rec["skip_reason"] = "MISSING_OR_INVALID_VXX_15_BARS"
        else:
            v = vxx.loc[day]
            rec.update(vxx_open=float(v.vxx_open), vxx_close=float(v.vxx_close), vxx_m1_count=int(v.vxx_m1_count))
            if v.vxx_close == v.vxx_open:
                rec["skip_reason"] = "ZERO_VXX_SIGNAL"
            else:
                rec["direction"] = -1 if v.vxx_close > v.vxx_open else 1
                if day not in us500.index or pd.isna(us500.loc[day].get("cash_open_bid")):
                    rec["skip_reason"] = "MISSING_US500_CASH_OPEN"
                elif pd.isna(us500.loc[day].get("entry_bid")):
                    rec["skip_reason"] = "MISSING_US500_ENTRY"
                elif pd.isna(us500.loc[day].get("exit_bid")):
                    rec["skip_reason"] = "MISSING_US500_EXIT"
                else:
                    rec["cash_open_bid"] = float(us500.loc[day, "cash_open_bid"])
                    rec["entry_bid"] = float(us500.loc[day, "entry_bid"])
                    rec["exit_bid"] = float(us500.loc[day, "exit_bid"])
                    rec["gross_points"] = rec["direction"] * (rec["exit_bid"] - rec["entry_bid"])
                    rec["base_net_points"] = rec["gross_points"] - 0.51
                    rec["stress_net_points"] = rec["gross_points"] - 1.02
                    rec["skip_reason"] = ""
        records.append(rec)
    signals = pd.DataFrame(records)
    trades = signals.loc[signals.skip_reason.eq("")].copy()
    calibration = trades.loc[trades.quarter.str.startswith("2022")]
    gated = trades.loc[trades.quarter.isin(["2023Q1", "2023Q2", "2023Q3"])]
    quarterly = {name: stats(part) for name, part in trades.groupby("quarter")}
    pval = bootstrap_p(gated.base_net_points.to_numpy(dtype=float)) if len(gated) else 1.0
    gates = {
        "G-VF1": bool(len(gated) >= 150 and gated.ny_date.nunique() == len(gated)),
        "G-VF2": bool(len(gated) and gated.stress_net_points.mean() > 0),
        "G-VF3": bool(pval <= 0.01),
        "G-VF4": all(q in quarterly and quarterly[q]["base_total_points"] > 0 for q in ("2023Q1", "2023Q2", "2023Q3")),
    }
    result = {"vxx_source": vxx_source, "us500_sha256": us500_sha,
              "spec_sha256": sha256(OUT / "SPEC_VXX_FRD_MORNING_LEAD.txt"),
              "gate_sha256": sha256(OUT / "GATE_VXX_FRD_MORNING_LEAD.txt"),
              "candidate_dates": len(signals), "executed_dates": len(trades),
              "skip_reasons": signals.loc[signals.skip_reason.ne(""), "skip_reason"].value_counts().to_dict(),
              "calibration_2022q4": stats(calibration), "gated_2023q1_q3": stats(gated),
              "bootstrap_p_nonpositive": pval, "quarterly": quarterly,
              "gates": gates, "verdict": "PASS" if all(gates.values()) else "FAIL",
              "caveat": "Vendor-derived 2022-23 VXX M1 archive and US500 BID M5 fixed-spread proxy only; no recent year, native execution, account feed, stop or prop pace validation."}
    signals.to_csv(OUT / "vxx_signals.csv", index=False)
    trades.to_csv(OUT / "vxx_trades.csv", index=False)
    (OUT / "run_meta.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_path.write_text("REPORT: HYPOTHESIS 80 VXX FRD MORNING LEAD\n"
                           + json.dumps(result, indent=2, allow_nan=False)
                           + "\nFINAL PHASE-1 VERDICT: " + result["verdict"] + "\n", encoding="utf-8")
    print("FINAL PHASE-1 VERDICT", result["verdict"])
    print("GATES", gates)
    print("FUNNEL", len(signals), len(trades), result["skip_reasons"])
    print("GATED", result["gated_2023q1_q3"], "p", pval)
    print("QUARTERLY", {quarter: round(row["base_total_points"], 3) for quarter, row in quarterly.items()})


if __name__ == "__main__":
    main()
