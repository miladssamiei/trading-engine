"""Frozen H77: completed Bitcoin weekend sign to Monday US500 cash return."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/us500_btc_weekend_monday_v1"
US500 = ROOT / "data/results/orb_2021_present/us500tick_m5.csv"
EXPECTED_US500_SHA = "a8ee68f1a0010a1be9704c0d59fd5382aa798846cdae4d519b799ae21ec66c2d"
FIELDS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trade_count", "taker_base", "taker_quote", "ignore"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_btc() -> tuple[pd.DataFrame, dict]:
    manifest_path = OUT / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["records"]
    if len(records) != 64 or any(rec["status"] != "OK" for rec in records):
        raise ValueError("Incomplete frozen Binance source manifest")
    frames = []
    for rec in records:
        path = ROOT / rec["file"]
        if sha256(path) != rec["sha256"] or rec["sha256"] != rec["published_sha256"]:
            raise ValueError(f"Binance source checksum mismatch: {path}")
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != 1 or not names[0].endswith(".csv"):
                raise ValueError(f"Invalid Binance archive shape: {path}")
            frame = pd.read_csv(archive.open(names[0]), header=None, names=FIELDS)
        if len(frame) != rec["rows"]:
            raise ValueError(f"Binance source row mismatch: {path}")
        frames.append(frame)
    btc = pd.concat(frames, ignore_index=True)
    raw = pd.to_numeric(btc.open_time, errors="raise").astype("int64")
    ms = raw < 10**14
    utc = pd.Series(pd.NaT, index=btc.index, dtype="datetime64[ns, UTC]")
    utc.loc[ms] = pd.to_datetime(raw.loc[ms], unit="ms", utc=True)
    utc.loc[~ms] = pd.to_datetime(raw.loc[~ms], unit="us", utc=True)
    if utc.isna().any() or utc.duplicated().any() or not utc.is_monotonic_increasing:
        raise ValueError("BTC hour timestamp integrity failed")
    if not ((utc.dt.minute == 0) & (utc.dt.second == 0)).all():
        raise ValueError("BTC source has non-hour bar opens")
    btc["close"] = pd.to_numeric(btc.close, errors="raise")
    btc["volume"] = pd.to_numeric(btc.volume, errors="raise")
    btc["trade_count"] = pd.to_numeric(btc.trade_count, errors="raise")
    if not np.isfinite(btc[["close", "volume", "trade_count"]].to_numpy(dtype=float)).all():
        raise ValueError("BTC source has nonfinite values")
    if (btc.close <= 0).any():
        raise ValueError("BTC source has nonpositive close")
    btc["ny"] = utc.dt.tz_convert("America/New_York")
    btc["date"] = btc.ny.dt.date
    btc["weekday"] = btc.ny.dt.weekday
    btc["hour"] = btc.ny.dt.hour
    source = {
        "manifest_sha256": sha256(manifest_path),
        "months": len(records), "rows": len(btc),
        "first_utc": str(utc.iloc[0]), "last_utc": str(utc.iloc[-1]),
    }
    return btc, source


def load_us500() -> tuple[pd.DataFrame, str]:
    source_hash = sha256(US500)
    if source_hash != EXPECTED_US500_SHA:
        raise ValueError("Frozen US500 BID M5 source hash mismatch")
    tape = pd.read_csv(US500, usecols=["time", "open"])
    ny = pd.to_datetime(tape.time, unit="s", utc=True).dt.tz_convert("America/New_York")
    keep = ny.dt.weekday.eq(0) & (((ny.dt.hour == 9) & (ny.dt.minute == 30)) | ((ny.dt.hour == 15) & (ny.dt.minute == 55)))
    tape = tape.loc[keep, ["open"]].copy()
    tape["date"] = ny.loc[keep].dt.date
    tape["wall"] = ny.loc[keep].dt.hour * 60 + ny.loc[keep].dt.minute
    if tape.duplicated(["date", "wall"]).any():
        raise ValueError("Duplicate US500 Monday M5 endpoint")
    wide = tape.pivot(index="date", columns="wall", values="open").rename(columns={570: "entry_bid", 955: "exit_bid"})
    return wide, source_hash


def stats(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"trades": 0, "base_total_points": 0.0}
    return {
        "trades": int(len(frame)),
        "gross_mean_points": float(frame.gross_points.mean()),
        "base_mean_points": float(frame.base_net_points.mean()),
        "base_total_points": float(frame.base_net_points.sum()),
        "stress_mean_points": float(frame.stress_net_points.mean()),
        "stress_total_points": float(frame.stress_net_points.sum()),
        "base_win_rate": float((frame.base_net_points > 0).mean()),
        "longs": int((frame.direction == 1).sum()),
        "shorts": int((frame.direction == -1).sum()),
    }


def bootstrap_p(values: np.ndarray) -> float:
    rng = np.random.default_rng(20261013)
    nonpositive = 0
    for _ in range(10000):
        nonpositive += int(values[rng.integers(len(values), size=len(values))].mean() <= 0)
    return nonpositive / 10000


def main() -> None:
    report_path = OUT / "REPORT_BTC_WEEKEND_MONDAY.txt"
    if report_path.exists():
        raise FileExistsError(f"Frozen report already exists: {report_path}")
    btc, source = load_btc()
    endpoints, us500_sha = load_us500()
    selected = btc.loc[((btc.weekday == 4) | (btc.weekday == 6)) & (btc.hour == 16), ["date", "weekday", "close", "volume", "trade_count"]]
    if selected.duplicated(["date", "weekday"]).any():
        raise ValueError("Duplicate NY weekend signal bar")
    friday = selected.loc[selected.weekday == 4].set_index("date")
    sunday = selected.loc[selected.weekday == 6].set_index("date")
    monday_dates = [ts.date() for ts in pd.date_range("2021-01-04", "2026-04-27", freq="W-MON")]
    records = []
    for monday in monday_dates:
        fri_date = monday - timedelta(days=3)
        sun_date = monday - timedelta(days=1)
        rec = {"ny_date": monday.isoformat(), "friday_date": fri_date.isoformat(), "sunday_date": sun_date.isoformat()}
        if fri_date not in friday.index or sun_date not in sunday.index:
            rec["skip_reason"] = "MISSING_BTC_BAR"
        else:
            f = friday.loc[fri_date]
            s = sunday.loc[sun_date]
            rec.update(friday_btc_close=float(f.close), sunday_btc_close=float(s.close),
                       friday_btc_volume=float(f.volume), sunday_btc_volume=float(s.volume),
                       friday_btc_trades=int(f.trade_count), sunday_btc_trades=int(s.trade_count))
            if min(f.volume, s.volume, f.trade_count, s.trade_count) <= 0:
                rec["skip_reason"] = "NO_BTC_TRADE"
            elif s.close == f.close:
                rec["skip_reason"] = "ZERO_BTC_SIGNAL"
            else:
                rec["direction"] = 1 if s.close > f.close else -1
                rec["btc_weekend_pct"] = 100 * (s.close / f.close - 1)
                if monday not in endpoints.index or pd.isna(endpoints.loc[monday].get("entry_bid")):
                    rec["skip_reason"] = "MISSING_US500_ENTRY"
                elif pd.isna(endpoints.loc[monday].get("exit_bid")):
                    rec["skip_reason"] = "MISSING_US500_EXIT"
                else:
                    rec["entry_bid"] = float(endpoints.loc[monday, "entry_bid"])
                    rec["exit_bid"] = float(endpoints.loc[monday, "exit_bid"])
                    rec["gross_points"] = rec["direction"] * (rec["exit_bid"] - rec["entry_bid"])
                    rec["base_net_points"] = rec["gross_points"] - 0.51
                    rec["stress_net_points"] = rec["gross_points"] - 1.02
                    rec["skip_reason"] = ""
        records.append(rec)
    signals = pd.DataFrame(records)
    trades = signals.loc[signals.skip_reason.eq("")].copy()
    trades["year"] = pd.to_datetime(trades.ny_date).dt.year
    cal = trades.loc[trades.year <= 2022]
    gated = trades.loc[trades.year >= 2023]
    annual = {str(y): stats(part) for y, part in trades.groupby("year")}
    pval = bootstrap_p(gated.base_net_points.to_numpy(dtype=float)) if len(gated) else 1.0
    gates = {
        "G-BW1": bool(len(gated) >= 150 and gated.ny_date.nunique() == len(gated)),
        "G-BW2": bool(len(gated) and gated.stress_net_points.mean() > 0),
        "G-BW3": bool(pval <= 0.01),
        "G-BW4": all(str(y) in annual and annual[str(y)]["base_total_points"] > 0 for y in (2023, 2024, 2025)),
    }
    result = {
        "source": source, "us500_sha256": us500_sha,
        "spec_sha256": sha256(OUT / "SPEC_BTC_WEEKEND_MONDAY.txt"),
        "gate_sha256": sha256(OUT / "GATE_BTC_WEEKEND_MONDAY.txt"),
        "candidate_mondays": len(signals), "executed_mondays": len(trades),
        "skip_reasons": signals.loc[signals.skip_reason.ne(""), "skip_reason"].value_counts().to_dict(),
        "calibration_2021_2022": stats(cal), "gated_2023_plus": stats(gated),
        "bootstrap_p_nonpositive": pval, "annual": annual, "gates": gates,
        "verdict": "PASS" if all(gates.values()) else "FAIL",
        "caveat": "BID M5 and fixed spread proxy; no native execution, stop, floating loss, FTMO feed or Challenge pace validation. Repeated historical project search requires forward confirmation even on Phase-1 PASS.",
    }
    signals.to_csv(OUT / "monday_signals.csv", index=False)
    trades.to_csv(OUT / "monday_trades.csv", index=False)
    (OUT / "run_meta.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_path.write_text("REPORT: HYPOTHESIS 77 BTC WEEKEND -> MONDAY US500 CASH\n"
                           + json.dumps(result, indent=2, allow_nan=False)
                           + "\nFINAL PHASE-1 VERDICT: " + result["verdict"] + "\n", encoding="utf-8")
    print("FINAL PHASE-1 VERDICT", result["verdict"])
    print("GATES", gates)
    print("FUNNEL", len(signals), len(trades), result["skip_reasons"])
    print("GATED", result["gated_2023_plus"], "p", pval)
    print("ANNUAL", {year: round(row["base_total_points"], 3) for year, row in annual.items()})


if __name__ == "__main__":
    main()
